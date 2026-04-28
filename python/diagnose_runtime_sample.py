#!/usr/bin/env python3
"""
Single-sample runtime dynamics diagnostic tool.

This script is meant for real-robot comparison rather than aggregate RMSE-only
testing. Given one state sample (either typed in manually or selected from a
processed real-data CSV), it reports:

- Y_M, Y_C, Y_Gx, Y_Gy, Y_Gz, Y_F
- per-component torque contributions
- total predicted torque
- optional measured torque and prediction error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "python"))

from load_model import URDFLoader
from run_pipeline import DEFAULT_URDF_PATH
from runtime_dynamics import GravityConfig, InertialParameterVector, RobotDynamicsModel


def _default_identified_json(root: Path) -> Path:
    identified_dir = root / "datasets" / "identified"
    preferred = [
        identified_dir / "theta_hat_real_sensed_latest.json",
        identified_dir / "theta_hat_real_sensor_latest.json",
        identified_dir / "theta_hat_synthetic_latest.json",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate
    matches = sorted(identified_dir.glob("theta_hat_*_latest.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No identified parameter JSON found in {identified_dir}. "
        "Run run_pipeline.py first or pass --identified-json explicitly."
    )


def _default_processed_path(root: Path) -> Path:
    preferred = [
        root / "datasets" / "real" / "processed" / "real_sensed_combined_processed.csv",
        root / "datasets" / "real" / "processed" / "real_sensor_combined_processed.csv",
        root / "datasets" / "synthetic" / "processed" / "synthetic_panda_run01_proc.csv",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No processed dataset found. Pass --processed-path explicitly.")


def _vector_from_row(row: pd.Series, prefix: str, num_joints: int) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}_{joint_idx}"]) for joint_idx in range(1, num_joints + 1)], dtype=float)


def _load_processed_sample(processed_path: Path, row_index: int, num_joints: int) -> dict[str, Any]:
    df = pd.read_csv(processed_path)
    if not 0 <= row_index < len(df):
        raise IndexError(f"row_index={row_index} out of range for processed dataset with {len(df)} rows.")
    row = df.iloc[int(row_index)]
    gravity = (
        np.asarray([float(row["gravity_x"]), float(row["gravity_y"]), float(row["gravity_z"])], dtype=float)
        if all(column in row.index for column in ("gravity_x", "gravity_y", "gravity_z"))
        else None
    )
    tau_measured = (
        np.asarray([float(row[f"tau_{joint_idx}"]) for joint_idx in range(1, num_joints + 1)], dtype=float)
        if all(f"tau_{joint_idx}" in row.index for joint_idx in range(1, num_joints + 1))
        else None
    )
    return {
        "source": "processed_row",
        "processed_path": str(processed_path),
        "row_index": int(row_index),
        "source_file": row.get("source_file"),
        "trajectory_id": row.get("trajectory_id"),
        "q": _vector_from_row(row, "q", num_joints),
        "dq": _vector_from_row(row, "dq", num_joints),
        "ddq": _vector_from_row(row, "ddq", num_joints),
        "gravity": gravity,
        "tau_measured": tau_measured,
    }


def _load_manual_sample(args: argparse.Namespace, num_joints: int) -> dict[str, Any]:
    q = np.asarray(args.q, dtype=float).reshape(num_joints)
    dq = np.asarray(args.dq, dtype=float).reshape(num_joints)
    ddq = np.asarray(args.ddq, dtype=float).reshape(num_joints)
    gravity = np.asarray(args.gravity, dtype=float).reshape(3) if args.gravity is not None else None
    tau_measured = np.asarray(args.tau_measured, dtype=float).reshape(num_joints) if args.tau_measured is not None else None
    return {
        "source": "manual",
        "q": q,
        "dq": dq,
        "ddq": ddq,
        "gravity": gravity,
        "tau_measured": tau_measured,
    }


def _format_vector(vector: np.ndarray) -> str:
    return np.array2string(np.asarray(vector, dtype=float), precision=6, suppress_small=False, separator=", ")


def _print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"\n{name} shape={matrix.shape}")
    print(np.array2string(np.asarray(matrix, dtype=float), precision=6, suppress_small=False, max_line_width=160))


def _prompt_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    prompt_text = f"{prompt}{suffix}: "
    try:
        value = input(prompt_text).strip()
    except EOFError:
        try:
            with open("/dev/tty", "r+", encoding="utf-8", errors="ignore") as tty:
                tty.write(prompt_text)
                tty.flush()
                value = tty.readline().strip()
        except OSError as exc:
            raise RuntimeError(
                "Interactive input is unavailable in the current process. "
                "If you are using `conda run`, try either:\n"
                "  1. `conda run --no-capture-output -n pinocchio_py python python/diagnose_runtime_sample.py`\n"
                "  2. `python python/diagnose_runtime_sample.py --row-index 0`\n"
                "  3. or pass explicit `--q/--dq/--ddq/--gravity` arguments.\n"
                f"Original TTY error: {exc}"
            ) from exc
    if not value and default is not None:
        return default
    return value


def _prompt_int(prompt: str, default: int | None = None) -> int:
    raw = _prompt_text(prompt, None if default is None else str(default))
    return int(raw)


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    default_text = "y" if default else "n"
    raw = _prompt_text(f"{prompt} (y/n)", default_text).lower()
    return raw in {"y", "yes", "1", "true"}


def _prompt_float_vector(prompt: str, expected_len: int, default: np.ndarray | None = None) -> np.ndarray:
    default_text = None
    if default is not None:
        default_text = " ".join(f"{float(value):.12g}" for value in np.asarray(default, dtype=float).reshape(expected_len))
    raw = _prompt_text(prompt, default_text)
    values = [float(item) for item in raw.replace(",", " ").split()]
    if len(values) != expected_len:
        raise ValueError(f"{prompt} expects {expected_len} values, got {len(values)}.")
    return np.asarray(values, dtype=float)


def _interactive_config(root: Path, num_joints: int) -> argparse.Namespace:
    print("=" * 88)
    print("Interactive runtime sample diagnostic")
    print("=" * 88)
    mode = _prompt_text("Select mode: processed row or manual", "processed").strip().lower()
    identified_default = str(_default_identified_json(root))
    urdf_default = str(DEFAULT_URDF_PATH)
    identified_json = _prompt_text("identified_json path", identified_default)
    urdf_path = _prompt_text("URDF path", urdf_default)
    show_matrices = _prompt_yes_no("Show Y matrices", default=False)
    json_output = _prompt_yes_no("Print JSON output", default=False)

    if mode in {"processed", "row", "processed_row", "p"}:
        processed_default = str(_default_processed_path(root))
        processed_path = _prompt_text("processed CSV path", processed_default)
        row_index = _prompt_int("row index", 0)
        return argparse.Namespace(
            identified_json=identified_json,
            urdf_path=urdf_path,
            processed_path=processed_path,
            row_index=row_index,
            q=None,
            dq=None,
            ddq=None,
            gravity=None,
            tau_measured=None,
            show_matrices=show_matrices,
            json_output=json_output,
        )

    if mode in {"manual", "m"}:
        q = _prompt_float_vector("q[1..7] in rad", num_joints)
        dq = _prompt_float_vector("dq[1..7] in rad/s", num_joints)
        ddq = _prompt_float_vector("ddq[1..7] in rad/s^2", num_joints)
        gravity = _prompt_float_vector("gravity [gx gy gz] in base frame", 3, default=np.array([9.81, 0.0, 0.0], dtype=float))
        tau_measured = None
        if _prompt_yes_no("Provide measured torque for comparison", default=False):
            tau_measured = _prompt_float_vector("tau_measured[1..7] in N·m", num_joints)
        return argparse.Namespace(
            identified_json=identified_json,
            urdf_path=urdf_path,
            processed_path=None,
            row_index=None,
            q=q.tolist(),
            dq=dq.tolist(),
            ddq=ddq.tolist(),
            gravity=gravity.tolist(),
            tau_measured=None if tau_measured is None else tau_measured.tolist(),
            show_matrices=show_matrices,
            json_output=json_output,
        )

    raise ValueError(f"Unsupported interactive mode: {mode!r}. Expected 'processed' or 'manual'.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose one runtime dynamics sample against model components.")
    parser.add_argument("--identified-json", default=None, help="Path to theta_hat_*.json; defaults to latest identified result.")
    parser.add_argument("--urdf-path", default=str(DEFAULT_URDF_PATH), help="URDF path used to build the robot model.")
    parser.add_argument("--processed-path", default=None, help="Processed CSV path for row-based diagnostics.")
    parser.add_argument("--row-index", type=int, default=None, help="Row index in processed CSV to inspect.")
    parser.add_argument("--q", nargs=7, type=float, help="Manual q[1..7] in rad.")
    parser.add_argument("--dq", nargs=7, type=float, help="Manual dq[1..7] in rad/s.")
    parser.add_argument("--ddq", nargs=7, type=float, help="Manual ddq[1..7] in rad/s^2.")
    parser.add_argument("--gravity", nargs=3, type=float, help="Manual base-frame gravity vector [gx gy gz].")
    parser.add_argument("--tau-measured", nargs=7, type=float, help="Optional measured torque [tau1..tau7] in N·m.")
    parser.add_argument("--show-matrices", action="store_true", help="Print Y_M / Y_C / Y_Gx / Y_Gy / Y_Gz / Y_F matrices.")
    parser.add_argument("--json-output", action="store_true", help="Dump the full diagnostic payload as JSON.")
    args = parser.parse_args()

    interactive_mode = (
        args.row_index is None
        and args.processed_path is None
        and args.q is None
        and args.dq is None
        and args.ddq is None
    )
    if interactive_mode:
        args = _interactive_config(project_root, 7)

    identified_json = Path(args.identified_json) if args.identified_json else _default_identified_json(project_root)
    identified_payload = json.loads(identified_json.read_text(encoding="utf-8"))

    loader = URDFLoader(str(Path(args.urdf_path)), gravity_vector=[9.81, 0.0, 0.0])
    robot_model = loader.build_robot_model()
    num_joints = robot_model.num_joints
    parameter_vector = InertialParameterVector.from_identification_result(identified_payload, num_joints=num_joints)
    runtime_model = RobotDynamicsModel(robot_model)

    use_row_mode = args.row_index is not None or args.processed_path is not None
    if use_row_mode:
        if args.row_index is None:
            raise ValueError("Row-based diagnostics require --row-index.")
        processed_path = Path(args.processed_path) if args.processed_path else _default_processed_path(project_root)
        sample = _load_processed_sample(processed_path, args.row_index, num_joints)
    else:
        if args.q is None or args.dq is None or args.ddq is None:
            raise ValueError("Manual diagnostics require --q, --dq, and --ddq.")
        sample = _load_manual_sample(args, num_joints)

    gravity = sample["gravity"]
    if gravity is None:
        gravity_payload = identified_payload.get("gravity_config", {}).get("gravity_vector_base")
        gravity = np.asarray(gravity_payload, dtype=float) if gravity_payload is not None else np.asarray(robot_model.gravity_vector, dtype=float)
    gravity_config = GravityConfig.from_any(gravity)

    breakdown = runtime_model.compute_tau_breakdown(
        sample["q"],
        sample["dq"],
        sample["ddq"],
        parameter_vector,
        gravity_config=gravity_config,
    )

    tau_measured = sample.get("tau_measured")
    error = None if tau_measured is None else tau_measured - breakdown["tau_total"]

    if args.json_output:
        payload = {
            "identified_json": str(identified_json),
            "sample": {
                "source": sample["source"],
                "metadata": {key: value for key, value in sample.items() if key not in {"q", "dq", "ddq", "gravity", "tau_measured"}},
                "q": sample["q"].tolist(),
                "dq": sample["dq"].tolist(),
                "ddq": sample["ddq"].tolist(),
                "gravity_vector_base": gravity_config.gravity_vector_base.tolist(),
                "tau_measured": None if tau_measured is None else tau_measured.tolist(),
            },
            "breakdown": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in breakdown.items()
            },
            "error": None if error is None else error.tolist(),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("=" * 88)
    print("Runtime Sample Diagnostic")
    print("=" * 88)
    print(f"identified_json: {identified_json}")
    print(f"sample_source:   {sample['source']}")
    if sample["source"] == "processed_row":
        print(f"processed_path:  {sample['processed_path']}")
        print(f"row_index:       {sample['row_index']}")
        print(f"source_file:     {sample.get('source_file')}")
        print(f"trajectory_id:   {sample.get('trajectory_id')}")
    print(f"gravity_base:    {_format_vector(gravity_config.gravity_vector_base)}")
    print(f"q:               {_format_vector(sample['q'])}")
    print(f"dq:              {_format_vector(sample['dq'])}")
    print(f"ddq:             {_format_vector(sample['ddq'])}")

    per_joint = pd.DataFrame(
        {
            "tau_M": breakdown["tau_M"],
            "tau_C": breakdown["tau_C"],
            "gx*Y_Gx": breakdown["tau_Gx"],
            "gy*Y_Gy": breakdown["tau_Gy"],
            "gz*Y_Gz": breakdown["tau_Gz"],
            "tau_gravity": breakdown["tau_gravity"],
            "tau_friction": breakdown["tau_friction"],
            "tau_pred": breakdown["tau_total"],
        },
        index=[f"J{joint_idx}" for joint_idx in range(1, num_joints + 1)],
    )
    if tau_measured is not None:
        per_joint["tau_measured"] = tau_measured
        per_joint["error"] = error
        per_joint["abs_error"] = np.abs(error)

    print("\nPer-joint torque breakdown [N·m]:")
    print(per_joint.round(6).to_string())

    if tau_measured is not None:
        print("\nError summary:")
        print(f"  max_abs_error: {float(np.max(np.abs(error))):.6f} N·m")
        print(f"  mean_abs_error: {float(np.mean(np.abs(error))):.6f} N·m")
        print(f"  rmse: {float(np.sqrt(np.mean(error ** 2))):.6f} N·m")

    if args.show_matrices:
        _print_matrix("Y_M", breakdown["Y_M"])
        _print_matrix("Y_C", breakdown["Y_C"])
        _print_matrix("Y_Gx", breakdown["Y_Gx"])
        _print_matrix("Y_Gy", breakdown["Y_Gy"])
        _print_matrix("Y_Gz", breakdown["Y_Gz"])
        _print_matrix("Y_F", breakdown["Y_F"])
        print("\ntheta_rigid:")
        print(np.array2string(breakdown["theta_rigid"], precision=6, suppress_small=False, max_line_width=160))
        print("\ntheta_friction:")
        print(np.array2string(breakdown["theta_friction"], precision=6, suppress_small=False, max_line_width=160))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
