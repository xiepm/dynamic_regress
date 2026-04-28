#!/usr/bin/env python3
"""
Side-by-side comparison of ridge+base vs constrained+full identification.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "python"))

from export_inverse_dynamics_code import export_identified_inverse_dynamics_cpp
from load_model import URDFLoader
from runtime_dynamics import GravityConfig, InertialParameterVector, PayloadMode
import pipeline_identification as pipeline_id
import pipeline_postprocess as pipeline_post
from run_pipeline import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CUTOFF_HZ,
    DEFAULT_GRAVITY_VECTOR,
    DEFAULT_REAL_DATA_DIR,
    DEFAULT_REAL_TORQUE_SOURCE,
    DEFAULT_SAMPLING_FREQ,
    DEFAULT_URDF_PATH,
    _prepare_real_dataset,
    _prepare_synthetic_dataset,
    _resolve_data_source,
    _split_dataframe_for_learning,
)


def _compile_harness(binary_path: Path, generated_cpp_path: Path) -> None:
    generated_header_name = generated_cpp_path.with_suffix(".h").name
    generated_class_name = generated_cpp_path.stem
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        f"-DGENERATED_DYNAMICS_HEADER=\"{generated_header_name}\"",
        f"-DGENERATED_DYNAMICS_CLASS={generated_class_name}",
        "-I",
        str(project_root / "output" / "test" / "rtos_stub"),
        "-I",
        str(generated_cpp_path.parent),
        str(generated_cpp_path),
        str(project_root / "output" / "test" / "cpp_harness.cpp"),
        "-o",
        str(binary_path),
    ]
    subprocess.run(command, cwd=project_root, check=True)


def _write_single_sample(input_path: Path, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> None:
    with input_path.open("w", encoding="utf-8") as handle:
        values = [*q.tolist(), *dq.tolist(), *ddq.tolist()]
        handle.write(" ".join(f"{value:.17g}" for value in values))
        handle.write("\n")


def _run_harness(binary_path: Path, input_path: Path, gravity: np.ndarray) -> dict[str, list[float]]:
    result = subprocess.run(
        [
            str(binary_path),
            str(input_path),
            f"{float(gravity[0]):.17g}",
            f"{float(gravity[1]):.17g}",
            f"{float(gravity[2]):.17g}",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    parsed: dict[str, list[float]] = {}
    for raw_line in result.stdout.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        parsed[parts[0]] = [float(value) for value in parts[1:]]
    return parsed


def _joint_feasibility(parameter_vector: InertialParameterVector) -> list[dict]:
    rows = []
    for joint_idx in range(parameter_vector.num_joints):
        phi = parameter_vector.get_link_phi(joint_idx)
        inertia = np.array(
            [
                [phi[4], phi[5], phi[6]],
                [phi[5], phi[7], phi[8]],
                [phi[6], phi[8], phi[9]],
            ],
            dtype=float,
        )
        eigvals = np.linalg.eigvalsh(0.5 * (inertia + inertia.T))
        rows.append(
            {
                "joint": f"J{joint_idx + 1}",
                "mass": float(phi[0]),
                "mass_positive": bool(phi[0] > 0.0),
                "inertia_psd": bool(np.all(eigvals >= -1e-9)),
                "triangle_ok": bool(
                    (phi[7] + phi[9] - phi[4] >= -1e-9)
                    and (phi[4] + phi[9] - phi[7] >= -1e-9)
                    and (phi[4] + phi[7] - phi[9] >= -1e-9)
                ),
                "min_inertia_eig": float(np.min(eigvals)),
            }
        )
    return rows


def _prepare_dataset(args, robot_model):
    effective_data_source = _resolve_data_source(args.data_source, Path(args.real_data_dir))
    if effective_data_source == "real":
        df_processed = _prepare_real_dataset(
            robot_model,
            Path(args.real_data_dir),
            args.sampling_freq,
            args.cutoff_hz,
            args.real_torque_source,
        )
    else:
        df_processed = _prepare_synthetic_dataset(robot_model, args.sampling_freq, args.cutoff_hz)
    splits = _split_dataframe_for_learning(df_processed, seed=args.seed)
    return effective_data_source, df_processed, splits


def _run_mode(
    *,
    robot_model,
    splits,
    effective_data_source: str,
    real_torque_source: str,
    gravity,
    parameterization: str,
    solver_method: str,
    identification_mode: str,
):
    bundle = pipeline_id.run_identification_stage(
        robot_model=robot_model,
        splits=splits,
        effective_data_source=effective_data_source,
        parameterization=parameterization,
        identification_mode=identification_mode,
        solver_method=solver_method,
        real_torque_source=real_torque_source,
        payload_model=None,
        effective_payload_mode=PayloadMode.NONE,
        subtract_known_payload_gravity=False,
    )
    split_eval = pipeline_id.evaluate_identification_splits(bundle["identifier"], bundle["result"], splits)
    overfit = pipeline_id.summarize_overfitting(split_eval)
    return bundle, split_eval, overfit


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ridge+base and constrained+full on the same dataset.")
    parser.add_argument("--data-source", default="real", choices=["auto", "real", "synthetic"])
    parser.add_argument("--real-data-dir", default=str(DEFAULT_REAL_DATA_DIR))
    parser.add_argument("--real-torque-source", default=DEFAULT_REAL_TORQUE_SOURCE, choices=["sensed", "sensor"])
    parser.add_argument("--sampling-freq", type=float, default=DEFAULT_SAMPLING_FREQ)
    parser.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ)
    parser.add_argument("--urdf-path", default=str(DEFAULT_URDF_PATH))
    parser.add_argument("--config-path", default=None if DEFAULT_CONFIG_PATH is None else str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--gravity", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sample-index", type=int, default=0, help="Index within the test split for single-sample output comparison.")
    args = parser.parse_args()

    effective_gravity = args.gravity or DEFAULT_GRAVITY_VECTOR
    loader = URDFLoader(
        str(Path(args.urdf_path)),
        str(Path(args.config_path)) if args.config_path else None,
        gravity_vector=effective_gravity,
    )
    robot_model = loader.build_robot_model()
    effective_data_source, df_processed, splits = _prepare_dataset(args, robot_model)

    ridge_bundle, ridge_eval, ridge_overfit = _run_mode(
        robot_model=robot_model,
        splits=splits,
        effective_data_source=effective_data_source,
        real_torque_source=args.real_torque_source,
        gravity=effective_gravity,
        parameterization="base",
        solver_method="ridge",
        identification_mode="rigid_body_friction",
    )
    constrained_bundle, constrained_eval, constrained_overfit = _run_mode(
        robot_model=robot_model,
        splits=splits,
        effective_data_source=effective_data_source,
        real_torque_source=args.real_torque_source,
        gravity=effective_gravity,
        parameterization="full",
        solver_method="constrained",
        identification_mode="rigid_body_friction",
    )

    summary_table = pd.DataFrame(
        [
            {
                "mode": "ridge+base",
                "train_rmse": ridge_eval["train"]["global_rmse"],
                "val_rmse": ridge_eval["val"]["global_rmse"],
                "test_rmse": ridge_eval["test"]["global_rmse"],
                "physical_sanity": bool(ridge_bundle["result"]["physical_sanity"]["is_valid"]),
                "optimizer_success": bool(ridge_bundle["result"]["optimizer_success"]),
            },
            {
                "mode": "constrained+full",
                "train_rmse": constrained_eval["train"]["global_rmse"],
                "val_rmse": constrained_eval["val"]["global_rmse"],
                "test_rmse": constrained_eval["test"]["global_rmse"],
                "physical_sanity": bool(constrained_bundle["result"]["physical_sanity"]["is_valid"]),
                "optimizer_success": bool(constrained_bundle["result"]["optimizer_success"]),
            },
        ]
    )

    test_df = splits["test"].reset_index(drop=True)
    if not 0 <= args.sample_index < len(test_df):
        raise IndexError(f"sample_index={args.sample_index} out of range for test split with {len(test_df)} rows.")
    sample = test_df.iloc[int(args.sample_index)]
    q = np.asarray([sample[f"q_{idx}"] for idx in range(1, robot_model.num_joints + 1)], dtype=float)
    dq = np.asarray([sample[f"dq_{idx}"] for idx in range(1, robot_model.num_joints + 1)], dtype=float)
    ddq = np.asarray([sample[f"ddq_{idx}"] for idx in range(1, robot_model.num_joints + 1)], dtype=float)
    gravity = np.asarray(
        [sample.get("gravity_x", effective_gravity[0]), sample.get("gravity_y", effective_gravity[1]), sample.get("gravity_z", effective_gravity[2])],
        dtype=float,
    )
    tau_measured = np.asarray([sample[f"tau_{idx}"] for idx in range(1, robot_model.num_joints + 1)], dtype=float)

    compare_root = project_root / "output" / "solver_mode_compare"
    ridge_dir = compare_root / "ridge_base"
    constrained_dir = compare_root / "constrained_full"
    ridge_dir.mkdir(parents=True, exist_ok=True)
    constrained_dir.mkdir(parents=True, exist_ok=True)

    ridge_paths = export_identified_inverse_dynamics_cpp(
        robot_model=robot_model,
        result=ridge_bundle["result"],
        urdf_path=Path(args.urdf_path),
        output_dir=ridge_dir,
        result_stem="ridge_base",
        class_name="ridgeBaseCompareDynamics",
        parameter_vector=ridge_bundle["result"]["parameter_vector"],
        gravity_config=GravityConfig.from_any(gravity),
    )
    constrained_paths = export_identified_inverse_dynamics_cpp(
        robot_model=robot_model,
        result=constrained_bundle["result"],
        urdf_path=Path(args.urdf_path),
        output_dir=constrained_dir,
        result_stem="constrained_full",
        class_name="constrainedFullCompareDynamics",
        parameter_vector=constrained_bundle["result"]["parameter_vector"],
        gravity_config=GravityConfig.from_any(gravity),
    )

    with tempfile.TemporaryDirectory(prefix="solver_mode_compare_", dir=str(compare_root)) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        input_path = temp_dir / "sample.txt"
        ridge_bin = temp_dir / "ridge_compare"
        constrained_bin = temp_dir / "constrained_compare"
        _write_single_sample(input_path, q, dq, ddq)
        _compile_harness(ridge_bin, ridge_paths["project_cpp"])
        _compile_harness(constrained_bin, constrained_paths["project_cpp"])
        ridge_output = _run_harness(ridge_bin, input_path, gravity)
        constrained_output = _run_harness(constrained_bin, input_path, gravity)

    sample_compare = pd.DataFrame(
        {
            "tau_measured": tau_measured,
            "ridge_tau_pred": ridge_output["tau_pred"],
            "constrained_tau_pred": constrained_output["tau_pred"],
            "ridge_error": tau_measured - np.asarray(ridge_output["tau_pred"], dtype=float),
            "constrained_error": tau_measured - np.asarray(constrained_output["tau_pred"], dtype=float),
        },
        index=[f"J{idx}" for idx in range(1, robot_model.num_joints + 1)],
    )

    ridge_joint = pd.DataFrame(_joint_feasibility(ridge_bundle["result"]["parameter_vector"])).set_index("joint")
    constrained_joint = pd.DataFrame(_joint_feasibility(constrained_bundle["result"]["parameter_vector"])).set_index("joint")
    joint_compare = pd.concat(
        [
            ridge_joint.add_prefix("ridge_"),
            constrained_joint.add_prefix("constrained_"),
        ],
        axis=1,
    )

    print("\n" + "=" * 88)
    print("Solver Mode Comparison")
    print("=" * 88)
    print(f"data_source:         {effective_data_source}")
    print(f"real_torque_source:  {args.real_torque_source if effective_data_source == 'real' else 'synthetic'}")
    print(f"gravity_base:        {np.asarray(gravity, dtype=float).tolist()}")
    print(f"test_sample_index:   {args.sample_index}")
    print("\nRMSE / sanity summary:")
    print(summary_table.round(6).to_string(index=False))
    print("\nPer-joint feasibility:")
    print(joint_compare.round(6).to_string())
    print("\nSingle-sample torque comparison (exported C++ output):")
    print(sample_compare.round(6).to_string())

    diagnostics_dir = project_root / "output" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": effective_data_source,
        "real_torque_source": args.real_torque_source if effective_data_source == "real" else None,
        "gravity_base": gravity.tolist(),
        "summary_table": summary_table.to_dict(orient="records"),
        "joint_compare": joint_compare.reset_index().to_dict(orient="records"),
        "sample_compare": sample_compare.reset_index().to_dict(orient="records"),
        "ridge_overfit": ridge_overfit,
        "constrained_overfit": constrained_overfit,
        "ridge_export": {key: str(value) for key, value in ridge_paths.items()},
        "constrained_export": {key: str(value) for key, value in constrained_paths.items()},
    }
    output_path = diagnostics_dir / "solver_mode_comparison.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved comparison payload: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
