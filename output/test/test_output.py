#!/usr/bin/env python3
"""
Test the final generated output/<class_name>.cpp with optional runtime payload inputs.

This script does not regenerate code. It compiles the current generated C++ file,
optionally injects payload mass/COM through RTOS-style `parms`, and compares the
generated-code output against the Python runtime dynamics reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "python"))

from load_model import URDFLoader
from output.test.test_cpp_consistency import (  # type: ignore
    _compile_harness,
    _run_harness,
    _summarize_difference,
    _summarize_prediction_quality,
    _write_harness_input,
)
from output.test.test_cpp_payload_override import (  # type: ignore
    _apply_payload_to_rtos_last_link,
    _build_rtos_parms,
    _rtos_parms_to_parameter_vector,
)
from run_pipeline import (
    DEFAULT_CUTOFF_HZ,
    DEFAULT_EXPORT_CLASS_NAME,
    DEFAULT_GRAVITY_VECTOR,
    DEFAULT_REAL_DATA_DIR,
    DEFAULT_REAL_TORQUE_SOURCE,
    DEFAULT_SAMPLING_FREQ,
    DEFAULT_URDF_PATH,
    _prepare_real_dataset,
    _split_dataframe_for_learning,
)
from runtime_dynamics import GravityConfig, InertialParameterVector, RobotDynamicsModel


def _detect_generated_class_name() -> str | None:
    cpp_files = sorted(
        path for path in (project_root / "output").glob("*.cpp")
        if path.is_file()
    )
    if len(cpp_files) == 1:
        return cpp_files[0].stem
    return None


def _load_parameter_vector(theta_path: Path, num_joints: int) -> InertialParameterVector:
    payload = json.loads(theta_path.read_text(encoding="utf-8"))
    theta_full = np.asarray(payload["theta_hat_full"], dtype=float)
    return InertialParameterVector.from_theta_full(theta_full, num_joints)


def _format_joint_values(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile and test the final generated output/<class_name>.cpp with optional runtime payload overrides."
    )
    parser.add_argument("--export-class-name", default=None)
    parser.add_argument("--urdf-path", default=str(DEFAULT_URDF_PATH))
    parser.add_argument("--theta-path", default=str(project_root / "datasets" / "identified" / "theta_hat_real_sensed_latest.json"))
    parser.add_argument("--real-data-dir", default=str(DEFAULT_REAL_DATA_DIR))
    parser.add_argument("--real-torque-source", default=DEFAULT_REAL_TORQUE_SOURCE)
    parser.add_argument("--sampling-freq", type=float, default=DEFAULT_SAMPLING_FREQ)
    parser.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ)
    parser.add_argument("--gravity", nargs=3, type=float, default=DEFAULT_GRAVITY_VECTOR, metavar=("GX", "GY", "GZ"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sample-limit", type=int, default=128)
    parser.add_argument("--payload-mass", type=float, default=0.0)
    parser.add_argument("--payload-com", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("CX", "CY", "CZ"))
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    export_class_name = args.export_class_name or _detect_generated_class_name() or DEFAULT_EXPORT_CLASS_NAME
    generated_cpp_path = project_root / "output" / f"{export_class_name}.cpp"
    generated_header_path = project_root / "output" / f"{export_class_name}.h"
    if not generated_cpp_path.exists() or not generated_header_path.exists():
        print(f"未找到生成文件：{generated_cpp_path.name} / {generated_header_path.name}")
        print("请先运行 run_pipeline.py 生成最终 output 代码。")
        return 1

    loader = URDFLoader(str(Path(args.urdf_path)), gravity_vector=list(args.gravity))
    robot_model = loader.build_robot_model()
    runtime_model = RobotDynamicsModel(robot_model)

    parameter_vector = _load_parameter_vector(Path(args.theta_path), robot_model.num_joints)
    base_parms = _build_rtos_parms(parameter_vector)
    active_parms = np.asarray(base_parms, dtype=float)
    using_payload_override = float(args.payload_mass) > 0.0
    if using_payload_override:
        active_parms = _apply_payload_to_rtos_last_link(
            active_parms,
            mass=float(args.payload_mass),
            com=np.asarray(args.payload_com, dtype=float),
        )

    active_parameter_vector = _rtos_parms_to_parameter_vector(active_parms, robot_model.num_joints)

    df_processed = _prepare_real_dataset(
        robot_model,
        Path(args.real_data_dir),
        args.sampling_freq,
        args.cutoff_hz,
        args.real_torque_source,
    )
    split_df = _split_dataframe_for_learning(df_processed)[args.split].reset_index(drop=True)
    if args.sample_limit > 0:
        split_df = split_df.head(args.sample_limit).copy()

    q_matrix = np.column_stack([split_df[f"q_{joint_idx}"].values for joint_idx in range(1, 8)])
    dq_matrix = np.column_stack([split_df[f"dq_{joint_idx}"].values for joint_idx in range(1, 8)])
    ddq_matrix = np.column_stack([split_df[f"ddq_{joint_idx}"].values for joint_idx in range(1, 8)])

    python_prediction = np.zeros((len(split_df), robot_model.num_joints), dtype=float)
    gravity_config = GravityConfig.from_any(args.gravity)
    for sample_idx in range(len(split_df)):
        python_prediction[sample_idx, :] = runtime_model.compute_tau_robot(
            q_matrix[sample_idx],
            dq_matrix[sample_idx],
            ddq_matrix[sample_idx],
            active_parameter_vector,
            gravity_config=gravity_config,
        )

    with tempfile.TemporaryDirectory(prefix="test_output_generated_", dir=str(project_root / "output" / "test")) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        input_path = temp_dir / "samples.txt"
        parms_path = temp_dir / "parms.txt"
        binary_path = temp_dir / "generated_dynamics_harness"
        _write_harness_input(split_df, input_path)
        if using_payload_override:
            parms_path.write_text(" ".join(f"{value:.17g}" for value in active_parms), encoding="utf-8")
        _compile_harness(binary_path, generated_cpp_path)
        cpp_prediction = _run_harness(
            binary_path,
            input_path,
            list(args.gravity),
            parms_path if using_payload_override else None,
        )

    if cpp_prediction.shape != python_prediction.shape:
        raise RuntimeError(
            f"Shape mismatch between generated C++ and Python reference: "
            f"{cpp_prediction.shape} vs {python_prediction.shape}"
        )

    diff_summary = _summarize_difference(python_prediction, cpp_prediction)
    quality_summary = _summarize_prediction_quality(split_df, cpp_prediction)

    print("=" * 72)
    print("final generated dynamics test")
    print("=" * 72)
    print(f"Generated C++: {generated_cpp_path}")
    print(f"Split: {args.split}")
    print(f"Samples compared: {len(split_df)}")
    print(f"Gravity: {list(args.gravity)}")
    print(f"Runtime payload override: {'enabled' if using_payload_override else 'disabled'}")
    if using_payload_override:
        print(f"Payload mass: {args.payload_mass}")
        print(f"Payload COM: {list(args.payload_com)}")
    print(f"Max abs error (C++ vs Python): {diff_summary['max_abs_error']:.12e}")
    print(f"RMSE (C++ vs Python):          {diff_summary['rmse']:.12e}")
    print(f"MAE  (C++ vs Python):          {diff_summary['mae']:.12e}")
    print(
        "Per-joint max abs error: "
        + ", ".join(f"J{idx + 1}={value:.3e}" for idx, value in enumerate(diff_summary["joint_max_abs_error"]))
    )
    worst_sample = diff_summary["worst_sample_index"]
    print(
        "Worst sample inputs: "
        f"q={_format_joint_values(q_matrix[worst_sample])}, "
        f"dq={_format_joint_values(dq_matrix[worst_sample])}, "
        f"ddq={_format_joint_values(ddq_matrix[worst_sample])}"
    )
    print(f"Worst sample Python tau: {_format_joint_values(python_prediction[worst_sample])}")
    print(f"Worst sample C++ tau:    {_format_joint_values(cpp_prediction[worst_sample])}")
    print(f"Global RMSE vs measured tau:   {quality_summary['global_rmse_vs_measured']:.6f} N·m")
    print(f"Global MAE  vs measured tau:   {quality_summary['global_mae_vs_measured']:.6f} N·m")
    print("=" * 72)

    if diff_summary["max_abs_error"] > args.tolerance:
        print(
            f"FAILED: max_abs_error={diff_summary['max_abs_error']:.12e} exceeds tolerance "
            f"{args.tolerance:.12e}"
        )
        return 1

    print(f"PASSED: final generated code matches Python reference within tolerance {args.tolerance:.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
