#!/usr/bin/env python3
"""
真正编译并调用当前导出的 output/<class_name>.cpp，然后与 Python 参考实现逐样本对比。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "python"))

from export_inverse_dynamics_code import export_identified_inverse_dynamics_cpp
from identify_parameters import ParameterIdentifier
from load_model import URDFLoader
from run_pipeline import (
    DEFAULT_CUTOFF_HZ,
    DEFAULT_DATA_SOURCE,
    DEFAULT_EXPORT_CLASS_NAME,
    DEFAULT_GRAVITY_VECTOR,
    DEFAULT_IDENTIFICATION_MODE,
    DEFAULT_PARAMETERIZATION,
    DEFAULT_REAL_DATA_DIR,
    DEFAULT_REAL_TORQUE_SOURCE,
    DEFAULT_SAMPLING_FREQ,
    DEFAULT_URDF_PATH,
    _prepare_real_dataset,
    _prepare_synthetic_dataset,
    _resolve_data_source,
    _split_dataframe_for_learning,
)


def _build_reference_pipeline(
    *,
    data_source: str,
    real_data_dir: Path,
    real_torque_source: str,
    sampling_freq: float,
    cutoff_hz: float,
    urdf_path: Path,
    gravity_vector: list[float],
    parameterization: str,
    identification_mode: str,
    export_class_name: str,
):
    effective_data_source = _resolve_data_source(data_source, real_data_dir)

    loader = URDFLoader(str(urdf_path), None, gravity_vector=gravity_vector)
    robot_model = loader.build_robot_model()

    if effective_data_source == "real":
        df_processed = _prepare_real_dataset(
            robot_model,
            real_data_dir,
            sampling_freq,
            cutoff_hz,
            real_torque_source,
        )
        result_stem = f"real_{real_torque_source}"
    else:
        df_processed = _prepare_synthetic_dataset(robot_model, sampling_freq, cutoff_hz)
        result_stem = "synthetic"

    splits = _split_dataframe_for_learning(df_processed)

    identifier = ParameterIdentifier(robot_model, parameterization=parameterization)
    result = identifier.identify_parameters(
        splits["train"],
        method="ridge",
        reference_parameters=None,
        ridge_lambda=1e-4,
    )

    generated = export_identified_inverse_dynamics_cpp(
        robot_model=robot_model,
        result=result,
        urdf_path=urdf_path,
        output_dir=project_root / "output",
        result_stem=result_stem,
        class_name=export_class_name,
        generation_metadata={
            "generated_at": "cpp_consistency_test",
            "robot_name": robot_model.name,
            "data_source": effective_data_source,
            "real_torque_source": real_torque_source if effective_data_source == "real" else None,
            "parameterization": parameterization,
            "identification_mode": identification_mode,
            "nonlinear_compensation": "disabled_for_export",
        },
    )

    return {
        "robot_model": robot_model,
        "effective_data_source": effective_data_source,
        "processed_df": df_processed,
        "splits": splits,
        "identifier": identifier,
        "result": result,
        "generated": generated,
    }


def _predict_python(split_df, identifier, result) -> np.ndarray:
    return identifier.predict_torques(split_df, result)


def _write_harness_input(split_df, destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as handle:
        for _, row in split_df.iterrows():
            values = []
            for prefix in ("q", "dq", "ddq"):
                for joint_idx in range(1, 8):
                    values.append(f"{float(row[f'{prefix}_{joint_idx}']):.17g}")
            handle.write(" ".join(values))
            handle.write("\n")


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
        str(project_root / "output"),
        str(generated_cpp_path),
        str(project_root / "output" / "test" / "cpp_harness.cpp"),
        "-o",
        str(binary_path),
    ]
    subprocess.run(command, cwd=project_root, check=True)


def _run_harness(
    binary_path: Path,
    input_path: Path,
    gravity_vector: list[float],
    parms_path: Path | None = None,
) -> np.ndarray:
    command = [
        str(binary_path),
        str(input_path),
        f"{gravity_vector[0]:.17g}",
        f"{gravity_vector[1]:.17g}",
        f"{gravity_vector[2]:.17g}",
    ]
    if parms_path is not None:
        command.append(str(parms_path))
    result = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        if not tokens or tokens[0] != "tau_pred":
            continue
        rows.append([float(value) for value in tokens[1:]])
    return np.asarray(rows, dtype=float)


def _summarize_difference(python_prediction: np.ndarray, cpp_prediction: np.ndarray) -> dict[str, float]:
    diff = cpp_prediction - python_prediction
    flat_index = int(np.argmax(np.abs(diff)))
    sample_index, joint_index = np.unravel_index(flat_index, diff.shape)
    joint_max_abs = np.max(np.abs(diff), axis=0)
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "worst_sample_index": int(sample_index),
        "worst_joint_index": int(joint_index),
        "worst_signed_error": float(diff[sample_index, joint_index]),
        "joint_max_abs_error": joint_max_abs.tolist(),
    }


def _summarize_prediction_quality(split_df, prediction: np.ndarray) -> dict[str, float]:
    tau_measured = np.column_stack([split_df[f"tau_{joint_idx}"].values for joint_idx in range(1, 8)])
    error = prediction - tau_measured
    return {
        "global_rmse_vs_measured": float(np.sqrt(np.mean(error ** 2))),
        "global_mae_vs_measured": float(np.mean(np.abs(error))),
    }


def _format_joint_values(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and validate the current generated output/<class_name>.cpp against Python reference.")
    parser.add_argument("--data-source", default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--real-data-dir", default=str(DEFAULT_REAL_DATA_DIR))
    parser.add_argument("--real-torque-source", default=DEFAULT_REAL_TORQUE_SOURCE)
    parser.add_argument("--sampling-freq", type=float, default=DEFAULT_SAMPLING_FREQ)
    parser.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ)
    parser.add_argument("--urdf-path", default=str(DEFAULT_URDF_PATH))
    parser.add_argument("--gravity", nargs=3, type=float, default=DEFAULT_GRAVITY_VECTOR, metavar=("GX", "GY", "GZ"))
    parser.add_argument("--parameterization", default=DEFAULT_PARAMETERIZATION)
    parser.add_argument("--identification-mode", default=DEFAULT_IDENTIFICATION_MODE)
    parser.add_argument("--export-class-name", default=DEFAULT_EXPORT_CLASS_NAME)
    parser.add_argument("--query-gravity", nargs=3, type=float, default=None, metavar=("QGX", "QGY", "QGZ"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sample-limit", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    pipeline = _build_reference_pipeline(
        data_source=args.data_source,
        real_data_dir=Path(args.real_data_dir),
        real_torque_source=args.real_torque_source,
        sampling_freq=args.sampling_freq,
        cutoff_hz=args.cutoff_hz,
        urdf_path=Path(args.urdf_path),
        gravity_vector=list(args.gravity),
        parameterization=args.parameterization,
        identification_mode=args.identification_mode,
        export_class_name=args.export_class_name,
    )

    split_df = pipeline["splits"][args.split].reset_index(drop=True)
    if args.sample_limit > 0:
        split_df = split_df.head(args.sample_limit).copy()
    query_gravity = list(args.query_gravity) if args.query_gravity is not None else list(args.gravity)
    split_df['gravity_x'] = float(query_gravity[0])
    split_df['gravity_y'] = float(query_gravity[1])
    split_df['gravity_z'] = float(query_gravity[2])

    python_prediction = _predict_python(
        split_df,
        pipeline["identifier"],
        pipeline["result"],
    )

    with tempfile.TemporaryDirectory(prefix="identified_dynamics_test_", dir=str(project_root / "output" / "test")) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        input_path = temp_dir / "samples.txt"
        binary_path = temp_dir / "identified_dynamics_harness"
        _write_harness_input(split_df, input_path)
        _compile_harness(binary_path, Path(pipeline["generated"]["project_cpp"]))
        cpp_prediction = _run_harness(binary_path, input_path, query_gravity)

    if cpp_prediction.shape != python_prediction.shape:
        raise RuntimeError(
            f"Shape mismatch between C++ and Python predictions: "
            f"{cpp_prediction.shape} vs {python_prediction.shape}"
        )

    diff_summary = _summarize_difference(python_prediction, cpp_prediction)
    quality_summary = _summarize_prediction_quality(split_df, cpp_prediction)

    print("=" * 72)
    print("identifiedDynamics.cpp consistency check")
    print("=" * 72)
    print(f"Data source: {pipeline['effective_data_source']}")
    print(f"Split: {args.split}")
    print(f"Samples compared: {len(split_df)}")
    print(f"Train gravity: {list(args.gravity)}")
    print(f"Query gravity: {query_gravity}")
    print(f"Generated C++: {pipeline['generated']['project_cpp']}")
    print(f"Max abs error (C++ vs Python): {diff_summary['max_abs_error']:.12e}")
    print(f"RMSE (C++ vs Python):          {diff_summary['rmse']:.12e}")
    print(f"MAE  (C++ vs Python):          {diff_summary['mae']:.12e}")
    print(
        "Worst mismatch: "
        f"sample={diff_summary['worst_sample_index']}, "
        f"joint={diff_summary['worst_joint_index'] + 1}, "
        f"signed_error={diff_summary['worst_signed_error']:.12e}"
    )
    print(
        "Per-joint max abs error: "
        + ", ".join(f"J{idx + 1}={value:.3e}" for idx, value in enumerate(diff_summary["joint_max_abs_error"]))
    )
    worst_sample = diff_summary["worst_sample_index"]
    print(
        "Worst sample inputs: "
        f"q={_format_joint_values(split_df.loc[worst_sample, [f'q_{idx}' for idx in range(1, 8)]].to_numpy())}, "
        f"dq={_format_joint_values(split_df.loc[worst_sample, [f'dq_{idx}' for idx in range(1, 8)]].to_numpy())}, "
        f"ddq={_format_joint_values(split_df.loc[worst_sample, [f'ddq_{idx}' for idx in range(1, 8)]].to_numpy())}"
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

    print(f"PASSED: C++ output matches Python reference within tolerance {args.tolerance:.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
