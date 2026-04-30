#!/usr/bin/env python3
"""
读取静止位置 CSV，按 sevendofDynamics::calculateEstimateJointToqrues 批量计算估计扭矩，
并与表中 J1-J6 的 FilEstimateTor 列对比，结果保存到 output/test。
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]


def _deg_to_rad(value: float) -> float:
    return value * math.pi / 180.0


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


def _detect_generated_class_name() -> str | None:
    cpp_files = sorted(
        path for path in (project_root / "output").glob("*.cpp")
        if path.is_file()
    )
    if len(cpp_files) == 1:
        return cpp_files[0].stem
    return None


def _joint_columns() -> list[dict[str, str]]:
    columns = []
    for joint_idx in range(1, 7):
        columns.append(
            {
                "q": f" J{joint_idx} Actual ACS_{joint_idx}_2_ACS(deg)",
                "dq": f" J{joint_idx} Actual JointVel_{joint_idx}_8_VEL(deg/s)",
                "ddq": f" J{joint_idx} Joint FilAcc_{joint_idx}_10_VEL(deg/s)",
                "estimate": f" J{joint_idx} FilEstimateTor_{joint_idx}_16_TQR(N)",
            }
        )
    return columns


def _write_samples_from_csv(csv_path: Path, input_path: Path) -> tuple[list[float], list[list[float]]]:
    times: list[float] = []
    reference_estimates: list[list[float]] = []
    joint_columns = _joint_columns()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        with input_path.open("w", encoding="utf-8") as output_handle:
            for row in reader:
                try:
                    time_value = float(row["time"])
                    q = [_deg_to_rad(float(row[column["q"]])) for column in joint_columns] + [0.0]
                    dq = [_deg_to_rad(float(row[column["dq"]])) for column in joint_columns] + [0.0]
                    ddq = [_deg_to_rad(float(row[column["ddq"]])) for column in joint_columns] + [0.0]
                    estimate = [float(row[column["estimate"]]) for column in joint_columns]
                except (KeyError, TypeError, ValueError):
                    continue
                times.append(time_value)
                reference_estimates.append(estimate)
                values = [*q, *dq, *ddq]
                output_handle.write(" ".join(f"{value:.17g}" for value in values))
                output_handle.write("\n")
    return times, reference_estimates


def _run_harness(binary_path: Path, input_path: Path, gravity: tuple[float, float, float]) -> list[list[float]]:
    command = [
        str(binary_path),
        str(input_path),
        f"{gravity[0]:.17g}",
        f"{gravity[1]:.17g}",
        f"{gravity[2]:.17g}",
    ]
    result = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    predictions: list[list[float]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] != "tau_pred":
            continue
        predictions.append([float(value) for value in parts[1:7]])
    return predictions


def _save_comparison_csv(
    output_path: Path,
    times: list[float],
    references: list[list[float]],
    predictions: list[list[float]],
) -> None:
    fieldnames = ["time"]
    for joint_idx in range(1, 7):
        fieldnames.extend(
            [
                f"J{joint_idx}_csv_estimate",
                f"J{joint_idx}_calc_estimate",
                f"J{joint_idx}_diff",
            ]
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for time_value, reference_row, predicted_row in zip(times, references, predictions):
            row: dict[str, float] = {"time": time_value}
            for joint_idx in range(6):
                csv_estimate = reference_row[joint_idx]
                calc_estimate = predicted_row[joint_idx]
                row[f"J{joint_idx + 1}_csv_estimate"] = csv_estimate
                row[f"J{joint_idx + 1}_calc_estimate"] = calc_estimate
                row[f"J{joint_idx + 1}_diff"] = calc_estimate - csv_estimate
            writer.writerow(row)


def _print_summary(references: list[list[float]], predictions: list[list[float]]) -> None:
    print("Joint comparison summary:")
    for joint_idx in range(6):
        reference_values = [row[joint_idx] for row in references]
        predicted_values = [row[joint_idx] for row in predictions]
        diffs = [predicted - reference for predicted, reference in zip(predicted_values, reference_values)]
        mae = statistics.fmean(abs(diff) for diff in diffs)
        rmse = math.sqrt(statistics.fmean(diff * diff for diff in diffs))
        max_abs_diff = max(abs(diff) for diff in diffs)
        mean_diff = statistics.fmean(diffs)
        print(
            f"  J{joint_idx + 1}: "
            f"mean_diff={mean_diff:.6f}, "
            f"mae={mae:.6f}, "
            f"rmse={rmse:.6f}, "
            f"max_abs_diff={max_abs_diff:.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对比静止位置 CSV 中的 FilEstimateTor 与 sevendofDynamics::calculateEstimateJointToqrues 结果。"
    )
    parser.add_argument(
        "--csv-path",
        default="/home/xpm/Desktop/dataprocess/静止位置数据.csv",
        help="输入 CSV 路径。",
    )
    parser.add_argument(
        "--export-class-name",
        default=None,
        help="当前导出的 C++ 类名，对应 output/<类名>.cpp 和 .h。",
    )
    parser.add_argument(
        "--output-csv",
        default=str(project_root / "output" / "test" / "static_position_estimate_compare.csv"),
        help="保存对比结果的 CSV 路径。",
    )
    parser.add_argument(
        "--gravity",
        nargs=3,
        type=float,
        default=(-9.81, 0.0, 0.0),
        metavar=("GX", "GY", "GZ"),
        help="传给 sevendofDynamics 的基坐标重力向量，默认使用生成类构造中的值。",
    )
    args = parser.parse_args()

    export_class_name = args.export_class_name or _detect_generated_class_name()
    if export_class_name is None:
        print("无法自动判断当前导出的类名，请显式传入 --export-class-name。")
        return 1

    generated_cpp_path = project_root / "output" / f"{export_class_name}.cpp"
    generated_header_path = project_root / "output" / f"{export_class_name}.h"
    csv_path = Path(args.csv_path)
    output_csv = Path(args.output_csv)

    if not generated_cpp_path.exists() or not generated_header_path.exists():
        print(f"未找到生成文件：{generated_cpp_path.name} / {generated_header_path.name}")
        return 1
    if not csv_path.exists():
        print(f"未找到输入 CSV：{csv_path}")
        return 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="compare_static_csv_", dir=str(project_root / "output" / "test")) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        binary_path = temp_dir / "query_harness"
        input_path = temp_dir / "samples.txt"
        _compile_harness(binary_path, generated_cpp_path)

        times, references = _write_samples_from_csv(csv_path, input_path)
        predictions = _run_harness(binary_path, input_path, tuple(float(value) for value in args.gravity))

    if not times:
        print("CSV 中没有成功解析出任何样本。")
        return 1
    if len(predictions) != len(times):
        print(f"样本数不一致：输入 {len(times)} 条，输出 {len(predictions)} 条。")
        return 1

    _save_comparison_csv(output_csv, times, references, predictions)
    print(f"saved_comparison_csv: {output_csv}")
    print(f"sample_count: {len(times)}")
    _print_summary(references, predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
