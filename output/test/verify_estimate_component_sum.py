#!/usr/bin/env python3
"""
验证当前导出的 sevendofDynamics 中：

    calculateEstimateJointToqrues(...)

得到的 tau_pred，是否与

    tau_M + tau_C + tau_gravity + tau_friction

逐样本一致。
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
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
            }
        )
    return columns


def _write_samples_from_csv(csv_path: Path, input_path: Path) -> list[float]:
    times: list[float] = []
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
                except (KeyError, TypeError, ValueError):
                    continue
                times.append(time_value)
                values = [*q, *dq, *ddq]
                output_handle.write(" ".join(f"{value:.17g}" for value in values))
                output_handle.write("\n")
    return times


def _run_harness(
    binary_path: Path,
    input_path: Path,
    gravity: tuple[float, float, float],
) -> list[dict[str, list[float]]]:
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
    required_names = {
        "tau_M",
        "tau_C",
        "tau_gravity",
        "tau_friction",
        "tau_pred",
    }
    samples: list[dict[str, list[float]]] = []
    current: dict[str, list[float]] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        name = parts[0]
        values = [float(value) for value in parts[1:]]
        current[name] = values
        if required_names.issubset(current.keys()):
            samples.append(
                {
                    key: current[key][:7]
                    for key in ("tau_M", "tau_C", "tau_gravity", "tau_friction", "tau_pred")
                }
            )
            current = {}
    return samples


def _save_verification_csv(output_path: Path, times: list[float], samples: list[dict[str, list[float]]]) -> None:
    fieldnames = ["time"]
    for joint_idx in range(1, 8):
        fieldnames.extend(
            [
                f"J{joint_idx}_tau_pred",
                f"J{joint_idx}_tau_sum",
                f"J{joint_idx}_tau_diff",
            ]
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for time_value, sample in zip(times, samples):
            row: dict[str, float] = {"time": time_value}
            for joint_idx in range(7):
                tau_sum = (
                    sample["tau_M"][joint_idx]
                    + sample["tau_C"][joint_idx]
                    + sample["tau_gravity"][joint_idx]
                    + sample["tau_friction"][joint_idx]
                )
                tau_pred = sample["tau_pred"][joint_idx]
                row[f"J{joint_idx + 1}_tau_pred"] = tau_pred
                row[f"J{joint_idx + 1}_tau_sum"] = tau_sum
                row[f"J{joint_idx + 1}_tau_diff"] = tau_pred - tau_sum
            writer.writerow(row)


def _print_summary(samples: list[dict[str, list[float]]]) -> None:
    print("Verification summary:")
    for joint_idx in range(7):
        diffs = []
        for sample in samples:
            tau_sum = (
                sample["tau_M"][joint_idx]
                + sample["tau_C"][joint_idx]
                + sample["tau_gravity"][joint_idx]
                + sample["tau_friction"][joint_idx]
            )
            diffs.append(sample["tau_pred"][joint_idx] - tau_sum)
        mae = statistics.fmean(abs(diff) for diff in diffs)
        rmse = math.sqrt(statistics.fmean(diff * diff for diff in diffs))
        max_abs_diff = max(abs(diff) for diff in diffs)
        mean_diff = statistics.fmean(diffs)
        print(
            f"  J{joint_idx + 1}: "
            f"mean_diff={mean_diff:.12g}, "
            f"mae={mae:.12g}, "
            f"rmse={rmse:.12g}, "
            f"max_abs_diff={max_abs_diff:.12g}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="验证 tau_pred 是否等于 tau_M + tau_C + tau_gravity + tau_friction。"
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
        default=str(project_root / "output" / "test" / "estimate_component_sum_verify.csv"),
        help="保存验证结果的 CSV 路径。",
    )
    parser.add_argument(
        "--gravity",
        nargs=3,
        type=float,
        default=(-9.81, 0.0, 0.0),
        metavar=("GX", "GY", "GZ"),
        help="传给 sevendofDynamics 的基坐标重力向量。",
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

    with tempfile.TemporaryDirectory(prefix="verify_estimate_sum_", dir=str(project_root / "output" / "test")) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        binary_path = temp_dir / "query_harness"
        input_path = temp_dir / "samples.txt"
        _compile_harness(binary_path, generated_cpp_path)
        times = _write_samples_from_csv(csv_path, input_path)
        samples = _run_harness(binary_path, input_path, tuple(float(value) for value in args.gravity))

    if not times:
        print("CSV 中没有成功解析出任何样本。")
        return 1
    if len(samples) != len(times):
        print(f"样本数不一致：输入 {len(times)} 条，输出 {len(samples)} 条。")
        return 1

    _save_verification_csv(output_csv, times, samples)
    print(f"saved_verification_csv: {output_csv}")
    print(f"sample_count: {len(samples)}")
    _print_summary(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
