#!/usr/bin/env python3
"""
交互式调用当前 output 导出的 C++ 动力学函数，并打印估计扭矩。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]


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


def _prompt_float(prompt: str, default: float | None = None) -> float:
    while True:
        suffix = f" [直接回车使用默认值 {default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("输入无效，请输入数字。")


def _prompt_gravity_vector() -> list[float]:
    print("\n请输入重力方向（基坐标系，m/s²）。")
    print("支持两种方式：")
    print("1. 直接输入一整行，例如：0,0,-9.81")
    print("2. 直接回车后，再逐个输入 gx / gy / gz")
    raw = input("  重力向量 [默认 0,0,-9.81]: ").strip()
    if not raw:
        return [
            _prompt_float("  重力 gx", 0.0),
            _prompt_float("  重力 gy", 0.0),
            _prompt_float("  重力 gz", -9.81),
        ]
    if "," in raw:
        try:
            values = [float(part.strip()) for part in raw.split(",")]
        except ValueError:
            print("输入无效，未能解析为 3 个数字，请重新输入。")
            return _prompt_gravity_vector()
        if len(values) != 3:
            print("输入无效，重力向量必须正好包含 3 个值，请重新输入。")
            return _prompt_gravity_vector()
        return values
    try:
        gx = float(raw)
    except ValueError:
        print("输入无效，请重新输入。")
        return _prompt_gravity_vector()
    gy = _prompt_float("  重力 gy", 0.0)
    gz = _prompt_float("  重力 gz", -9.81)
    return [gx, gy, gz]


def _prompt_joint_vector(name: str) -> list[float]:
    print(f"\n请输入 {name}，共 7 个关节，按 J1 到 J7 逐个输入。")
    values = []
    for joint_idx in range(1, 8):
        value = _prompt_float(f"  {name} - J{joint_idx}")
        values.append(value)
    return values


def _write_single_sample(input_path: Path, q: list[float], dq: list[float], ddq: list[float]) -> None:
    with input_path.open("w", encoding="utf-8") as handle:
        values = [*q, *dq, *ddq]
        handle.write(" ".join(f"{value:.17g}" for value in values))
        handle.write("\n")


def _run_single_query(binary_path: Path, input_path: Path, gravity: list[float]) -> dict[str, list[float]]:
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
    payload: dict[str, list[float]] = {}
    for raw_line in result.stdout.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        payload[parts[0]] = [float(value) for value in parts[1:]]
    return payload


def _format_vector(values: list[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description="交互式调用当前 output 导出的动力学函数。")
    parser.add_argument(
        "--export-class-name",
        default=None,
        help="当前导出的 C++ 类名，对应 output/<类名>.cpp 和 .h。",
    )
    parser.add_argument(
        "--show-physical",
        action="store_true",
        help="打印“纯物理项扭矩”标签。当前第一阶段 output 默认就是纯物理模型。",
    )
    args = parser.parse_args()

    export_class_name = args.export_class_name or _detect_generated_class_name()
    if export_class_name is None:
        print("无法自动判断当前导出的类名。")
        print("请显式传入，例如：--export-class-name sevendofDynamics")
        return 1

    generated_cpp_path = project_root / "output" / f"{export_class_name}.cpp"
    generated_header_path = project_root / "output" / f"{export_class_name}.h"
    if not generated_cpp_path.exists() or not generated_header_path.exists():
        print(f"未找到生成文件：{generated_cpp_path.name} / {generated_header_path.name}")
        print("请先运行 run_pipeline.py 生成对应的 output 文件。")
        return 1

    with tempfile.TemporaryDirectory(prefix="query_output_torque_", dir=str(project_root / "output" / "test")) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        binary_path = temp_dir / "query_harness"
        input_path = temp_dir / "single_sample.txt"
        _compile_harness(binary_path, generated_cpp_path)

        print("=" * 72)
        print("交互式 output 扭矩查询")
        print("=" * 72)
        print(f"当前导出类名: {export_class_name}")
        print(f"当前导出文件: {generated_cpp_path}")
        print("下面请按提示输入重力方向、关节角、角速度、角加速度。")
        print("单位由你自己保证与导出模型一致。通常应使用 rad / rad/s / rad/s^2。")
        print("输入完成后会调用 output 中的 C++ 函数并打印估计扭矩。")
        print("想结束时，在“是否继续”处输入 n。")

        while True:
            gravity = _prompt_gravity_vector()
            q = _prompt_joint_vector("关节角 q")
            dq = _prompt_joint_vector("关节角速度 dq")
            ddq = _prompt_joint_vector("关节角加速度 ddq")

            _write_single_sample(input_path, q, dq, ddq)
            outputs = _run_single_query(binary_path, input_path, gravity)
            tau_gx = [gravity[0] * value for value in outputs["tau_Gx_unit"]]
            tau_gy = [gravity[1] * value for value in outputs["tau_Gy_unit"]]
            tau_gz = [gravity[2] * value for value in outputs["tau_Gz_unit"]]

            print("\n" + "=" * 72)
            print("output 生成函数单样本测试")
            print("=" * 72)
            print("sample_source:   manual")
            print(f"gravity_base:    {_format_vector(gravity)}")
            print(f"q:               {_format_vector(q)}")
            print(f"dq:              {_format_vector(dq)}")
            print(f"ddq:             {_format_vector(ddq)}")

            print("\n分项扭矩结果如下：")
            for joint_idx in range(7):
                print(f"  J{joint_idx + 1}:")
                print(f"    tau_M        = {outputs['tau_M'][joint_idx]:.12f}")
                print(f"    tau_C        = {outputs['tau_C'][joint_idx]:.12f}")
                print(f"    g_x * Y_Gx   = {tau_gx[joint_idx]:.12f}")
                print(f"    g_y * Y_Gy   = {tau_gy[joint_idx]:.12f}")
                print(f"    g_z * Y_Gz   = {tau_gz[joint_idx]:.12f}")
                print(f"    tau_gravity  = {outputs['tau_gravity'][joint_idx]:.12f}")
                print(f"    tau_friction = {outputs['tau_friction'][joint_idx]:.12f}")
                print(f"    tau_pred     = {outputs['tau_pred'][joint_idx]:.12f}")

            again = input("\n是否继续下一组输入？[y/N]: ").strip().lower()
            if again not in {"y", "yes"}:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
