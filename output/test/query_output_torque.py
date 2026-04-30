#!/usr/bin/env python3
"""
交互式或命令行方式调用当前 output 导出的 C++ 动力学函数，并打印估计扭矩。
支持通过 RTOS 风格 `parms` 在运行时注入 payload 质量和质心。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "python"))

from output.test.test_cpp_payload_override import (  # type: ignore
    _apply_payload_to_rtos_last_link,
    _build_rtos_parms,
)
from runtime_dynamics import InertialParameterVector


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


def _safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError as exc:
        raise RuntimeError(
            "当前进程没有可用的交互式 stdin。"
            " 如果你在用 `conda run`，请改用 `conda run --no-capture-output ...`，"
            "或者先 `conda activate pinocchio_py` 再直接执行脚本，"
            "或者通过命令行参数一次性传入 `--gravity --q --dq --ddq`。"
        ) from exc


def _prompt_float(prompt: str, default: float | None = None) -> float:
    while True:
        suffix = f" [直接回车使用默认值 {default}]" if default is not None else ""
        raw = _safe_input(f"{prompt}{suffix}: ").strip()
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
    raw = _safe_input("  重力向量 [默认 0,0,-9.81]: ").strip()
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


def _prompt_payload_override() -> tuple[float, list[float] | None]:
    raw = _safe_input("\n这次计算是否临时加入外部 payload（质量 + 质心）？[y/N]: ").strip().lower()
    if raw not in {"y", "yes"}:
        return 0.0, None
    mass = _prompt_float("  payload 质量 mass (kg)", 0.0)
    if mass <= 0.0:
        return 0.0, None
    print("  请输入 payload 质心位置 COM，单位 m，表达在末端参考坐标系。")
    com = [
        _prompt_float("    COM x", 0.0),
        _prompt_float("    COM y", 0.0),
        _prompt_float("    COM z", 0.0),
    ]
    return float(mass), com


def _load_parameter_vector(theta_path: Path, num_joints: int = 7) -> InertialParameterVector:
    payload = json.loads(theta_path.read_text(encoding="utf-8"))
    theta_full = np.asarray(payload["theta_hat_full"], dtype=float)
    return InertialParameterVector.from_theta_full(theta_full, num_joints)


def _build_runtime_parms(theta_path: Path, payload_mass: float, payload_com: list[float] | None) -> np.ndarray | None:
    if payload_mass <= 0.0 or payload_com is None:
        return None
    parameter_vector = _load_parameter_vector(theta_path)
    base_parms = _build_rtos_parms(parameter_vector)
    return _apply_payload_to_rtos_last_link(
        base_parms,
        mass=float(payload_mass),
        com=np.asarray(payload_com, dtype=float),
    )


def _write_single_sample(input_path: Path, q: list[float], dq: list[float], ddq: list[float]) -> None:
    with input_path.open("w", encoding="utf-8") as handle:
        values = [*q, *dq, *ddq]
        handle.write(" ".join(f"{value:.17g}" for value in values))
        handle.write("\n")


def _write_parms_file(parms_path: Path, parms: np.ndarray | None) -> None:
    if parms is None:
        return
    parms_path.write_text(" ".join(f"{value:.17g}" for value in np.asarray(parms, dtype=float)), encoding="utf-8")


def _run_single_query(
    binary_path: Path,
    input_path: Path,
    gravity: list[float],
    parms_path: Path | None = None,
) -> dict[str, list[float]]:
    command = [
        str(binary_path),
        str(input_path),
        f"{gravity[0]:.17g}",
        f"{gravity[1]:.17g}",
        f"{gravity[2]:.17g}",
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


def _resolve_vector_arg(values: list[float] | None, expected_len: int, flag_name: str) -> list[float] | None:
    if values is None:
        return None
    if len(values) != expected_len:
        raise ValueError(f"{flag_name} 需要正好 {expected_len} 个数字。")
    return [float(value) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description="交互式或命令行方式调用当前 output 导出的动力学函数。")
    parser.add_argument(
        "--export-class-name",
        default=None,
        help="当前导出的 C++ 类名，对应 output/<类名>.cpp 和 .h。",
    )
    parser.add_argument(
        "--theta-path",
        default=str(project_root / "datasets" / "identified" / "theta_hat_real_sensed_latest.json"),
        help="当启用 payload 运行时覆盖时，用于构造默认 RTOS parms 的 theta_hat_full 文件。",
    )
    parser.add_argument("--gravity", nargs=3, type=float, default=None, metavar=("GX", "GY", "GZ"))
    parser.add_argument("--q", nargs=7, type=float, default=None, metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"))
    parser.add_argument("--dq", nargs=7, type=float, default=None, metavar=("DQ1", "DQ2", "DQ3", "DQ4", "DQ5", "DQ6", "DQ7"))
    parser.add_argument("--ddq", nargs=7, type=float, default=None, metavar=("DDQ1", "DDQ2", "DDQ3", "DDQ4", "DDQ5", "DDQ6", "DDQ7"))
    parser.add_argument("--payload-mass", type=float, default=0.0)
    parser.add_argument("--payload-com", nargs=3, type=float, default=None, metavar=("CX", "CY", "CZ"))
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

    try:
        gravity_arg = _resolve_vector_arg(args.gravity, 3, "--gravity")
        q_arg = _resolve_vector_arg(args.q, 7, "--q")
        dq_arg = _resolve_vector_arg(args.dq, 7, "--dq")
        ddq_arg = _resolve_vector_arg(args.ddq, 7, "--ddq")
        payload_com_arg = _resolve_vector_arg(args.payload_com, 3, "--payload-com")
    except ValueError as exc:
        print(str(exc))
        return 1

    with tempfile.TemporaryDirectory(prefix="query_output_torque_", dir=str(project_root / "output" / "test")) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        binary_path = temp_dir / "query_harness"
        input_path = temp_dir / "single_sample.txt"
        parms_path = temp_dir / "runtime_parms.txt"
        _compile_harness(binary_path, generated_cpp_path)

        print("=" * 72)
        print("交互式 output 扭矩查询")
        print("=" * 72)
        print(f"当前导出类名: {export_class_name}")
        print(f"当前导出文件: {generated_cpp_path}")
        print("单位由你自己保证与导出模型一致。通常应使用 rad / rad/s / rad/s^2。")
        if not sys.stdin.isatty():
            print("检测到当前 stdin 不是交互式终端。")
            print("如果你想保留交互模式，请优先使用：")
            print(f"  conda run --no-capture-output -n pinocchio_py python output/test/query_output_torque.py --export-class-name {export_class_name}")
            print("或者先激活环境后直接运行脚本。")
            print("当前脚本也支持直接通过命令行参数传入 --gravity --q --dq --ddq。")
            if gravity_arg is None or q_arg is None or dq_arg is None or ddq_arg is None:
                print("\n当前运行方式下没有可用的交互式 stdin，且未提供完整的命令行输入参数。")
                print("请二选一：")
                print("1. 改成交互式打开：")
                print(f"   conda run --no-capture-output -n pinocchio_py python output/test/query_output_torque.py --export-class-name {export_class_name}")
                print("2. 直接一次性传入输入量，例如：")
                print(
                    "   conda run -n pinocchio_py python output/test/query_output_torque.py "
                    f"--export-class-name {export_class_name} "
                    "--gravity 9.81 0 0 "
                    "--q 0 0 0 0 0 0 0 "
                    "--dq 0 0 0 0 0 0 0 "
                    "--ddq 0 0 0 0 0 0 0"
                )
                print("如果要同时测试 payload 运行时覆盖，可继续追加：")
                print("   --payload-mass 1.25 --payload-com 0 0 0.08")
                return 1
        else:
            print("下面请按提示输入重力方向、关节角、角速度、角加速度。")
            print("现在也支持输入 payload 质量和质心作为运行时覆盖。")
            print("想结束时，在“是否继续”处输入 n。")

        while True:
            try:
                gravity = gravity_arg if gravity_arg is not None else _prompt_gravity_vector()
                q = q_arg if q_arg is not None else _prompt_joint_vector("关节角 q")
                dq = dq_arg if dq_arg is not None else _prompt_joint_vector("关节角速度 dq")
                ddq = ddq_arg if ddq_arg is not None else _prompt_joint_vector("关节角加速度 ddq")
                if payload_com_arg is not None or args.payload_mass > 0.0:
                    payload_mass = float(args.payload_mass)
                    payload_com = payload_com_arg
                else:
                    payload_mass, payload_com = _prompt_payload_override() if sys.stdin.isatty() else (0.0, None)
            except RuntimeError as exc:
                print(str(exc))
                return 1

            runtime_parms = _build_runtime_parms(Path(args.theta_path), payload_mass, payload_com)
            _write_single_sample(input_path, q, dq, ddq)
            if runtime_parms is not None:
                _write_parms_file(parms_path, runtime_parms)
            outputs = _run_single_query(binary_path, input_path, gravity, parms_path if runtime_parms is not None else None)
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
            if runtime_parms is None:
                print("payload_runtime: disabled")
            else:
                print("payload_runtime: enabled")
                print(f"payload_mass:    {payload_mass:.6g}")
                print(f"payload_com:     {_format_vector(payload_com or [0.0, 0.0, 0.0])}")

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

            if gravity_arg is not None and q_arg is not None and dq_arg is not None and ddq_arg is not None:
                break

            again = _safe_input("\n是否继续下一组输入？[y/N]: ").strip().lower()
            if again not in {"y", "yes"}:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
