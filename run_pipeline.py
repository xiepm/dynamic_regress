#!/usr/bin/env python3
"""
机器人动力学参数辨识流水线总入口。

这个文件负责把模型加载、golden data 生成、真实/合成数据处理、参数辨识、残差补偿、
稳定性评估和可视化结果导出串成一条完整工程流程。它位于项目最上层，被命令行直接调用，
同时也可以作为 Python API 被其他脚本复用；下游真正的动力学实现和数据处理细节分别委托给
`load_model.py`、`process_measured_data.py`、`identify_parameters.py` 和
`residual_compensation.py`。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

# 当前脚本所在目录，也就是项目根目录。
# 后面所有相对路径（模型、数据集输出目录等）都基于它来拼接。
project_root = Path(__file__).parent

# 让 Python 能直接 import `python/` 目录下的模块。
# 这样当前目录不用额外安装成 package，也能把工程跑起来。
sys.path.insert(0, str(project_root / "python"))

from generate_golden_data import GoldenDataGenerator
from export_inverse_dynamics_code import export_identified_inverse_dynamics_cpp
from identify_parameters import ParameterIdentifier
from load_model import URDFLoader, print_model_info
import pipeline_identification as pipeline_id
import pipeline_postprocess as pipeline_post
from process_measured_data import (
    MeasuredDataProcessor,
    compute_excitation_diagnostics,
    create_synthetic_measured_data,
)
from residual_compensation import MLPCompensator, ResidualAnalyzer, ResidualCompensator
from runtime_dynamics import GravityConfig, PayloadMode, PayloadModel, RobotDynamicsModel


# 当前版本只支持这两种参数化形式：
# - base: 只辨识可辨识的基参数，通常更稳健；
# - full: 保留完整参数形式，便于研究和对比。
SUPPORTED_PARAMETERIZATIONS = ('base', 'full')
DEFAULT_PARAMETERIZATION = 'base'
# 这里要区分“数据来源”和“运行模式”：
# - 从业务上说，项目目前只有两类数据来源：synthetic（合成）和 real（真实测量）；
# - 从命令行运行上说，多了一个 auto 模式，它只是帮你自动在这两类来源里做选择。
#   换句话说，auto 不是第三种数据，只是一个“自动判断用哪种数据来源”的开关。
SUPPORTED_DATA_SOURCES = ('auto', 'synthetic', 'real')
DEFAULT_DATA_SOURCE = 'auto'
SUPPORTED_REAL_TORQUE_SOURCES = ('sensed', 'sensor')
REAL_TORQUE_UNIT_SCALE = {
    # q*_SensedTorque 已经是 N·m。
    'sensed': 1.0,
    # q*_JointSensorTorque 是传感器原始值，需要乘 0.06 才是 N·m。
    'sensor': 0.06,
}

# 默认重力方向直接写在代码里，方便固定安装方式长期使用。
# 这里推荐直接写“基坐标系下的任意三维重力向量”，而不是只依赖预设关键字。
#
# 当前默认配置成一个标准侧装示例：
# - [ -9.81, 0.0, 0.0 ]  表示 -X 方向朝下
#
# 如果以后安装方式不是正交的 6 个典型方向，比如带倾角安装，
# 直接把下面这个三元组改成你实际测得/标定得到的重力方向即可。
#
# 常见示例：
# - [ 0.0, 0.0, -9.81 ]   正装
# - [ 0.0, 0.0,  9.81 ]   倒装
# - [ -9.81, 0.0, 0.0 ]   侧装（-X 朝下）
# - [  9.81, 0.0, 0.0 ]   侧装（+X 朝下）
# - [ 0.0, -9.81, 0.0 ]   侧装（-Y 朝下）
# - [ 0.0,  9.81, 0.0 ]   侧装（+Y 朝下）
# - [ -6.93, 0.0, -6.93 ]  45 度倾斜安装示例
#
# 命令行如果显式传了 `--gravity`，会覆盖这里的默认值；
# 如果不传，就按这个代码里的默认安装方式运行。
DEFAULT_GRAVITY_VECTOR = [9.81, 0.0, 0.0]

# 默认真实数据扭矩来源。
#
# - 'sensed'：使用 q*_SensedTorque，通常是通过电流估计出的扭矩，默认已是 N·m；
# - 'sensor'：使用 q*_JointSensorTorque，通常是扭矩传感器原始值，本项目会自动乘 0.06 转成 N·m。
#
# 命令行如果显式传了 `--real-torque-source`，会覆盖这里的默认值；
# 如果不传，就按这里的默认扭矩来源运行。
DEFAULT_REAL_TORQUE_SOURCE = 'sensed'

# 其他常用运行默认项。
#
# 这些默认值都会被命令行参数覆盖；如果不传命令行参数，就按这里运行。
DEFAULT_ROBOT_NAME = None
DEFAULT_IDENTIFICATION_MODE = 'rigid_body_friction'
DEFAULT_REAL_DATA_DIR = project_root / "datasets" / "real" / "raw"
DEFAULT_SAMPLING_FREQ = 100.0
DEFAULT_CUTOFF_HZ = 15.0
DEFAULT_URDF_PATH = project_root / "models" / "05_urdf" / "urdf" / "05_urdf_temp.urdf"
DEFAULT_CONFIG_PATH = None
DEFAULT_EXPORT_CLASS_NAME = "sevendofDynamics"
DEFAULT_PAYLOAD_MODE = PayloadMode.NONE.value
SUPPORTED_SOLVER_METHODS = ('ols', 'wls', 'ridge', 'constrained')
DEFAULT_SOLVER_METHOD = 'ridge'


def _build_payload_model(
    *,
    robot_model,
    payload_mass: float,
    payload_com: Iterable[float] | None,
    payload_reference_link: str | None,
    payload_com_frame: str | None,
):
    if payload_mass <= 0.0:
        return None
    if payload_com is None:
        raise ValueError("payload_com must be provided when payload_mass > 0.")
    return PayloadModel(
        mass=float(payload_mass),
        com_position=np.asarray(payload_com, dtype=float),
        com_frame=payload_com_frame or (payload_reference_link or robot_model.ee_link),
        reference_link=payload_reference_link or robot_model.ee_link,
        inertia_about_com=None,
        is_point_mass_approx=True,
    )


def _subtract_known_payload_gravity(
    df: pd.DataFrame,
    robot_model,
    payload_model: PayloadModel | None,
    gravity_vector,
) -> pd.DataFrame:
    if payload_model is None:
        return df

    runtime_model = RobotDynamicsModel(robot_model)
    gravity_config = GravityConfig.from_any(gravity_vector)
    corrected = df.copy()
    q_matrix = np.column_stack([corrected[f'q_{joint_idx}'].values for joint_idx in range(1, robot_model.num_joints + 1)])

    payload_tau = np.zeros((len(corrected), robot_model.num_joints), dtype=float)
    for sample_idx in range(len(corrected)):
        payload_tau[sample_idx, :] = runtime_model.compute_payload_gravity_torque(
            q_matrix[sample_idx],
            payload_model,
            gravity_config=gravity_config,
        )

    for joint_idx in range(1, robot_model.num_joints + 1):
        corrected[f'tau_{joint_idx}'] = corrected[f'tau_{joint_idx}'].values - payload_tau[:, joint_idx - 1]
    return corrected


def _parse_timestamp_to_seconds(value) -> float:
    """
    把常见时间格式转成秒数。

    这里兼容两类输入：
    - 纯数字秒数，例如 `12.345`
    - 带冒号的时间串，例如 `09:47.0` 或 `01:09:47.0`

    之所以单独写这个函数，是因为真实日志里的 timestamp 格式不一定统一，
    先在这里“兜底”转换，后面的同步和重采样逻辑就能保持简单。
    """
    if value is None:
        return np.nan

    text = str(value).strip()
    if not text:
        return np.nan

    try:
        return float(text)
    except ValueError:
        pass

    if ':' not in text:
        return np.nan

    total_seconds = 0.0
    for part in text.split(':'):
        total_seconds = total_seconds * 60.0 + float(part)
    return total_seconds


def _build_monotonic_timestamps(raw_timestamp: pd.Series, sampling_freq: float) -> np.ndarray:
    """
    把原始时间列规范成严格递增的秒数时间轴。

    真实数据里常见两个问题：
    1. timestamp 是字符串格式，不能直接拿来做数值插值；
    2. timestamp 分辨率太粗，连续多行可能完全一样，不满足严格递增。

    所以这里会优先尝试解析原始时间；
    如果发现无法解析，或者时间没有严格递增，就退回到“按采样频率均匀生成时间轴”。
    这样做虽然牺牲了一部分原始时间信息，但能保证整条 pipeline 先稳定跑通。
    """
    parsed = np.array([_parse_timestamp_to_seconds(value) for value in raw_timestamp], dtype=float)
    finite_mask = np.isfinite(parsed)

    if finite_mask.sum() != len(parsed):
        print("Timestamp column contains invalid values; falling back to uniform sample-index timestamps.")
        return np.arange(len(raw_timestamp), dtype=float) / sampling_freq

    parsed = parsed - parsed[0]
    if len(parsed) <= 1:
        return parsed

    # 很多真实日志虽然带 timestamp，但分辨率不够高，会出现大量重复值。
    # 这种情况下直接重采样容易失败，所以退回到“按采样频率均匀生成时间轴”。
    if np.any(np.diff(parsed) <= 0.0):
        print("Timestamp column is not strictly increasing; using uniform timestamps derived from sample index.")
        return np.arange(len(raw_timestamp), dtype=float) / sampling_freq

    return parsed


def _resolve_real_column(df_raw: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in df_raw.columns:
            return candidate
    if required:
        raise ValueError(f"Missing required column. Tried: {', '.join(candidates)}")
    return None


def _normalize_real_measured_dataframe(
    df_raw: pd.DataFrame,
    num_joints: int,
    sampling_freq: float,
    torque_source: str,
) -> pd.DataFrame:
    """
    把真实 CSV 映射成项目内部统一使用的列名。

    兼容两类外部真实文件列名：
    - 旧格式：q1_pos / q1_vel / q1_acc / q1_sensedTorque / q1_cur
    - 新格式：q1_JointPos / q1_JointVel / q1_JointAcc / q1_SensedTorque /
              q1_JointSensorTorque / q1_JointCur

    项目内部统一列名：
    - q_1 / dq_1 / ddq_1 / tau_1 / current_1

    这样做的目的，是让后面的辨识、残差分析、评估模块都只面对一种固定格式，
    不需要每个模块都去适配外部 CSV 的命名差异。
    """
    if 'timestamp' not in df_raw.columns:
        raise ValueError("Real measured CSV must contain a 'timestamp' column.")

    normalized = {
        'timestamp': _build_monotonic_timestamps(df_raw['timestamp'], sampling_freq),
    }

    # 真实数据和动力学模型必须统一单位。
    #
    # 当前真实 CSV 约定是：
    # - 位置：角度（deg）
    # - 速度：度每秒（deg/s）
    # - 加速度：度每二次方秒（deg/s^2）
    # - 力矩：牛米（N·m）
    #
    # 而 Pinocchio / 回归矩阵 / 逆动力学辨识默认使用的是国际单位制：
    # - q:   rad
    # - dq:  rad/s
    # - ddq: rad/s^2
    # - tau: N·m
    #
    # 所以这里必须把角度系变量统一转成弧度系，扭矩则保持不变。
    unit_scale = {
        'q': np.pi / 180.0,
        'dq': np.pi / 180.0,
        'ddq': np.pi / 180.0,
        'tau': 1.0,
        'current': 1.0,
    }

    if torque_source not in SUPPORTED_REAL_TORQUE_SOURCES:
        raise ValueError(f"Unsupported real torque source: {torque_source}")

    torque_candidates_by_source = {
        'sensed': lambda joint_idx: [f'q{joint_idx}_SensedTorque', f'q{joint_idx}_sensedTorque'],
        'sensor': lambda joint_idx: [f'q{joint_idx}_JointSensorTorque'],
    }
    unit_scale['tau'] = REAL_TORQUE_UNIT_SCALE[torque_source]

    for joint_idx in range(1, num_joints + 1):
        column_candidates = {
            'q': [f'q{joint_idx}_JointPos', f'q{joint_idx}_pos'],
            'dq': [f'q{joint_idx}_JointVel', f'q{joint_idx}_vel'],
            'ddq': [f'q{joint_idx}_JointAcc', f'q{joint_idx}_acc'],
            'tau': torque_candidates_by_source[torque_source](joint_idx),
            'current': [f'q{joint_idx}_JointCur', f'q{joint_idx}_cur'],
        }

        for target_prefix, candidates in column_candidates.items():
            source_col = _resolve_real_column(df_raw, candidates, required=target_prefix != 'current')
            target_col = f'{target_prefix}_{joint_idx}'
            if source_col is None:
                normalized[target_col] = np.zeros(len(df_raw), dtype=float)
                continue
            values = pd.to_numeric(df_raw[source_col], errors='coerce')
            normalized[target_col] = values * unit_scale[target_prefix]

    df_normalized = pd.DataFrame(normalized)

    # `current_*` 目前先保留下来，方便后续扩展电流相关分析；
    # 但在本版辨识主链里它还不是必需列，所以 dropna 时不把它算进“硬性必填”。
    essential_columns = [col for col in df_normalized.columns if not col.startswith('current_')]
    before_drop = len(df_normalized)
    df_normalized = df_normalized.dropna(subset=essential_columns).reset_index(drop=True)
    dropped = before_drop - len(df_normalized)
    if dropped > 0:
        print(f"Dropped {dropped} rows with invalid numeric values from real measured data.")
    print(
        "Converted real-data units to SI: q->rad, dq->rad/s, ddq->rad/s^2, "
        f"tau converted to N·m with scale={REAL_TORQUE_UNIT_SCALE[torque_source]} "
        f"(source: {torque_source})"
    )
    return df_normalized


def _load_real_data_files(real_data_dir: Path) -> list[Path]:
    # 统一从一个目录里批量读取所有真实 CSV。
    # 这种约定比“在代码里手填单个文件路径”更适合长期维护：
    # - 新增数据时不需要改代码；
    # - 多次实验文件可以集中管理；
    # - 更适合后续做批量对比和版本留档。
    csv_files = sorted(path for path in real_data_dir.glob('*.csv') if path.is_file())
    if not csv_files:
        raise FileNotFoundError(
            f"No real data CSV files found in {real_data_dir}. "
            "Please place one or more files there."
        )
    return csv_files


def _clear_real_derived_csv_files(*directories: Path) -> None:
    # normalized/processed 是 raw 的派生结果；每次按当前 raw 重新生成，避免旧文件混在新结果里。
    removed_count = 0
    for directory in directories:
        for csv_path in directory.glob('*.csv'):
            csv_path.unlink()
            removed_count += 1
    if removed_count:
        print(f"Cleared {removed_count} stale real-data derived CSV file(s).")


def _clear_legacy_generated_artifacts(
    identified_dir: Path,
    residual_dir: Path,
) -> None:
    """
    删除旧版本遗留下来的结果文件别名，避免目录里长期堆积冗余产物。

    当前主流程已经统一写入：
    - `theta_hat_real_sensed*.json`
    - `evaluation_real_sensed*_splits.json`
    - `compensation_result_real_sensed*.json`
    - `stability_eval_real_sensed.json` / `stability_eval_latest.json`

    因而下面这些旧文件名如果还存在，基本都属于历史遗留，不再由当前代码维护。
    """
    legacy_paths = [
        identified_dir / "theta_hat_real.json",
        identified_dir / "evaluation_real_splits.json",
        identified_dir / "stability_eval.json",
        residual_dir / "compensation_result_real.json",
    ]
    removed = []
    for path in legacy_paths:
        if path.exists():
            path.unlink()
            removed.append(path.name)
    if removed:
        print("Cleared legacy generated artifact(s): " + ", ".join(sorted(removed)))


def _attach_gravity_columns(df: pd.DataFrame, gravity_vector) -> pd.DataFrame:
    """
    为数据表补上逐样本重力分量列。
    """
    gravity = np.asarray(gravity_vector, dtype=float)
    if gravity.shape != (3,):
        raise ValueError(f"Gravity vector must have shape (3,), got {gravity.shape}")
    frame = df.copy()
    frame['gravity_x'] = float(gravity[0])
    frame['gravity_y'] = float(gravity[1])
    frame['gravity_z'] = float(gravity[2])
    return frame


def _prepare_synthetic_dataset(robot_model, sampling_freq: float, cutoff_hz: float) -> pd.DataFrame:
    # synthetic 分支保留，是为了两件事：
    # 1. 没有真实数据时，新人也能先把整条链路跑通；
    # 2. 改算法后可以快速做回归测试，确认是不是代码逻辑本身出了问题。
    #
    # 注意这里的目录结构表达的是：
    # - `synthetic/` 是一种数据来源；
    # - `raw/` 和 `processed/` 是这种数据来源内部的处理阶段。
    print("\n[STEP 5] Creating and processing synthetic data...")
    print("-" * 78)

    raw_dir = project_root / "datasets" / "synthetic" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "synthetic_panda_run01.csv"

    df_raw = create_synthetic_measured_data(robot_model, num_samples=1600, sampling_freq=sampling_freq, seed=2026)
    df_raw.to_csv(raw_path, index=False)
    print(f"Saved raw synthetic data to {raw_path}")

    processor = MeasuredDataProcessor(num_joints=robot_model.num_joints)
    df_synced = processor.synchronize_timestamps(df_raw, reference_freq=sampling_freq)
    df_filtered = processor.apply_low_pass_filter(df_synced, cutoff_hz=cutoff_hz, sampling_freq=sampling_freq)
    df_diff = processor.differentiate_position(df_filtered, sampling_freq=sampling_freq)

    proc_dir = project_root / "datasets" / "synthetic" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    proc_path = proc_dir / "synthetic_panda_run01_proc.csv"
    df_processed = processor.clean_and_export(df_diff, str(proc_path))
    df_processed = _attach_gravity_columns(df_processed, robot_model.gravity_vector)
    df_processed.to_csv(proc_path, index=False)
    df_processed['source_file'] = raw_path.name
    return df_processed


def _prepare_real_dataset(
    robot_model,
    real_data_dir: Path,
    sampling_freq: float,
    cutoff_hz: float,
    torque_source: str,
) -> pd.DataFrame:
    # real 分支专门负责“批量读取真实 CSV -> 统一格式 -> 清洗 -> 合并”。
    # 注意这里假设目录中的文件格式一致；如果后续出现不同采集协议，
    # 最好新增不同的适配函数，而不是把所有特殊情况都堆在一个函数里。
    #
    # 这里的目录关系同样是：
    # - `real/` 是一种数据来源；
    # - `raw/normalized/processed/` 是真实数据在项目内部流转的几个阶段目录。
    print("\n[STEP 5] Loading and processing real data...")
    print("-" * 78)
    print(f"Real torque source for identification: {torque_source}")

    processor = MeasuredDataProcessor(num_joints=robot_model.num_joints)
    csv_files = _load_real_data_files(real_data_dir)

    normalized_dir = project_root / "datasets" / "real" / "normalized"
    processed_dir = project_root / "datasets" / "real" / "processed"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    _clear_real_derived_csv_files(normalized_dir, processed_dir)

    processed_frames = []
    trajectory_offset = 0

    for csv_path in csv_files:
        # 每个文件单独做一次：
        # 1. 原始 CSV 读取
        # 2. 列名标准化
        # 3. 时间同步/滤波/清洗
        # 4. 保存单文件中间结果
        print(f"Processing real file: {csv_path.name}")
        df_raw = pd.read_csv(csv_path)
        df_normalized = _normalize_real_measured_dataframe(
            df_raw,
            robot_model.num_joints,
            sampling_freq,
            torque_source,
        )

        normalized_path = normalized_dir / f"{csv_path.stem}_{torque_source}_normalized.csv"
        df_normalized.to_csv(normalized_path, index=False)

        df_synced = processor.synchronize_timestamps(df_normalized, reference_freq=sampling_freq)
        df_filtered = processor.apply_low_pass_filter(df_synced, cutoff_hz=cutoff_hz, sampling_freq=sampling_freq)

        processed_path = processed_dir / f"{csv_path.stem}_{torque_source}_processed.csv"
        df_processed_single = processor.clean_and_export(df_filtered, str(processed_path))
        if df_processed_single.empty:
            print(f"Skipped {csv_path.name}: no valid samples after cleaning.")
            continue
        df_processed_single = _attach_gravity_columns(df_processed_single, robot_model.gravity_vector)

        df_processed_single['source_file'] = csv_path.name
        df_processed_single['source_path'] = str(csv_path)
        df_processed_single['torque_source'] = torque_source
        if 'trajectory_id' in df_processed_single.columns:
            # 不同文件内部可能都会从 trajectory_id=0 开始编号。
            # 这里做一个全局偏移，避免多个文件合并后轨迹编号冲突。
            df_processed_single['trajectory_id'] = df_processed_single['trajectory_id'] + trajectory_offset
            trajectory_offset = int(df_processed_single['trajectory_id'].max()) + 1
        df_processed_single.to_csv(processed_path, index=False)
        processed_frames.append(df_processed_single)

    if not processed_frames:
        raise ValueError("All real data files were filtered out during cleaning. Please check the raw signals or thresholds.")

    combined = pd.concat(processed_frames, ignore_index=True)
    combined_path = processed_dir / f"real_{torque_source}_combined_processed.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Combined processed real dataset saved to {combined_path}")
    print(f"Loaded {len(csv_files)} real file(s), total valid samples: {len(combined)}")
    # 新增：[Diag] 打印 q/dq/ddq 的关键统计和一致性检查，便于排查真实数据质量。
    q_cols = [f'q_{i}' for i in range(1, robot_model.num_joints + 1)]
    dq_cols = [f'dq_{i}' for i in range(1, robot_model.num_joints + 1)]
    ddq_cols = [f'ddq_{i}' for i in range(1, robot_model.num_joints + 1)]
    dt = 1.0 / sampling_freq
    for joint_idx in range(1, robot_model.num_joints + 1):
        q_values = combined[f'q_{joint_idx}'].values
        dq_values = combined[f'dq_{joint_idx}'].values
        ddq_values = combined[f'ddq_{joint_idx}'].values
        dq_from_q = np.gradient(q_values, dt, edge_order=2)
        dq_diff_rms = float(np.sqrt(np.mean((dq_values - dq_from_q) ** 2)))
        print(
            f"[Diag] joint {joint_idx}: "
            f"q_range=[{q_values.min():.3f}, {q_values.max():.3f}] rad, "
            f"dq_rms={np.sqrt(np.mean(dq_values ** 2)):.3f} rad/s, "
            f"ddq_rms={np.sqrt(np.mean(ddq_values ** 2)):.3f} rad/s^2"
        )
        print(f"[Diag] joint {joint_idx}: dq consistency RMS diff={dq_diff_rms:.3f} rad/s")
        if dq_diff_rms > 0.1:
            print(f"[Diag] Warning: dq/ddq inconsistency detected on joint {joint_idx}, RMS diff={dq_diff_rms:.3f}")
    return combined


def _resolve_data_source(data_source: str, real_data_dir: Path) -> str:
    """根据命令行模式和真实数据目录内容决定本次实际使用的数据来源。"""
    # `auto` 模式是给日常使用准备的：
    # - 真实目录里已经放了 CSV，就直接走真实数据流程；
    # - 如果目录还是空的，就自动回退成 synthetic。
    if data_source == 'auto':
        return 'real' if any(real_data_dir.glob('*.csv')) else 'synthetic'
    return data_source


def _split_dataframe_for_learning(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 123,
) -> dict[str, pd.DataFrame]:
    """
    将处理后的总数据切分为 train / val / test。

    Parameters
    ----------
    df : pd.DataFrame
        处理完成后的总数据集。
    train_ratio : float
        训练集比例。
    val_ratio : float
        验证集比例。
    seed : int
        随机种子。

    Returns
    -------
    dict[str, pd.DataFrame]
        含 `train`、`val`、`test` 三份数据表的字典。

    Raises
    ------
    ValueError
        当比例非法或某个 split 最终为空时抛出。
    """
    # 这里负责把处理后的总数据集切成 train / val / test。
    # 默认比例是 70% / 15% / 15%。
    #
    # 如果数据里已经有 trajectory_id，就优先按整段轨迹来切，
    # 避免一条连续运动轨迹同时落进训练集和测试集，导致评估“虚高”。
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.0.")

    test_ratio = 1.0 - train_ratio - val_ratio
    rng = np.random.default_rng(seed)

    if 'trajectory_id' in df.columns and df['trajectory_id'].nunique() >= 3:
        # 修复：先把所有轨迹 ID 随机打散，再按轨迹数比例切分，
        # 避免旧版贪心策略把 test 变成“最后剩下的一小撮轨迹”。
        trajectory_ids = rng.permutation(df['trajectory_id'].drop_duplicates().to_numpy())
        n_trajectories = len(trajectory_ids)
        n_train_ids = int(n_trajectories * train_ratio)
        n_val_ids = int(n_trajectories * val_ratio)

        assigned_groups = {
            'train': trajectory_ids[:n_train_ids].tolist(),
            'val': trajectory_ids[n_train_ids:n_train_ids + n_val_ids].tolist(),
            'test': trajectory_ids[n_train_ids + n_val_ids:].tolist(),
        }

        splits = {
            split_name: df[df['trajectory_id'].isin(group_ids)].reset_index(drop=True)
            for split_name, group_ids in assigned_groups.items()
        }
    else:
        # 如果没有足够多的轨迹段，就退回到普通随机切分。
        indices = rng.permutation(len(df))
        n_train = int(len(df) * train_ratio)
        n_val = int(len(df) * val_ratio)
        splits = {
            'train': df.iloc[indices[:n_train]].reset_index(drop=True),
            'val': df.iloc[indices[n_train:n_train + n_val]].reset_index(drop=True),
            'test': df.iloc[indices[n_train + n_val:]].reset_index(drop=True),
        }

    # 真实数据里经常会出现“总轨迹数不多、但某几段特别长”的情况。
    # 上面的轨迹级贪心切分在这种情况下可能把某个 split 分空。
    # 如果发生这种边界情况，就退回到样本级随机切分，优先保证流程能跑通并且三份集合都存在。
    if any(split_df.empty for split_df in splits.values()):
        print("Trajectory-level split produced an empty subset; falling back to sample-level random split.")
        indices = rng.permutation(len(df))
        n_train = max(1, int(len(df) * train_ratio))
        n_val = max(1, int(len(df) * val_ratio))
        n_test = len(df) - n_train - n_val

        # 如果样本数非常少，优先保证 val/test 至少各有 1 条，再把剩余给 train。
        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            else:
                n_val -= 1
        if n_val <= 0:
            n_val = 1
            if n_train > 1:
                n_train -= 1
            else:
                n_test -= 1

        splits = {
            'train': df.iloc[indices[:n_train]].reset_index(drop=True),
            'val': df.iloc[indices[n_train:n_train + n_val]].reset_index(drop=True),
            'test': df.iloc[n_train + n_val:n_train + n_val + n_test].reset_index(drop=True),
        }

    for split_name, split_df in splits.items():
        if split_df.empty:
            raise ValueError(f"Dataset split '{split_name}' is empty. Please provide more data or adjust split ratios.")

    print("\nDataset split for identification:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val:   {len(splits['val'])} samples")
    print(f"  Test:  {len(splits['test'])} samples")
    return splits


def _evaluate_identification_splits(identifier, result: dict, splits: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    在 train / val / test 三个子集上分别评估辨识误差。

    Parameters
    ----------
    identifier : ParameterIdentifier
        已初始化的辨识器。
    result : dict
        辨识结果字典。
    splits : dict[str, pd.DataFrame]
        数据集切分结果。

    Returns
    -------
    dict[str, dict]
        每个 split 对应一份误差评估结果。
    """
    # 用同一组已辨识参数，分别在 train / val / test 上做评估。
    # 这样终端和 JSON 里就能直接横向比较三份数据上的误差表现。
    split_metrics = {}
    for split_name, split_df in splits.items():
        print(f"\nEvaluating identification on {split_name} split...")
        split_metrics[split_name] = identifier.evaluate_identification(split_df, result)
    return split_metrics


def _summarize_overfitting(split_metrics: dict[str, dict]) -> dict[str, float | bool]:
    """
    基于 train / val / test RMSE 给出一个简单过拟合提示。

    Parameters
    ----------
    split_metrics : dict[str, dict]
        三个子集上的误差评估结果。

    Returns
    -------
    dict[str, float | bool]
        含 RMSE 比值和 `possible_overfit` 标志的摘要字典。
    """
    # 这里给一个“够用但简单”的过拟合提示：
    # 如果验证集或测试集 RMSE 明显高于训练集，就标记 possible_overfit=True。
    #
    # 注意它只是启发式提示，不代表严格统计结论。
    # 真正做研究时，还是建议结合更多工况、更多实验轮次一起判断。
    train_rmse = split_metrics['train']['global_rmse']
    val_rmse = split_metrics['val']['global_rmse']
    test_rmse = split_metrics['test']['global_rmse']
    val_ratio = float(val_rmse / train_rmse) if train_rmse > 0 else np.inf
    test_ratio = float(test_rmse / train_rmse) if train_rmse > 0 else np.inf

    summary = {
        'train_rmse': float(train_rmse),
        'val_rmse': float(val_rmse),
        'test_rmse': float(test_rmse),
        'val_to_train_rmse_ratio': val_ratio,
        'test_to_train_rmse_ratio': test_ratio,
        'possible_overfit': bool(val_ratio > 1.25 or test_ratio > 1.25),
    }
    return summary


def _print_per_joint_summary(title: str, metrics: dict[str, dict], metric_keys: list[str]) -> None:
    """
    按关节顺序打印一组指标摘要。

    Parameters
    ----------
    title : str
        输出标题。
    metrics : dict[str, dict]
        指标字典。
    metric_keys : list[str]
        需要抽取并打印的指标键名列表。

    Returns
    -------
    None
        结果直接打印到终端。
    """
    # 新增：把每关节指标集中打印，避免用户只能看到全局误差而看不到具体是哪几个关节拖后腿。
    print(f"\n{title}")
    joint_labels = []
    for metric_key in metric_keys:
        joint_labels.extend(metrics.get(metric_key, {}).keys())
    ordered_labels = sorted(set(joint_labels), key=lambda label: int(label.split('_')[-1]))

    for label in ordered_labels:
        joint_idx = int(label.split('_')[-1])
        parts = []
        for metric_key in metric_keys:
            value = metrics.get(metric_key, {}).get(label)
            if value is None:
                continue
            if 'improvement' in metric_key:
                parts.append(f"{metric_key}={value:.2f}%")
            else:
                parts.append(f"{metric_key}={value:.6f} N·m")
        if parts:
            print(f"  Joint {joint_idx}: " + ", ".join(parts))


def _joint_metric_array(metric_map: dict[str, float]) -> list[float]:
    """把 `joint_i -> value` 的字典按关节编号稳定转换为数组。"""
    # 新增：把 joint_1 ... joint_n 这种字典稳定转换成数组，方便 notebook 直接按关节顺序画图。
    ordered_items = sorted(metric_map.items(), key=lambda item: int(item[0].split('_')[-1]))
    return [float(value) for _, value in ordered_items]


def _write_visualization_payload(
    robot_model,
    effective_data_source: str,
    result_stem: str,
    split_evaluation: dict[str, dict],
    compensation_eval: dict[str, dict],
    mlp_eval: dict[str, dict],
    stability_eval: dict,
) -> Path:
    """
    导出 notebook 可直接消费的最新可视化结果文件。

    Parameters
    ----------
    robot_model : RobotModel
        当前机器人模型。
    effective_data_source : str
        本次实际使用的数据来源。
    result_stem : str
        本次输出文件使用的结果前缀。
    split_evaluation : dict[str, dict]
        辨识误差评估结果。
    compensation_eval : dict[str, dict]
        线性补偿评估结果。
    mlp_eval : dict[str, dict]
        MLP 补偿评估结果。
    stability_eval : dict
        多随机种子稳定性评估结果。

    Returns
    -------
    Path
        最新可视化 JSON 的输出路径。
    """
    # 新增：给 notebook 输出一份固定路径的“最新可视化结果”，每次运行直接覆盖旧结果。
    vis_dir = project_root / "datasets" / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = vis_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    identification_test = split_evaluation['test']
    linear_test = compensation_eval['test']
    mlp_test = mlp_eval['test']
    joint_count = len(identification_test['joint_rmse'])
    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'robot_name': robot_model.name,
        'data_source': effective_data_source,
        'result_stem': result_stem,
        'joints': [f'J{joint_idx}' for joint_idx in range(1, joint_count + 1)],
        'stages': ['Identification', '+ Linear comp.', '+ MLP comp.'],
        'paths': {
            'figure_dir': str(figure_dir),
            'latest_results_file': str(vis_dir / "latest_results.json"),
            'snapshot_results_file': str(vis_dir / f"results_{result_stem}.json"),
        },
        'identification': {
            'test': {
                'global_rmse': float(identification_test['global_rmse']),
                'global_mae': float(identification_test['global_mae']),
                'joint_rmse': _joint_metric_array(identification_test['joint_rmse']),
                'joint_mae': _joint_metric_array(identification_test['joint_mae']),
            },
        },
        'linear': {
            'test': {
                'global_rmse': float(linear_test['rmse']),
                'global_mae': float(linear_test['mae']),
                'improvement_percent': float(linear_test['improvement_percent']),
                'joint_rmse': _joint_metric_array(linear_test['joint_rmse']),
                'joint_mae': _joint_metric_array(linear_test['joint_mae']),
                'joint_improvement_percent': _joint_metric_array(linear_test['joint_improvement_percent']),
            },
        },
        'mlp': {
            'test': {
                'global_rmse': float(mlp_test['rmse']),
                'global_mae': float(mlp_test['mae']),
                'improvement_percent': float(mlp_test['improvement_percent']),
                'joint_rmse': _joint_metric_array(mlp_test['joint_rmse']),
                'joint_mae': _joint_metric_array(mlp_test['joint_mae']),
                'joint_improvement_percent': _joint_metric_array(mlp_test['joint_improvement_percent']),
            },
        },
        'stability': {
            'splits': ['Train', 'Val', 'Test'],
            'rmse_mean': [
                float(stability_eval['summary']['train']['mean']),
                float(stability_eval['summary']['val']['mean']),
                float(stability_eval['summary']['test']['mean']),
            ],
            'rmse_std': [
                float(stability_eval['summary']['train']['std']),
                float(stability_eval['summary']['val']['std']),
                float(stability_eval['summary']['test']['std']),
            ],
        },
        'global_pipeline': {
            'rmse': [
                float(identification_test['global_rmse']),
                float(linear_test['rmse']),
                float(mlp_test['rmse']),
            ],
            'mae': [
                float(identification_test['global_mae']),
                float(linear_test['mae']),
                float(mlp_test['mae']),
            ],
            'improvement_percent': [
                float(linear_test['improvement_percent']),
                float(mlp_test['improvement_percent']),
            ],
        },
    }

    snapshot_path = vis_dir / f"results_{result_stem}.json"
    with open(snapshot_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)

    output_path = vis_dir / "latest_results.json"
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(f"[Diag] Visualization payload updated: {output_path}")
    print(f"[Diag] Visualization timestamped snapshot saved: {snapshot_path}")
    return output_path


def _evaluate_residual_compensation_by_split(analyzer, compensator, identifier, result: dict, splits: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    在 train / val / test 三个子集上评估线性残差补偿器。

    Parameters
    ----------
    analyzer : ResidualAnalyzer
        残差分析器。
    compensator : ResidualCompensator
        线性补偿器。
    identifier : ParameterIdentifier
        主辨识器。
    result : dict
        主辨识结果。
    splits : dict[str, pd.DataFrame]
        数据集切分结果。

    Returns
    -------
    dict[str, dict]
        含训练摘要与 train/val/test 评估结果的字典。
    """
    # 残差补偿也沿用同样的 train / val / test 思路：
    # - 在 train 上训练线性补偿器
    # - 在 train / val / test 上分别看 MAE、RMSE 和 improvement
    #
    # 这样可以直接观察：
    # - 补偿器是不是只在训练集上看起来很好；
    # - 到验证/测试集时效果是否还能保持。
    residual_splits = {}
    for split_name, split_df in splits.items():
        residual_df = analyzer.compute_residuals(split_df, identifier, result)
        residual_splits[split_name] = analyzer.feature_engineering(residual_df)

    # 修复：补偿器默认先做 5 折交叉验证挑选岭回归正则强度，再在 train 上重训。
    train_info = compensator.train_with_cross_validation(residual_splits['train'])
    evaluations = {
        'train_fit': train_info,
        'train': compensator.evaluate_compensator(residual_splits['train']),
        'val': compensator.evaluate_compensator(residual_splits['val']),
        'test': compensator.evaluate_compensator(residual_splits['test']),
    }
    return evaluations


def run_stability_evaluation(df, identifier, result, n_seeds: int = 5) -> dict:
    """
    在多个随机种子下重复切分并评估辨识稳定性。

    Parameters
    ----------
    df : pd.DataFrame
        完整处理后数据集。
    identifier : ParameterIdentifier
        辨识器实例。
    result : dict
        一次主辨识结果，用于复用方法名和正则参数配置。
    n_seeds : int
        参与稳定性测试的随机种子个数。

    Returns
    -------
    dict
        含每个种子的 RMSE 和均值/标准差摘要的结果字典。
    """
    # 新增：多随机种子稳定性测试，用于判断辨识结果是否对数据划分敏感。
    seeds = [123, 42, 7, 2024, 999][:n_seeds]
    per_seed = []
    for seed in seeds:
        splits = _split_dataframe_for_learning(df, seed=seed)
        seed_result = identifier.identify_parameters(
            splits['train'],
            method=result['method'].lower(),
            reference_parameters=result.get('reference_parameters'),
            ridge_lambda=result.get('ridge_lambda', 0.0) or 1e-4,
        )
        split_eval = _evaluate_identification_splits(identifier, seed_result, splits)
        per_seed.append({
            'seed': seed,
            'train_rmse': split_eval['train']['global_rmse'],
            'val_rmse': split_eval['val']['global_rmse'],
            'test_rmse': split_eval['test']['global_rmse'],
        })

    metrics = {}
    for split_name in ('train', 'val', 'test'):
        values = np.array([entry[f'{split_name}_rmse'] for entry in per_seed], dtype=float)
        metrics[split_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        }

    print(f"\n[Diag] Stability evaluation ({len(seeds)} seeds):")
    print(f"[Diag]   Train RMSE: mean={metrics['train']['mean']:.3f}, std={metrics['train']['std']:.3f}")
    print(f"[Diag]   Val   RMSE: mean={metrics['val']['mean']:.3f}, std={metrics['val']['std']:.3f}")
    print(f"[Diag]   Test  RMSE: mean={metrics['test']['mean']:.3f}, std={metrics['test']['std']:.3f}")
    return {
        'seeds': seeds,
        'per_seed': per_seed,
        'summary': metrics,
    }


def _run_identification_stage(
    *,
    robot_model,
    splits: dict[str, pd.DataFrame],
    effective_data_source: str,
    parameterization: str,
    identification_mode: str,
    solver_method: str,
    real_torque_source: str,
    payload_model,
    effective_payload_mode: PayloadMode,
    subtract_known_payload_gravity: bool,
) -> dict:
    """
    Execute the core identification stage only.

    This stage stops at parameter estimation and does not mix in result
    serialization, compensation, export, or visualization side effects.
    """
    print("\n[STEP 6] Identifying parameters...")
    print("-" * 78)
    identifier = ParameterIdentifier(robot_model, parameterization=parameterization)
    reference_parameters = None if effective_data_source == 'real' else robot_model.full_parameter_vector()
    result = identifier.identify_parameters(
        splits['train'],
        method=solver_method,
        reference_parameters=reference_parameters,
        ridge_lambda=1e-4,
    )
    return {
        'identifier': identifier,
        'result': result,
        'reference_parameters': reference_parameters,
        'context': {
            'parameterization': parameterization,
            'identification_mode': identification_mode,
            'data_source': effective_data_source,
            'real_torque_source': real_torque_source if effective_data_source == 'real' else None,
            'payload_mode': effective_payload_mode.value,
            'payload_model': None if payload_model is None else payload_model.to_dict(),
            'subtract_known_payload_gravity': bool(subtract_known_payload_gravity),
        },
    }


def _write_identification_artifacts(
    *,
    identified_dir: Path,
    result_stem: str,
    result_stem_base: str,
    robot_model,
    identification_bundle: dict,
    split_evaluation: dict[str, dict],
    overfitting_summary: dict,
) -> None:
    """Persist identification parameters and split evaluation artifacts."""
    result = identification_bundle['result']
    context = identification_bundle['context']
    theta_payload = {
        'robot_name': robot_model.name,
        'identification_mode': context['identification_mode'],
        'parameterization': context['parameterization'],
        'data_source': context['data_source'],
        'real_torque_source': context['real_torque_source'],
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'rank': result['rank'],
        'full_regressor_rank': result['full_regressor_rank'],
        'base_parameter_count': result['base_parameter_count'],
        'condition_number': result['condition_number'],
        'active_condition_number': result['active_condition_number'],
        'active_parameter_indices': result['active_parameter_indices'].tolist(),
        'base_column_indices': result['base_column_indices'].tolist(),
        'base_transform_full_from_beta': np_to_list(result['base_transform_full_from_beta']),
        'theta_hat': np_to_list(result['theta_hat']),
        'theta_hat_full': np_to_list(result['theta_hat_full']),
        'pi_full_hat': np_to_list(result['pi_full_hat']),
        'beta_hat': np_to_list(result['beta_hat']),
        'parameter_vector': result['parameter_vector'].to_dict(),
        'gravity_config': result['gravity_config'],
        'physical_sanity': result['physical_sanity'],
        'reference_parameters': None if result['reference_parameters'] is None else np_to_list(result['reference_parameters']),
        'method': result['method'],
        'ridge_lambda': result['ridge_lambda'],
        'optimizer_success': result['optimizer_success'],
        'optimizer_status': result['optimizer_status'],
        'optimizer_iterations': result['optimizer_iterations'],
        'payload_mode': context['payload_mode'],
        'payload_model': context['payload_model'],
        'subtract_known_payload_gravity': context['subtract_known_payload_gravity'],
    }
    with open(identified_dir / f"theta_hat_{result_stem}.json", 'w', encoding='utf-8') as handle:
        json.dump(theta_payload, handle, indent=2)
    with open(identified_dir / f"theta_hat_{result_stem_base}_latest.json", 'w', encoding='utf-8') as handle:
        json.dump(theta_payload, handle, indent=2)

    evaluation_payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'split_metrics': split_evaluation,
        'overfitting_summary': overfitting_summary,
    }
    with open(identified_dir / f"evaluation_{result_stem}_splits.json", 'w', encoding='utf-8') as handle:
        json.dump(evaluation_payload, handle, indent=2)
    with open(identified_dir / f"evaluation_{result_stem_base}_latest_splits.json", 'w', encoding='utf-8') as handle:
        json.dump(evaluation_payload, handle, indent=2)


def _run_post_identification_stage(
    *,
    robot_model,
    df_processed: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    identification_bundle: dict,
    effective_data_source: str,
    real_torque_source: str,
    identification_mode: str,
    effective_gravity,
    effective_payload_mode: PayloadMode,
    payload_model,
    resolved_urdf_path: Path,
    parameterization: str,
    export_class_name: str,
) -> dict:
    """
    Execute all workflows that consume identification results.

    This keeps evaluation/export/compensation concerns out of the core
    identification stage so the pipeline stays easier to read and extend.
    """
    identifier = identification_bundle['identifier']
    result = identification_bundle['result']

    split_evaluation = _evaluate_identification_splits(identifier, result, splits)
    overfitting_summary = _summarize_overfitting(split_evaluation)
    _print_per_joint_summary(
        "Identification per-joint metrics (test split):",
        split_evaluation['test'],
        ['joint_rmse', 'joint_mae'],
    )

    identified_dir = project_root / "datasets" / "identified"
    identified_dir.mkdir(parents=True, exist_ok=True)
    residual_dir = project_root / "datasets" / "residual"
    residual_dir.mkdir(parents=True, exist_ok=True)
    _clear_legacy_generated_artifacts(identified_dir, residual_dir)
    result_stem_base = f"real_{real_torque_source}" if effective_data_source == 'real' else 'synthetic'
    result_stem = result_stem_base

    _write_identification_artifacts(
        identified_dir=identified_dir,
        result_stem=result_stem,
        result_stem_base=result_stem_base,
        robot_model=robot_model,
        identification_bundle=identification_bundle,
        split_evaluation=split_evaluation,
        overfitting_summary=overfitting_summary,
    )

    print("\n[STEP 7-8] Residual analysis and compensation...")
    print("-" * 78)
    analyzer = ResidualAnalyzer(robot_model)
    compensator = ResidualCompensator(num_joints=robot_model.num_joints)
    compensation_eval = _evaluate_residual_compensation_by_split(analyzer, compensator, identifier, result, splits)
    mlp_compensator = MLPCompensator(num_joints=robot_model.num_joints)
    residual_splits = {}
    for split_name, split_df in splits.items():
        residual_df = analyzer.compute_residuals(split_df, identifier, result)
        residual_splits[split_name] = analyzer.feature_engineering(residual_df)
    mlp_train_info = mlp_compensator.train(residual_splits['train'])
    mlp_eval = {
        'train_fit': mlp_train_info,
        'train': mlp_compensator.evaluate(residual_splits['train']),
        'val': mlp_compensator.evaluate(residual_splits['val']),
        'test': mlp_compensator.evaluate(residual_splits['test']),
    }
    print("\nCompensator comparison (test set):")
    print(f"  Linear    MAE: {compensation_eval['test']['mae']:.3f} N·m  Improvement: {compensation_eval['test']['improvement_percent']:.1f}%")
    print(f"  MLP       MAE: {mlp_eval['test']['mae']:.3f} N·m  Improvement: {mlp_eval['test']['improvement_percent']:.1f}%")
    _print_per_joint_summary(
        "Linear compensator per-joint metrics (test split):",
        compensation_eval['test'],
        ['joint_rmse', 'joint_mae', 'joint_improvement_percent'],
    )
    _print_per_joint_summary(
        "MLP compensator per-joint metrics (test split):",
        mlp_eval['test'],
        ['joint_rmse', 'joint_mae', 'joint_improvement_percent'],
    )
    if mlp_eval['test']['improvement_percent'] - compensation_eval['test']['improvement_percent'] > 15.0:
        print("Suggestion: nonlinear compensation shows significant benefit, consider a deeper/wider MLP")

    generated_dynamics_paths = export_identified_inverse_dynamics_cpp(
        robot_model=robot_model,
        result=result,
        urdf_path=resolved_urdf_path,
        output_dir=project_root / "output",
        result_stem=result_stem,
        class_name=export_class_name,
        parameter_vector=result['parameter_vector'],
        gravity_config=GravityConfig.from_any(effective_gravity),
        payload_model=payload_model,
        payload_mode=effective_payload_mode,
        generation_metadata={
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'robot_name': robot_model.name,
            'data_source': effective_data_source,
            'real_torque_source': real_torque_source if effective_data_source == 'real' else None,
            'parameterization': parameterization,
            'identification_mode': identification_mode,
            'payload_mode': effective_payload_mode.value,
            'nonlinear_compensation': 'disabled_for_export',
        },
    )
    print("Generated identified inverse-dynamics code with explicit gravity decomposition:")
    for label, path in generated_dynamics_paths.items():
        print(f"  {label}: {path}")

    compensation_payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'linear': compensation_eval,
        'mlp': mlp_eval,
    }
    with open(residual_dir / f"compensation_result_{result_stem}.json", 'w', encoding='utf-8') as handle:
        json.dump(compensation_payload, handle, indent=2)
    with open(residual_dir / f"compensation_result_{result_stem_base}_latest.json", 'w', encoding='utf-8') as handle:
        json.dump(compensation_payload, handle, indent=2)

    stability_eval = run_stability_evaluation(df_processed, identifier, result, n_seeds=5)
    with open(identified_dir / f"stability_eval_{result_stem}.json", 'w', encoding='utf-8') as handle:
        json.dump(stability_eval, handle, indent=2)
    with open(identified_dir / "stability_eval_latest.json", 'w', encoding='utf-8') as handle:
        json.dump(stability_eval, handle, indent=2)
    _write_visualization_payload(
        robot_model=robot_model,
        effective_data_source=effective_data_source,
        result_stem=result_stem,
        split_evaluation=split_evaluation,
        compensation_eval=compensation_eval,
        mlp_eval=mlp_eval,
        stability_eval=stability_eval,
    )

    return {
        'split_evaluation': split_evaluation,
        'overfitting_summary': overfitting_summary,
        'compensation_eval': compensation_eval,
        'mlp_eval': mlp_eval,
        'stability_eval': stability_eval,
        'generated_dynamics_paths': generated_dynamics_paths,
        'result_stem': result_stem,
        'result_stem_base': result_stem_base,
    }


def _print_pipeline_summary(
    *,
    robot_model,
    df_processed: pd.DataFrame,
    result: dict,
    post_bundle: dict,
) -> None:
    """Print the compact end-of-run summary."""
    split_evaluation = post_bundle['split_evaluation']
    overfitting_summary = post_bundle['overfitting_summary']
    compensation_eval = post_bundle['compensation_eval']
    mlp_eval = post_bundle['mlp_eval']
    generated_dynamics_paths = post_bundle['generated_dynamics_paths']

    print("\n" + "=" * 78)
    print(" PIPELINE COMPLETED")
    print("=" * 78)
    print(f"  Robot: {robot_model.name}")
    print(f"  Samples processed: {len(df_processed)}")
    print(f"  Full parameters: {result['num_parameters_full']}")
    print(f"  Active parameters: {result['num_parameters_active']}")
    print(f"  Rank: {result['full_regressor_rank']}")
    print(f"  Condition number: {result['condition_number']:.2e}")
    print(f"  Active condition number: {result['active_condition_number']:.2e}")
    print(f"  Train RMSE: {split_evaluation['train']['global_rmse']:.6f} N·m")
    print(f"  Val RMSE:   {split_evaluation['val']['global_rmse']:.6f} N·m")
    print(f"  Test RMSE:  {split_evaluation['test']['global_rmse']:.6f} N·m")
    print(f"  Possible overfit: {overfitting_summary['possible_overfit']}")
    print(f"  Linear compensation test improvement: {compensation_eval['test']['improvement_percent']:.2f}%")
    print(f"  MLP compensation test improvement: {mlp_eval['test']['improvement_percent']:.2f}%")
    print(f"  Generated dynamics code: {generated_dynamics_paths['project_cpp']}")
    print("=" * 78 + "\n")


def run_pipeline(
    robot_name: str | None = DEFAULT_ROBOT_NAME,
    identification_mode: str = DEFAULT_IDENTIFICATION_MODE,
    parameterization: str = DEFAULT_PARAMETERIZATION,
    data_source: str = DEFAULT_DATA_SOURCE,
    real_data_dir: str | None = None,
    real_torque_source: str = DEFAULT_REAL_TORQUE_SOURCE,
    sampling_freq: float = DEFAULT_SAMPLING_FREQ,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    urdf_path: str | None = None,
    config_path: str | None = None,
    gravity: str | None = None,
    export_class_name: str = DEFAULT_EXPORT_CLASS_NAME,
    solver_method: str = DEFAULT_SOLVER_METHOD,
    payload_mode: str = DEFAULT_PAYLOAD_MODE,
    payload_mass: float = 0.0,
    payload_com: Iterable[float] | None = None,
    payload_reference_link: str | None = None,
    payload_com_frame: str | None = None,
    subtract_known_payload_gravity: bool = False,
):
    """
    执行整条动力学参数辨识流水线。

    Parameters
    ----------
    robot_name : str | None
        可选机器人名称；如果与 URDF 推断结果不一致，最终以 URDF 为准。
    identification_mode : str
        当前支持的辨识模式。
    parameterization : str
        参数化方式，支持 `base` 或 `full`。
    data_source : str
        数据来源模式，支持 `auto`、`synthetic`、`real`。
    real_data_dir : str | None
        真实数据目录；为空时使用默认目录。
    real_torque_source : str
        真实数据辨识使用的扭矩来源，支持 `sensed` 或 `sensor`。
    sampling_freq : float
        采样频率，单位 Hz。
    cutoff_hz : float
        低通滤波截止频率，单位 Hz。
    urdf_path : str | None
        可选 URDF 路径。
    config_path : str | None
        可选 yaml 配置路径。
    gravity : str | None
        可选重力方向配置；为空时使用代码里写死的默认重力向量。
    export_class_name : str
        导出的 C++ 类名，同时也会决定生成的头/源文件名。
    solver_method : str
        参数辨识求解方法，支持 `ols`、`wls`、`ridge`、`constrained`。
    payload_mode : str
        payload 处理模式，支持 `none`、`lumped_last_link`、`external_wrench`。
    payload_mass : float
        已知 payload 质量，单位 kg。
    payload_com : Iterable[float] | None
        payload 质心位置，默认相对 payload_reference_link 表达。
    payload_reference_link : str | None
        payload 参考连杆，默认使用末端连杆。
    payload_com_frame : str | None
        payload COM 所在坐标系名称。
    subtract_known_payload_gravity : bool
        若为 True，则在辨识本体参数前先扣除已知 payload 重力项。

    Returns
    -------
    dict
        一份精简版结果摘要，便于其他脚本通过 Python API 复用。

    Raises
    ------
    ValueError
        当辨识模式、参数化方式或数据来源不受支持时抛出。
    """
    # 当前辨识主链仍然只支持 rigid_body_friction 这个模式；
    # 但机器人模型本身不再强绑 Panda，只要能从 URDF 中正确读出关节动力学参数即可。
    if identification_mode != 'rigid_body_friction':
        raise ValueError("Only identification_mode='rigid_body_friction' is currently supported.")
    if parameterization not in SUPPORTED_PARAMETERIZATIONS:
        raise ValueError(f"Unsupported parameterization: {parameterization}")
    if data_source not in SUPPORTED_DATA_SOURCES:
        raise ValueError(f"Unsupported data_source: {data_source}")
    if real_torque_source not in SUPPORTED_REAL_TORQUE_SOURCES:
        raise ValueError(f"Unsupported real_torque_source: {real_torque_source}")
    if solver_method not in SUPPORTED_SOLVER_METHODS:
        raise ValueError(f"Unsupported solver_method: {solver_method}")
    effective_payload_mode = PayloadMode.from_value(payload_mode)

    # 真实数据统一约定放在一个目录里，默认是：
    # `datasets/real/raw/`
    #
    # 这么做的好处是：
    # - 管理方便，新增文件不需要改代码；
    # - 多次实验文件能集中存放；
    # - 更容易做批量处理、统计和留档。
    #
    # 目录层级上也更清楚：
    # - `datasets/synthetic/` 和 `datasets/real/` 是两类平行的数据来源；
    # - 它们各自下面再区分 raw / processed 等处理阶段。
    #
    # 如果你以后想换目录，也可以通过 `--real-data-dir` 传进来。
    real_data_path = Path(real_data_dir) if real_data_dir else DEFAULT_REAL_DATA_DIR
    effective_data_source = _resolve_data_source(data_source, real_data_path)
    effective_gravity = gravity or DEFAULT_GRAVITY_VECTOR

    print("\n" + "=" * 78)
    print(" PHYSICALLY CONSISTENT ROBOT DYNAMICS IDENTIFICATION PIPELINE")
    print("=" * 78 + "\n")
    print(f"Requested robot name: {robot_name or 'auto-from-urdf'}")
    print(f"Identification mode: {identification_mode}")
    print(f"Parameterization: {parameterization}\n")
    print(f"Data source: {effective_data_source}")
    if effective_data_source == 'real':
        # 这里打印出来，是为了避免“代码实际读了哪个目录”不清楚。
        print(f"Real data directory: {real_data_path}")
        print(f"Real torque source: {real_torque_source}")
    print(f"Gravity vector: {effective_gravity}")
    print(f"Solver method: {solver_method}")
    print(f"Payload mode: {effective_payload_mode.value}")
    if payload_mass > 0.0:
        print(f"Payload mass: {payload_mass} kg")
        print(f"Payload COM: {payload_com}")
    print(f"Sampling frequency: {sampling_freq} Hz")
    print(f"Low-pass cutoff: {cutoff_hz} Hz\n")

    # ------------------------------------------------------------------
    # STEP 1: 载入机器人模型
    # ------------------------------------------------------------------
    # 这里会读取：
    # - URDF：机器人连杆/关节/惯性等结构描述
    # - YAML 配置：项目自己补充的模型配置
    #
    # 现在模型入口统一支持“直接给一个 URDF 文件路径”。
    # config_path 变成可选项：
    # - 如果提供 yaml，就优先读取里面的 active joints / base / ee 等元信息；
    # - 如果不提供，就直接从 URDF 自动推断。
    print("[STEP 1] Loading robot model from URDF...")
    print("-" * 78)
    resolved_urdf_path = Path(urdf_path) if urdf_path else DEFAULT_URDF_PATH
    resolved_config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    loader = URDFLoader(
        str(resolved_urdf_path),
        str(resolved_config_path) if resolved_config_path else None,
        gravity_vector=effective_gravity,
    )
    robot_model = loader.build_robot_model()
    if robot_name and robot_name != robot_model.name:
        print(f"Warning: requested robot_name='{robot_name}' but URDF resolved to '{robot_model.name}'. Using URDF metadata.")
    print_model_info(robot_model)

    if effective_data_source != 'real':
        # ------------------------------------------------------------------
        # STEP 2: 生成 golden data（标准对照数据）
        # ------------------------------------------------------------------
        # 这一步只在 synthetic / 调试场景下保留。
        # 它的目标不是“模拟真实采集”，而是生成一批可控参考数据，
        # 方便验证动力学实现、做回归测试、排查某次改动是否把公式改坏了。
        print("[STEP 2] Generating golden data...")
        print("-" * 78)
        golden_dir = project_root / "datasets" / "golden"
        golden_dir.mkdir(parents=True, exist_ok=True)
        generator = GoldenDataGenerator(robot_model)

        fixed_cases = generator.generate_fixed_cases(num_cases=12)
        random_cases = generator.generate_random_cases(num_cases=96, seed=42)
        trajectory_cases = generator.generate_trajectory_cases(num_points=120)

        generator.export_to_json(fixed_cases, str(golden_dir / "fixed_cases.json"))
        generator.export_to_json(random_cases, str(golden_dir / "random_cases_seed42.json"))
        generator.export_to_json(trajectory_cases, str(golden_dir / "trajectory_cases.json"))
        print(f"Generated {len(fixed_cases)} fixed, {len(random_cases)} random, {len(trajectory_cases)} trajectory cases")
    else:
        print("[STEP 2] Skipping golden-data generation for real-data identification.")
        print("-" * 78)

    # ------------------------------------------------------------------
    # STEP 5: 获取并处理测量数据
    # ------------------------------------------------------------------
    # 现在这里同时支持两种“真正的数据来源”：
    # - synthetic: 自动生成合成数据，适合先验证代码链路；
    # - real:      从单独目录批量读取真实 CSV，适合正式辨识分析。
    #
    # `auto` 只是命令行层面的自动选择模式，不算新的数据来源。
    #
    # 对真实数据来说，推荐把所有同格式 CSV 都放到：
    # `datasets/real/raw/`
    # 脚本会自动逐个处理并合并。
    if effective_data_source == 'real':
        real_data_path.mkdir(parents=True, exist_ok=True)
        df_processed = _prepare_real_dataset(
            robot_model,
            real_data_path,
            sampling_freq,
            cutoff_hz,
            real_torque_source,
        )
    else:
        df_processed = _prepare_synthetic_dataset(robot_model, sampling_freq, cutoff_hz)

    payload_model = _build_payload_model(
        robot_model=robot_model,
        payload_mass=payload_mass,
        payload_com=payload_com,
        payload_reference_link=payload_reference_link,
        payload_com_frame=payload_com_frame,
    )
    if subtract_known_payload_gravity:
        df_processed = _subtract_known_payload_gravity(
            df_processed,
            robot_model,
            payload_model,
            effective_gravity,
        )

    # 激励度诊断：粗略判断数据对参数辨识“够不够有信息量”。
    # 如果某些关节运动太单一，或者数据变化不充分，辨识结果通常会很差。
    excitation = compute_excitation_diagnostics(df_processed, robot_model.num_joints)
    print("Excitation diagnostics:")
    for key, value in excitation.items():
        print(f"  {key}: {value:.6f}")

    # 在正式辨识前先切分 train / val / test。
    # 这里尽量按 trajectory 切分，避免同一段连续轨迹同时出现在训练和测试里。
    splits = _split_dataframe_for_learning(df_processed)

    identification_bundle = pipeline_id.run_identification_stage(
        robot_model=robot_model,
        splits=splits,
        effective_data_source=effective_data_source,
        parameterization=parameterization,
        identification_mode=identification_mode,
        solver_method=solver_method,
        real_torque_source=real_torque_source,
        payload_model=payload_model,
        effective_payload_mode=effective_payload_mode,
        subtract_known_payload_gravity=subtract_known_payload_gravity,
    )
    result = identification_bundle['result']

    post_bundle = pipeline_post.run_post_identification_stage(
        project_root=project_root,
        robot_model=robot_model,
        df_processed=df_processed,
        splits=splits,
        identification_bundle=identification_bundle,
        effective_data_source=effective_data_source,
        real_torque_source=real_torque_source,
        identification_mode=identification_mode,
        effective_gravity=effective_gravity,
        effective_payload_mode=effective_payload_mode,
        payload_model=payload_model,
        resolved_urdf_path=resolved_urdf_path,
        parameterization=parameterization,
        export_class_name=export_class_name,
        split_dataframe_fn=_split_dataframe_for_learning,
    )

    pipeline_post.print_pipeline_summary(
        robot_model=robot_model,
        df_processed=df_processed,
        result=result,
        post_bundle=post_bundle,
    )

    # 返回一个“精简版结果摘要”。
    # 这样如果以后别人想从别的脚本里调用 `run_pipeline()`，
    # 就不用再去解析终端输出或 JSON 文件，直接拿这个返回值即可。
    return {
        'robot_name': robot_model.name,
        'data_source': effective_data_source,
        'real_torque_source': real_torque_source if effective_data_source == 'real' else None,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'parameterization': parameterization,
        'rank': result['full_regressor_rank'],
        'base_parameter_count': result['base_parameter_count'],
        'condition_number': result['condition_number'],
        'active_condition_number': result['active_condition_number'],
        'train_rmse': post_bundle['split_evaluation']['train']['global_rmse'],
        'val_rmse': post_bundle['split_evaluation']['val']['global_rmse'],
        'test_rmse': post_bundle['split_evaluation']['test']['global_rmse'],
        'possible_overfit': post_bundle['overfitting_summary']['possible_overfit'],
        'residual_improvement_test': post_bundle['compensation_eval']['test']['improvement_percent'],
        'export_class_name': export_class_name,
        'generated_dynamics_paths': {key: str(path) for key, path in post_bundle['generated_dynamics_paths'].items()},
    }


def np_to_list(value):
    """
    把 numpy 数组或标量转换成 JSON 友好的 Python 原生类型。

    Parameters
    ----------
    value : Any
        待转换对象。

    Returns
    -------
    list
        适合 `json.dump` 的原生列表结构。
    """
    # `json.dump` 不能直接可靠地处理 numpy 数组/标量，
    # 所以统一转成 Python 原生 list，避免序列化报错。
    return np.asarray(value, dtype=float).tolist()


def parse_args():
    """
    解析命令行参数。

    Returns
    -------
    argparse.Namespace
        已完成默认值填充和合法性检查的参数对象。
    """
    # 命令行参数入口。
    # 目前 choices 故意收得很紧，是为了避免传入当前版本根本不支持的配置。
    # 如果后面扩展了新机器人/新模式/新参数化形式，记得同步修改这里。
    parser = argparse.ArgumentParser(description="Run the robot dynamics identification pipeline.")
    parser.add_argument('--robot-name', default=DEFAULT_ROBOT_NAME)
    parser.add_argument('--identification-mode', default=DEFAULT_IDENTIFICATION_MODE, choices=['rigid_body_friction'])
    parser.add_argument('--parameterization', default=DEFAULT_PARAMETERIZATION, choices=SUPPORTED_PARAMETERIZATIONS)
    parser.add_argument('--solver-method', default=DEFAULT_SOLVER_METHOD, choices=SUPPORTED_SOLVER_METHODS)
    parser.add_argument('--data-source', default=DEFAULT_DATA_SOURCE, choices=SUPPORTED_DATA_SOURCES)
    parser.add_argument('--real-data-dir', default=str(DEFAULT_REAL_DATA_DIR))
    parser.add_argument(
        '--real-torque-source',
        default=DEFAULT_REAL_TORQUE_SOURCE,
        choices=SUPPORTED_REAL_TORQUE_SOURCES,
        help=(
            "真实数据辨识使用的扭矩来源："
            "sensed 使用 q*_SensedTorque（电流估计扭矩），"
            "sensor 使用 q*_JointSensorTorque（扭矩传感器估计扭矩）。"
        ),
    )
    parser.add_argument('--sampling-freq', type=float, default=DEFAULT_SAMPLING_FREQ)
    parser.add_argument('--cutoff-hz', type=float, default=DEFAULT_CUTOFF_HZ)
    parser.add_argument('--urdf-path', default=str(DEFAULT_URDF_PATH))
    parser.add_argument('--config-path', default=None if DEFAULT_CONFIG_PATH is None else str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        '--export-class-name',
        default=DEFAULT_EXPORT_CLASS_NAME,
        help="导出的 C++ 类名，同时也会决定生成的 .h/.cpp 文件名。",
    )
    parser.add_argument(
        '--gravity',
        default=None,
        help=(
            "重力向量（基坐标系，m/s²）。支持预设关键字或逗号分隔三元组。\n"
            "预设关键字: upright(默认), inverted, wall_x, wall_x_pos, wall_y, wall_y_pos\n"
            "示例: --gravity inverted  或  --gravity '0,0,9.81'"
        ),
    )
    parser.add_argument('--payload-mode', default=DEFAULT_PAYLOAD_MODE, choices=[mode.value for mode in PayloadMode])
    parser.add_argument('--payload-mass', type=float, default=0.0)
    parser.add_argument('--payload-com', nargs=3, type=float, default=None, metavar=('CX', 'CY', 'CZ'))
    parser.add_argument('--payload-reference-link', default=None)
    parser.add_argument('--payload-com-frame', default=None)
    parser.add_argument(
        '--subtract-known-payload-gravity',
        action='store_true',
        help="若提供已知 payload，则在本体辨识前先扣除其重力项。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        # 作为脚本直接运行时，走命令行参数；
        # 如果是被别的 Python 文件 import，则不会进入这里。
        args = parse_args()
        run_pipeline(
            robot_name=args.robot_name,
            identification_mode=args.identification_mode,
            parameterization=args.parameterization,
            solver_method=args.solver_method,
            data_source=args.data_source,
            real_data_dir=args.real_data_dir,
            real_torque_source=args.real_torque_source,
            sampling_freq=args.sampling_freq,
            cutoff_hz=args.cutoff_hz,
            urdf_path=args.urdf_path,
            config_path=args.config_path,
            gravity=args.gravity,
            export_class_name=args.export_class_name,
            payload_mode=args.payload_mode,
            payload_mass=args.payload_mass,
            payload_com=args.payload_com,
            payload_reference_link=args.payload_reference_link,
            payload_com_frame=args.payload_com_frame,
            subtract_known_payload_gravity=args.subtract_known_payload_gravity,
        )
    except Exception as exc:
        # 出错时打印完整堆栈，方便定位是哪一步、哪一行失败。
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
