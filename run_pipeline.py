#!/usr/bin/env python3
"""
Physically consistent robot dynamics identification pipeline.

这个文件是整个项目的“总调度入口”：
1. 先加载机器人模型；
2. 再生成一批标准/对照用的 golden data；
3. 然后构造并处理测量数据；
4. 接着做动力学参数辨识；
5. 最后做残差分析与补偿。

如果你是第一次接手这个项目，建议先只读 `run_pipeline()`：
- 它基本按照执行顺序从上到下展开；
- 每一段都会把中间结果写到 `datasets/` 下面；
- 大多数“想改流程”的需求，通常都只需要从这里改参数或替换某一步调用。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# 当前脚本所在目录，也就是项目根目录。
# 后面所有相对路径（模型、数据集输出目录等）都基于它来拼接。
project_root = Path(__file__).parent

# 让 Python 能直接 import `python/` 目录下的模块。
# 这样当前目录不用额外安装成 package，也能把工程跑起来。
sys.path.insert(0, str(project_root / "python"))

from generate_golden_data import GoldenDataGenerator
from identify_parameters import ParameterIdentifier
from load_model import URDFLoader, print_model_info
from process_measured_data import (
    MeasuredDataProcessor,
    compute_excitation_diagnostics,
    create_synthetic_measured_data,
)
from residual_compensation import MLPCompensator, ResidualAnalyzer, ResidualCompensator


# 当前版本只支持这两种参数化形式：
# - base: 只辨识可辨识的基参数，通常更稳健；
# - full: 保留完整参数形式，便于研究和对比。
SUPPORTED_PARAMETERIZATIONS = ('base', 'full')
# 这里要区分“数据来源”和“运行模式”：
# - 从业务上说，项目目前只有两类数据来源：synthetic（合成）和 real（真实测量）；
# - 从命令行运行上说，多了一个 auto 模式，它只是帮你自动在这两类来源里做选择。
#   换句话说，auto 不是第三种数据，只是一个“自动判断用哪种数据来源”的开关。
SUPPORTED_DATA_SOURCES = ('auto', 'synthetic', 'real')


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


def _normalize_real_measured_dataframe(df_raw: pd.DataFrame, num_joints: int, sampling_freq: float) -> pd.DataFrame:
    """
    把真实 CSV 映射成项目内部统一使用的列名。

    外部真实文件列名示例：
    - q1_pos / q1_vel / q1_acc / q1_sensedTorque / q1_cur

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
    # 当前这套真实 CSV 约定是：
    # - 位置 `q*_pos`：角度（deg）
    # - 速度 `q*_vel`：度每秒（deg/s）
    # - 加速度 `q*_acc`：度每二次方秒（deg/s^2）
    # - 力矩 `q*_sensedTorque`：牛米（N·m）
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

    # 这里定义“外部真实列名”到“内部统一列名”的映射规则。
    # 如果以后采集系统升级、列名改了，优先改这里，而不是去动后面的算法模块。
    required_suffixes = {
        'q': 'pos',
        'dq': 'vel',
        'ddq': 'acc',
        'tau': 'sensedTorque',
        'current': 'cur',
    }

    for joint_idx in range(1, num_joints + 1):
        for target_prefix, source_suffix in required_suffixes.items():
            source_col = f'q{joint_idx}_{source_suffix}'
            if target_prefix in ('tau', 'current'):
                source_col = f'q{joint_idx}_{source_suffix}'
            if source_col not in df_raw.columns:
                if target_prefix == 'current':
                    normalized[f'current_{joint_idx}'] = np.zeros(len(df_raw), dtype=float)
                    continue
                raise ValueError(f"Missing required column in real measured CSV: {source_col}")

            target_col = f'{target_prefix}_{joint_idx}'
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
    print("Converted real-data units to SI: q->rad, dq->rad/s, ddq->rad/s^2, tau kept in N·m")
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
    df_processed['source_file'] = raw_path.name
    return df_processed


def _prepare_real_dataset(robot_model, real_data_dir: Path, sampling_freq: float, cutoff_hz: float) -> pd.DataFrame:
    # real 分支专门负责“批量读取真实 CSV -> 统一格式 -> 清洗 -> 合并”。
    # 注意这里假设目录中的文件格式一致；如果后续出现不同采集协议，
    # 最好新增不同的适配函数，而不是把所有特殊情况都堆在一个函数里。
    #
    # 这里的目录关系同样是：
    # - `real/` 是一种数据来源；
    # - `raw/normalized/processed/` 是真实数据在项目内部流转的几个阶段目录。
    print("\n[STEP 5] Loading and processing real data...")
    print("-" * 78)

    processor = MeasuredDataProcessor(num_joints=robot_model.num_joints)
    csv_files = _load_real_data_files(real_data_dir)

    normalized_dir = project_root / "datasets" / "real" / "normalized"
    processed_dir = project_root / "datasets" / "real" / "processed"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

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
        df_normalized = _normalize_real_measured_dataframe(df_raw, robot_model.num_joints, sampling_freq)

        normalized_path = normalized_dir / f"{csv_path.stem}_normalized.csv"
        df_normalized.to_csv(normalized_path, index=False)

        df_synced = processor.synchronize_timestamps(df_normalized, reference_freq=sampling_freq)
        df_filtered = processor.apply_low_pass_filter(df_synced, cutoff_hz=cutoff_hz, sampling_freq=sampling_freq)

        processed_path = processed_dir / f"{csv_path.stem}_processed.csv"
        df_processed_single = processor.clean_and_export(df_filtered, str(processed_path))
        if df_processed_single.empty:
            print(f"Skipped {csv_path.name}: no valid samples after cleaning.")
            continue

        df_processed_single['source_file'] = csv_path.name
        df_processed_single['source_path'] = str(csv_path)
        if 'trajectory_id' in df_processed_single.columns:
            # 不同文件内部可能都会从 trajectory_id=0 开始编号。
            # 这里做一个全局偏移，避免多个文件合并后轨迹编号冲突。
            df_processed_single['trajectory_id'] = df_processed_single['trajectory_id'] + trajectory_offset
            trajectory_offset = int(df_processed_single['trajectory_id'].max()) + 1
        processed_frames.append(df_processed_single)

    if not processed_frames:
        raise ValueError("All real data files were filtered out during cleaning. Please check the raw signals or thresholds.")

    combined = pd.concat(processed_frames, ignore_index=True)
    combined_path = processed_dir / "real_combined_processed.csv"
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
    # 用同一组已辨识参数，分别在 train / val / test 上做评估。
    # 这样终端和 JSON 里就能直接横向比较三份数据上的误差表现。
    split_metrics = {}
    for split_name, split_df in splits.items():
        print(f"\nEvaluating identification on {split_name} split...")
        split_metrics[split_name] = identifier.evaluate_identification(split_df, result)
    return split_metrics


def _summarize_overfitting(split_metrics: dict[str, dict]) -> dict[str, float | bool]:
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
    # 新增：把 joint_1 ... joint_n 这种字典稳定转换成数组，方便 notebook 直接按关节顺序画图。
    ordered_items = sorted(metric_map.items(), key=lambda item: int(item[0].split('_')[-1]))
    return [float(value) for _, value in ordered_items]


def _write_visualization_payload(
    robot_model,
    effective_data_source: str,
    split_evaluation: dict[str, dict],
    compensation_eval: dict[str, dict],
    mlp_eval: dict[str, dict],
    stability_eval: dict,
) -> Path:
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
        'joints': [f'J{joint_idx}' for joint_idx in range(1, joint_count + 1)],
        'stages': ['Identification', '+ Linear comp.', '+ MLP comp.'],
        'paths': {
            'figure_dir': str(figure_dir),
            'latest_results_file': str(vis_dir / "latest_results.json"),
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

    output_path = vis_dir / "latest_results.json"
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(f"[Diag] Visualization payload updated: {output_path}")
    return output_path


def _evaluate_residual_compensation_by_split(analyzer, compensator, identifier, result: dict, splits: dict[str, pd.DataFrame]) -> dict[str, dict]:
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


def run_pipeline(
    robot_name: str | None = None,
    identification_mode: str = 'rigid_body_friction',
    parameterization: str = 'base',
    data_source: str = 'auto',
    real_data_dir: str | None = None,
    sampling_freq: float = 100.0,
    cutoff_hz: float = 15.0,
    urdf_path: str | None = None,
    config_path: str | None = None,
):
    # 当前辨识主链仍然只支持 rigid_body_friction 这个模式；
    # 但机器人模型本身不再强绑 Panda，只要能从 URDF 中正确读出关节动力学参数即可。
    if identification_mode != 'rigid_body_friction':
        raise ValueError("Only identification_mode='rigid_body_friction' is currently supported.")
    if parameterization not in SUPPORTED_PARAMETERIZATIONS:
        raise ValueError(f"Unsupported parameterization: {parameterization}")
    if data_source not in SUPPORTED_DATA_SOURCES:
        raise ValueError(f"Unsupported data_source: {data_source}")

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
    real_data_path = Path(real_data_dir) if real_data_dir else project_root / "datasets" / "real" / "raw"
    effective_data_source = _resolve_data_source(data_source, real_data_path)

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
    resolved_urdf_path = Path(urdf_path) if urdf_path else project_root / "models" / "05_urdf" / "urdf" / "05_urdf_temp.urdf"
    resolved_config_path = Path(config_path) if config_path else None
    loader = URDFLoader(str(resolved_urdf_path), str(resolved_config_path) if resolved_config_path else None)
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
        df_processed = _prepare_real_dataset(robot_model, real_data_path, sampling_freq, cutoff_hz)
    else:
        df_processed = _prepare_synthetic_dataset(robot_model, sampling_freq, cutoff_hz)

    # 激励度诊断：粗略判断数据对参数辨识“够不够有信息量”。
    # 如果某些关节运动太单一，或者数据变化不充分，辨识结果通常会很差。
    excitation = compute_excitation_diagnostics(df_processed, robot_model.num_joints)
    print("Excitation diagnostics:")
    for key, value in excitation.items():
        print(f"  {key}: {value:.6f}")

    # 在正式辨识前先切分 train / val / test。
    # 这里尽量按 trajectory 切分，避免同一段连续轨迹同时出现在训练和测试里。
    splits = _split_dataframe_for_learning(df_processed)

    # ------------------------------------------------------------------
    # STEP 6: 动力学参数辨识
    # ------------------------------------------------------------------
    # 这一步会根据处理后的测量数据，估计动力学参数。
    # 当前默认方法是 OLS（普通最小二乘），适合先跑通基线流程。
    #
    # 如果后面你想尝试岭回归、加权最小二乘、带约束优化等方法，
    # 通常入口就是这里的 `method` 或 `ParameterIdentifier` 内部实现。
    print("\n[STEP 6] Identifying parameters...")
    print("-" * 78)
    identifier = ParameterIdentifier(robot_model, parameterization=parameterization)
    reference_parameters = None if effective_data_source == 'real' else robot_model.full_parameter_vector()
    # 修复：默认切到 ridge，降低病态法方程下纯 OLS 的数值不稳定性。
    result = identifier.identify_parameters(
        splits['train'],
        method='ridge',
        reference_parameters=reference_parameters,
        ridge_lambda=1e-4,
    )

    # 训练只在 train 集上做，但评估会同时看 train / val / test。
    # 这样就能比较：
    # - 训练误差是否很低；
    # - 验证/测试误差是否明显变差；
    # - 是否有过拟合迹象。
    split_evaluation = _evaluate_identification_splits(identifier, result, splits)
    overfitting_summary = _summarize_overfitting(split_evaluation)
    _print_per_joint_summary(
        "Identification per-joint metrics (test split):",
        split_evaluation['test'],
        ['joint_rmse', 'joint_mae'],
    )

    identified_dir = project_root / "datasets" / "identified"
    identified_dir.mkdir(parents=True, exist_ok=True)
    result_stem = 'real' if effective_data_source == 'real' else 'synthetic'

    # 这里保存“辨识出的参数结果”。
    # 之所以手工组织成字典再写 JSON，是为了让输出结构清晰、稳定，
    # 后续画图、对比实验、写报告时都更容易复用。
    with open(identified_dir / f"theta_hat_{result_stem}.json", 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'robot_name': robot_model.name,
                'identification_mode': identification_mode,
                'parameterization': parameterization,
                'data_source': effective_data_source,
                'rank': result['rank'],
                'full_regressor_rank': result['full_regressor_rank'],
                'base_parameter_count': result['base_parameter_count'],
                'condition_number': result['condition_number'],
                'active_condition_number': result['active_condition_number'],
                'active_parameter_indices': result['active_parameter_indices'].tolist(),
                'theta_hat': np_to_list(result['theta_hat']),
                'theta_hat_full': np_to_list(result['theta_hat_full']),
                'reference_parameters': None if result['reference_parameters'] is None else np_to_list(result['reference_parameters']),
                'method': result['method'],
                'ridge_lambda': result['ridge_lambda'],
            },
            handle,
            indent=2,
        )

    # 这里保存按数据集切分后的评估结果，以及一个简单的过拟合诊断摘要。
    with open(identified_dir / f"evaluation_{result_stem}_splits.json", 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'split_metrics': split_evaluation,
                'overfitting_summary': overfitting_summary,
            },
            handle,
            indent=2,
        )

    # ------------------------------------------------------------------
    # STEP 7-8: 残差分析与补偿
    # ------------------------------------------------------------------
    # 即便刚体 + 摩擦模型已经辨识完成，真实力矩里仍然可能有解释不了的残差。
    # 这一步会：
    # 1. 先构建残差学习数据集；
    # 2. 再训练一个简单补偿器；
    # 3. 最后看补偿后误差有没有下降。
    print("\n[STEP 7-8] Residual analysis and compensation...")
    print("-" * 78)
    analyzer = ResidualAnalyzer(robot_model)
    compensator = ResidualCompensator(num_joints=robot_model.num_joints)
    compensation_eval = _evaluate_residual_compensation_by_split(analyzer, compensator, identifier, result, splits)
    # 新增：非线性补偿优先改成 MLP，更贴近你现在希望尝试的神经网络路线。
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

    residual_dir = project_root / "datasets" / "residual"
    residual_dir.mkdir(parents=True, exist_ok=True)

    # 单独保存补偿效果，方便和“未补偿时的辨识结果”做横向比较。
    with open(residual_dir / f"compensation_result_{result_stem}.json", 'w', encoding='utf-8') as handle:
        json.dump({'linear': compensation_eval, 'mlp': mlp_eval}, handle, indent=2)

    stability_eval = run_stability_evaluation(df_processed, identifier, result, n_seeds=5)
    with open(identified_dir / "stability_eval.json", 'w', encoding='utf-8') as handle:
        json.dump(stability_eval, handle, indent=2)
    _write_visualization_payload(
        robot_model=robot_model,
        effective_data_source=effective_data_source,
        split_evaluation=split_evaluation,
        compensation_eval=compensation_eval,
        mlp_eval=mlp_eval,
        stability_eval=stability_eval,
    )

    # 最后在终端打印一份概要，方便快速看这次运行是否正常、效果大概如何。
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
    print("=" * 78 + "\n")

    # 返回一个“精简版结果摘要”。
    # 这样如果以后别人想从别的脚本里调用 `run_pipeline()`，
    # 就不用再去解析终端输出或 JSON 文件，直接拿这个返回值即可。
    return {
        'robot_name': robot_model.name,
        'data_source': effective_data_source,
        'parameterization': parameterization,
        'rank': result['full_regressor_rank'],
        'base_parameter_count': result['base_parameter_count'],
        'condition_number': result['condition_number'],
        'active_condition_number': result['active_condition_number'],
        'train_rmse': split_evaluation['train']['global_rmse'],
        'val_rmse': split_evaluation['val']['global_rmse'],
        'test_rmse': split_evaluation['test']['global_rmse'],
        'possible_overfit': overfitting_summary['possible_overfit'],
        'residual_improvement_test': compensation_eval['test']['improvement_percent'],
    }


def np_to_list(value):
    # `json.dump` 不能直接可靠地处理 numpy 数组/标量，
    # 所以统一转成 Python 原生 list，避免序列化报错。
    return np.asarray(value, dtype=float).tolist()


def parse_args():
    # 命令行参数入口。
    # 目前 choices 故意收得很紧，是为了避免传入当前版本根本不支持的配置。
    # 如果后面扩展了新机器人/新模式/新参数化形式，记得同步修改这里。
    parser = argparse.ArgumentParser(description="Run the robot dynamics identification pipeline.")
    parser.add_argument('--robot-name', default=None)
    parser.add_argument('--identification-mode', default='rigid_body_friction', choices=['rigid_body_friction'])
    parser.add_argument('--parameterization', default='base', choices=SUPPORTED_PARAMETERIZATIONS)
    parser.add_argument('--data-source', default='auto', choices=SUPPORTED_DATA_SOURCES)
    parser.add_argument('--real-data-dir', default=str(project_root / "datasets" / "real" / "raw"))
    parser.add_argument('--sampling-freq', type=float, default=100.0)
    parser.add_argument('--cutoff-hz', type=float, default=15.0)
    parser.add_argument('--urdf-path', default=str(project_root / "models" / "05_urdf" / "urdf" / "05_urdf_temp.urdf"))
    parser.add_argument('--config-path', default=None)
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
            data_source=args.data_source,
            real_data_dir=args.real_data_dir,
            sampling_freq=args.sampling_freq,
            cutoff_hz=args.cutoff_hz,
            urdf_path=args.urdf_path,
            config_path=args.config_path,
        )
    except Exception as exc:
        # 出错时打印完整堆栈，方便定位是哪一步、哪一行失败。
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
