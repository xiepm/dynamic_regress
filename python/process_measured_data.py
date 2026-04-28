"""
处理真实测量数据，并生成与辨识主链兼容的合成运行数据。

这个模块位于流水线的 Step 5，由 `run_pipeline.py` 在模型加载之后调用。
它负责把真实或 synthetic 数据整理成统一的 `timestamp / q / dq / ddq / tau`
表格格式，并完成同步、滤波、求导、样本清洗和激励度诊断，供后续参数辨识与残差补偿
模块直接消费。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from generate_golden_data import coulomb_sign
from load_model import pin


class MeasuredDataProcessor:
    """
    处理测量数据的工具类。

    这个类把“同步、滤波、求导、清洗”这些时序预处理步骤集中管理，目的是让上游数据入口
    和下游辨识器之间通过统一表格接口解耦。类本身几乎不持久化状态，关键状态只有
    `num_joints`，用于约束列名和循环维度。
    """

    def __init__(self, num_joints: int = 7):
        """
        记录关节数量，用于驱动后续列处理逻辑。

        Parameters
        ----------
        num_joints : int
            机器人主动关节数。
        """
        self.num_joints = num_joints

    def synchronize_timestamps(self, df: pd.DataFrame, reference_freq: float = 100.0) -> pd.DataFrame:
        """
        把原始时间戳重采样到统一参考频率。

        Parameters
        ----------
        df : pd.DataFrame
            含 `timestamp` 列的原始数据表。
        reference_freq : float
            目标重采样频率，单位 Hz。

        Returns
        -------
        pd.DataFrame
            在统一时间轴上插值后的数据表。
        """
        df_local = df.copy()
        df_local['timestamp'] = pd.to_numeric(df_local['timestamp'])
        df_local = df_local.sort_values('timestamp').set_index('timestamp')
        # 统一时间轴是后续滤波、求导和数据集对齐的前提，否则不同文件间采样间隔不可比较。
        target_times = np.arange(df_local.index[0], df_local.index[-1] + 1e-12, 1.0 / reference_freq)
        synced = df_local.reindex(df_local.index.union(target_times)).interpolate(method='index').reindex(target_times)
        synced = synced.reset_index().rename(columns={'index': 'timestamp'})
        print(f"Synchronized to {reference_freq}Hz: {len(synced)} samples")
        return synced

    def apply_low_pass_filter(
        self,
        df: pd.DataFrame,
        cutoff_hz: float = 15.0,
        sampling_freq: float = 100.0,
    ) -> pd.DataFrame:
        """
        对位置、速度、加速度和扭矩列做低通滤波。

        Parameters
        ----------
        df : pd.DataFrame
            待滤波数据表。
        cutoff_hz : float
            截止频率，单位 Hz。
        sampling_freq : float
            采样频率，单位 Hz。

        Returns
        -------
        pd.DataFrame
            滤波后的数据表；如果样本太少，会直接返回原始副本。
        """
        if len(df) < 5:
            print("Skipped low-pass filter: not enough samples")
            return df.copy()

        # Savitzky-Golay 需要奇数窗口；窗口过短时宁可跳过滤波，也不做失真的局部拟合。
        window_length = int(max(7, round(2 * sampling_freq / cutoff_hz)))
        window_length = min(window_length, len(df) if len(df) % 2 == 1 else len(df) - 1)
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 5:
            print("Skipped low-pass filter: effective window too short")
            return df.copy()
        polyorder = min(3, window_length - 2)

        filtered = df.copy()
        filterable_columns = [
            col for col in df.columns
            if col.startswith('q_')
            or col.startswith('dq_')
            or col.startswith('ddq_')
            or col.startswith('tau_')
        ]
        for column in filterable_columns:
            filtered[column] = savgol_filter(df[column].values, window_length=window_length, polyorder=polyorder)
        print(f"Applied low-pass filter: cutoff={cutoff_hz}Hz, window={window_length}, columns={len(filterable_columns)}")
        return filtered

    def differentiate_position(self, df: pd.DataFrame, sampling_freq: float = 100.0) -> pd.DataFrame:
        """
        通过数值求导从位置列恢复速度和加速度列。

        Parameters
        ----------
        df : pd.DataFrame
            至少含 `q_i` 列的数据表。
        sampling_freq : float
            采样频率，单位 Hz。

        Returns
        -------
        pd.DataFrame
            新增 `dq_i` 和 `ddq_i` 列的数据表副本。
        """
        dt = 1.0 / sampling_freq
        differentiated = df.copy()
        for joint_idx in range(1, self.num_joints + 1):
            q_col = f'q_{joint_idx}'
            differentiated[f'dq_{joint_idx}'] = np.gradient(differentiated[q_col].values, dt, edge_order=2)
            differentiated[f'ddq_{joint_idx}'] = np.gradient(differentiated[f'dq_{joint_idx}'].values, dt, edge_order=2)
        print(f"Differentiated {self.num_joints} joints into dq and ddq")
        return differentiated

    def detect_invalid_samples(
        self,
        df: pd.DataFrame,
        acceleration_margin: float = 3.0,
        min_acceleration_limit: float = 5.0,
    ) -> np.ndarray:
        """
        检测无效样本。

        当前策略刻意不再做“速度阈值筛除”，原因是之前那套阈值来自当前数据本身的最大速度，
        逻辑上几乎不会真正筛掉异常点，容易造成“看起来有检查，其实基本没作用”的假象。

        现在只保留加速度筛查，并改成：
        - 每个关节分别估计自己的加速度统计尺度
        - 用 `observed_max * acceleration_margin` 形成自适应阈值
        - 再和 `min_acceleration_limit` 取最大值，避免动作太小时阈值过紧
        """
        valid_mask = np.ones(len(df), dtype=bool)
        acceleration_cols = [f'ddq_{i}' for i in range(1, self.num_joints + 1)]

        observed_acceleration = np.max(np.abs(df[acceleration_cols].values), axis=0)
        acceleration_threshold = np.maximum(observed_acceleration * acceleration_margin, min_acceleration_limit)

        for index, column in enumerate(acceleration_cols):
            valid_mask &= np.abs(df[column].values) <= acceleration_threshold[index]

        invalid_count = int((~valid_mask).sum())
        print(f"Detected {invalid_count} invalid samples ({100 * invalid_count / len(df):.1f}%)")
        print(f"Adaptive acceleration thresholds: {np.array2string(acceleration_threshold, precision=3)}")
        return valid_mask

    def clean_and_export(self, df: pd.DataFrame, output_path: str, min_trajectory_length: int = 40) -> pd.DataFrame:
        """
        根据有效样本掩码切分轨迹段，并导出清洗结果。

        Parameters
        ----------
        df : pd.DataFrame
            待清洗的数据表。
        output_path : str
            输出 CSV 路径。
        min_trajectory_length : int
            轨迹段最小长度，小于该值的连续有效片段会被丢弃。

        Returns
        -------
        pd.DataFrame
            仅保留有效轨迹段的清洗后数据表。
        """
        # 先重置索引，确保下面基于位置的起止区间与 DataFrame 标签一致。
        cleaned = df.reset_index(drop=True).copy()
        cleaned['valid_mask'] = self.detect_invalid_samples(cleaned)
        cleaned['trajectory_id'] = -1

        current_id = 0
        start = None
        for row_index, is_valid in enumerate(cleaned['valid_mask'].values):
            if is_valid and start is None:
                start = row_index
            elif (not is_valid) and start is not None:
                if row_index - start >= min_trajectory_length:
                    cleaned.loc[start:row_index - 1, 'trajectory_id'] = current_id
                    current_id += 1
                start = None

        if start is not None and len(cleaned) - start >= min_trajectory_length:
            cleaned.loc[start:, 'trajectory_id'] = current_id

        cleaned = cleaned[cleaned['trajectory_id'] >= 0].reset_index(drop=True)
        cleaned.to_csv(output_path, index=False)
        print(f"After cleaning: {len(cleaned)} valid samples, {cleaned['trajectory_id'].nunique()} trajectories")
        if not cleaned.empty:
            trajectory_lengths = cleaned.groupby('trajectory_id').size().to_numpy()
            print(
                "Trajectory lengths: "
                f"min={trajectory_lengths.min()}, "
                f"max={trajectory_lengths.max()}, "
                f"mean={trajectory_lengths.mean():.1f}"
            )
        print(f"Saved to {output_path}")
        return cleaned


def _build_exciting_trajectory(num_samples: int, num_joints: int, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造一段多频正弦激励轨迹，用于 synthetic 数据生成。"""
    time = np.arange(num_samples) * dt
    q = np.zeros((num_samples, num_joints))
    dq = np.zeros((num_samples, num_joints))
    ddq = np.zeros((num_samples, num_joints))
    phases = np.linspace(0.0, np.pi / 3.0, num_joints)

    for joint_idx in range(num_joints):
        freq1 = 0.25 + 0.05 * joint_idx
        freq2 = 0.55 + 0.04 * joint_idx
        amp1 = 0.45 - 0.03 * joint_idx
        amp2 = 0.18 - 0.01 * joint_idx
        phase = phases[joint_idx]

        q[:, joint_idx] = (
            amp1 * np.sin(2 * np.pi * freq1 * time + phase)
            + amp2 * np.sin(2 * np.pi * freq2 * time + 0.3 + phase)
        )
        dq[:, joint_idx] = (
            amp1 * 2 * np.pi * freq1 * np.cos(2 * np.pi * freq1 * time + phase)
            + amp2 * 2 * np.pi * freq2 * np.cos(2 * np.pi * freq2 * time + 0.3 + phase)
        )
        ddq[:, joint_idx] = (
            -amp1 * (2 * np.pi * freq1) ** 2 * np.sin(2 * np.pi * freq1 * time + phase)
            - amp2 * (2 * np.pi * freq2) ** 2 * np.sin(2 * np.pi * freq2 * time + 0.3 + phase)
        )
    return q, dq, ddq


def create_synthetic_measured_data(
    robot_model,
    num_samples: int = 1500,
    sampling_freq: float = 100.0,
    torque_noise_std: float = 0.02,
    timestamp_jitter_std: float = 1e-4,
    seed: int = 2026,
) -> pd.DataFrame:
    """
    生成一段与辨识主链一致口径的 synthetic 测量数据。

    Parameters
    ----------
    robot_model : RobotModel
        已加载好的机器人模型。
    num_samples : int
        样本数。
    sampling_freq : float
        采样频率，单位 Hz。
    torque_noise_std : float
        扭矩高斯噪声标准差，单位 N·m。
    timestamp_jitter_std : float
        时间戳抖动标准差，单位 s。
    seed : int
        随机种子。

    Returns
    -------
    pd.DataFrame
        含时间戳、位置、真实速度/加速度和扭矩的 synthetic 数据表。
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / sampling_freq
    q, dq, ddq = _build_exciting_trajectory(num_samples, robot_model.num_joints, dt)

    tau = np.zeros_like(q)
    data = robot_model.pinocchio_model.createData()
    for sample_idx in range(num_samples):
        tau_rigid = np.array(pin.rnea(robot_model.pinocchio_model, data, q[sample_idx], dq[sample_idx], ddq[sample_idx]), dtype=float)
        tau_friction = robot_model.damping * dq[sample_idx] + robot_model.friction * coulomb_sign(dq[sample_idx])
        tau[sample_idx] = tau_rigid + tau_friction

    # 给 synthetic 数据加一点噪声和时间抖动，是为了让它更接近真实采样而不是“完美仿真值”。
    tau += rng.normal(0.0, torque_noise_std, size=tau.shape)
    timestamps = np.arange(num_samples) * dt + rng.normal(0.0, timestamp_jitter_std, size=num_samples)
    timestamps[0] = 0.0
    timestamps = np.maximum.accumulate(timestamps)

    frame = {'timestamp': timestamps}
    for joint_idx in range(robot_model.num_joints):
        frame[f'q_{joint_idx + 1}'] = q[:, joint_idx]
        frame[f'dq_true_{joint_idx + 1}'] = dq[:, joint_idx]
        frame[f'ddq_true_{joint_idx + 1}'] = ddq[:, joint_idx]
        frame[f'tau_{joint_idx + 1}'] = tau[:, joint_idx]
    return pd.DataFrame(frame)


def compute_excitation_diagnostics(df: pd.DataFrame, num_joints: int) -> Dict[str, float]:
    """
    计算一组简单的激励度诊断指标。

    Parameters
    ----------
    df : pd.DataFrame
        处理后的数据集。
    num_joints : int
        主动关节数。

    Returns
    -------
    Dict[str, float]
        平均关节跨度、速度/加速度 RMS、零速度占比等诊断量。
    """
    q_cols = [f'q_{i}' for i in range(1, num_joints + 1)]
    dq_cols = [f'dq_{i}' for i in range(1, num_joints + 1)]
    ddq_cols = [f'ddq_{i}' for i in range(1, num_joints + 1)]

    q_span = float(np.mean(df[q_cols].max() - df[q_cols].min()))
    dq_rms = float(np.sqrt(np.mean(df[dq_cols].values ** 2)))
    ddq_rms = float(np.sqrt(np.mean(df[ddq_cols].values ** 2)))
    zero_velocity_ratio = float(np.mean(np.abs(df[dq_cols].values) < 1e-3))
    return {
        'mean_joint_span_rad': q_span,
        'dq_rms': dq_rms,
        'ddq_rms': ddq_rms,
        'zero_velocity_ratio': zero_velocity_ratio,
    }
