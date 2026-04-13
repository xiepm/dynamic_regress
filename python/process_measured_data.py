"""
Step 5: Process measured data and generate physically consistent synthetic runs.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from generate_golden_data import coulomb_sign
from load_model import pin


class MeasuredDataProcessor:
    """Processing utilities for measured or synthetic Panda data."""

    def __init__(self, num_joints: int = 7):
        self.num_joints = num_joints

    def synchronize_timestamps(self, df: pd.DataFrame, reference_freq: float = 100.0) -> pd.DataFrame:
        df_local = df.copy()
        df_local['timestamp'] = pd.to_numeric(df_local['timestamp'])
        df_local = df_local.sort_values('timestamp').set_index('timestamp')
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
        window_length = int(max(7, round(2 * sampling_freq / cutoff_hz)))
        if window_length % 2 == 0:
            window_length += 1
        polyorder = min(3, window_length - 2)

        filtered = df.copy()
        for column in [col for col in df.columns if col.startswith('q_') or col.startswith('tau_')]:
            filtered[column] = savgol_filter(df[column].values, window_length=window_length, polyorder=polyorder)
        print(f"Applied low-pass filter: cutoff={cutoff_hz}Hz, window={window_length}")
        return filtered

    def differentiate_position(self, df: pd.DataFrame, sampling_freq: float = 100.0) -> pd.DataFrame:
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
        velocity_margin: float = 1.15,
        acceleration_limit: float = 8.0,
    ) -> np.ndarray:
        valid_mask = np.ones(len(df), dtype=bool)
        velocity_cols = [f'dq_{i}' for i in range(1, self.num_joints + 1)]
        acceleration_cols = [f'ddq_{i}' for i in range(1, self.num_joints + 1)]

        observed_velocity = np.max(np.abs(df[velocity_cols].values), axis=0)
        velocity_threshold = np.maximum(observed_velocity * velocity_margin, 0.5)

        for index, column in enumerate(velocity_cols):
            valid_mask &= np.abs(df[column].values) <= velocity_threshold[index]
        for column in acceleration_cols:
            valid_mask &= np.abs(df[column].values) <= acceleration_limit

        invalid_count = int((~valid_mask).sum())
        print(f"Detected {invalid_count} invalid samples ({100 * invalid_count / len(df):.1f}%)")
        return valid_mask

    def clean_and_export(self, df: pd.DataFrame, output_path: str, min_trajectory_length: int = 40) -> pd.DataFrame:
        cleaned = df.copy()
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
        print(f"Saved to {output_path}")
        return cleaned


def _build_exciting_trajectory(num_samples: int, num_joints: int, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    """Create a synthetic run using the same Panda dynamics used for identification."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / sampling_freq
    q, dq, ddq = _build_exciting_trajectory(num_samples, robot_model.num_joints, dt)

    tau = np.zeros_like(q)
    data = robot_model.pinocchio_model.createData()
    for sample_idx in range(num_samples):
        tau_rigid = np.array(pin.rnea(robot_model.pinocchio_model, data, q[sample_idx], dq[sample_idx], ddq[sample_idx]), dtype=float)
        tau_friction = robot_model.damping * dq[sample_idx] + robot_model.friction * coulomb_sign(dq[sample_idx])
        tau[sample_idx] = tau_rigid + tau_friction

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
