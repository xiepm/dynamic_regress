"""
Step 7-8: Residual analysis and lightweight compensation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class ResidualAnalyzer:
    """Analyze residuals produced by the same identification model."""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.num_joints = robot_model.num_joints

    def compute_residuals(self, df: pd.DataFrame, identifier, result: Dict) -> pd.DataFrame:
        tau_pred = identifier.predict_torques(df, result)
        tau_meas = np.column_stack([df[f'tau_{i}'].values for i in range(1, self.num_joints + 1)])
        residual = tau_meas - tau_pred

        frame = df.copy()
        for joint_idx in range(self.num_joints):
            frame[f'tau_pred_{joint_idx + 1}'] = tau_pred[:, joint_idx]
            frame[f'residual_tau_{joint_idx + 1}'] = residual[:, joint_idx]
        return frame

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        for joint_idx in range(1, self.num_joints + 1):
            frame[f'abs_dq_{joint_idx}'] = np.abs(df[f'dq_{joint_idx}'])
            frame[f'abs_ddq_{joint_idx}'] = np.abs(df[f'ddq_{joint_idx}'])
            frame[f'q_dq_prod_{joint_idx}'] = df[f'q_{joint_idx}'] * df[f'dq_{joint_idx}']
        frame['motion_magnitude'] = np.linalg.norm(
            np.column_stack([df[f'dq_{joint_idx}'].values for joint_idx in range(1, self.num_joints + 1)]),
            axis=1,
        )
        return frame

    def build_residual_dataset(self, df: pd.DataFrame, identifier, result: Dict, test_split: float = 0.2, val_split: float = 0.1) -> Dict:
        residual_df = self.compute_residuals(df, identifier, result)
        feature_df = self.feature_engineering(residual_df)

        n_samples = len(feature_df)
        n_test = int(n_samples * test_split)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_test - n_val

        indices = np.random.default_rng(123).permutation(n_samples)
        datasets = {
            'train': feature_df.iloc[indices[:n_train]].reset_index(drop=True),
            'val': feature_df.iloc[indices[n_train:n_train + n_val]].reset_index(drop=True),
            'test': feature_df.iloc[indices[n_train + n_val:]].reset_index(drop=True),
        }
        print("Residual dataset split:")
        print(f"  Train: {len(datasets['train'])} samples")
        print(f"  Val:   {len(datasets['val'])} samples")
        print(f"  Test:  {len(datasets['test'])} samples")
        return datasets


class ResidualCompensator:
    """Linear residual model for quick sanity checks."""

    def __init__(self, num_joints: int):
        self.num_joints = num_joints
        self.weights = None
        self.bias = None
        self.feature_cols = None

    def _feature_columns(self, df: pd.DataFrame):
        return [
            col for col in df.columns
            if col.startswith('q_') or col.startswith('dq_') or col.startswith('ddq_')
            or col.startswith('abs_') or col.startswith('q_dq_prod_') or col == 'motion_magnitude'
        ]

    def train_linear_compensator(self, df_train: pd.DataFrame) -> Dict:
        self.feature_cols = self._feature_columns(df_train)
        residual_cols = [f'residual_tau_{i + 1}' for i in range(self.num_joints)]
        X = df_train[self.feature_cols].values
        y = df_train[residual_cols].values

        X_aug = np.column_stack([X, np.ones(len(X))])
        theta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        self.weights = theta[:-1]
        self.bias = theta[-1]

        y_pred = X @ self.weights + self.bias
        train_mae = float(np.mean(np.abs(y - y_pred)))
        print("Linear compensator trained:")
        print(f"  Features: {X.shape[1]}")
        print(f"  Train MAE: {train_mae:.6f} N·m")
        return {'type': 'linear', 'train_mae': train_mae, 'n_features': int(X.shape[1])}

    def evaluate_compensator(self, df_test: pd.DataFrame) -> Dict:
        if self.weights is None:
            raise RuntimeError("Compensator not trained")

        residual_cols = [f'residual_tau_{i + 1}' for i in range(self.num_joints)]
        X = df_test[self.feature_cols].values
        y = df_test[residual_cols].values
        y_pred = X @ self.weights + self.bias
        err = y - y_pred

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        baseline_mae = float(np.mean(np.abs(y)))
        improvement_percent = 100.0 * max(0.0, 1.0 - mae / baseline_mae) if baseline_mae > 0 else 0.0
        print("Compensator Evaluation:")
        print(f"  MAE: {mae:.6f} N·m")
        print(f"  RMSE: {rmse:.6f} N·m")
        print(f"  Improvement: {improvement_percent:.2f}%")
        return {
            'mae': mae,
            'rmse': rmse,
            'baseline_mae': baseline_mae,
            'improvement_percent': improvement_percent,
        }
