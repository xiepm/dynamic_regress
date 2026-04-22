"""
Step 7-8: Residual analysis and lightweight compensation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


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
            # 修复：增加 sin/cos 位置特征，增强对周期性和非线性摩擦的表达能力。
            frame[f'sin_q_{joint_idx}'] = np.sin(df[f'q_{joint_idx}'])
            frame[f'cos_q_{joint_idx}'] = np.cos(df[f'q_{joint_idx}'])
        # 修复：增加相邻关节速度交叉项，提升耦合摩擦/传动效应的表达能力。
        for joint_idx in range(1, self.num_joints):
            frame[f'dq_pair_{joint_idx}_{joint_idx + 1}'] = df[f'dq_{joint_idx}'] * df[f'dq_{joint_idx + 1}']
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
        self.lambda_reg = None

    def _feature_columns(self, df: pd.DataFrame):
        return [
            col for col in df.columns
            if col.startswith('q_') or col.startswith('dq_') or col.startswith('ddq_')
            or col.startswith('abs_') or col.startswith('q_dq_prod_')
            or col.startswith('sin_q_') or col.startswith('cos_q_') or col.startswith('dq_pair_')
            or col == 'motion_magnitude'
        ]

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray, lambda_reg: float) -> tuple[np.ndarray, np.ndarray]:
        X_aug = np.column_stack([X, np.ones(len(X))])
        regularizer = np.eye(X_aug.shape[1], dtype=float) * lambda_reg
        # 不正则化偏置项，避免整体残差均值被强行压缩。
        regularizer[-1, -1] = 0.0
        theta = np.linalg.solve(X_aug.T @ X_aug + regularizer, X_aug.T @ y)
        return theta[:-1], theta[-1]

    def train_linear_compensator(self, df_train: pd.DataFrame, lambda_reg: float = 1e-2) -> Dict:
        self.feature_cols = self._feature_columns(df_train)
        residual_cols = [f'residual_tau_{i + 1}' for i in range(self.num_joints)]
        X = df_train[self.feature_cols].values
        y = df_train[residual_cols].values

        # 修复：使用带 L2 正则的岭回归，缓解高维线性特征在训练集上过拟合。
        self.weights, self.bias = self._fit_ridge(X, y, lambda_reg=lambda_reg)
        self.lambda_reg = lambda_reg

        y_pred = X @ self.weights + self.bias
        train_mae = float(np.mean(np.abs(y - y_pred)))
        print("Linear compensator trained:")
        print(f"  Features: {X.shape[1]}")
        print(f"  Lambda: {lambda_reg:.4g}")
        print(f"  Train MAE: {train_mae:.6f} N·m")
        return {
            'type': 'linear_ridge',
            'train_mae': train_mae,
            'n_features': int(X.shape[1]),
            'lambda_reg': float(lambda_reg),
        }

    def train_with_cross_validation(self, df_train: pd.DataFrame) -> Dict:
        candidate_lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        self.feature_cols = self._feature_columns(df_train)
        residual_cols = [f'residual_tau_{i + 1}' for i in range(self.num_joints)]
        X = df_train[self.feature_cols].values
        y = df_train[residual_cols].values

        rng = np.random.default_rng(123)
        indices = rng.permutation(len(df_train))
        folds = np.array_split(indices, 5)

        best_lambda = candidate_lambdas[0]
        best_score = np.inf
        cv_scores = {}

        for lambda_reg in candidate_lambdas:
            fold_maes = []
            for fold_idx in range(len(folds)):
                val_idx = folds[fold_idx]
                train_idx = np.concatenate([folds[i] for i in range(len(folds)) if i != fold_idx])

                weights, bias = self._fit_ridge(X[train_idx], y[train_idx], lambda_reg=lambda_reg)
                y_val_pred = X[val_idx] @ weights + bias
                fold_maes.append(float(np.mean(np.abs(y[val_idx] - y_val_pred))))

            mean_mae = float(np.mean(fold_maes))
            cv_scores[str(lambda_reg)] = mean_mae
            if mean_mae < best_score:
                best_score = mean_mae
                best_lambda = lambda_reg

        print("Residual compensator cross-validation:")
        for lambda_reg in candidate_lambdas:
            print(f"  lambda={lambda_reg:.4g}: mean val MAE={cv_scores[str(lambda_reg)]:.6f} N·m")
        print(f"  Selected lambda={best_lambda:.4g}")

        train_info = self.train_linear_compensator(df_train, lambda_reg=best_lambda)
        train_info['cv_scores'] = cv_scores
        return train_info

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
        improvement_percent = 100.0 * (1.0 - mae / baseline_mae) if baseline_mae > 0 else 0.0
        joint_mae = np.mean(np.abs(err), axis=0)
        joint_rmse = np.sqrt(np.mean(err ** 2, axis=0))
        baseline_joint_mae = np.mean(np.abs(y), axis=0)
        joint_improvement_percent = []
        for joint_idx in range(self.num_joints):
            baseline_value = float(baseline_joint_mae[joint_idx])
            if baseline_value > 0:
                joint_improvement = 100.0 * (1.0 - float(joint_mae[joint_idx]) / baseline_value)
            else:
                joint_improvement = 0.0
            joint_improvement_percent.append(joint_improvement)
        print("Compensator Evaluation:")
        print(f"  MAE: {mae:.6f} N·m")
        print(f"  RMSE: {rmse:.6f} N·m")
        print(f"  Improvement: {improvement_percent:.2f}%")
        print("  Per-joint metrics:")
        for joint_idx in range(self.num_joints):
            print(
                f"    Joint {joint_idx + 1}: "
                f"RMSE={joint_rmse[joint_idx]:.6f} N·m, "
                f"MAE={joint_mae[joint_idx]:.6f} N·m, "
                f"Improvement={joint_improvement_percent[joint_idx]:.2f}%"
            )
        if improvement_percent < 0:
            print("Warning: compensator is making predictions worse on this split")
        return {
            'mae': mae,
            'rmse': rmse,
            'baseline_mae': baseline_mae,
            'improvement_percent': improvement_percent,
            'joint_mae': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(joint_mae)},
            'joint_rmse': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(joint_rmse)},
            'joint_baseline_mae': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(baseline_joint_mae)},
            'joint_improvement_percent': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(joint_improvement_percent)},
        }


class MLPCompensator:
    """Simple multi-output MLP residual model as the default nonlinear compensator."""

    def __init__(
        self,
        num_joints: int,
        hidden_layer_sizes: tuple[int, ...] = (64, 64),
        alpha: float = 1e-4,
        max_iter: int = 500,
        random_state: int = 42,
    ):
        self.num_joints = num_joints
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.feature_cols = None
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = None

    def _feature_columns(self, df: pd.DataFrame):
        return [
            col for col in df.columns
            if col.startswith('q_') or col.startswith('dq_') or col.startswith('ddq_')
            or col.startswith('abs_') or col.startswith('q_dq_prod_')
            or col.startswith('sin_q_') or col.startswith('cos_q_') or col.startswith('dq_pair_')
            or col == 'motion_magnitude'
        ]

    def train(self, df_train: pd.DataFrame) -> Dict:
        self.feature_cols = self._feature_columns(df_train)
        residual_cols = [f'residual_tau_{i + 1}' for i in range(self.num_joints)]
        X = df_train[self.feature_cols].values
        y = df_train[residual_cols].values

        # 新增：MLP 对输入/输出尺度更敏感，先标准化能显著提升训练稳定性。
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=self.alpha,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=1e-3,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled, y_scaled)

        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        train_mae = float(np.mean(np.abs(y - y_pred)))
        print("MLP compensator trained:")
        print(f"  Features: {X.shape[1]}")
        print(f"  Hidden layers: {self.hidden_layer_sizes}")
        print(f"  Alpha: {self.alpha:.4g}")
        print(f"  Iterations: {self.model.n_iter_}")
        print(f"  Train MAE: {train_mae:.6f} N·m")
        return {
            'type': 'mlp',
            'train_mae': train_mae,
            'n_features': int(X.shape[1]),
            'hidden_layer_sizes': list(self.hidden_layer_sizes),
            'alpha': float(self.alpha),
            'max_iter': int(self.max_iter),
            'n_iter': int(self.model.n_iter_),
        }

    def evaluate(self, df_test: pd.DataFrame) -> Dict:
        if self.model is None:
            raise RuntimeError("MLP compensator not trained")

        residual_cols = [f'residual_tau_{i + 1}' for i in range(self.num_joints)]
        X = df_test[self.feature_cols].values
        y = df_test[residual_cols].values
        X_scaled = self.x_scaler.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        err = y - y_pred

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        baseline_mae = float(np.mean(np.abs(y)))
        improvement_percent = 100.0 * (1.0 - mae / baseline_mae) if baseline_mae > 0 else 0.0
        joint_mae = np.mean(np.abs(err), axis=0)
        joint_rmse = np.sqrt(np.mean(err ** 2, axis=0))
        baseline_joint_mae = np.mean(np.abs(y), axis=0)
        joint_improvement_percent = []
        for joint_idx in range(self.num_joints):
            baseline_value = float(baseline_joint_mae[joint_idx])
            if baseline_value > 0:
                joint_improvement = 100.0 * (1.0 - float(joint_mae[joint_idx]) / baseline_value)
            else:
                joint_improvement = 0.0
            joint_improvement_percent.append(joint_improvement)
        print("MLP Compensator Evaluation:")
        print(f"  MAE: {mae:.6f} N·m")
        print(f"  RMSE: {rmse:.6f} N·m")
        print(f"  Improvement: {improvement_percent:.2f}%")
        print("  Per-joint metrics:")
        for joint_idx in range(self.num_joints):
            print(
                f"    Joint {joint_idx + 1}: "
                f"RMSE={joint_rmse[joint_idx]:.6f} N·m, "
                f"MAE={joint_mae[joint_idx]:.6f} N·m, "
                f"Improvement={joint_improvement_percent[joint_idx]:.2f}%"
            )
        if improvement_percent < 0:
            print("Warning: compensator is making predictions worse on this split")
        return {
            'mae': mae,
            'rmse': rmse,
            'baseline_mae': baseline_mae,
            'improvement_percent': improvement_percent,
            'joint_mae': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(joint_mae)},
            'joint_rmse': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(joint_rmse)},
            'joint_baseline_mae': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(baseline_joint_mae)},
            'joint_improvement_percent': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(joint_improvement_percent)},
        }


class RandomForestCompensator:
    """Random-forest residual model as a nonlinear baseline."""

    def __init__(self, num_joints: int, n_estimators: int = 100, max_depth: int = 8, random_state: int = 42):
        self.num_joints = num_joints
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.feature_cols = None
        self.model = None

    def _feature_columns(self, df: pd.DataFrame):
        return [
            col for col in df.columns
            if col.startswith('q_') or col.startswith('dq_') or col.startswith('ddq_')
            or col.startswith('abs_') or col.startswith('q_dq_prod_')
            or col.startswith('sin_q_') or col.startswith('cos_q_') or col.startswith('dq_pair_')
            or col == 'motion_magnitude'
        ]

    def train(self, df_train: pd.DataFrame) -> Dict:
        self.feature_cols = self._feature_columns(df_train)
        residual_cols = [f'residual_tau_{i + 1}' for i in range(self.num_joints)]
        X = df_train[self.feature_cols].values
        y = df_train[residual_cols].values

        # 新增：随机森林作为非线性残差补偿基线，和线性模型做并排比较。
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X, y)

        y_pred = self.model.predict(X)
        train_mae = float(np.mean(np.abs(y - y_pred)))
        print("Random-forest compensator trained:")
        print(f"  Features: {X.shape[1]}")
        print(f"  Train MAE: {train_mae:.6f} N·m")
        return {
            'type': 'random_forest',
            'train_mae': train_mae,
            'n_features': int(X.shape[1]),
            'n_estimators': int(self.n_estimators),
            'max_depth': int(self.max_depth),
        }

    def evaluate(self, df_test: pd.DataFrame) -> Dict:
        if self.model is None:
            raise RuntimeError("Random-forest compensator not trained")

        residual_cols = [f'residual_tau_{i + 1}' for i in range(self.num_joints)]
        X = df_test[self.feature_cols].values
        y = df_test[residual_cols].values
        y_pred = self.model.predict(X)
        err = y - y_pred

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        baseline_mae = float(np.mean(np.abs(y)))
        improvement_percent = 100.0 * (1.0 - mae / baseline_mae) if baseline_mae > 0 else 0.0
        joint_mae = np.mean(np.abs(err), axis=0)
        joint_rmse = np.sqrt(np.mean(err ** 2, axis=0))
        baseline_joint_mae = np.mean(np.abs(y), axis=0)
        joint_improvement_percent = []
        for joint_idx in range(self.num_joints):
            baseline_value = float(baseline_joint_mae[joint_idx])
            if baseline_value > 0:
                joint_improvement = 100.0 * (1.0 - float(joint_mae[joint_idx]) / baseline_value)
            else:
                joint_improvement = 0.0
            joint_improvement_percent.append(joint_improvement)
        print("Random-forest Compensator Evaluation:")
        print(f"  MAE: {mae:.6f} N·m")
        print(f"  RMSE: {rmse:.6f} N·m")
        print(f"  Improvement: {improvement_percent:.2f}%")
        print("  Per-joint metrics:")
        for joint_idx in range(self.num_joints):
            print(
                f"    Joint {joint_idx + 1}: "
                f"RMSE={joint_rmse[joint_idx]:.6f} N·m, "
                f"MAE={joint_mae[joint_idx]:.6f} N·m, "
                f"Improvement={joint_improvement_percent[joint_idx]:.2f}%"
            )
        if improvement_percent < 0:
            print("Warning: compensator is making predictions worse on this split")
        return {
            'mae': mae,
            'rmse': rmse,
            'baseline_mae': baseline_mae,
            'improvement_percent': improvement_percent,
            'joint_mae': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(joint_mae)},
            'joint_rmse': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(joint_rmse)},
            'joint_baseline_mae': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(baseline_joint_mae)},
            'joint_improvement_percent': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(joint_improvement_percent)},
        }
