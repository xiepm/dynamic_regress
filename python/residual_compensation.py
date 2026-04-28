"""
残差分析与补偿模块。

这个模块对应流水线的 Step 7-8，由 `run_pipeline.py` 在主动力学参数辨识结束后调用。
它复用 `identify_parameters.py` 的 torque 预测结果，把“模型解释不了的剩余误差”整理成
残差学习问题，再分别交给线性 ridge 补偿器和 MLP 补偿器建模，用于评估主模型之外的
非线性误差是否还能被进一步吸收。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class ResidualAnalyzer:
    """
    负责从主辨识模型中提取残差并构造残差特征。

    这个类只做“数据准备”，不参与任何参数训练。这样设计是为了让线性补偿器、MLP
    补偿器等下游模型共享同一套残差定义和特征工程，避免不同补偿器之间比较口径不一致。
    """

    def __init__(self, robot_model):
        """
        保存机器人维度信息，供残差列构造使用。

        Parameters
        ----------
        robot_model : RobotModel
            已加载好的机器人模型对象。
        """
        self.robot_model = robot_model
        self.num_joints = robot_model.num_joints

    def compute_residuals(self, df: pd.DataFrame, identifier, result: Dict) -> pd.DataFrame:
        """
        计算观测力矩与主模型预测力矩之间的残差。

        Parameters
        ----------
        df : pd.DataFrame
            待计算残差的数据集。
        identifier : ParameterIdentifier
            主动力学辨识器。
        result : Dict
            主辨识结果字典。

        Returns
        -------
        pd.DataFrame
            在原始数据基础上新增 `tau_pred_i` 和 `residual_tau_i` 的表。
        """
        tau_pred = identifier.predict_torques(df, result)
        tau_meas = np.column_stack([df[f'tau_{i}'].values for i in range(1, self.num_joints + 1)])
        residual = tau_meas - tau_pred

        frame = df.copy()
        for joint_idx in range(self.num_joints):
            frame[f'tau_pred_{joint_idx + 1}'] = tau_pred[:, joint_idx]
            frame[f'residual_tau_{joint_idx + 1}'] = residual[:, joint_idx]
        return frame

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为残差补偿模型生成衍生特征。

        Parameters
        ----------
        df : pd.DataFrame
            至少包含 `q_i`、`dq_i`、`ddq_i` 和残差列的数据表。

        Returns
        -------
        pd.DataFrame
            增加绝对值、乘积、三角函数和相邻关节耦合项后的特征表。
        """
        frame = df.copy()
        for joint_idx in range(1, self.num_joints + 1):
            frame[f'abs_dq_{joint_idx}'] = np.abs(df[f'dq_{joint_idx}'])
            frame[f'abs_ddq_{joint_idx}'] = np.abs(df[f'ddq_{joint_idx}'])
            frame[f'q_dq_prod_{joint_idx}'] = df[f'q_{joint_idx}'] * df[f'dq_{joint_idx}']
            # sin/cos 特征可以更自然地表达周期性位姿依赖误差，而不是把角度当线性量。
            frame[f'sin_q_{joint_idx}'] = np.sin(df[f'q_{joint_idx}'])
            frame[f'cos_q_{joint_idx}'] = np.cos(df[f'q_{joint_idx}'])
        # 相邻关节速度乘积用来近似耦合摩擦、柔性传动或串联关节联动带来的残差。
        for joint_idx in range(1, self.num_joints):
            frame[f'dq_pair_{joint_idx}_{joint_idx + 1}'] = df[f'dq_{joint_idx}'] * df[f'dq_{joint_idx + 1}']
        frame['motion_magnitude'] = np.linalg.norm(
            np.column_stack([df[f'dq_{joint_idx}'].values for joint_idx in range(1, self.num_joints + 1)]),
            axis=1,
        )
        return frame

    def build_residual_dataset(self, df: pd.DataFrame, identifier, result: Dict, test_split: float = 0.2, val_split: float = 0.1) -> Dict:
        """
        从整张数据表构造 train/val/test 残差学习数据集。

        Parameters
        ----------
        df : pd.DataFrame
            原始数据表。
        identifier : ParameterIdentifier
            主辨识器。
        result : Dict
            主辨识结果。
        test_split : float
            测试集占比。
        val_split : float
            验证集占比。

        Returns
        -------
        Dict
            含 `train`、`val`、`test` 三份特征表的字典。
        """
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
    """
    线性 ridge 残差补偿器。

    这个类的目标不是追求最强拟合能力，而是提供一个可解释、训练快、便于做 sanity check
    的线性基线。它持有训练得到的权重、偏置和特征列名，这些状态会在 `train_*` 方法中
    初始化，在 `evaluate_compensator()` 中复用。
    """

    def __init__(self, num_joints: int):
        """
        初始化线性补偿器。

        Parameters
        ----------
        num_joints : int
            主动关节数。
        """
        self.num_joints = num_joints
        self.weights = None
        self.bias = None
        self.feature_cols = None
        self.lambda_reg = None

    def _feature_columns(self, df: pd.DataFrame):
        """选出当前补偿器会使用的特征列名。"""
        return [
            col for col in df.columns
            if col.startswith('q_') or col.startswith('dq_') or col.startswith('ddq_')
            or col.startswith('abs_') or col.startswith('q_dq_prod_')
            or col.startswith('sin_q_') or col.startswith('cos_q_') or col.startswith('dq_pair_')
            or col == 'motion_magnitude'
        ]

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray, lambda_reg: float) -> tuple[np.ndarray, np.ndarray]:
        """求解带偏置项的多输出 ridge 回归。"""
        X_aug = np.column_stack([X, np.ones(len(X))])
        regularizer = np.eye(X_aug.shape[1], dtype=float) * lambda_reg
        # 偏置项不参与正则化，否则容易把整体残差均值错误地往 0 拉。
        regularizer[-1, -1] = 0.0
        theta = np.linalg.solve(X_aug.T @ X_aug + regularizer, X_aug.T @ y)
        return theta[:-1], theta[-1]

    def train_linear_compensator(self, df_train: pd.DataFrame, lambda_reg: float = 1e-2) -> Dict:
        """
        在训练集上拟合线性 ridge 残差补偿器。

        Parameters
        ----------
        df_train : pd.DataFrame
            训练特征表。
        lambda_reg : float
            L2 正则强度。

        Returns
        -------
        Dict
            训练 MAE、特征数和正则参数等摘要信息。
        """
        self.feature_cols = self._feature_columns(df_train)
        residual_cols = [f'residual_tau_{i + 1}' for i in range(self.num_joints)]
        X = df_train[self.feature_cols].values
        y = df_train[residual_cols].values

        # 线性特征维度已经不低，直接 OLS 容易把训练集噪声也学进去。
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
        """
        用 5 折交叉验证自动选择 ridge 正则强度。

        Parameters
        ----------
        df_train : pd.DataFrame
            训练特征表。

        Returns
        -------
        Dict
            最优 lambda 下的训练摘要，并附带每个候选值的交叉验证分数。
        """
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
        """
        评估线性补偿器在指定数据集上的效果。

        Parameters
        ----------
        df_test : pd.DataFrame
            待评估特征表。

        Returns
        -------
        Dict
            含全局与每关节 MAE/RMSE/提升率的评估结果。

        Raises
        ------
        RuntimeError
            当模型尚未训练时抛出。
        """
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
    """
    多输出 MLP 残差补偿器。

    它是当前主流程里的默认非线性补偿模型，目标是在不过多引入工程复杂度的前提下，
    给线性基线提供一个更强的非线性对照。类内部持有输入/输出标准化器和训练好的
    `MLPRegressor`，以保证训练和评估阶段使用完全一致的数据变换。
    """

    def __init__(
        self,
        num_joints: int,
        hidden_layer_sizes: tuple[int, ...] = (64, 64),
        alpha: float = 1e-4,
        max_iter: int = 500,
        random_state: int = 42,
    ):
        """
        初始化 MLP 补偿器及其超参数。

        Parameters
        ----------
        num_joints : int
            主动关节数。
        hidden_layer_sizes : tuple[int, ...]
            隐藏层结构。
        alpha : float
            L2 正则强度。
        max_iter : int
            最大训练轮数。
        random_state : int
            随机种子。
        """
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
        """选出 MLP 会消费的输入特征列。"""
        return [
            col for col in df.columns
            if col.startswith('q_') or col.startswith('dq_') or col.startswith('ddq_')
            or col.startswith('abs_') or col.startswith('q_dq_prod_')
            or col.startswith('sin_q_') or col.startswith('cos_q_') or col.startswith('dq_pair_')
            or col == 'motion_magnitude'
        ]

    def train(self, df_train: pd.DataFrame) -> Dict:
        """
        在训练集上拟合 MLP 残差补偿器。

        Parameters
        ----------
        df_train : pd.DataFrame
            训练特征表。

        Returns
        -------
        Dict
            训练 MAE、隐藏层结构、迭代次数等训练摘要。
        """
        self.feature_cols = self._feature_columns(df_train)
        residual_cols = [f'residual_tau_{i + 1}' for i in range(self.num_joints)]
        X = df_train[self.feature_cols].values
        y = df_train[residual_cols].values

        # MLP 对量纲和数值范围很敏感；输入输出一起标准化，训练更稳定。
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
        """
        评估 MLP 补偿器在指定数据集上的效果。

        Parameters
        ----------
        df_test : pd.DataFrame
            待评估特征表。

        Returns
        -------
        Dict
            含全局与每关节 MAE/RMSE/提升率的结果字典。

        Raises
        ------
        RuntimeError
            当模型尚未训练时抛出。
        """
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

    def export_state(self) -> Dict:
        """
        导出 C++ 前向推理所需的全部 MLP 状态。

        这里不保存 sklearn 对象本身，而是把标准化参数、权重和偏置转成普通
        Python 列表，方便代码生成器展开成 controller 侧可直接编译的静态数组。
        """
        if self.model is None:
            raise RuntimeError("MLP compensator not trained")
        if self.feature_cols is None:
            raise RuntimeError("MLP feature columns are not initialized")

        return {
            'type': 'mlp',
            'activation': 'relu',
            'feature_cols': list(self.feature_cols),
            'x_mean': self.x_scaler.mean_.astype(float).tolist(),
            'x_scale': self.x_scaler.scale_.astype(float).tolist(),
            'y_mean': self.y_scaler.mean_.astype(float).tolist(),
            'y_scale': self.y_scaler.scale_.astype(float).tolist(),
            'coefs': [coef.astype(float).tolist() for coef in self.model.coefs_],
            'intercepts': [bias.astype(float).tolist() for bias in self.model.intercepts_],
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
