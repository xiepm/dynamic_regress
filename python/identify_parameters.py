"""
基于回归矩阵的动力学参数辨识模块。

这个模块位于流水线的 Step 6，由 `run_pipeline.py` 在数据清洗和切分完成后调用。
它依赖 `generate_golden_data.py` 中的摩擦符号函数、`load_model.py` 提供的
Pinocchio 模型，以及预处理后的 `q / dq / ddq / tau` 数据，负责构造回归矩阵、
选择可辨识参数列，并输出 base/full 两种口径下的辨识结果和误差评估。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import lstsq, qr

from generate_golden_data import coulomb_sign
from load_model import pin
from runtime_dynamics import GravityConfig, InertialParameterVector, ValidationTools


class RegressorBuilder:
    """
    构造“刚体动力学 + 摩擦项”的线性回归矩阵。

    设计上把回归矩阵构造独立成一个类，是为了让“动力学建模”和“参数求解器”解耦：
    上层 `ParameterIdentifier` 只关心解线性系统，不关心每一列是如何拼出来的。
    """

    def __init__(self, robot_model, model_options: Dict | None = None):
        """
        初始化回归矩阵构造器。

        Parameters
        ----------
        robot_model : RobotModel
            含有 Pinocchio 模型、关节数量和摩擦先验的机器人对象。
        """
        self.robot_model = robot_model
        self.num_joints = robot_model.num_joints
        self.model_options = self._normalize_model_options(model_options)
        self.num_rigid_params = 10 * self.num_joints
        self.num_friction_params = 2 * self.num_joints
        self.num_bias_params = self.num_joints if self.model_options['include_bias'] else 0
        self.num_hold_params = self.num_joints if self.model_options['include_hold'] else 0
        self.total_params = (
            self.num_rigid_params
            + self.num_friction_params
            + self.num_bias_params
            + self.num_hold_params
        )

    def _normalize_model_options(self, model_options: Dict | None) -> Dict:
        """
        规范化实验性附加建模选项。

        默认保持当前线上口径：只包含 rigid-body + friction。
        """
        options = {
            'include_bias': False,
            'include_hold': False,
            'hold_model': 'indicator',
            'hold_velocity_epsilon': 0.02,
            'stribeck_velocity_scale': 0.05,
        }
        if model_options:
            options.update(model_options)

        if options['hold_model'] not in {'indicator', 'stribeck'}:
            raise ValueError(
                f"Unsupported hold_model: {options['hold_model']}. "
                "Expected 'indicator' or 'stribeck'."
            )
        if options['hold_velocity_epsilon'] <= 0.0:
            raise ValueError("hold_velocity_epsilon must be > 0.")
        if options['stribeck_velocity_scale'] <= 0.0:
            raise ValueError("stribeck_velocity_scale must be > 0.")
        return options

    def _gravity_matrix_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        从数据表中读取逐样本重力向量。

        兼容旧数据：如果缺少 `gravity_x/y/z`，则退回到 robot_model 当前重力方向。
        """
        gravity_cols = ['gravity_x', 'gravity_y', 'gravity_z']
        if all(column in df.columns for column in gravity_cols):
            gravity = df[gravity_cols].values.astype(float)
            if gravity.shape != (len(df), 3):
                raise ValueError(f"Invalid gravity matrix shape: {gravity.shape}")
            return gravity
        return np.tile(np.asarray(self.robot_model.gravity_vector, dtype=float), (len(df), 1))

    def _rigid_regressor_with_conditions(
        self,
        data,
        q_sample: np.ndarray,
        dq_sample: np.ndarray,
        ddq_sample: np.ndarray,
        gravity_vector: np.ndarray,
    ) -> np.ndarray:
        model = self.robot_model.pinocchio_model
        previous_gravity = model.gravity.linear.copy()
        try:
            model.gravity.linear = np.asarray(gravity_vector, dtype=float).copy()
            return np.asarray(
                pin.computeJointTorqueRegressor(
                    model,
                    data,
                    q_sample,
                    dq_sample,
                    ddq_sample,
                ),
                dtype=float,
            ).reshape(self.num_joints, self.num_rigid_params)
        finally:
            model.gravity.linear = previous_gravity

    def _friction_block(self, dq_sample: np.ndarray) -> np.ndarray:
        friction_block = np.zeros((self.num_joints, self.num_friction_params))
        dq_sign = coulomb_sign(dq_sample)
        for joint_idx in range(self.num_joints):
            friction_block[joint_idx, joint_idx] = dq_sample[joint_idx]
            friction_block[joint_idx, self.num_joints + joint_idx] = dq_sign[joint_idx]
        return friction_block

    def _bias_block(self) -> np.ndarray:
        return np.eye(self.num_joints, dtype=float)

    def _hold_block(self, dq_sample: np.ndarray) -> np.ndarray:
        hold_block = np.zeros((self.num_joints, self.num_joints), dtype=float)
        abs_dq = np.abs(np.asarray(dq_sample, dtype=float))

        if self.model_options['hold_model'] == 'indicator':
            activation = (abs_dq <= self.model_options['hold_velocity_epsilon']).astype(float)
        else:
            velocity_scale = self.model_options['stribeck_velocity_scale']
            activation = np.exp(-np.square(abs_dq / velocity_scale))

        for joint_idx in range(self.num_joints):
            hold_block[joint_idx, joint_idx] = activation[joint_idx]
        return hold_block

    def build_regressor_components(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        构造显式的 M/C/G/F 分解回归矩阵分量。
        """
        q = np.column_stack([df[f'q_{i}'].values for i in range(1, self.num_joints + 1)])
        dq = np.column_stack([df[f'dq_{i}'].values for i in range(1, self.num_joints + 1)])
        ddq = np.column_stack([df[f'ddq_{i}'].values for i in range(1, self.num_joints + 1)])
        gravity = self._gravity_matrix_from_dataframe(df)

        num_samples = len(df)
        shape = (num_samples * self.num_joints, self.num_rigid_params)
        Y_M = np.zeros(shape)
        Y_C = np.zeros(shape)
        Y_Gx = np.zeros(shape)
        Y_Gy = np.zeros(shape)
        Y_Gz = np.zeros(shape)
        Y_F = np.zeros((num_samples * self.num_joints, self.num_friction_params))
        Y_B = np.zeros((num_samples * self.num_joints, self.num_bias_params)) if self.num_bias_params else None
        Y_H = np.zeros((num_samples * self.num_joints, self.num_hold_params)) if self.num_hold_params else None

        data = self.robot_model.pinocchio_model.createData()
        zero_dq = np.zeros(self.num_joints, dtype=float)
        zero_ddq = np.zeros(self.num_joints, dtype=float)
        zero_gravity = np.zeros(3, dtype=float)
        unit_x = np.array([1.0, 0.0, 0.0], dtype=float)
        unit_y = np.array([0.0, 1.0, 0.0], dtype=float)
        unit_z = np.array([0.0, 0.0, 1.0], dtype=float)

        for sample_idx in range(num_samples):
            row_slice = slice(sample_idx * self.num_joints, (sample_idx + 1) * self.num_joints)
            q_sample = q[sample_idx]
            dq_sample = dq[sample_idx]
            ddq_sample = ddq[sample_idx]

            Y_M[row_slice, :] = self._rigid_regressor_with_conditions(
                data,
                q_sample,
                zero_dq,
                ddq_sample,
                zero_gravity,
            )
            Y_C[row_slice, :] = self._rigid_regressor_with_conditions(
                data,
                q_sample,
                dq_sample,
                zero_ddq,
                zero_gravity,
            )
            Y_Gx[row_slice, :] = self._rigid_regressor_with_conditions(
                data,
                q_sample,
                zero_dq,
                zero_ddq,
                unit_x,
            )
            Y_Gy[row_slice, :] = self._rigid_regressor_with_conditions(
                data,
                q_sample,
                zero_dq,
                zero_ddq,
                unit_y,
            )
            Y_Gz[row_slice, :] = self._rigid_regressor_with_conditions(
                data,
                q_sample,
                zero_dq,
                zero_ddq,
                unit_z,
            )
            Y_F[row_slice, :] = self._friction_block(dq_sample)
            if Y_B is not None:
                Y_B[row_slice, :] = self._bias_block()
            if Y_H is not None:
                Y_H[row_slice, :] = self._hold_block(dq_sample)

        components = {
            'gravity': gravity,
            'Y_M': Y_M,
            'Y_C': Y_C,
            'Y_Gx': Y_Gx,
            'Y_Gy': Y_Gy,
            'Y_Gz': Y_Gz,
            'Y_F': Y_F,
        }
        if Y_B is not None:
            components['Y_B'] = Y_B
        if Y_H is not None:
            components['Y_H'] = Y_H
        return components

    def assemble_regressor_from_components(self, components: Dict[str, np.ndarray]) -> np.ndarray:
        """
        按 `YM + YC + gx*YGx + gy*YGy + gz*YGz + YF` 组装总回归矩阵。
        """
        gravity = np.asarray(components['gravity'], dtype=float)
        num_samples = gravity.shape[0]
        Phi = np.zeros((num_samples * self.num_joints, self.total_params))

        for sample_idx in range(num_samples):
            row_slice = slice(sample_idx * self.num_joints, (sample_idx + 1) * self.num_joints)
            gx, gy, gz = gravity[sample_idx]
            Phi[row_slice, :self.num_rigid_params] = (
                components['Y_M'][row_slice, :]
                + components['Y_C'][row_slice, :]
                + gx * components['Y_Gx'][row_slice, :]
                + gy * components['Y_Gy'][row_slice, :]
                + gz * components['Y_Gz'][row_slice, :]
            )
            col_start = self.num_rigid_params
            col_end = col_start + self.num_friction_params
            Phi[row_slice, col_start:col_end] = components['Y_F'][row_slice, :]

            if self.num_bias_params:
                col_start = col_end
                col_end = col_start + self.num_bias_params
                Phi[row_slice, col_start:col_end] = components['Y_B'][row_slice, :]

            if self.num_hold_params:
                col_start = col_end
                col_end = col_start + self.num_hold_params
                Phi[row_slice, col_start:col_end] = components['Y_H'][row_slice, :]
        return Phi

    def build_regressor_matrix_legacy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        保留旧的“整体 regressor”实现，用于数值对照测试。
        """
        q = np.column_stack([df[f'q_{i}'].values for i in range(1, self.num_joints + 1)])
        dq = np.column_stack([df[f'dq_{i}'].values for i in range(1, self.num_joints + 1)])
        ddq = np.column_stack([df[f'ddq_{i}'].values for i in range(1, self.num_joints + 1)])
        tau = np.column_stack([df[f'tau_{i}'].values for i in range(1, self.num_joints + 1)])
        gravity = self._gravity_matrix_from_dataframe(df)

        num_samples = len(df)
        Phi = np.zeros((num_samples * self.num_joints, self.total_params))
        tau_flat = tau.reshape(-1)
        data = self.robot_model.pinocchio_model.createData()
        model = self.robot_model.pinocchio_model
        previous_gravity = model.gravity.linear.copy()

        try:
            for sample_idx in range(num_samples):
                model.gravity.linear = gravity[sample_idx].copy()
                Y = np.asarray(
                    pin.computeJointTorqueRegressor(
                        model,
                        data,
                        q[sample_idx],
                        dq[sample_idx],
                        ddq[sample_idx],
                    ),
                    dtype=float,
                ).reshape(self.num_joints, self.num_rigid_params)
                row_slice = slice(sample_idx * self.num_joints, (sample_idx + 1) * self.num_joints)
                Phi[row_slice, :self.num_rigid_params] = Y
                col_start = self.num_rigid_params
                col_end = col_start + self.num_friction_params
                Phi[row_slice, col_start:col_end] = self._friction_block(dq[sample_idx])
                if self.num_bias_params:
                    col_start = col_end
                    col_end = col_start + self.num_bias_params
                    Phi[row_slice, col_start:col_end] = self._bias_block()
                if self.num_hold_params:
                    col_start = col_end
                    col_end = col_start + self.num_hold_params
                    Phi[row_slice, col_start:col_end] = self._hold_block(dq[sample_idx])
        finally:
            model.gravity.linear = previous_gravity

        return Phi, tau_flat

    def build_regressor_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据处理后的时序数据构造辨识回归矩阵。

        Parameters
        ----------
        df : pd.DataFrame
            至少包含 `q_i`、`dq_i`、`ddq_i`、`tau_i` 列的样本表。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            `Phi` 和拉平成一维的观测力矩 `tau_flat`。
        """
        tau = np.column_stack([df[f'tau_{i}'].values for i in range(1, self.num_joints + 1)])
        tau_flat = tau.reshape(-1)
        components = self.build_regressor_components(df)
        Phi = self.assemble_regressor_from_components(components)

        print(f"Built physically consistent regressor: shape {Phi.shape}")
        return Phi, tau_flat


class ParameterIdentifier:
    """
    管理参数辨识与误差评估的主类。

    它持有 `RegressorBuilder` 作为内部依赖，先负责构造回归矩阵，再根据
    `base/full` 参数化和求解方法（OLS / ridge）给出参数估计，最后提供统一的
    torque 预测和评估接口，供主流程与残差补偿模块复用。
    """

    def __init__(self, robot_model, parameterization: str = 'base', model_options: Dict | None = None):
        """
        初始化辨识器。

        Parameters
        ----------
        robot_model : RobotModel
            机器人模型对象。
        parameterization : str
            参数化方式，支持 `base` 或 `full`。
        """
        self.robot_model = robot_model
        self.num_joints = robot_model.num_joints
        self.parameterization = parameterization
        self.model_options = model_options or {}
        self.regressor = RegressorBuilder(robot_model, model_options=self.model_options)
        self.solver = IdentificationSolver()


class IdentificationSolver:
    """
    Linear identification solver façade.

    The solver supports both unconstrained linear least-squares variants and a
    physically constrained optimizer over the full inertial parameter vector.
    """

    @staticmethod
    def _inertia_constraints_from_theta(theta: np.ndarray, num_joints: int, epsilon: float) -> np.ndarray:
        """
        Build per-link physical-feasibility inequality values.

        Each returned element is expected to stay >= 0:
        - positive mass
        - non-negative diagonal inertias
        - positive 2x2 principal minors
        - positive determinant
        - triangle inequalities
        """
        theta = np.asarray(theta, dtype=float).reshape(-1)
        constraints = []
        rigid = theta[:10 * num_joints].reshape(num_joints, 10)
        for link_idx in range(num_joints):
            phi = rigid[link_idx]
            mass = float(phi[0])
            ixx, ixy, ixz, iyy, iyz, izz = [float(value) for value in phi[4:10]]
            det_xy = ixx * iyy - ixy * ixy
            det_xz = ixx * izz - ixz * ixz
            det_yz = iyy * izz - iyz * iyz
            det_full = (
                ixx * (iyy * izz - iyz * iyz)
                - ixy * (ixy * izz - iyz * ixz)
                + ixz * (ixy * iyz - iyy * ixz)
            )
            constraints.extend(
                [
                    mass - epsilon,
                    ixx - epsilon,
                    iyy - epsilon,
                    izz - epsilon,
                    det_xy - epsilon,
                    det_xz - epsilon,
                    det_yz - epsilon,
                    det_full - epsilon,
                    (iyy + izz - ixx) - epsilon,
                    (ixx + izz - iyy) - epsilon,
                    (ixx + iyy - izz) - epsilon,
                ]
            )
        return np.asarray(constraints, dtype=float)

    @staticmethod
    def _inertia_constraints_and_jacobian(
        theta: np.ndarray,
        num_joints: int,
        epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return physical-feasibility inequalities and their Jacobian.

        The constraint ordering matches `_inertia_constraints_from_theta`.
        Each row of the Jacobian corresponds to the gradient of one inequality
        with respect to the full parameter vector.
        """
        theta = np.asarray(theta, dtype=float).reshape(-1)
        num_constraints_per_link = 11
        constraints = np.zeros(num_constraints_per_link * num_joints, dtype=float)
        jacobian = np.zeros((num_constraints_per_link * num_joints, theta.size), dtype=float)
        rigid = theta[:10 * num_joints].reshape(num_joints, 10)

        for link_idx in range(num_joints):
            row0 = link_idx * num_constraints_per_link
            col0 = link_idx * 10
            phi = rigid[link_idx]
            mass = float(phi[0])
            ixx, ixy, ixz, iyy, iyz, izz = [float(value) for value in phi[4:10]]

            det_xy = ixx * iyy - ixy * ixy
            det_xz = ixx * izz - ixz * ixz
            det_yz = iyy * izz - iyz * iyz
            det_full = (
                ixx * (iyy * izz - iyz * iyz)
                - ixy * (ixy * izz - iyz * ixz)
                + ixz * (ixy * iyz - iyy * ixz)
            )

            rows = constraints[row0:row0 + num_constraints_per_link]
            rows[:] = [
                mass - epsilon,
                ixx - epsilon,
                iyy - epsilon,
                izz - epsilon,
                det_xy - epsilon,
                det_xz - epsilon,
                det_yz - epsilon,
                det_full - epsilon,
                (iyy + izz - ixx) - epsilon,
                (ixx + izz - iyy) - epsilon,
                (ixx + iyy - izz) - epsilon,
            ]

            jacobian[row0 + 0, col0 + 0] = 1.0
            jacobian[row0 + 1, col0 + 4] = 1.0
            jacobian[row0 + 2, col0 + 7] = 1.0
            jacobian[row0 + 3, col0 + 9] = 1.0

            jacobian[row0 + 4, col0 + 4] = iyy
            jacobian[row0 + 4, col0 + 5] = -2.0 * ixy
            jacobian[row0 + 4, col0 + 7] = ixx

            jacobian[row0 + 5, col0 + 4] = izz
            jacobian[row0 + 5, col0 + 6] = -2.0 * ixz
            jacobian[row0 + 5, col0 + 9] = ixx

            jacobian[row0 + 6, col0 + 7] = izz
            jacobian[row0 + 6, col0 + 8] = -2.0 * iyz
            jacobian[row0 + 6, col0 + 9] = iyy

            jacobian[row0 + 7, col0 + 4] = iyy * izz - iyz * iyz
            jacobian[row0 + 7, col0 + 5] = 2.0 * (ixz * iyz - ixy * izz)
            jacobian[row0 + 7, col0 + 6] = 2.0 * (ixy * iyz - iyy * ixz)
            jacobian[row0 + 7, col0 + 7] = ixx * izz - ixz * ixz
            jacobian[row0 + 7, col0 + 8] = 2.0 * (ixy * ixz - ixx * iyz)
            jacobian[row0 + 7, col0 + 9] = ixx * iyy - ixy * ixy

            jacobian[row0 + 8, col0 + 4] = -1.0
            jacobian[row0 + 8, col0 + 7] = 1.0
            jacobian[row0 + 8, col0 + 9] = 1.0

            jacobian[row0 + 9, col0 + 4] = 1.0
            jacobian[row0 + 9, col0 + 7] = -1.0
            jacobian[row0 + 9, col0 + 9] = 1.0

            jacobian[row0 + 10, col0 + 4] = 1.0
            jacobian[row0 + 10, col0 + 7] = 1.0
            jacobian[row0 + 10, col0 + 9] = -1.0

        return constraints, jacobian

    def _build_constrained_bounds(
        self,
        num_params: int,
        num_joints: int,
        *,
        min_mass: float,
        min_friction: float,
        max_abs_rigid: float,
        max_abs_friction: float,
    ) -> Bounds:
        lower = np.full(num_params, -np.inf, dtype=float)
        upper = np.full(num_params, np.inf, dtype=float)

        rigid_count = 10 * num_joints
        lower[:rigid_count] = -max_abs_rigid
        upper[:rigid_count] = max_abs_rigid
        for joint_idx in range(num_joints):
            lower[10 * joint_idx] = min_mass

        viscous_start = rigid_count
        viscous_end = viscous_start + num_joints
        coulomb_end = viscous_end + num_joints
        if num_params >= viscous_end:
            lower[viscous_start:viscous_end] = min_friction
            upper[viscous_start:viscous_end] = max_abs_friction
        if num_params >= coulomb_end:
            lower[viscous_end:coulomb_end] = min_friction
            upper[viscous_end:coulomb_end] = max_abs_friction

        return Bounds(lower, upper)

    @staticmethod
    def _project_inertia_matrix(
        inertia_matrix: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        """
        Project a symmetric inertia matrix to a PSD matrix whose principal
        moments satisfy the triangle inequalities.
        """
        symmetric = 0.5 * (np.asarray(inertia_matrix, dtype=float) + np.asarray(inertia_matrix, dtype=float).T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        principal = np.maximum(eigenvalues, epsilon)
        i1, i2, i3 = [float(value) for value in principal]

        a = max(0.5 * (i2 + i3 - i1), epsilon)
        b = max(0.5 * (i1 + i3 - i2), epsilon)
        c = max(0.5 * (i1 + i2 - i3), epsilon)
        projected_principal = np.asarray([b + c, a + c, a + b], dtype=float)
        return eigenvectors @ np.diag(projected_principal) @ eigenvectors.T

    def _project_theta_physical(
        self,
        theta: np.ndarray,
        *,
        num_joints: int,
        min_mass: float,
        inertia_epsilon: float,
        min_friction: float,
        max_abs_rigid: float,
        max_abs_friction: float,
    ) -> np.ndarray:
        """
        Project a full parameter vector to the physically admissible set used by
        the constrained solver.
        """
        projected = np.asarray(theta, dtype=float).copy().reshape(-1)
        rigid_count = 10 * num_joints

        for joint_idx in range(num_joints):
            offset = 10 * joint_idx
            projected[offset:offset + 10] = np.clip(
                projected[offset:offset + 10],
                -max_abs_rigid,
                max_abs_rigid,
            )
            projected[offset] = max(float(projected[offset]), min_mass)

            inertia_matrix = np.asarray(
                [
                    [projected[offset + 4], projected[offset + 5], projected[offset + 6]],
                    [projected[offset + 5], projected[offset + 7], projected[offset + 8]],
                    [projected[offset + 6], projected[offset + 8], projected[offset + 9]],
                ],
                dtype=float,
            )
            inertia_projected = self._project_inertia_matrix(inertia_matrix, inertia_epsilon)
            projected[offset + 4] = float(inertia_projected[0, 0])
            projected[offset + 5] = float(inertia_projected[0, 1])
            projected[offset + 6] = float(inertia_projected[0, 2])
            projected[offset + 7] = float(inertia_projected[1, 1])
            projected[offset + 8] = float(inertia_projected[1, 2])
            projected[offset + 9] = float(inertia_projected[2, 2])

        viscous_start = rigid_count
        viscous_end = viscous_start + num_joints
        coulomb_end = viscous_end + num_joints
        if projected.size >= viscous_end:
            projected[viscous_start:viscous_end] = np.clip(
                projected[viscous_start:viscous_end],
                min_friction,
                max_abs_friction,
            )
        if projected.size >= coulomb_end:
            projected[viscous_end:coulomb_end] = np.clip(
                projected[viscous_end:coulomb_end],
                min_friction,
                max_abs_friction,
            )

        return projected

    def _solve_constrained(
        self,
        Phi: np.ndarray,
        tau: np.ndarray,
        *,
        num_joints: int,
        ridge_lambda: float,
        sample_weights: np.ndarray | None,
        initial_guess: np.ndarray | None = None,
        min_mass: float = 1e-6,
        inertia_epsilon: float = 1e-9,
        min_friction: float = 0.0,
        max_abs_rigid: float = 1e4,
        max_abs_friction: float = 1e4,
        maxiter: int = 500,
        step_scale: float = 1.0,
        xtol: float = 1e-9,
        ftol: float = 1e-12,
    ) -> Dict[str, np.ndarray | float | int | str]:
        Phi_solve = np.asarray(Phi, dtype=float)
        tau_solve = np.asarray(tau, dtype=float).reshape(-1)
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=float).reshape(-1)
            if weights.shape[0] != Phi_solve.shape[0]:
                raise ValueError(
                    f"sample_weights length {weights.shape[0]} must match equation count {Phi_solve.shape[0]}."
                )
            if np.any(weights < 0.0):
                raise ValueError("sample_weights must be non-negative.")
            sqrt_w = np.sqrt(weights)
            Phi_solve = Phi_solve * sqrt_w[:, None]
            tau_solve = tau_solve * sqrt_w

        if initial_guess is not None:
            x0 = np.asarray(initial_guess, dtype=float).reshape(-1)
        else:
            init = self.solve(Phi_solve, tau_solve, method='ridge', ridge_lambda=ridge_lambda)
            x0 = np.asarray(init['theta_hat'], dtype=float)
        hessian = Phi_solve.T @ Phi_solve + ridge_lambda * np.eye(Phi_solve.shape[1], dtype=float)
        linear_term = Phi_solve.T @ tau_solve
        constant_term = float(tau_solve @ tau_solve)
        spectral_radius = float(np.max(np.linalg.eigvalsh(hessian)))
        base_step = step_scale / max(spectral_radius, 1e-12)

        def objective(theta: np.ndarray) -> float:
            theta = np.asarray(theta, dtype=float)
            return 0.5 * float(theta @ (hessian @ theta)) - float(linear_term @ theta) + 0.5 * constant_term

        theta_hat = self._project_theta_physical(
            x0,
            num_joints=num_joints,
            min_mass=min_mass,
            inertia_epsilon=inertia_epsilon,
            min_friction=min_friction,
            max_abs_rigid=max_abs_rigid,
            max_abs_friction=max_abs_friction,
        )
        previous_objective = objective(theta_hat)
        optimizer_success = False
        optimizer_status = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
        optimizer_iterations = int(maxiter)

        for iteration in range(1, int(maxiter) + 1):
            gradient = hessian @ theta_hat - linear_term
            step = base_step
            candidate = theta_hat
            candidate_objective = previous_objective
            for _ in range(25):
                projected_candidate = self._project_theta_physical(
                    theta_hat - step * gradient,
                    num_joints=num_joints,
                    min_mass=min_mass,
                    inertia_epsilon=inertia_epsilon,
                    min_friction=min_friction,
                    max_abs_rigid=max_abs_rigid,
                    max_abs_friction=max_abs_friction,
                )
                trial_objective = objective(projected_candidate)
                if trial_objective <= previous_objective + 1e-14:
                    candidate = projected_candidate
                    candidate_objective = trial_objective
                    break
                step *= 0.5

            parameter_delta = float(np.linalg.norm(candidate - theta_hat))
            objective_delta = abs(previous_objective - candidate_objective)
            theta_hat = candidate
            previous_objective = candidate_objective

            if parameter_delta <= xtol * (1.0 + float(np.linalg.norm(theta_hat))):
                optimizer_success = True
                optimizer_status = "Projected constrained solver converged (parameter step tolerance)."
                optimizer_iterations = iteration
                break
            if objective_delta <= ftol * (1.0 + abs(previous_objective)):
                optimizer_success = True
                optimizer_status = "Projected constrained solver converged (objective tolerance)."
                optimizer_iterations = iteration
                break

        if not optimizer_success:
            projected_stationary = self._project_theta_physical(
                theta_hat - base_step * (hessian @ theta_hat - linear_term),
                num_joints=num_joints,
                min_mass=min_mass,
                inertia_epsilon=inertia_epsilon,
                min_friction=min_friction,
                max_abs_rigid=max_abs_rigid,
                max_abs_friction=max_abs_friction,
            )
            stationarity_gap = float(np.linalg.norm(projected_stationary - theta_hat))
            parameter_vector = InertialParameterVector.from_theta_full(theta_hat, num_joints=num_joints)
            sanity = ValidationTools.parameter_sanity_check(parameter_vector)
            if stationarity_gap <= 1e-5 * (1.0 + float(np.linalg.norm(theta_hat))) and sanity['is_valid']:
                optimizer_success = True
                optimizer_status = "Projected constrained solver reached a feasible stationary point."
            elif sanity['is_valid']:
                optimizer_success = True
                optimizer_status = (
                    "Projected constrained solver reached the iteration limit but produced a feasible physical solution."
                )

        residual_vector = tau_solve - Phi_solve @ theta_hat
        singular_values = np.linalg.svd(Phi_solve, compute_uv=False)
        active_condition_number = (
            float(singular_values[0] / singular_values[-1])
            if len(singular_values) > 1 and singular_values[-1] > 0
            else np.inf
        )
        return {
            'theta_hat': theta_hat,
            'residual_vector': residual_vector,
            'rank': int(np.linalg.matrix_rank(Phi_solve)),
            'singular_values': singular_values,
            'active_condition_number': active_condition_number,
            'ridge_lambda_used': float(ridge_lambda),
            'optimizer_success': bool(optimizer_success),
            'optimizer_status': str(optimizer_status),
            'optimizer_iterations': int(optimizer_iterations),
        }

    def solve(
        self,
        Phi: np.ndarray,
        tau: np.ndarray,
        *,
        method: str = 'ols',
        ridge_lambda: float = 1e-4,
        sample_weights: np.ndarray | None = None,
        num_joints: int | None = None,
        initial_guess: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray | float | int]:
        method_lower = method.lower()
        Phi_solve = np.asarray(Phi, dtype=float)
        tau_solve = np.asarray(tau, dtype=float).reshape(-1)

        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=float).reshape(-1)
            if weights.shape[0] != Phi_solve.shape[0]:
                raise ValueError(
                    f"sample_weights length {weights.shape[0]} must match equation count {Phi_solve.shape[0]}."
                )
            if np.any(weights < 0.0):
                raise ValueError("sample_weights must be non-negative.")
            sqrt_w = np.sqrt(weights)
            Phi_solve = Phi_solve * sqrt_w[:, None]
            tau_solve = tau_solve * sqrt_w

        if method_lower == 'ols':
            theta_hat, residuals, solved_rank, singular_values = lstsq(Phi_solve, tau_solve)
            residual_vector = tau_solve - Phi_solve @ theta_hat
            ridge_lambda_used = 0.0
        elif method_lower == 'wls':
            theta_hat, residuals, solved_rank, singular_values = lstsq(Phi_solve, tau_solve)
            residual_vector = tau_solve - Phi_solve @ theta_hat
            ridge_lambda_used = 0.0
            del residuals
        elif method_lower == 'ridge':
            normal_matrix = Phi_solve.T @ Phi_solve + ridge_lambda * np.eye(Phi_solve.shape[1], dtype=float)
            theta_hat = np.linalg.solve(normal_matrix, Phi_solve.T @ tau_solve)
            residual_vector = tau_solve - Phi_solve @ theta_hat
            solved_rank = int(np.linalg.matrix_rank(Phi_solve))
            singular_values = np.linalg.svd(Phi_solve, compute_uv=False)
            ridge_lambda_used = float(ridge_lambda)
        elif method_lower == 'constrained':
            if num_joints is None:
                raise ValueError("num_joints must be provided for constrained identification.")
            return self._solve_constrained(
                Phi_solve,
                tau_solve,
                num_joints=num_joints,
                ridge_lambda=ridge_lambda,
                sample_weights=None,
                initial_guess=initial_guess,
            )
        else:
            raise ValueError(f"Unsupported identification method: {method}")

        active_condition_number = (
            float(singular_values[0] / singular_values[-1])
            if len(singular_values) > 1 and singular_values[-1] > 0
            else np.inf
        )
        return {
            'theta_hat': np.asarray(theta_hat, dtype=float),
            'residual_vector': np.asarray(residual_vector, dtype=float),
            'rank': int(solved_rank),
            'singular_values': np.asarray(singular_values, dtype=float),
            'active_condition_number': active_condition_number,
            'ridge_lambda_used': ridge_lambda_used,
        }

    def _select_base_columns(self, Phi: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, int]:
        """用带 pivoting 的 QR 从完整列空间里选出可辨识列。"""
        _, R, pivots = qr(Phi, pivoting=True, mode='economic')
        diag = np.abs(np.diag(R))
        if diag.size == 0:
            return np.array([], dtype=int), 0
        threshold = tol * max(diag[0], 1.0)
        rank = int(np.sum(diag > threshold))
        return np.sort(np.array(pivots[:rank], dtype=int)), rank

    def identify_parameters(
        self,
        df: pd.DataFrame,
        method: str = 'ols',
        reference_parameters: np.ndarray | None = None,
        ridge_lambda: float = 1e-4,
        sample_weights: np.ndarray | None = None,
    ) -> Dict:
        """
        根据样本数据求解动力学参数。

        Parameters
        ----------
        df : pd.DataFrame
            处理后的训练集数据。
        method : str
            求解方法，支持 `ols` 和 `ridge`。
        reference_parameters : np.ndarray | None
            可选参考参数；synthetic 模式下可作为对照，real 模式通常留空。
        ridge_lambda : float
            ridge 正则强度，仅在 `method='ridge'` 时生效。

        Returns
        -------
        Dict
            包含参数估计、秩、条件数、残差等指标的结果字典。

        Raises
        ------
        ValueError
            当 `method` 不在支持列表中时抛出。
        """
        print("\nParameter Identification")
        print("=" * 70)
        print(f"Robot: {self.robot_model.name}")
        print(f"Parameterization: {self.parameterization}")

        Phi, tau_flat = self.regressor.build_regressor_matrix(df)
        U, singular_values, _ = np.linalg.svd(Phi, full_matrices=False)
        condition_number = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else np.inf

        full_rank = int(np.sum(singular_values > 1e-8 * singular_values[0])) if singular_values.size else 0
        base_columns, base_rank = self.solver._select_base_columns(Phi)
        if method.lower() == 'constrained' and self.parameterization != 'full':
            raise ValueError(
                "constrained solver currently requires parameterization='full', "
                "because physical constraints are expressed in the full inertial-parameter space."
            )
        if self.parameterization == 'base':
            active_columns = base_columns
            active_Phi = Phi[:, active_columns]
        else:
            active_columns = np.arange(Phi.shape[1], dtype=int)
            active_Phi = Phi

        method_upper = method.upper()
        initial_guess = None
        if method.lower() == 'constrained':
            try:
                initial_guess = np.asarray(self.robot_model.full_parameter_vector(), dtype=float)
                if initial_guess.shape[0] != active_Phi.shape[1]:
                    initial_guess = None
            except Exception:
                initial_guess = None
        solve_result = self.solver.solve(
            active_Phi,
            tau_flat,
            method=method,
            ridge_lambda=ridge_lambda,
            sample_weights=sample_weights,
            num_joints=self.num_joints,
            initial_guess=initial_guess,
        )
        theta_active = np.asarray(solve_result['theta_hat'], dtype=float)
        solved_rank = int(solve_result['rank'])
        s_active = np.asarray(solve_result['singular_values'], dtype=float)
        active_condition_number = float(solve_result['active_condition_number'])
        ridge_lambda_used = float(solve_result['ridge_lambda_used'])
        theta_full = np.zeros(Phi.shape[1], dtype=float)
        theta_full[active_columns] = theta_active

        tau_pred = Phi[:, active_columns] @ theta_active
        residual_vector = tau_flat - tau_pred
        parameter_vector = InertialParameterVector.from_theta_full(theta_full, self.num_joints)
        base_transform = np.zeros((Phi.shape[1], len(active_columns)), dtype=float)
        for beta_idx, full_idx in enumerate(active_columns):
            base_transform[full_idx, beta_idx] = 1.0
        gravity_config = GravityConfig.from_any(self.robot_model.gravity_vector)
        physical_sanity = ValidationTools.parameter_sanity_check(parameter_vector)

        result = {
            'theta_hat': theta_active,
            'theta_hat_full': theta_full,
            'pi_full_hat': theta_full,
            'beta_hat': theta_active,
            # 注意：对 synthetic 数据，可以把生成数据时使用的参数当作“参考真值”；
            # 但对 real 数据，URDF 里的惯性/摩擦通常只是先验或初值，不能当作真实答案。
            # 所以这里统一存成 reference_parameters，由调用方决定是否提供。
            'reference_parameters': None if reference_parameters is None else np.asarray(reference_parameters, dtype=float),
            'active_parameter_indices': active_columns,
            'base_column_indices': base_columns,
            'base_transform_full_from_beta': base_transform,
            'parameterization': self.parameterization,
            'rank': int(solved_rank),
            'full_regressor_rank': full_rank,
            'base_parameter_count': int(base_rank),
            'condition_number': condition_number,
            'active_condition_number': active_condition_number,
            'singular_values': singular_values.tolist(),
            'num_parameters_full': int(Phi.shape[1]),
            'num_parameters_active': int(len(active_columns)),
            'num_samples': int(len(df)),
            'num_equations': int(Phi.shape[0]),
            'residual_norm': float(np.linalg.norm(residual_vector)),
            'method': method_upper,
            'ridge_lambda': ridge_lambda_used,
            'model_options': dict(self.regressor.model_options),
            'gravity_config': gravity_config.to_dict(),
            'physical_sanity': physical_sanity,
            'parameter_vector': parameter_vector,
            'optimizer_success': bool(solve_result.get('optimizer_success', True)),
            'optimizer_status': solve_result.get('optimizer_status', 'not_applicable'),
            'optimizer_iterations': int(solve_result.get('optimizer_iterations', 0)),
        }

        print(f"Equations: {result['num_equations']}")
        print(f"Full parameters: {result['num_parameters_full']}")
        print(f"Active parameters: {result['num_parameters_active']}")
        print(f"Full regressor rank: {result['full_regressor_rank']}")
        print(f"Condition number: {result['condition_number']:.2e}")
        print(f"Active condition number: {result['active_condition_number']:.2e}")
        if result['full_regressor_rank'] < result['num_parameters_full']:
            print("Warning: regressor is rank deficient; only a base parameter subset is identifiable.")

        return result

    def predict_torques(self, df: pd.DataFrame, result: Dict) -> np.ndarray:
        """
        用已辨识参数回放给定数据集的关节力矩。

        Parameters
        ----------
        df : pd.DataFrame
            待预测数据集。
        result : Dict
            `identify_parameters()` 返回的结果字典。

        Returns
        -------
        np.ndarray
            形状为 `(num_samples, num_joints)` 的预测力矩矩阵。
        """
        Phi, _ = self.regressor.build_regressor_matrix(df)
        active_columns = np.array(result['active_parameter_indices'], dtype=int)
        theta_active = np.array(result['theta_hat'], dtype=float)
        tau_flat = Phi[:, active_columns] @ theta_active
        return tau_flat.reshape(len(df), self.num_joints)

    def evaluate_identification(self, df: pd.DataFrame, result: Dict) -> Dict:
        """
        评估辨识结果在给定数据集上的力矩误差。

        Parameters
        ----------
        df : pd.DataFrame
            待评估数据集。
        result : Dict
            `identify_parameters()` 返回的结果字典。

        Returns
        -------
        Dict
            含全局 RMSE/MAE、每关节误差及可选参考参数误差的字典。
        """
        tau_meas = np.column_stack([df[f'tau_{i}'].values for i in range(1, self.num_joints + 1)])
        tau_pred = self.predict_torques(df, result)
        error = tau_meas - tau_pred

        joint_rmse = np.sqrt(np.mean(error ** 2, axis=0))
        joint_mae = np.mean(np.abs(error), axis=0)

        evaluation = {
            'global_rmse': float(np.sqrt(np.mean(error ** 2))),
            'global_mae': float(np.mean(np.abs(error))),
            'joint_rmse': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(joint_rmse)},
            'joint_mae': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(joint_mae)},
        }

        reference_parameters = result.get('reference_parameters')
        if reference_parameters is not None:
            theta_reference = np.array(reference_parameters, dtype=float)
            theta_est_full = np.array(result['theta_hat_full'], dtype=float)
            rigid_count = self.regressor.num_rigid_params
            evaluation.update({
                'theta_reference_error_norm_full': float(np.linalg.norm(theta_est_full - theta_reference)),
                'theta_reference_error_norm_rigid': float(np.linalg.norm(theta_est_full[:rigid_count] - theta_reference[:rigid_count])),
                'theta_reference_error_norm_friction': float(np.linalg.norm(theta_est_full[rigid_count:] - theta_reference[rigid_count:])),
            })

        print("\nIdentification Evaluation:")
        print(f"  Global RMSE: {evaluation['global_rmse']:.6f} N·m")
        print(f"  Global MAE:  {evaluation['global_mae']:.6f} N·m")
        # 新增：按关节打印 RMSE/MAE，方便直接看出是哪几个关节误差更突出。
        print("  Per-joint metrics:")
        for joint_idx in range(self.num_joints):
            print(
                f"    Joint {joint_idx + 1}: "
                f"RMSE={joint_rmse[joint_idx]:.6f} N·m, "
                f"MAE={joint_mae[joint_idx]:.6f} N·m"
            )
        if reference_parameters is not None:
            print(f"  Theta reference error norm (full): {evaluation['theta_reference_error_norm_full']:.6f}")
        return evaluation


# Keep the public orchestration entry point stable after introducing
# `IdentificationSolver` as the lower-level numerical backend.
ParameterIdentifier.identify_parameters = IdentificationSolver.identify_parameters
ParameterIdentifier.predict_torques = IdentificationSolver.predict_torques
ParameterIdentifier.evaluate_identification = IdentificationSolver.evaluate_identification
