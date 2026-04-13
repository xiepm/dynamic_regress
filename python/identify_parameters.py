"""
Step 6: Physically consistent parameter identification.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import lstsq, qr

from generate_golden_data import coulomb_sign
from load_model import pin


class RegressorBuilder:
    """Build the rigid-body inverse-dynamics regressor plus friction columns."""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.num_joints = robot_model.num_joints
        self.num_rigid_params = 10 * self.num_joints
        self.num_friction_params = 2 * self.num_joints
        self.total_params = self.num_rigid_params + self.num_friction_params

    def build_regressor_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        q = np.column_stack([df[f'q_{i}'].values for i in range(1, self.num_joints + 1)])
        dq = np.column_stack([df[f'dq_{i}'].values for i in range(1, self.num_joints + 1)])
        ddq = np.column_stack([df[f'ddq_{i}'].values for i in range(1, self.num_joints + 1)])
        tau = np.column_stack([df[f'tau_{i}'].values for i in range(1, self.num_joints + 1)])

        num_samples = len(df)
        Phi = np.zeros((num_samples * self.num_joints, self.total_params))
        tau_flat = tau.reshape(-1)
        data = self.robot_model.pinocchio_model.createData()

        for sample_idx in range(num_samples):
            Y = np.asarray(
                pin.computeJointTorqueRegressor(
                    self.robot_model.pinocchio_model,
                    data,
                    q[sample_idx],
                    dq[sample_idx],
                    ddq[sample_idx],
                ),
                dtype=float,
            ).reshape(self.num_joints, self.num_rigid_params)

            friction_block = np.zeros((self.num_joints, self.num_friction_params))
            dq_sign = coulomb_sign(dq[sample_idx])
            for joint_idx in range(self.num_joints):
                friction_block[joint_idx, joint_idx] = dq[sample_idx, joint_idx]
                friction_block[joint_idx, self.num_joints + joint_idx] = dq_sign[joint_idx]

            row_slice = slice(sample_idx * self.num_joints, (sample_idx + 1) * self.num_joints)
            Phi[row_slice, :self.num_rigid_params] = Y
            Phi[row_slice, self.num_rigid_params:] = friction_block

        print(f"Built physically consistent regressor: shape {Phi.shape}")
        return Phi, tau_flat


class ParameterIdentifier:
    """Identify rigid-body and friction parameters with full or base parameterization."""

    def __init__(self, robot_model, parameterization: str = 'base'):
        self.robot_model = robot_model
        self.num_joints = robot_model.num_joints
        self.parameterization = parameterization
        self.regressor = RegressorBuilder(robot_model)

    def _select_base_columns(self, Phi: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, int]:
        _, R, pivots = qr(Phi, pivoting=True, mode='economic')
        diag = np.abs(np.diag(R))
        if diag.size == 0:
            return np.array([], dtype=int), 0
        threshold = tol * max(diag[0], 1.0)
        rank = int(np.sum(diag > threshold))
        return np.sort(np.array(pivots[:rank], dtype=int)), rank

    def identify_parameters(self, df: pd.DataFrame, method: str = 'ols') -> Dict:
        print("\nParameter Identification")
        print("=" * 70)
        print(f"Robot: {self.robot_model.name}")
        print(f"Parameterization: {self.parameterization}")

        Phi, tau_flat = self.regressor.build_regressor_matrix(df)
        U, singular_values, _ = np.linalg.svd(Phi, full_matrices=False)
        condition_number = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else np.inf

        full_rank = int(np.sum(singular_values > 1e-8 * singular_values[0])) if singular_values.size else 0
        base_columns, base_rank = self._select_base_columns(Phi)
        if self.parameterization == 'base':
            active_columns = base_columns
            active_Phi = Phi[:, active_columns]
        else:
            active_columns = np.arange(Phi.shape[1], dtype=int)
            active_Phi = Phi

        theta_active, residuals, solved_rank, s_active = lstsq(active_Phi, tau_flat)
        active_condition_number = float(s_active[0] / s_active[-1]) if len(s_active) > 1 and s_active[-1] > 0 else np.inf
        theta_full = np.zeros(Phi.shape[1], dtype=float)
        theta_full[active_columns] = theta_active

        tau_pred = Phi[:, active_columns] @ theta_active
        residual_vector = tau_flat - tau_pred

        result = {
            'theta_hat': theta_active,
            'theta_hat_full': theta_full,
            'theta_true': self.robot_model.full_parameter_vector(),
            'active_parameter_indices': active_columns,
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
            'method': method.upper(),
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
        Phi, _ = self.regressor.build_regressor_matrix(df)
        active_columns = np.array(result['active_parameter_indices'], dtype=int)
        theta_active = np.array(result['theta_hat'], dtype=float)
        tau_flat = Phi[:, active_columns] @ theta_active
        return tau_flat.reshape(len(df), self.num_joints)

    def evaluate_identification(self, df: pd.DataFrame, result: Dict) -> Dict:
        tau_meas = np.column_stack([df[f'tau_{i}'].values for i in range(1, self.num_joints + 1)])
        tau_pred = self.predict_torques(df, result)
        error = tau_meas - tau_pred

        joint_rmse = np.sqrt(np.mean(error ** 2, axis=0))
        theta_true = np.array(result['theta_true'], dtype=float)
        theta_est_full = np.array(result['theta_hat_full'], dtype=float)
        rigid_count = self.regressor.num_rigid_params

        evaluation = {
            'global_rmse': float(np.sqrt(np.mean(error ** 2))),
            'global_mae': float(np.mean(np.abs(error))),
            'joint_rmse': {f'joint_{idx + 1}': float(value) for idx, value in enumerate(joint_rmse)},
            'joint_mae': {
                f'joint_{idx + 1}': float(np.mean(np.abs(error[:, idx])))
                for idx in range(self.num_joints)
            },
            'theta_error_norm_full': float(np.linalg.norm(theta_est_full - theta_true)),
            'theta_error_norm_rigid': float(np.linalg.norm(theta_est_full[:rigid_count] - theta_true[:rigid_count])),
            'theta_error_norm_friction': float(np.linalg.norm(theta_est_full[rigid_count:] - theta_true[rigid_count:])),
        }

        print("\nIdentification Evaluation:")
        print(f"  Global RMSE: {evaluation['global_rmse']:.6f} N·m")
        print(f"  Global MAE:  {evaluation['global_mae']:.6f} N·m")
        print(f"  Theta error norm (full): {evaluation['theta_error_norm_full']:.6f}")
        return evaluation
