"""
Step 2: Generate physically consistent golden data.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from load_model import pin


def coulomb_sign(values: np.ndarray, deadzone: float = 1e-4) -> np.ndarray:
    sign = np.zeros_like(values)
    sign[values > deadzone] = 1.0
    sign[values < -deadzone] = -1.0
    return sign


class GoldenDataGenerator:
    """Generate golden inverse-dynamics samples from the Panda model."""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.pin_model = robot_model.pinocchio_model
        self.pin_data = robot_model.pinocchio_data
        self.num_joints = robot_model.num_joints

    def compute_dynamics(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute M(q), g(q), rigid torque and full torque."""
        pin.crba(self.pin_model, self.pin_data, q)
        M_raw = self.pin_data.M
        M = np.array(M_raw.dense()) if hasattr(M_raw, 'dense') else np.array(M_raw)

        g = np.array(pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q), dtype=float)
        tau_rigid = np.array(pin.rnea(self.pin_model, self.pin_data, q, dq, ddq), dtype=float)
        tau_friction = self.robot_model.damping * dq + self.robot_model.friction * coulomb_sign(dq)
        tau_full = tau_rigid + tau_friction
        return M, g, tau_rigid, tau_full

    def _make_record(
        self,
        case_id: int,
        case_type: str,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
    ) -> Dict:
        M, g, tau_rigid, tau_full = self.compute_dynamics(q, dq, ddq)
        return {
            'case_id': case_id,
            'case_type': case_type,
            'q': q.tolist(),
            'dq': dq.tolist(),
            'ddq': ddq.tolist(),
            'M': M.tolist(),
            'g': g.tolist(),
            'tau_rigid': tau_rigid.tolist(),
            'tau': tau_full.tolist(),
        }

    def generate_fixed_cases(self, num_cases: int = 12) -> List[Dict]:
        cases = []
        q_zero = np.zeros(self.num_joints)
        dq_zero = np.zeros(self.num_joints)
        ddq_nom = np.full(self.num_joints, 0.2)
        cases.append(self._make_record(0, 'zero_pose', q_zero, dq_zero, ddq_nom))
        case_id = 1

        for joint_idx, joint_name in enumerate(self.robot_model.joint_names):
            limits = self.robot_model.joint_limits[joint_name]
            for label, ratio in [('lower', 0.8), ('upper', 0.8)]:
                q = np.zeros(self.num_joints)
                q[joint_idx] = limits[label] * ratio
                dq = np.zeros(self.num_joints)
                ddq = np.zeros(self.num_joints)
                ddq[joint_idx] = 0.5
                cases.append(self._make_record(case_id, f'limit_{joint_idx + 1}_{label}', q, dq, ddq))
                case_id += 1

        return cases[:num_cases]

    def generate_random_cases(self, num_cases: int = 128, seed: int = 42) -> List[Dict]:
        rng = np.random.default_rng(seed)
        lowers = np.array([self.robot_model.joint_limits[name]['lower'] for name in self.robot_model.joint_names])
        uppers = np.array([self.robot_model.joint_limits[name]['upper'] for name in self.robot_model.joint_names])
        vmax = np.array([self.robot_model.joint_limits[name]['velocity'] for name in self.robot_model.joint_names])
        amax = 0.6 * vmax

        cases = []
        for case_id in range(num_cases):
            q = rng.uniform(lowers * 0.7, uppers * 0.7)
            dq = rng.uniform(-vmax * 0.5, vmax * 0.5)
            ddq = rng.uniform(-amax, amax)
            cases.append(self._make_record(case_id, 'random', q, dq, ddq))
        return cases

    def generate_trajectory_cases(self, num_points: int = 200) -> List[Dict]:
        lowers = np.array([self.robot_model.joint_limits[name]['lower'] for name in self.robot_model.joint_names])
        uppers = np.array([self.robot_model.joint_limits[name]['upper'] for name in self.robot_model.joint_names])
        amplitudes = 0.25 * np.minimum(np.abs(lowers), np.abs(uppers))
        phases = np.linspace(0.0, np.pi / 2.0, self.num_joints)
        base_freq = np.linspace(0.5, 1.1, self.num_joints)
        time = np.linspace(0.0, 8.0, num_points)

        cases = []
        for case_id, t in enumerate(time):
            q = amplitudes * np.sin(base_freq * t + phases)
            dq = amplitudes * base_freq * np.cos(base_freq * t + phases)
            ddq = -amplitudes * (base_freq ** 2) * np.sin(base_freq * t + phases)
            cases.append(self._make_record(case_id, 'trajectory', q, dq, ddq))
        return cases

    def export_to_json(self, cases: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(cases, handle, indent=2)
        print(f"Exported {len(cases)} cases to {output_path}")
