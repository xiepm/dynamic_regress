"""
生成用于回归测试和基线对照的 golden data。

这个模块位于整条辨识流水线的 Step 2，由 `run_pipeline.py` 在 synthetic /
调试模式下调用。它依赖 `load_model.py` 中构建好的 `RobotModel` 和
Pinocchio 模型，通过同一套逆动力学接口生成一批“已知输入、已知输出”的标准样本，
方便后续验证动力学实现、辨识求解和数据处理链路是否退化。
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from load_model import pin


def coulomb_sign(values: np.ndarray, deadzone: float = 1e-4) -> np.ndarray:
    """把速度向量映射成带死区的库仑摩擦符号项。"""
    sign = np.zeros_like(values)
    sign[values > deadzone] = 1.0
    sign[values < -deadzone] = -1.0
    return sign


class GoldenDataGenerator:
    """
    基于当前机器人模型生成标准逆动力学样本。

    这个类不直接参与真实数据辨识，而是提供一套“可重复、可控”的样本生成器，
    用来给回归测试、单元检查和数值对照提供统一输入。核心思路是尽量复用辨识主链
    里同一套 Pinocchio 动力学接口，避免 golden data 和正式流程来自两套不同实现。
    """

    def __init__(self, robot_model):
        """
        保存机器人模型及其 Pinocchio 句柄。

        Parameters
        ----------
        robot_model : RobotModel
            已经从 URDF 构建完成的机器人模型对象。
        """
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
        """
        计算当前状态下的惯性矩阵、重力项和力矩。

        Parameters
        ----------
        q : np.ndarray
            关节位置，单位 rad。
        dq : np.ndarray
            关节速度，单位 rad/s。
        ddq : np.ndarray
            关节加速度，单位 rad/s^2。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            依次为 `M(q)`、`g(q)`、纯刚体力矩 `tau_rigid`、带摩擦的总力矩 `tau_full`。
        """
        pin.crba(self.pin_model, self.pin_data, q)
        M_raw = self.pin_data.M
        # 兼容 Pinocchio 在不同版本里对 CRBA 输出矩阵的存储形式差异。
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
        """把一次动力学计算整理成可序列化的标准记录。"""
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
        """
        生成若干具有明确物理含义的固定工况样本。

        Parameters
        ----------
        num_cases : int
            最终保留的样本数上限。

        Returns
        -------
        List[Dict]
            每个元素都是一条可直接导出为 JSON 的样本记录。
        """
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
        """
        在关节限位与速度范围内随机采样动力学状态。

        Parameters
        ----------
        num_cases : int
            需要生成的随机样本数。
        seed : int
            随机种子，用于保证样本可复现。

        Returns
        -------
        List[Dict]
            随机工况样本列表。
        """
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
        """
        生成一段平滑激励轨迹上的连续样本。

        Parameters
        ----------
        num_points : int
            轨迹离散点数。

        Returns
        -------
        List[Dict]
            轨迹样本列表，适合检查连续运动下的动力学输出。
        """
        lowers = np.array([self.robot_model.joint_limits[name]['lower'] for name in self.robot_model.joint_names])
        uppers = np.array([self.robot_model.joint_limits[name]['upper'] for name in self.robot_model.joint_names])
        # 振幅取关节限位的一部分，避免 golden 轨迹一开始就落在极端姿态附近。
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
        """
        把 golden data 导出为 JSON 文件。

        Parameters
        ----------
        cases : List[Dict]
            待导出的样本列表。
        output_path : str
            输出 JSON 路径。

        Returns
        -------
        None
            结果直接写入磁盘。
        """
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(cases, handle, indent=2)
        print(f"Exported {len(cases)} cases to {output_path}")
