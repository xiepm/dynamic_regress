"""
统一的增强参数模型定义。

本文件不再保留旧版 9-param / 12-param / 15-param / 18-param 的并行口径，
而是统一采用下面这套机器人动力学参数结构：

每连杆理论参数（10个）：
1.  质量 m
2.  一阶质量矩 hx = m * cx
3.  一阶质量矩 hy = m * cy
4.  一阶质量矩 hz = m * cz
5.  惯性 Ixx
6.  惯性 Ixy
7.  惯性 Ixz
8.  惯性 Iyy
9.  惯性 Iyz
10. 惯性 Izz

每关节摩擦参数：
11. 粘性摩擦 fv
12. 库仑摩擦 fc
    或者使用不对称库仑摩擦时：
12. 正向库仑摩擦 fc_pos
13. 反向库仑摩擦 fc_neg

电机参数（可选）：
- 转子惯量 Jr

因此每关节参数数目有三种常见配置：
- 12个：10 刚体 + 1 粘性摩擦 + 1 对称库仑摩擦
- 13个：10 刚体 + 1 粘性摩擦 + 2 不对称库仑摩擦
- 13个：10 刚体 + 1 粘性摩擦 + 1 对称库仑摩擦 + 1 转子惯量
- 14个：10 刚体 + 1 粘性摩擦 + 2 不对称库仑摩擦 + 1 转子惯量
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RigidBodyParams:
    """
    单个连杆的 10 个理论惯性参数容器。

    这里把质量、质量矩和惯性张量拆成显式字段，是为了让参数结构和后续回归矩阵列布局
    一一对应，避免用匿名数组传来传去时丢失物理含义。
    """

    mass: float
    hx: float
    hy: float
    hz: float
    ixx: float
    ixy: float
    ixz: float
    iyy: float
    iyz: float
    izz: float

    def to_array(self) -> np.ndarray:
        """
        按统一列顺序把参数打平成数组。

        Returns
        -------
        np.ndarray
            长度为 10 的参数向量。
        """
        return np.array(
            [
                self.mass,
                self.hx,
                self.hy,
                self.hz,
                self.ixx,
                self.ixy,
                self.ixz,
                self.iyy,
                self.iyz,
                self.izz,
            ],
            dtype=float,
        )

    def to_dict(self) -> Dict:
        """
        把参数转换成更适合日志或 JSON 展示的字典结构。

        Returns
        -------
        Dict
            包含质量、质量矩和惯性张量字段的字典。
        """
        return {
            'mass': self.mass,
            'first_moments': [self.hx, self.hy, self.hz],
            'inertia_tensor': [self.ixx, self.ixy, self.ixz, self.iyy, self.iyz, self.izz],
        }


@dataclass
class JointFrictionParams:
    """
    关节摩擦参数容器。

    这里显式支持“对称库仑摩擦”和“不对称库仑摩擦”两种口径，目的是让同一套参数模型
    能覆盖工程里常见的摩擦建模变体，而不需要维护多份互不兼容的数据结构。
    """

    viscous: float
    coulomb_positive: float
    coulomb_negative: Optional[float] = None

    @property
    def asymmetric(self) -> bool:
        """判断当前摩擦模型是否区分正反向库仑摩擦。"""
        return self.coulomb_negative is not None

    def to_array(self) -> np.ndarray:
        """
        按当前摩擦模型配置导出参数数组。

        Returns
        -------
        np.ndarray
            对称模型返回 2 维，不对称模型返回 3 维。
        """
        if self.asymmetric:
            return np.array([self.viscous, self.coulomb_positive, self.coulomb_negative], dtype=float)
        return np.array([self.viscous, self.coulomb_positive], dtype=float)

    def to_dict(self) -> Dict:
        """
        转换成适合打印或序列化的字典格式。

        Returns
        -------
        Dict
            摩擦参数字典。
        """
        data = {
            'viscous': self.viscous,
            'coulomb_positive': self.coulomb_positive,
        }
        if self.asymmetric:
            data['coulomb_negative'] = self.coulomb_negative
        return data


@dataclass
class MotorParams:
    """可选电机参数，目前仅包含转子惯量。"""

    rotor_inertia: float

    def to_array(self) -> np.ndarray:
        """把电机参数导出为数组。"""
        return np.array([self.rotor_inertia], dtype=float)

    def to_dict(self) -> Dict:
        """把电机参数导出为字典。"""
        return {'rotor_inertia': self.rotor_inertia}


class ParameterModel:
    """
    统一参数模型管理器。

    两个可选开关：
    - asymmetric_coulomb: 是否区分正反向库仑摩擦
    - include_rotor_inertia: 是否纳入电机转子惯量
    """

    def __init__(self, asymmetric_coulomb: bool = False, include_rotor_inertia: bool = False):
        """
        根据两个开关组合出统一参数模型。

        Parameters
        ----------
        asymmetric_coulomb : bool
            是否启用正反向分离的库仑摩擦参数。
        include_rotor_inertia : bool
            是否把电机转子惯量纳入每关节参数。
        """
        self.asymmetric_coulomb = asymmetric_coulomb
        self.include_rotor_inertia = include_rotor_inertia

        self.rigid_body_params = 10
        self.viscous_friction_params = 1
        self.coulomb_params = 2 if asymmetric_coulomb else 1
        self.rotor_params = 1 if include_rotor_inertia else 0

        self.params_per_joint = (
            self.rigid_body_params
            + self.viscous_friction_params
            + self.coulomb_params
            + self.rotor_params
        )

    @property
    def model_name(self) -> str:
        """返回当前参数模型的简短名称。"""
        parts = ['rigid10', 'viscous1']
        parts.append('coulomb2' if self.asymmetric_coulomb else 'coulomb1')
        if self.include_rotor_inertia:
            parts.append('rotor1')
        return '+'.join(parts)

    def summary(self) -> Dict:
        """
        汇总当前参数模型的配置。

        Returns
        -------
        Dict
            各类参数数量与总参数数目摘要。
        """
        return {
            'model_name': self.model_name,
            'rigid_body_params_per_link': self.rigid_body_params,
            'viscous_friction_params_per_joint': self.viscous_friction_params,
            'coulomb_friction_params_per_joint': self.coulomb_params,
            'rotor_inertia_params_per_joint': self.rotor_params,
            'params_per_joint': self.params_per_joint,
        }

    def parameter_names_per_joint(self) -> List[str]:
        """
        给出每个关节参数的列名顺序。

        Returns
        -------
        List[str]
            当前配置下单关节参数名称列表。
        """
        names = [
            'm',
            'hx',
            'hy',
            'hz',
            'Ixx',
            'Ixy',
            'Ixz',
            'Iyy',
            'Iyz',
            'Izz',
            'fv',
        ]
        if self.asymmetric_coulomb:
            names.extend(['fc_pos', 'fc_neg'])
        else:
            names.append('fc')
        if self.include_rotor_inertia:
            names.append('Jr')
        return names

    @classmethod
    def print_supported_variants(cls):
        """
        打印当前统一参数模型支持的几个常见变体。

        Returns
        -------
        None
            结果直接输出到终端。
        """
        print("\n" + "=" * 80)
        print("统一参数模型的支持变体")
        print("=" * 80)
        variants = [
            cls(asymmetric_coulomb=False, include_rotor_inertia=False),
            cls(asymmetric_coulomb=True, include_rotor_inertia=False),
            cls(asymmetric_coulomb=False, include_rotor_inertia=True),
            cls(asymmetric_coulomb=True, include_rotor_inertia=True),
        ]
        for variant in variants:
            summary = variant.summary()
            print(f"\n【{summary['model_name']}】")
            print(f"  每关节参数数: {summary['params_per_joint']}")
            print(f"  参数列表: {', '.join(variant.parameter_names_per_joint())}")


class EnhancedRegressorBuilder:
    """
    基于统一参数模型的增强回归矩阵构建器。

    注意：
    - 这里仍然是教学/占位性质的线性回归结构示例；
    - 它表达的是“参数组织形式”和“各参数在回归矩阵中的列布局”；
    - 不应把这里的近似系数当作严格的刚体动力学解析式。
    """

    def __init__(
        self,
        num_joints: int = 6,
        asymmetric_coulomb: bool = False,
        include_rotor_inertia: bool = False,
    ):
        """
        初始化增强回归矩阵构建器。

        Parameters
        ----------
        num_joints : int
            主动关节数。
        asymmetric_coulomb : bool
            是否使用不对称库仑摩擦。
        include_rotor_inertia : bool
            是否纳入转子惯量项。
        """
        self.num_joints = num_joints
        self.model = ParameterModel(
            asymmetric_coulomb=asymmetric_coulomb,
            include_rotor_inertia=include_rotor_inertia,
        )
        self.params_per_joint = self.model.params_per_joint
        self.total_params = self.num_joints * self.params_per_joint

        print(f"Using enhanced parameter model: {self.model.model_name}")
        print(f"  Joints: {self.num_joints}")
        print(f"  Parameters per joint: {self.params_per_joint}")
        print(f"  Total parameters: {self.total_params}")

    def build_regressor_row(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        """
        为统一参数模型构建一行回归矩阵。

        列布局固定为：
        [10个刚体参数 | 1个粘性摩擦 | 1或2个库仑摩擦 | 可选1个转子惯量]

        Parameters
        ----------
        q : np.ndarray
            单个样本的关节位置。
        dq : np.ndarray
            单个样本的关节速度。
        ddq : np.ndarray
            单个样本的关节加速度。

        Returns
        -------
        np.ndarray
            单个样本对应的一行增强回归向量。
        """
        phi_row = np.zeros(self.total_params, dtype=float)

        for joint_idx in range(self.num_joints):
            base = joint_idx * self.params_per_joint
            qj = q[joint_idx]
            dqj = dq[joint_idx]
            ddqj = ddq[joint_idx]

            # 10 个连杆理论参数
            phi_row[base + 0] = ddqj                 # m
            phi_row[base + 1] = 0.1 * ddqj          # hx
            phi_row[base + 2] = 0.1 * ddqj          # hy
            phi_row[base + 3] = 0.1 * ddqj          # hz
            phi_row[base + 4] = ddqj                # Ixx
            phi_row[base + 5] = qj * ddqj           # Ixy
            phi_row[base + 6] = dqj * ddqj          # Ixz
            phi_row[base + 7] = ddqj                # Iyy
            phi_row[base + 8] = qj * dqj            # Iyz
            phi_row[base + 9] = ddqj                # Izz

            # 粘性摩擦
            phi_row[base + 10] = dqj

            next_col = base + 11

            # 库仑摩擦
            if self.model.asymmetric_coulomb:
                phi_row[next_col] = 1.0 if dqj > 1e-3 else 0.0
                phi_row[next_col + 1] = -1.0 if dqj < -1e-3 else 0.0
                next_col += 2
            else:
                phi_row[next_col] = np.sign(dqj) if abs(dqj) > 1e-3 else 0.0
                next_col += 1

            # 可选电机转子惯量
            if self.model.include_rotor_inertia:
                phi_row[next_col] = ddqj

        return phi_row

    def build_regressor_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        逐样本构造整张增强回归矩阵。

        Parameters
        ----------
        df : pd.DataFrame
            含 `q_i / dq_i / ddq_i / tau_i` 列的数据表。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            回归矩阵 `Phi` 与对应的力矩矩阵 `tau_data`。
        """
        q_data = np.array([df[f'q_{i}'].values for i in range(1, self.num_joints + 1)]).T
        dq_data = np.array([df[f'dq_{i}'].values for i in range(1, self.num_joints + 1)]).T
        ddq_data = np.array([df[f'ddq_{i}'].values for i in range(1, self.num_joints + 1)]).T
        tau_data = np.array([df[f'tau_{i}'].values for i in range(1, self.num_joints + 1)]).T

        num_samples = len(df)
        Phi = np.zeros((num_samples, self.total_params), dtype=float)

        for sample_idx in range(num_samples):
            Phi[sample_idx, :] = self.build_regressor_row(
                q_data[sample_idx],
                dq_data[sample_idx],
                ddq_data[sample_idx],
            )

        print(f"Built enhanced regressor matrix: shape {Phi.shape}")
        print(f"  Samples: {num_samples}")
        print(f"  Parameters per joint: {self.params_per_joint}")
        print(f"  Total parameters: {self.total_params}")
        return Phi, tau_data


if __name__ == "__main__":
    ParameterModel.print_supported_variants()

    print("\n" + "=" * 80)
    print("参数模型详细说明")
    print("=" * 80)
    print("""
固定理论参数部分（每连杆 10 个）：
  1. 质量 m
  2. 一阶质量矩 hx
  3. 一阶质量矩 hy
  4. 一阶质量矩 hz
  5. 惯性 Ixx
  6. 惯性 Ixy
  7. 惯性 Ixz
  8. 惯性 Iyy
  9. 惯性 Iyz
 10. 惯性 Izz

固定摩擦参数部分（每关节）：
  - 粘性摩擦 fv × 1
  - 库仑摩擦：
      对称模型 fc × 1
      或不对称模型 fc_pos, fc_neg × 2

可选电机参数：
  - 转子惯量 Jr × 1
    """)

    variants = [
        ('标准模型', EnhancedRegressorBuilder(num_joints=6, asymmetric_coulomb=False, include_rotor_inertia=False)),
        ('不对称摩擦模型', EnhancedRegressorBuilder(num_joints=6, asymmetric_coulomb=True, include_rotor_inertia=False)),
        ('带转子惯量模型', EnhancedRegressorBuilder(num_joints=6, asymmetric_coulomb=False, include_rotor_inertia=True)),
        ('完整模型', EnhancedRegressorBuilder(num_joints=6, asymmetric_coulomb=True, include_rotor_inertia=True)),
    ]

    print("\n" + "=" * 80)
    print("模型变体总参数对比")
    print("=" * 80)
    rows = []
    for display_name, builder in variants:
        rows.append({
            '模型': display_name,
            '每关节参数数': builder.params_per_joint,
            '6关节总参数数': builder.total_params,
            '参数列表': ', '.join(builder.model.parameter_names_per_joint()),
        })
    print(pd.DataFrame(rows).to_string(index=False))
