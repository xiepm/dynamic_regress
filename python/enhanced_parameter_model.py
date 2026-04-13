"""
改进的参数模型：12个参数/joint (包含电机惯量)
Enhanced Parameter Model with Motor Inertia

每个joint的12个参数：
1-7.   刚体参数 (m, x_com, y_com, z_com, I_xx, I_yy, I_zz)
8.     J_motor - 电机绕动惯量 (kg·m²)
9.     r_gear  - 减速比 (无量纲)
10.    f_v_motor - 电机侧粘性摩擦 (N·m·s/rad)
11.    f_s_motor - 电机侧库伦摩擦 (N·m)
12.    efficiency - 传动效率 (0-1)

或者选择：
1-7.   刚体参数
8.     J_eff   - 等效电机惯量 = J_motor × r_gear² (kg·m²)
9-10.  f_v, f_s (关节侧总摩擦)
11.    f_v_motor - 电机侧额外摩擦
12.    friction_asym - 摩擦不对称性参数
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json


# ============================================================================
# 新的参数数据结构
# ============================================================================

@dataclass
class RigidBodyParams:
    """刚体参数 (7个)"""
    mass: float              # m
    com_x: float            # x_com
    com_y: float            # y_com
    com_z: float            # z_com
    inertia_xx: float       # I_xx
    inertia_yy: float       # I_yy
    inertia_zz: float       # I_zz
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.mass, self.com_x, self.com_y, self.com_z,
            self.inertia_xx, self.inertia_yy, self.inertia_zz
        ])
    
    def to_dict(self) -> Dict:
        return {
            'mass': self.mass,
            'com': [self.com_x, self.com_y, self.com_z],
            'inertia': [self.inertia_xx, self.inertia_yy, self.inertia_zz]
        }


@dataclass
class DrivetrainParams:
    """传动系统参数 (5个)"""
    J_motor: float           # 电机转动惯量
    r_gear: float            # 减速比
    f_v_motor: float         # 电机侧粘性摩擦
    f_s_motor: float         # 电机侧库伦摩擦
    efficiency: float        # 传动效率 (0-1)
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.J_motor, self.r_gear, self.f_v_motor, self.f_s_motor, self.efficiency
        ])
    
    def to_dict(self) -> Dict:
        return {
            'J_motor': self.J_motor,
            'r_gear': self.r_gear,
            'f_v_motor': self.f_v_motor,
            'f_s_motor': self.f_s_motor,
            'efficiency': self.efficiency
        }


@dataclass
class FrictionParams:
    """摩擦参数 (4个模型，可选)"""
    f_v: float               # 关节侧粘性摩擦
    f_s: float               # 关节侧库伦摩擦
    f_v_asym: float          # 摩擦不对称性 (速度依赖)
    f_s_asym: float          # 库伦摩擦不对称性
    
    def to_array(self) -> np.ndarray:
        return np.array([self.f_v, self.f_s, self.f_v_asym, self.f_s_asym])
    
    def to_dict(self) -> Dict:
        return {
            'f_v': self.f_v,
            'f_s': self.f_s,
            'f_v_asym': self.f_v_asym,
            'f_s_asym': self.f_s_asym
        }


class ParameterModel:
    """
    参数模型管理器，支持多种参数配置
    """
    
    # 参数模型配置
    MODELS = {
        '9-param': {
            'description': '基础模型（仅刚体+基础摩擦）',
            'rigid_body': True,      # 7个
            'drivetrain': False,     # 0个
            'friction': False,       # 0个
            'advanced': False,       # 0个
            'total': 9,
            'suitable_for': '教学、模拟、初步验证'
        },
        '12-param': {
            'description': '工业模型（包含电机惯量）',
            'rigid_body': True,      # 7个
            'drivetrain': True,      # 5个
            'friction': False,       # 0个
            'advanced': False,       # 0个
            'total': 12,
            'suitable_for': '实际机械臂、中等精度'
        },
        '13-param': {
            'description': '改进型（包含电机和库伦摩擦）',
            'rigid_body': True,      # 7个
            'drivetrain': True,      # 4个 (J_motor, r_gear, f_v_motor, f_s_motor)
            'friction': False,       # 0个
            'advanced': False,       # 0个
            'total': 13,
            'suitable_for': '高摩擦机械臂、精密应用'
        },
        '15-param': {
            'description': '高精度模型（包含非对角惯量）',
            'rigid_body': True,      # 7个
            'drivetrain': True,      # 4个
            'friction': False,       # 0个
            'advanced': True,        # 4个 (I_xy, I_xz, I_yz, efficiency)
            'total': 15,
            'suitable_for': '高精度应用、充足数据'
        },
        '18-param': {
            'description': '完整模型（所有主要参数）',
            'rigid_body': True,      # 7个
            'drivetrain': True,      # 4个
            'friction': True,        # 4个 (f_v, f_s, f_v_asym, f_s_asym)
            'advanced': True,        # 3个 (I_xy, I_xz, I_yz)
            'total': 18,
            'suitable_for': '研究、高精度、充足数据'
        }
    }
    
    @classmethod
    def print_models(cls):
        """打印所有支持的模型"""
        print("\n" + "="*80)
        print("支持的参数模型")
        print("="*80)
        for model_name, config in cls.MODELS.items():
            print(f"\n【{model_name}】 {config['total']}个参数")
            print(f"  描述: {config['description']}")
            print(f"  适用: {config['suitable_for']}")
            print(f"  包含:")
            if config['rigid_body']:
                print(f"    - 刚体参数 (7个): m, x_com, y_com, z_com, I_xx, I_yy, I_zz")
            if config['drivetrain']:
                print(f"    - 传动参数 (4个): J_motor, r_gear, f_v_motor, f_s_motor")
            if config['friction']:
                print(f"    - 摩擦参数 (4个): f_v, f_s, f_v_asym, f_s_asym")
            if config['advanced']:
                print(f"    - 高级参数 (3个): I_xy, I_xz, I_yz")


class EnhancedRegressorBuilder:
    """
    改进的回归矩阵构建器，支持12参数模型
    """
    
    def __init__(self, num_joints: int = 6, param_model: str = '12-param'):
        self.num_joints = num_joints
        self.param_model = param_model
        self.config = ParameterModel.MODELS[param_model]
        self.params_per_joint = self.config['total']
        self.total_params = num_joints * self.params_per_joint
        
        print(f"Using {param_model} model:")
        print(f"  Joints: {num_joints}")
        print(f"  Parameters per joint: {self.params_per_joint}")
        print(f"  Total parameters: {self.total_params}")
    
    def build_regressor_row_12param(self, q: np.ndarray, dq: np.ndarray,
                                    ddq: np.ndarray, 
                                    gravity_term: np.ndarray = None) -> np.ndarray:
        """
        为12参数模型构建回归矩阵的一行
        
        12个参数：
        [m, x_com, y_com, z_com, I_xx, I_yy, I_zz, J_motor, r_gear, f_v_motor, f_s_motor, efficiency]
        
        动力学方程：
        τ = M(q)·ddq + C(q,dq)·dq + g(q) + F_friction + τ_motor_eff
        
        其中关键项：
        - J_motor贡献: J_motor × r_gear² × ddq
        - 摩擦: (f_v + f_v_motor × r_gear²)·dq + (f_s + f_s_motor × r_gear²)·sign(dq)
        """
        phi_row = np.zeros(self.num_joints * self.params_per_joint)
        
        if gravity_term is None:
            gravity_term = np.zeros(self.num_joints)
        
        for i in range(self.num_joints):
            base_idx = i * self.params_per_joint
            
            # 第1-7个参数：刚体动力学（与原模型相同）
            # m: 加速度系数
            phi_row[base_idx + 0] = ddq[i]
            
            # x_com, y_com, z_com: 质心位置系数
            # 这些与q的二阶导数和Coriolis项有关
            # 简化：线性近似
            phi_row[base_idx + 1] = 0.1 * ddq[i]  # 近似系数
            phi_row[base_idx + 2] = 0.1 * ddq[i]
            phi_row[base_idx + 3] = 0.1 * ddq[i]
            
            # I_xx, I_yy, I_zz: 惯量系数
            # 与速度二次项和角加速度有关
            phi_row[base_idx + 4] = ddq[i]
            phi_row[base_idx + 5] = ddq[i]
            phi_row[base_idx + 6] = ddq[i]
            
            # 第8-12个参数：传动系统
            # 8: J_motor - 电机惯量对加速度的影响
            #    τ中的贡献: J_motor × r_gear² × ddq
            phi_row[base_idx + 7] = ddq[i]  # r_gear² 的系数被吸收到参数中
            
            # 9: r_gear - 减速比（非线性，难以处理）
            #    简化：假设r_gear已知，参数化为J_eff = J_motor × r_gear²
            #    则: phi_row[base_idx + 8] = ddq[i]  （但这样就与第8个重复）
            # 
            #    更好的做法：参数化为 [J_eff, f_v_eff, f_s_eff] 其中包含r_gear的影响
            #    或者假设r_gear已知，从外部输入
            phi_row[base_idx + 8] = 0  # 暂时设为0，因为通常r_gear是已知的
            
            # 10: f_v_motor - 电机侧粘性摩擦
            #     贡献: f_v_motor × r_gear² × dq
            phi_row[base_idx + 9] = dq[i]  # r_gear² 的系数在参数中
            
            # 11: f_s_motor - 电机侧库伦摩擦
            #     贡献: f_s_motor × r_gear² × sign(dq)
            phi_row[base_idx + 10] = np.sign(dq[i]) if abs(dq[i]) > 0.01 else 0
            
            # 12: efficiency - 传动效率（通常非线性）
            #     简化处理或设为常数
            phi_row[base_idx + 11] = 0  # 效率通常作为常数应用，不线性参数化
        
        return phi_row
    
    def build_regressor_row_alternative(self, q: np.ndarray, dq: np.ndarray,
                                       ddq: np.ndarray,
                                       r_gear: np.ndarray = None) -> np.ndarray:
        """
        更好的12参数模型：参数化为等效参数
        
        假设减速比r_gear是已知的，参数化为：
        θ = [m, x_com, y_com, z_com, I_xx, I_yy, I_zz, J_eff, f_v_eff, f_s_eff, 
             f_v_motor_eff, f_s_motor_eff]
        
        其中：
        - J_eff = J_motor × r_gear²
        - f_v_eff = f_v + f_v_motor × r_gear² （总粘性摩擦）
        - f_s_eff = f_s + f_s_motor × r_gear² （总库伦摩擦）
        - f_v_motor_eff = f_v_motor × r_gear²
        - f_s_motor_eff = f_s_motor × r_gear²
        """
        if r_gear is None:
            r_gear = np.ones(self.num_joints)  # 默认减速比为1
        
        phi_row = np.zeros(self.num_joints * self.params_per_joint)
        
        for i in range(self.num_joints):
            base_idx = i * self.params_per_joint
            r2 = r_gear[i] ** 2
            
            # 刚体参数（7个）
            phi_row[base_idx + 0] = ddq[i]      # m
            phi_row[base_idx + 1] = 0.1*ddq[i]  # x_com
            phi_row[base_idx + 2] = 0.1*ddq[i]  # y_com
            phi_row[base_idx + 3] = 0.1*ddq[i]  # z_com
            phi_row[base_idx + 4] = ddq[i]      # I_xx
            phi_row[base_idx + 5] = ddq[i]      # I_yy
            phi_row[base_idx + 6] = ddq[i]      # I_zz
            
            # 传动参数（5个）
            phi_row[base_idx + 7] = ddq[i] * r2  # J_eff = J_motor × r²
            phi_row[base_idx + 8] = dq[i]        # f_v_eff （关节侧）
            phi_row[base_idx + 9] = np.sign(dq[i]) if abs(dq[i]) > 0.01 else 0  # f_s_eff
            phi_row[base_idx + 10] = dq[i] * r2   # f_v_motor_eff （电机侧，经过减速比）
            phi_row[base_idx + 11] = np.sign(dq[i]) * r2 if abs(dq[i]) > 0.01 else 0  # f_s_motor_eff
        
        return phi_row

    def build_regressor_row_15param(self, q: np.ndarray, dq: np.ndarray,
                                    ddq: np.ndarray,
                                    r_gear: np.ndarray = None) -> np.ndarray:
        """
        15参数模型：在12参数基础上加入非对角惯量项

        θ = [m, x_com, y_com, z_com, I_xx, I_yy, I_zz,
             I_xy, I_xz, I_yz, J_eff, f_v_eff, f_s_eff,
             f_v_motor_eff, f_s_motor_eff]
        """
        if r_gear is None:
            r_gear = np.ones(self.num_joints)

        phi_row = np.zeros(self.num_joints * self.params_per_joint)

        for i in range(self.num_joints):
            base_idx = i * self.params_per_joint
            r2 = r_gear[i] ** 2
            dq_sign = np.sign(dq[i]) if abs(dq[i]) > 0.01 else 0

            phi_row[base_idx + 0] = ddq[i]        # m
            phi_row[base_idx + 1] = 0.1 * ddq[i]  # x_com
            phi_row[base_idx + 2] = 0.1 * ddq[i]  # y_com
            phi_row[base_idx + 3] = 0.1 * ddq[i]  # z_com
            phi_row[base_idx + 4] = ddq[i]        # I_xx
            phi_row[base_idx + 5] = ddq[i]        # I_yy
            phi_row[base_idx + 6] = ddq[i]        # I_zz

            # 非对角惯量项，用速度与加速度耦合项作为近似激励
            phi_row[base_idx + 7] = q[i] * ddq[i]   # I_xy
            phi_row[base_idx + 8] = dq[i] * ddq[i]  # I_xz
            phi_row[base_idx + 9] = q[i] * dq[i]    # I_yz

            phi_row[base_idx + 10] = ddq[i] * r2    # J_eff
            phi_row[base_idx + 11] = dq[i]          # f_v_eff
            phi_row[base_idx + 12] = dq_sign        # f_s_eff
            phi_row[base_idx + 13] = dq[i] * r2     # f_v_motor_eff
            phi_row[base_idx + 14] = dq_sign * r2   # f_s_motor_eff

        return phi_row
    
    def build_regressor_matrix(self, df: pd.DataFrame,
                              r_gear: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建完整的回归矩阵
        """
        # 提取数据
        q_data = np.array([df[f'q_{i}'].values for i in range(1, self.num_joints+1)]).T
        dq_data = np.array([df[f'dq_{i}'].values for i in range(1, self.num_joints+1)]).T
        ddq_data = np.array([df[f'ddq_{i}'].values for i in range(1, self.num_joints+1)]).T
        tau_data = np.array([df[f'tau_{i}'].values for i in range(1, self.num_joints+1)]).T
        
        num_samples = len(df)
        
        Phi = np.zeros((num_samples, self.total_params))
        tau_stacked = np.zeros((num_samples, self.num_joints))
        
        # 为每个样本构建回归矩阵行
        for n in range(num_samples):
            q = q_data[n]
            dq = dq_data[n]
            ddq = ddq_data[n]
            tau = tau_data[n]
            
            if self.param_model in ['9-param']:
                # 原始9参数模型（基础实现）
                phi_row = self._build_regressor_row_9param(q, dq, ddq)
            elif self.param_model == '15-param':
                phi_row = self.build_regressor_row_15param(q, dq, ddq, r_gear)
            else:
                # 使用改进的12参数模型
                phi_row = self.build_regressor_row_alternative(q, dq, ddq, r_gear)
            
            Phi[n, :] = phi_row
            tau_stacked[n, :] = tau
        
        print(f"Built regressor matrix: shape {Phi.shape}")
        print(f"  Samples: {num_samples}")
        print(f"  Parameters: {self.total_params}")
        
        return Phi, tau_stacked
    
    def _build_regressor_row_9param(self, q: np.ndarray, dq: np.ndarray,
                                    ddq: np.ndarray) -> np.ndarray:
        """9参数的基础实现"""
        phi_row = np.zeros(self.num_joints * 9)
        
        for i in range(self.num_joints):
            base_idx = i * 9
            phi_row[base_idx + 0] = ddq[i]
            phi_row[base_idx + 1] = 0.1*ddq[i]
            phi_row[base_idx + 2] = 0.1*ddq[i]
            phi_row[base_idx + 3] = 0.1*ddq[i]
            phi_row[base_idx + 4] = ddq[i]
            phi_row[base_idx + 5] = ddq[i]
            phi_row[base_idx + 6] = ddq[i]
            phi_row[base_idx + 7] = dq[i]
            phi_row[base_idx + 8] = np.sign(dq[i]) if abs(dq[i]) > 0.01 else 0
        
        return phi_row


# ============================================================================
# 使用示例和测试
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    # 打印所有支持的模型
    ParameterModel.print_models()
    
    print("\n" + "="*80)
    print("参数对比表")
    print("="*80)
    
    # 创建对比表
    comparison_data = []
    for model_name, config in ParameterModel.MODELS.items():
        comparison_data.append({
            '模型': model_name,
            '参数数': config['total'],
            '刚体': '是' if config['rigid_body'] else '否',
            '传动': '是' if config['drivetrain'] else '否',
            '摩擦': '是' if config['friction'] else '否',
            '高级': '是' if config['advanced'] else '否',
            '适用场景': config['suitable_for']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    print("\n" + "="*80)
    print("参数详细对比")
    print("="*80)
    
    model_details = {
        '9-param': [
            'm', 'x_com', 'y_com', 'z_com', 'I_xx', 'I_yy', 'I_zz', 'f_v', 'f_s'
        ],
        '12-param': [
            'm', 'x_com', 'y_com', 'z_com', 'I_xx', 'I_yy', 'I_zz',
            'J_motor', 'r_gear', 'f_v_motor', 'f_s_motor'
        ],
        '13-param': [
            'm', 'x_com', 'y_com', 'z_com', 'I_xx', 'I_yy', 'I_zz',
            'J_motor', 'r_gear', 'f_v_motor', 'f_s_motor', 'f_s'
        ],
        '15-param': [
            'm', 'x_com', 'y_com', 'z_com', 'I_xx', 'I_yy', 'I_zz', 'I_xy', 'I_xz', 'I_yz',
            'J_motor', 'r_gear', 'f_v_motor', 'f_s_motor'
        ],
        '18-param': [
            'm', 'x_com', 'y_com', 'z_com', 'I_xx', 'I_yy', 'I_zz', 'I_xy', 'I_xz', 'I_yz',
            'J_motor', 'r_gear', 'f_v_motor', 'f_s_motor', 'f_v', 'f_s', 'f_v_asym', 'f_s_asym'
        ]
    }
    
    for model_name, params in model_details.items():
        print(f"\n【{model_name}】")
        print(f"  参数数: {len(params)}")
        print(f"  参数列表: {', '.join(params)}")
        print(f"  总参数 (6关节): {len(params) * 6}")
    
    print("\n" + "="*80)
    print("建议用法")
    print("="*80)
    print("""
【学习和教学】→ 使用 9-param
  - 简单易懂
  - 不容易过拟合
  - 快速验证概念

【实际机械臂】→ 使用 12-param
  - 包含电机效应
  - 平衡精度和复杂性
  - 推荐用于工业应用

【高精度应用】→ 使用 15-param 或以上
  - 需要充足的数据 (5000+ 样本)
  - 高采样频率 (200Hz+)
  - 完整的轨迹激励

【研究应用】→ 使用 18-param 或自定义
  - 可根据具体系统定制
  - 验证可识别性
  - 进行不确定性分析
    """)
    
    # 创建EnhancedRegressorBuilder示例
    print("\n" + "="*80)
    print("创建不同参数模型的回归矩阵构建器")
    print("="*80)
    
    for model_name in ['9-param', '12-param', '13-param']:
        print(f"\n{model_name}:")
        builder = EnhancedRegressorBuilder(num_joints=6, param_model=model_name)
        print(f"  ✓ 创建成功")
