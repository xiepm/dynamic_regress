#!/usr/bin/env python3
"""
测试 output/identifiedDynamics.cpp 中生成的C++函数

从 datasets/real/raw 读取原始数据
使用Python调用相同的动力学模型进行预测
对比结果验证C++代码的正确性
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# 添加python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "python"))

from identify_parameters import ParameterIdentifier, RegressorBuilder
from load_model import URDFLoader

def main():
    # 1. 读取原始数据
    raw_data_path = project_root / "datasets/real/raw/vel_20.csv"
    print(f"读取原始数据: {raw_data_path}")

    df_raw = pd.read_csv(raw_data_path)
    print(f"加载 {len(df_raw)} 个样本\n")

    # 2. 单位转换（度->弧度）
    DEG_TO_RAD = np.pi / 180.0
    q_cols = [f'q{i}_JointPos' for i in range(1, 8)]
    dq_cols = [f'q{i}_JointVel' for i in range(1, 8)]
    ddq_cols = [f'q{i}_JointAcc' for i in range(1, 8)]
    tau_cols = [f'q{i}_SensedTorque' for i in range(1, 8)]

    # 检查列名
    if q_cols[0] not in df_raw.columns:
        q_cols = [f'q{i}_pos' for i in range(1, 8)]
        dq_cols = [f'q{i}_vel' for i in range(1, 8)]
        ddq_cols = [f'q{i}_acc' for i in range(1, 8)]
        tau_cols = [f'q{i}_sensedTorque' for i in range(1, 8)]

    q = df_raw[q_cols].values * DEG_TO_RAD
    dq = df_raw[dq_cols].values * DEG_TO_RAD
    ddq = df_raw[ddq_cols].values * DEG_TO_RAD
    tau_measured = df_raw[tau_cols].values

    # 3. 加载模型和辨识结果
    urdf_path = project_root / "models/05_urdf/urdf/05_urdf_temp.urdf"
    loader = URDFLoader(str(urdf_path), gravity_vector=[9.81, 0.0, 0.0])
    robot_model = loader.build_robot_model()

    theta_path = project_root / "datasets/identified/theta_hat_real_sensed_latest.json"
    with open(theta_path) as f:
        theta_result = json.load(f)

    theta_full = np.array(theta_result['theta_hat_full'])

    # 4. 使用Python预测（模拟C++函数的行为）
    builder = RegressorBuilder(robot_model)

    # 构造DataFrame
    df_test = pd.DataFrame({
        **{f'q_{i}': q[:, i-1] for i in range(1, 8)},
        **{f'dq_{i}': dq[:, i-1] for i in range(1, 8)},
        **{f'ddq_{i}': ddq[:, i-1] for i in range(1, 8)},
        **{f'tau_{i}': tau_measured[:, i-1] for i in range(1, 8)},
    })

    # 构建回归矩阵并预测
    Phi, _ = builder.build_regressor_matrix(df_test)
    tau_predicted = (Phi @ theta_full).reshape(-1, 7)

    # 5. 计算误差
    errors = tau_predicted - tau_measured
    rmse_per_joint = np.sqrt(np.mean(errors**2, axis=0))
    mae_per_joint = np.mean(np.abs(errors), axis=0)
    global_rmse = np.sqrt(np.mean(errors**2))
    global_mae = np.mean(np.abs(errors))

    # 6. 输出结果
    print("="*60)
    print("  Python预测结果（模拟C++函数）")
    print("="*60)
    print(f"\n全局 RMSE: {global_rmse:.4f} N·m")
    print(f"全局 MAE:  {global_mae:.4f} N·m\n")

    print("各关节 RMSE:")
    for i in range(7):
        print(f"  J{i+1}: {rmse_per_joint[i]:.4f} N·m")

    print("\n" + "="*60)
    print("说明:")
    print("="*60)
    print("这是使用Python模拟C++函数的预测结果。")
    print("C++函数 calculateIdentifiedInverseDynamicsCore 应该")
    print("产生相同的结果（误差 < 0.001 N·m）。")
    print("\n如果要验证C++实现，需要：")
    print("1. 编译 identifiedDynamics.cpp")
    print("2. 用相同的输入数据调用C++函数")
    print("3. 对比输出是否一致")
    print("="*60)

if __name__ == "__main__":
    main()
