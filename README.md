# 机械臂动力学参数辨识完整流程示例

## 项目概述

这是一个完整的机械臂动力学参数辨识系统实现，基于**UR5标准机械臂**。包含了从模型定义、金标准数据生成、实测数据处理、参数辨识到残差补偿的完整流程。

## 项目结构

```
robot_dynamics_example/
├── models/
│   ├── urdf/
│   │   └── ur5.urdf                 # UR5机械臂URDF模型
│   └── configs/
│       └── ur5_config.yaml          # 机械臂配置和参考参数
├── python/
│   ├── load_model.py                # [Step 1] 加载模型
│   ├── generate_golden_data.py       # [Step 2] 生成金标准数据
│   ├── process_measured_data.py      # [Step 5] 处理测量数据
│   ├── identify_parameters.py        # [Step 6] 参数辨识
│   └── residual_compensation.py      # [Step 7-8] 残差补偿
├── datasets/                         # 所有流动数据
│   ├── golden/                       # 金标准数据
│   ├── measured/                     # 实测数据（原始、同步、处理后）
│   ├── identified/                   # 辨识结果
│   └── residual/                     # 残差分析结果
└── run_pipeline.py                   # 完整流程脚本
```

## 完整流程说明

### Step 1: 加载模型 (load_model.py)

**作用**: 从URDF和配置文件加载机械臂模型

**关键类**:
- `URDFLoader`: URDF和配置加载器
- `RobotModel`: 机械臂模型数据结构
- `JointParameters`: 单关节参数

**输入**:
- `models/urdf/ur5.urdf`: UR5标准URDF
- `models/configs/ur5_config.yaml`: 配置和参考参数

**输出**:
- 内存中的RobotModel对象
- Pinocchio模型（用于动力学计算）

**示例**:
```python
from load_model import URDFLoader

loader = URDFLoader("models/urdf/ur5.urdf", "models/configs/ur5_config.yaml")
robot_model = loader.build_robot_model()
print(f"Loaded {robot_model.name} with {robot_model.num_joints} joints")
```

---

### Step 2: 生成金标准数据 (generate_golden_data.py)

**作用**: 从已知模型和参数生成参考样本

**关键类**:
- `GoldenDataGenerator`: 金标准数据生成器

**生成三类数据**:

1. **固定测试点** (fixed_cases.json)
   - 零点、各joint极值点
   - 用于基础验证

2. **随机点** (random_cases_seed42.json)
   - 关节角度、速度、加速度随机采样
   - 覆盖工作空间

3. **轨迹点** (trajectory_cases.json)
   - 模拟正弦轨迹的连续运动
   - 接近真实应用

**数据字段**:
```json
{
  "case_id": 0,
  "case_type": "fixed_zero",
  "q": [0, 0, 0, 0, 0, 0],           // 位置
  "dq": [0, 0, 0, 0, 0, 0],          // 速度
  "ddq": [0.1, 0.1, 0.1, ...],       // 加速度
  "tau": [0.5, 2.1, 1.2, ...]        // 力矩（通过动力学计算）
}
```

**示例**:
```python
from generate_golden_data import GoldenDataGenerator

generator = GoldenDataGenerator(pin_model, pin_data, num_joints=6)
cases = generator.generate_random_cases(num_cases=100, seed=42)
generator.export_to_json(cases, "datasets/golden/random_cases.json")
```

---

### Step 3-4: 验证C++实现 (在完整项目中)

在本示例中跳过，但在实际项目中应该用C++代码读取golden数据进行验证。

---

### Step 5: 处理测量数据 (process_measured_data.py)

**作用**: 处理机械臂采集的实测数据

**关键类**:
- `MeasuredDataProcessor`: 数据处理管道

**处理步骤**:

1. **加载原始数据** (raw/exp_*.csv)
   ```csv
   timestamp,q_1,q_2,...,q_6,tau_1,...,tau_6
   ```

2. **时间同步** (synced/exp_*_synced.parquet)
   - 统一采样频率
   - 线性插值

3. **低通滤波** (Savitzky-Golay)
   - 移除高频噪声
   - 保留运动特性

4. **数值微分**
   - 从位置计算速度 (dq)
   - 从速度计算加速度 (ddq)

5. **异常检测和清理**
   - 检测速度/加速度异常
   - 标记连续有效段

**输出数据** (processed/exp_*_proc.parquet):
```
timestamp | q_1 | q_2 | ... | dq_1 | dq_2 | ... | ddq_1 | ... | tau_1 | ... | valid_mask | trajectory_id
```

**示例**:
```python
from process_measured_data import MeasuredDataProcessor

processor = MeasuredDataProcessor(num_joints=6)
df = processor.load_raw_csv("datasets/measured/raw/exp_*.csv")
df = processor.synchronize_timestamps(df)
df = processor.apply_low_pass_filter(df, cutoff_hz=10.0)
df = processor.differentiate_position(df)
df_clean = processor.clean_and_export(df, "datasets/measured/processed/exp_proc.parquet")
```

---

### Step 6: 参数辨识 (identify_parameters.py)

**作用**: 用实测数据估计机械臂参数

**关键类**:
- `RegressorBuilder`: 构建回归矩阵
- `ParameterIdentifier`: 参数求解器

**数学模型**:

机械臂动力学方程：
```
tau = M(q)*ddq + C(q,dq)*dq + g(q)
```

简化为线性形式：
```
tau = Phi(q,dq,ddq) * theta
```

其中 `theta` 是参数向量，`Phi` 是回归矩阵。

**求解方法**:

1. **普通最小二乘法 (OLS)**
   ```
   theta_hat = (Phi^T Phi)^{-1} Phi^T tau
   ```

2. **加权最小二乘法 (WLS)** [推荐]
   ```
   theta_hat = (Phi^T W Phi)^{-1} Phi^T W tau
   ```
   权重基于运动强度

3. **岭回归 (Ridge)**
   ```
   theta_hat = (Phi^T Phi + lambda*I)^{-1} Phi^T tau
   ```

**输出**:

参数文件 (identified/parameters/theta_hat_*.json):
```json
{
  "theta_hat": [3.7, 0.5, 0.3, ...],  // 辨识的参数
  "method": "weighted_least_squares",
  "condition_number": 123.45,
  "num_joints": 6,
  "num_parameters": 18,
  "num_samples": 1500
}
```

评估报告 (identified/reports/evaluation_*.json):
```json
{
  "rmse": 0.0234,
  "mae": 0.0156,
  "max_error": 0.0521,
  "joint_errors": {
    "joint_1": {"rmse": 0.012, "mae": 0.008, "max_error": 0.031},
    ...
  }
}
```

**示例**:
```python
from identify_parameters import ParameterIdentifier

identifier = ParameterIdentifier(num_joints=6)
result = identifier.identify_parameters(
    df_processed, 
    method='weighted_ls'
)
theta_hat = result['theta_hat']
evaluation = identifier.evaluate_identification(df_processed, result)
print(f"Identification RMSE: {evaluation['rmse']:.4f} N·m")
```

---

### Step 7: 残差分析 (residual_compensation.py)

**作用**: 分析刚体模型的残差，找出模型的局限性

**残差定义**:
```
r = tau_meas - tau_pred(theta_hat)
```

**分析内容**:

1. **计算残差**: 实测力矩 - 预测力矩
2. **特征工程**:
   - 状态量: q, dq, ddq
   - 衍生特征: |dq|, sign(dq), q×dq等
   - 全局特征: 运动量等

3. **数据分割**:
   - 训练集 (train): 70%
   - 验证集 (val): 10%
   - 测试集 (test): 20%

**输出数据** (residual/datasets/residual_*.parquet):
```
q_1, ..., dq_1, ..., ddq_1, ... | tau_pred_1, ... | residual_tau_1, ... | 特征列
```

**示例**:
```python
from residual_compensation import ResidualAnalyzer

analyzer = ResidualAnalyzer(num_joints=6)
datasets = analyzer.build_residual_dataset(
    df_processed, theta_hat,
    test_split=0.2, val_split=0.1
)
```

---

### Step 8: 残差补偿 (residual_compensation.py)

**作用**: 训练数据驱动的补偿模型

**补偿模型**:

1. **线性补偿**
   ```
   r_pred = W * features + b
   tau_final = tau_pred + r_pred
   ```

2. **神经网络补偿**
   ```
   h = tanh(X @ W1 + b1)
   r_pred = h @ W2 + b2
   tau_final = tau_pred + r_pred
   ```

**训练流程**:
- 特征归一化
- 模型训练（梯度下降）
- 验证集监控
- 测试集评估

**评估指标**:
```json
{
  "mae": 0.0089,           // 补偿后平均误差
  "rmse": 0.0112,
  "mse": 0.0001,
  "original_mae": 0.0234,  // 补偿前误差
  "improvement_percent": 62.0  // 改进百分比
}
```

**示例**:
```python
from residual_compensation import ResidualCompensator

compensator = ResidualCompensator(num_joints=6)
compensator.train_linear_compensator(datasets['train'])
eval_result = compensator.evaluate_compensator(datasets['test'])
print(f"Improvement: {eval_result['improvement_percent']:.1f}%")
```

---

## 运行完整流程

### 方法1: 运行完整管道脚本

```bash
python run_pipeline.py
```

这将自动执行所有步骤并生成输出。

### 方法2: 逐步运行各模块

```bash
# Step 1: 加载模型
python python/load_model.py

# Step 2: 生成金标准数据 (需要pinocchio)
python python/generate_golden_data.py

# Step 5: 处理测量数据
python python/process_measured_data.py

# Step 6: 参数辨识
python python/identify_parameters.py

# Step 7-8: 残差补偿
python python/residual_compensation.py
```

### 方法3: 在Python脚本中集成

```python
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python"))

from load_model import URDFLoader
from process_measured_data import MeasuredDataProcessor, create_synthetic_measured_data
from identify_parameters import ParameterIdentifier
from residual_compensation import ResidualAnalyzer, ResidualCompensator

# 加载模型
loader = URDFLoader("models/urdf/ur5.urdf", "models/configs/ur5_config.yaml")
robot_model = loader.build_robot_model()

# 处理数据
processor = MeasuredDataProcessor(num_joints=6)
df_raw = create_synthetic_measured_data(num_samples=2000)
df_processed = processor.differentiate_position(
    processor.apply_low_pass_filter(df_raw)
)

# 参数辨识
identifier = ParameterIdentifier(num_joints=6)
result = identifier.identify_parameters(df_processed, method='weighted_ls')

# 残差补偿
analyzer = ResidualAnalyzer(num_joints=6)
datasets = analyzer.build_residual_dataset(df_processed, result['theta_hat'])

compensator = ResidualCompensator(num_joints=6)
compensator.train_linear_compensator(datasets['train'])
eval_result = compensator.evaluate_compensator(datasets['test'])

print(f"Improvement: {eval_result['improvement_percent']:.1f}%")
```

---

## 数据流图

```
models/urdf/ur5.urdf + config.yaml
    ↓
[Step 1] load_model.py → RobotModel
    ↓
[Step 2] generate_golden_data.py → datasets/golden/*
    ↓
[Step 3-4] C++ validation (可选)
    ↓
Real Robot Experiment
    ↓
datasets/measured/raw/*
    ↓
[Step 5] process_measured_data.py
    ↓
datasets/measured/processed/*
    ↓
[Step 6] identify_parameters.py
    ↓
Regressor Matrix Phi + tau_stacked
    ↓
Least Squares / WLS / Ridge
    ↓
datasets/identified/parameters/theta_hat_*
    ↓
[Step 6] 预测: tau_pred = Phi @ theta_hat
    ↓
datasets/identified/predictions/tau_pred_*
    ↓
[Step 7] residual_compensation.py
    ↓
datasets/residual/datasets/* (train/val/test)
    ↓
[Step 8] 训练补偿模型
    ↓
datasets/residual/models/compensator_model
    ↓
最终部署: tau_final = tau_pred + compensator(features)
```

---

## 关键参数配置

在 `models/configs/ur5_config.yaml` 中配置：

```yaml
identification:
  method: "weighted_least_squares"    # 选择辨识方法
  regularization_lambda: 0.001        # 岭回归的正则化参数
  use_joint_friction: true            # 是否考虑关节摩擦
  use_gravity: true                   # 是否考虑重力项
```

在各脚本中调整：

```python
# 处理数据参数
processor.apply_low_pass_filter(df, cutoff_hz=10.0, sampling_freq=100.0)

# 参数辨识方法
identifier.identify_parameters(df_processed, method='weighted_ls')

# 残差补偿参数
compensator.train_linear_compensator(df_train)  # 或用神经网络
```

---

## 输出解释

### 参数辨识结果

- **RMSE (Root Mean Square Error)**: 预测力矩与实测力矩的均方根误差
  - 值越小越好，表示模型拟合度好
  
- **MAE (Mean Absolute Error)**: 平均绝对误差
  - 更容易理解的误差度量

- **Condition Number**: 回归矩阵的条件数
  - < 100: 良好
  - 100-1000: 一般
  - > 1000: 数值不稳定

### 残差补偿结果

- **improvement_percent**: 补偿的改进百分比
  - 例如 62% 表示误差减少到原来的 38%

- **MAE 改进**: original_mae → mae 的减少
  - 原来误差 0.0234 → 补偿后 0.0089

---

## 扩展和改进

### 1. 使用真实机械臂数据

替换 `create_synthetic_measured_data()` 为实际采集的CSV数据：

```python
df_raw = processor.load_raw_csv("path/to/real_robot_data.csv")
```

### 2. 改进补偿模型

使用更复杂的模型：

```python
# 神经网络补偿
compensator.train_neural_network_compensator(
    datasets['train'],
    hidden_size=128,
    epochs=200
)
```

### 3. 考虑更复杂的动力学

在 `RegressorBuilder.build_regressor_row()` 中扩展参数：

```python
# 现在: [m, f_v, f_s] 共18个参数
# 可扩展为: [m, x_com, y_com, z_com, I_xx, I_yy, I_zz, f_v, f_s] 共54个参数
```

### 4. 跨平台部署

将训练的补偿模型导出为：
- ONNX (用于C++推理)
- TorchScript (用于PyTorch)
- JSON (用于自定义推理)

---

## 常见问题

**Q: 为什么我的参数辨识误差很大？**

A: 检查以下几点：
1. 测量数据质量是否好（是否有足够的激励）
2. 数据是否对齐（时间戳同步）
3. 是否有足够的样本数（至少1000）
4. 微分是否正确（考虑使用更大的滤波窗口）

**Q: 如何验证参数辨识的结果？**

A: 
1. 检查RMSE和MAE（与全局力矩幅度比较）
2. 可视化 tau_meas vs tau_pred
3. 检查残差的分布（应该接近高斯分布）
4. 验证条件数（< 100最好）

**Q: 需要多少训练数据？**

A: 
- 最少：500-1000样本
- 推荐：2000-5000样本
- 更好的结果：10000+样本

**Q: 可以对标准UR5直接使用这个代码吗？**

A: 可以，但需要：
1. 用真实的UR5 URDF替换ur5.urdf
2. 用真实参考参数更新config.yaml
3. 采集真实机械臂数据替换synthetic data

---

## 许可证

MIT License

---

## 参考文献

- Pinocchio: https://stack-of-tasks.github.io/pinocchio/
- Robot Dynamics: "Handbook of Robotics" (Springer)
- Parameter Identification: Ljung, "System Identification: Theory for the User"
