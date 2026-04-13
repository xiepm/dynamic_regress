# 快速开始指南

## 📋 项目概述

这是一个基于 **UR5 标准机械臂** 的完整动力学参数辨识系统，实现了从模型定义、数据处理到参数估计的全流程。

## 🚀 5分钟快速运行

### 1. 安装依赖

```bash
pip install numpy pandas scipy pyyaml

# 可选（用于生成金标准数据）
pip install pinocchio
```

### 2. 运行完整流程

```bash
python run_pipeline.py
```

### 3. 查看输出

```
datasets/
├── golden/
│   ├── fixed_cases.json           # 金标准数据
│   ├── random_cases_seed42.json
│   └── trajectory_cases.json
├── measured/
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 处理后数据
│   └── synced/
├── identified/                    # 辨识结果
│   ├── parameters/
│   │   └── theta_hat_synthetic.json
│   └── reports/
│       └── evaluation_synthetic.json
└── residual/                      # 残差补偿结果
    ├── datasets/
    └── compensation_result.json
```

---

## 📁 文件说明

### 核心模块

| 文件 | Step | 功能 |
|------|------|------|
| `load_model.py` | 1 | 加载URDF和配置文件 |
| `generate_golden_data.py` | 2 | 生成参考样本数据 |
| `process_measured_data.py` | 5 | 处理机械臂采集数据 |
| `identify_parameters.py` | 6 | 参数辨识（最小二乘法） |
| `residual_compensation.py` | 7-8 | 残差分析和补偿 |

### 模型文件

| 文件 | 说明 |
|------|------|
| `models/urdf/ur5.urdf` | UR5机械臂URDF模型（6DOF） |
| `models/configs/ur5_config.yaml` | 配置和参考参数 |

---

## 🔄 完整数据流

```
URDF + 配置文件
    ↓
加载模型 (Step 1)
    ↓
生成金标准数据 (Step 2)
    ↓
采集机械臂数据
    ↓
处理测量数据 (Step 5)
    → 同步、滤波、微分、清洗
    ↓
参数辨识 (Step 6)
    → 构建回归矩阵、最小二乘求解
    ↓
残差分析 (Step 7)
    → 计算 r = τ_meas - τ_pred
    ↓
训练补偿模型 (Step 8)
    → 线性或神经网络补偿
    ↓
最终结果：θ_hat + 补偿模型
```

---

## 💡 使用示例

### 示例1：只加载模型

```python
from python.load_model import URDFLoader

loader = URDFLoader("models/urdf/ur5.urdf", "models/configs/ur5_config.yaml")
robot = loader.build_robot_model()
print(f"Loaded {robot.name} with {robot.num_joints} joints")
```

### 示例2：生成golden数据

```python
from python.generate_golden_data import GoldenDataGenerator

generator = GoldenDataGenerator(pin_model, pin_data, num_joints=6)
cases = generator.generate_random_cases(num_cases=100, seed=42)
generator.export_to_json(cases, "datasets/golden/random.json")
```

### 示例3：处理测量数据

```python
from python.process_measured_data import MeasuredDataProcessor, create_synthetic_measured_data

processor = MeasuredDataProcessor(num_joints=6)
df_raw = create_synthetic_measured_data(num_samples=1000)
df_filtered = processor.apply_low_pass_filter(df_raw, cutoff_hz=10.0)
df_processed = processor.differentiate_position(df_filtered)
```

### 示例4：参数辨识

```python
from python.identify_parameters import ParameterIdentifier

identifier = ParameterIdentifier(num_joints=6)
result = identifier.identify_parameters(df_processed, method='weighted_ls')
print(f"RMSE: {result['rmse']:.4f} N·m")
```

### 示例5：残差补偿

```python
from python.residual_compensation import ResidualAnalyzer, ResidualCompensator

analyzer = ResidualAnalyzer(num_joints=6)
datasets = analyzer.build_residual_dataset(df_processed, result['theta_hat'])

compensator = ResidualCompensator(num_joints=6)
compensator.train_linear_compensator(datasets['train'])
eval_result = compensator.evaluate_compensator(datasets['test'])
print(f"改进: {eval_result['improvement_percent']:.1f}%")
```

---

## 📊 预期输出

运行完整流程后，你会看到：

```
=======================================================================
 ROBOT DYNAMICS IDENTIFICATION PIPELINE (UR5)
=======================================================================

[STEP 1] Loading Model...
Robot Model: UR5
Number of Joints: 6
[Joint 1] joint_1
  Mass: 3.700 kg
  CoM: [0. 0. 0.1]
  ...

[STEP 2] Generating Golden Data...
Generated 12 fixed cases
Generated 100 random cases
Generated 50 trajectory cases

[STEP 5] Processing Measured Data...
Loaded raw data: 500 samples
Synchronized to 100.0Hz: 500 samples
Applied low-pass filter: cutoff=10.0Hz
Differentiated: computed 6 velocities and 6 accelerations
Detected 5 invalid samples (1.0%)

[STEP 6] Identifying Parameters...
Method: weighted_least_squares
Samples: 475
Parameters: 18
Condition number: 45.23
RMSE: 0.0234 N·m
MAE: 0.0156 N·m

[STEP 7-8] Residual Analysis & Compensation...
Train: 333 samples
Val: 48 samples
Test: 94 samples

Linear compensator trained:
  Train MAE: 0.0089 N·m

Compensator Evaluation:
  MAE: 0.0089 N·m
  Improvement: 62.0%

=======================================================================
 PIPELINE COMPLETED SUCCESSFULLY
=======================================================================
```

---

## ⚙️ 自定义配置

### 修改参数辨识方法

在 `run_pipeline.py` 中修改：

```python
result = identifier.identify_parameters(
    df_processed,
    method='ridge',  # 可选: 'ols', 'weighted_ls', 'ridge'
    lambda=0.01      # ridge 参数
)
```

### 调整低通滤波频率

```python
df_filtered = processor.apply_low_pass_filter(
    df_synced, 
    cutoff_hz=5.0,   # 从10.0改为5.0（更平滑）
    sampling_freq=100.0
)
```

### 修改补偿器类型

```python
# 线性补偿（快速）
compensator.train_linear_compensator(datasets['train'])

# 神经网络补偿（精度高）
compensator.train_neural_network_compensator(
    datasets['train'],
    hidden_size=128,
    epochs=200
)
```

---

## 🔧 故障排查

### 问题1: ImportError: No module named 'pinocchio'

**解决**: Pinocchio 是可选的，用于生成golden数据。如果不需要可以跳过。

```bash
# 如果想使用：
pip install pinocchio
```

### 问题2: 数据处理后样本数为0

**原因**: 可能检测到太多异常样本。

**解决**: 调整异常检测阈值：

```python
valid_mask = processor.detect_invalid_samples(
    df,
    velocity_threshold=3.0,        # 从2.0改为3.0
    acceleration_threshold=8.0     # 从5.0改为8.0
)
```

### 问题3: 参数辨识的RMSE很大

**原因**: 
1. 数据激励不足
2. 采样频率过低
3. 特征工程不够好

**解决**:
1. 收集更多、更激励的数据
2. 提高采样频率到至少100Hz
3. 在 `residual_compensation.py` 中添加更多特征

---

## 📚 UR5机械臂信息

本项目使用标准的UR5机械臂模型：

- **关节数**: 6 (6DOF)
- **关节类型**: 都是旋转关节
- **主要参数**:
  - Shoulder Pan (Joint 1): ±180°
  - Shoulder Lift (Joint 2): ±180°
  - Elbow (Joint 3): ±180°
  - Wrist 1/2/3: ±180° 各自

详细的关节配置见 `models/configs/ur5_config.yaml`

---

## 🎯 关键指标解释

| 指标 | 含义 | 好的范围 |
|------|------|---------|
| RMSE | 预测误差均方根 | < 0.05 N·m |
| MAE | 平均绝对误差 | < 0.03 N·m |
| Condition Number | 矩阵的数值稳定性 | < 100 |
| Improvement % | 补偿后的改进 | > 50% |

---

## 🔗 文件关系图

```
run_pipeline.py
    ├─→ load_model.py
    │   └─→ models/urdf/ur5.urdf
    │   └─→ models/configs/ur5_config.yaml
    ├─→ generate_golden_data.py
    │   └─→ datasets/golden/*
    ├─→ process_measured_data.py
    │   └─→ datasets/measured/processed/*
    ├─→ identify_parameters.py
    │   └─→ datasets/identified/*
    └─→ residual_compensation.py
        └─→ datasets/residual/*
```

---

## 📖 详细文档

更详细的说明请参考 `README.md`

---

## 💬 示例代码段

### 只运行参数辨识部分

```python
import sys
from pathlib import Path
sys.path.insert(0, "python")

from process_measured_data import create_synthetic_measured_data, MeasuredDataProcessor
from identify_parameters import ParameterIdentifier

# 生成测试数据
df_raw = create_synthetic_measured_data(num_samples=1000)

# 处理
processor = MeasuredDataProcessor(num_joints=6)
df_processed = processor.differentiate_position(
    processor.apply_low_pass_filter(df_raw)
)

# 辨识
identifier = ParameterIdentifier(num_joints=6)
result = identifier.identify_parameters(df_processed, method='weighted_ls')
evaluation = identifier.evaluate_identification(df_processed, result)

print(f"RMSE: {evaluation['rmse']:.4f} N·m")
```

### 导入真实数据

```python
# 用真实CSV数据替换合成数据
df_raw = processor.load_raw_csv("path/to/your/robot_data.csv")

# 然后继续处理...
df_processed = processor.differentiate_position(
    processor.apply_low_pass_filter(df_raw)
)
```

---

## 🎓 学习路径

1. **第一步**: 运行 `python run_pipeline.py` 了解完整流程
2. **第二步**: 阅读各模块的代码注释，理解每个步骤
3. **第三步**: 用自己的数据替换synthetic data
4. **第四步**: 调整参数优化结果
5. **第五步**: 扩展模型（如添加更复杂的动力学）

---

## 📝 许可证

MIT License

---

**准备好了吗？运行 `python run_pipeline.py` 开始吧！** 🚀
