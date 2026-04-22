# 基于 URDF 的机器人动力学参数辨识示例

这个项目实现了一条基于 `URDF + Pinocchio` 的物理一致动力学辨识流程。  
当前主链已经不再绑定某一台特定机器人，核心思路是：

- 从一个 URDF 文件直接构建 Pinocchio 模型
- 用 `pin.rnea(...)` 计算逆动力学力矩
- 用 `pin.computeJointTorqueRegressor(...)` 构造刚体动力学回归矩阵
- 再额外拼接基础摩擦项：粘性摩擦 `dq` 与库伦摩擦 `sign(dq)`

项目默认支持两种参数化方式：

- `base`：只求解当前数据下真正可辨识的基参数子空间，推荐日常使用
- `full`：求解完整参数，适合理论对照，但通常更病态

## 当前设计重点

- 模型入口统一为 `URDF` 文件路径
- `yaml` 配置文件是可选的
- 数据来源只有两类：`synthetic` 和 `real`
- `raw / normalized / processed` 只是数据处理阶段，不是新的数据类型

## 关于 URDF 中 damping / friction 为 0

如果你使用的是“真实机器人对应的原始 URDF”，各关节的 `damping` 和 `friction` 读出来是 `0`，这是合理的。

这通常意味着：

- 该 URDF 主要描述几何、惯性、关节结构等基础信息
- 阻尼和摩擦还没有被单独建模或写回 URDF
- 这些量后续应通过辨识流程，从真实数据里估计出来

所以在本项目里：

- `URDF 中 damping / friction = 0` 不会被当成错误
- 它表示“当前模型还处在未辨识或未写回摩擦参数的初始状态”

## 项目结构

```text
dynamic_regress/
├── environment.yml
├── run_pipeline.py
├── models/
│   └── 05_urdf/
│       ├── config/
│       ├── meshes/
│       └── urdf/
│           └── 05_urdf_temp.urdf
├── python/
│   ├── load_model.py
│   ├── generate_golden_data.py
│   ├── process_measured_data.py
│   ├── identify_parameters.py
│   ├── residual_compensation.py
│   └── enhanced_parameter_model.py
└── datasets/
    ├── golden/
    ├── synthetic/
    ├── real/
    ├── identified/
    └── residual/
```

## 当前主流程

### Step 1: 加载模型

入口：[python/load_model.py](python/load_model.py)

- 直接从 `URDF` 构建 Pinocchio 模型
- 自动提取惯性参数、关节限位、阻尼和摩擦
- 如果提供了额外 yaml，则优先使用其中的 active joints、base/ee link 等元信息
- 如果没有 yaml，则直接从 URDF 自动推断 `robot_name / active_joint_names / base_link / ee_link`

这里要特别注意：

- 对真实机器人来说，URDF 中的惯性、阻尼、摩擦通常只能视为先验或初始模型
- 它们可以参与建模、生成回归矩阵、提供结构信息
- 但不应该在真实数据辨识场景里被当成“真实答案”去评价辨识误差

### Step 2: 生成 golden data

入口：[python/generate_golden_data.py](python/generate_golden_data.py)

- 这一步只在 synthetic / 调试场景下启用
- 用当前加载好的机器人模型生成一批标准参考样本
- 用于检查动力学实现是否稳定、回归结果是否退化
- 如果当前跑的是 `real` 数据模式，这一步会默认跳过

### Step 5: 获取并处理数据

入口：[python/process_measured_data.py](python/process_measured_data.py)

支持两类数据来源：

- `synthetic`：由当前模型自动生成合成数据
- `real`：从真实采集 CSV 批量读取

真实数据会依次经过：

- 原始 CSV 读取
- 列名标准化
- 单位转换（deg / deg/s / deg/s^2 -> rad / rad/s / rad/s^2）
- 时间同步
- 滤波
- 清洗
- 合并成统一处理后的数据集

当前真实数据默认约定：

- 位置：度 `deg`
- 速度：度每秒 `deg/s`
- 加速度：度每二次方秒 `deg/s^2`
- 扭矩：牛米 `N·m`

进入辨识主链前，会自动转换成动力学计算使用的 SI 单位：

- `q -> rad`
- `dq -> rad/s`
- `ddq -> rad/s^2`
- `tau` 保持 `N·m` 不变

### Step 6: 参数辨识

入口：[python/identify_parameters.py](python/identify_parameters.py)

最小二乘问题写成：

```text
Phi * theta = tau
```

其中：

- 刚体部分 `Phi_rigid` 来自 `pin.computeJointTorqueRegressor(...)`
- 摩擦部分来自各关节的 `dq` 和 `sign(dq)`
- 求解器使用 `scipy.linalg.lstsq`

如果选择 `base` 参数化，会先用带 pivoting 的 QR 选出可辨识列，再做 OLS。

### Step 7-8: 残差分析与补偿

入口：[python/residual_compensation.py](python/residual_compensation.py)

- 先计算主动力学模型的 torque 残差
- 再基于 `q, dq, ddq` 及若干衍生特征训练一个轻量线性补偿器

## 环境准备

推荐使用 `conda`。

### 方式 1：直接用 environment.yml

```bash
conda env create -f environment.yml
conda activate panda_id
```

### 方式 2：手动创建

```bash
conda create -n panda_id python=3.11 -y
conda activate panda_id
conda install -c conda-forge numpy pandas scipy pyyaml pinocchio -y
```

如果你希望沿用当前实际验证环境，也可以直接使用：

```bash
conda activate urdfly
```

## 运行方式

### 默认运行

当前默认模型已经切到你的新 URDF：

```bash
python run_pipeline.py
```

### 显式指定 URDF

```bash
python run_pipeline.py --urdf-path models/05_urdf/urdf/05_urdf_temp.urdf
```

### 同时指定真实数据

```bash
python run_pipeline.py --data-source real --urdf-path models/05_urdf/urdf/05_urdf_temp.urdf
```

### 如果还有额外 yaml 配置

```bash
python run_pipeline.py --urdf-path models/05_urdf/urdf/05_urdf_temp.urdf --config-path your_config.yaml
```

## 数据目录

```text
datasets/
├── synthetic/
│   ├── raw/
│   └── processed/
└── real/
    ├── raw/
    ├── normalized/
    └── processed/
```

说明：

- `synthetic` 和 `real` 是两类平行的数据来源
- `raw / normalized / processed` 是各自内部的处理阶段

真实采集数据统一放到：

```text
datasets/real/raw/
```

## 输出结果

主流程运行后，通常会生成：

- `datasets/golden/*.json`
- `datasets/synthetic/raw/*.csv`
- `datasets/synthetic/processed/*.csv`
- `datasets/real/normalized/*.csv`
- `datasets/real/processed/*.csv`
- `datasets/identified/theta_hat_*.json`
- `datasets/identified/evaluation_*_splits.json`
- `datasets/residual/compensation_result_*.json`

## 结果怎么理解

建议重点关注：

- `rank`
- `active_condition_number`
- `train_rmse`
- `val_rmse`
- `test_rmse`
- `possible_overfit`

一般来说：

- `base` 比 `full` 更适合作为工程结果
- 如果验证/测试误差明显高于训练误差，要警惕过拟合
- 如果 `active_condition_number` 很大，说明辨识仍然偏病态

## 关于增强参数模型

[python/enhanced_parameter_model.py](python/enhanced_parameter_model.py) 现在已经统一成固定参数口径：

- 每连杆理论参数：10 个
  质量 1 个
  一阶质量矩 3 个
  惯性张量 6 个
- 每关节摩擦参数：
  粘性摩擦 1 个
  库仑摩擦 1 个
  或正反向不对称库仑摩擦 2 个
- 电机参数（可选）：
  转子惯量 1 个

因此常见配置会是：

- 12 个/关节：10 刚体 + 1 粘性 + 1 对称库仑
- 13 个/关节：10 刚体 + 1 粘性 + 2 不对称库仑
- 13 个/关节：10 刚体 + 1 粘性 + 1 对称库仑 + 1 转子惯量
- 14 个/关节：10 刚体 + 1 粘性 + 2 不对称库仑 + 1 转子惯量

这个增强参数模型文件主要用于参数组织、模型变体说明和扩展实验，不直接替代当前主流程中的 Pinocchio 刚体回归实现。
