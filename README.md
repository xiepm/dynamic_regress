# Panda 动力学参数辨识示例

这个项目实现了一条基于 `Franka Panda` 和 `Pinocchio` 的物理一致动力学辨识流程。当前主链不再使用早期的 UR5 手写经验回归模型，而是统一使用：

- Panda 开源 URDF
- `pin.rnea(...)` 生成逆动力学力矩
- `pin.computeJointTorqueRegressor(...)` 构造刚体动力学回归矩阵
- 额外拼接基础摩擦项：粘性摩擦 `dq` 与库伦摩擦 `sign(dq)`

项目默认支持两种参数化方式：

- `base`：仅求解当前数据下可辨识的基参数子空间，推荐日常使用
- `full`：求解全部 84 个参数，通常会明显病态，只适合做理论对照

## 项目结构

```text
robot_dynamics_example/
├── environment.yml
├── run_pipeline.py
├── models/
│   ├── configs/
│   │   ├── panda_config.yaml
│   │   └── ur5_config.yaml
│   └── urdf/
│       ├── panda_arm_minimal.urdf
│       └── ur5.urdf
├── python/
│   ├── load_model.py
│   ├── generate_golden_data.py
│   ├── process_measured_data.py
│   ├── identify_parameters.py
│   ├── residual_compensation.py
│   └── enhanced_parameter_model.py
└── datasets/
```

## 当前主流程

### Step 1: 加载模型

入口：[python/load_model.py](python/load_model.py)

- 从 `models/urdf/panda_arm_minimal.urdf` 构建 Pinocchio 模型
- 从 URDF 中提取惯性参数、关节限位、阻尼和摩擦
- `panda_config.yaml` 只保留 active joints、base/ee link 等轻量元信息

### Step 2: 生成 golden data

入口：[python/generate_golden_data.py](python/generate_golden_data.py)

- 用 `pin.crba(...)` 计算 `M(q)`
- 用 `pin.computeGeneralizedGravity(...)` 计算 `g(q)`
- 用 `pin.rnea(...)` 计算刚体逆动力学力矩 `tau_rigid`
- 再叠加摩擦得到 `tau`

### Step 5: 生成并处理 synthetic measured data

入口：[python/process_measured_data.py](python/process_measured_data.py)

- 生成激励轨迹 `q, dq, ddq`
- 用同一个 Panda 动力学模型生成 synthetic torque
- 叠加小噪声和时间戳扰动
- 再做同步、滤波、数值微分和清洗

### Step 6: 参数辨识

入口：[python/identify_parameters.py](python/identify_parameters.py)

最小二乘问题写成：

```text
Phi * theta = tau
```

其中：

- 刚体部分 `Phi_rigid` 来自 `pin.computeJointTorqueRegressor(...)`
- 摩擦部分来自每个关节的 `dq` 和 `sign(dq)`
- 求解器使用 `scipy.linalg.lstsq`

如果选择 `base` 参数化，会先用带 pivoting 的 QR 选出可辨识列，再做 OLS。

### Step 7-8: 残差分析与补偿

入口：[python/residual_compensation.py](python/residual_compensation.py)

- 先计算主动力学模型的 torque 残差
- 再基于 `q, dq, ddq` 及若干衍生特征训练一个轻量线性补偿器

## 环境准备

推荐使用 `conda`，尤其是在 Windows 上运行 PyCharm 时。

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

## Windows + PyCharm 推荐做法

1. 先安装 `Miniconda` 或 `Anaconda`
2. 在 `Anaconda Prompt` 中执行：

```bash
conda env create -f environment.yml
conda activate panda_id
```

3. 打开 PyCharm，选择：

```text
File -> Settings -> Project -> Python Interpreter
```

4. `Add Interpreter -> Conda -> Existing environment`
5. 选择类似下面的解释器路径：

```text
C:\Users\你的用户名\miniconda3\envs\panda_id\python.exe
```

这样做的好处是：

- `pinocchio` 这类依赖由 conda 统一管理
- PyCharm 只负责使用现成环境，不直接安装复杂依赖
- 更容易在多台机器上复现

## 运行方式

### 默认运行

```bash
python run_pipeline.py
```

### 显式指定参数化方式

```bash
python run_pipeline.py --parameterization base
python run_pipeline.py --parameterization full
```

## 输出结果

主流程运行后，通常会生成：

- `datasets/golden/*.json`
- `datasets/measured/raw/synthetic_panda_run01.csv`
- `datasets/measured/processed/synthetic_panda_run01_proc.csv`
- `datasets/identified/theta_hat_synthetic.json`
- `datasets/identified/evaluation_synthetic.json`
- `datasets/residual/compensation_result.json`

典型输出会包含：

- `Full parameters`
- `Active parameters`
- `Rank`
- `Condition number`
- `Active condition number`
- `Global torque RMSE`
- `Residual compensation improvement`

## 怎么理解 base 和 full

- `full` 对 7 个关节的全部刚体参数和摩擦参数一起求解，总共 84 维
- `base` 只保留当前数据真正可辨识的参数子空间

在刚体动力学辨识里，`full` 病态是常见现象；因此工程上更应该关注：

- `base_parameter_count`
- `rank`
- `active_condition_number`
- `global_rmse`

## 说明

[python/enhanced_parameter_model.py](python/enhanced_parameter_model.py) 和旧的 `9-param / 12-param / 15-param` 逻辑目前仍保留在仓库中，主要用于历史参考，不属于当前主流程。
