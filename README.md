# 基于 URDF 的机器人动力学参数辨识示例

这个项目实现了一条基于 `URDF + Pinocchio` 的物理一致动力学辨识流程。  
当前主链已经不再绑定某一台特定机器人，核心思路是：

- 从一个 URDF 文件直接构建 Pinocchio 模型
- 用 `pin.rnea(...)` 计算逆动力学力矩
- 用 `pin.computeJointTorqueRegressor(...)` 构造刚体动力学回归矩阵
- 再额外拼接基础摩擦项：粘性摩擦 `dq` 与库伦摩擦 `sign(dq)`

## 新手建议先看

如果你现在最关心的是：

- 项目 clone 下来以后先执行什么命令
- conda 环境怎么激活
- 真实数据应该放到哪里
- 主程序怎么运行
- 可视化 notebook 应该怎么刷新最新结果

建议先直接看：

- [QUICKSTART.md](/home/xpm/projects/dynamic7dof/dynamic_regress/QUICKSTART.md)

那份文档已经按“从 clone 到运行，再到可视化”的实际操作顺序整理好了。

项目默认支持两种参数化方式：

- `base`：只求解当前数据下真正可辨识的基参数子空间，推荐日常使用
- `full`：求解完整参数，适合理论对照，但通常更病态

## 当前设计重点

- 模型入口统一为 `URDF` 文件路径
- `yaml` 配置文件是可选的
- 数据来源只有两类：`synthetic` 和 `real`
- `raw / normalized / processed` 只是数据处理阶段，不是新的数据类型
- 导出的 C++ 动力学类默认保持旧版 `LEGACY_RAW` 行为，同时可选启用内部 `ROBUST_NO_TIME` 运行模式

## Runtime RobustNoTime 说明

导出后的 `sevendofDynamics.h/.cpp` 现在支持一个默认关闭的内部鲁棒层：

- `LEGACY_RAW`：完全保持旧版行为
- `ROBUST_NO_TIME`：在不改变原有虚函数签名的前提下，增加输入检查、样本级平滑、`ddq` 幅值限幅、摩擦状态机、静止保持、`M/C` 门控和最近一次分项诊断 getter
- `ROBUST_TIMED_RESERVED`：仅预留，将来如果有可靠 `timestamp/dt` 或内部时间源，再扩展成真正的 time-aware 模式

`RobustNoTime` 可以做：

- 输入合法性检查
- 样本级平滑
- `ddq` 幅值限幅
- 摩擦状态机
- 静止保持
- `M/C` 门控
- 分项诊断输出

`RobustNoTime` 不能严格做：

- Hz 意义上的低通滤波
- 从 `q` 推导真实 `dq`
- 从 `dq` 推导真实 `ddq`
- 基于物理时间的 rate limit
- 严格时间常数滤波
- 真实 alpha-beta / Kalman 状态估计

离线辨识侧也新增了一个可选 `--robust-identification` 开关。默认关闭；开启后会在 processed dataframe 中追加
`q_used_i / dq_used_i / ddq_used_i / tau_used_i / friction_sign_i / hold_indicator_i / motion_state_i / sample_type`
这些增强列，而不破坏原有列契约。

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

## `run_pipeline.py` 命令行选项

当前主入口仍然是：

```bash
python run_pipeline.py [options]
```

内部流程现在已经拆分为：

- [python/pipeline_identification.py](/home/xpm/projects/dynamic7dof/dynamic_regress/python/pipeline_identification.py:1)：只负责辨识阶段
- [python/pipeline_postprocess.py](/home/xpm/projects/dynamic7dof/dynamic_regress/python/pipeline_postprocess.py:1)：只负责评估、导出、补偿和结果落盘

所以对外命令行入口不变，但内部职责已经分清。

### 选项总览

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `--robot-name` | `None` | 机器人名字标签，主要用于结果记录和日志展示。一般可以不传。 |
| `--identification-mode` | `rigid_body_friction` | 当前辨识模式。现版本只支持“刚体动力学 + 摩擦”联合辨识。 |
| `--parameterization {base,full}` | `base` | 参数化方式。`base` 只保留可辨识基参数，工程上更稳；`full` 保留完整物理参数，更适合物理解释、payload 等效和约束求解。 |
| `--solver-method {ols,wls,ridge,constrained}` | `ridge` | 求解器类型。`ols` 是普通最小二乘，`wls` 是加权最小二乘，`ridge` 是带 L2 正则的稳健版本，`constrained` 是带物理可行性约束的求解器。 |
| `--data-source {auto,synthetic,real}` | `auto` | 数据来源。`auto` 会自动判断优先使用哪类数据；`synthetic` 用模型合成数据；`real` 用真实采集数据。 |
| `--real-data-dir PATH` | `datasets/real/raw` | 真实数据原始 CSV 所在目录。只在 `--data-source real` 时生效。 |
| `--real-torque-source {sensed,sensor}` | `sensed` | 真实数据里使用哪一列扭矩作为辨识目标。`sensed` 对应 `q*_SensedTorque`，`sensor` 对应 `q*_JointSensorTorque`，后者会自动乘以标定比例。 |
| `--sampling-freq FLOAT` | `100.0` | 数据采样频率，单位 Hz。用于时间轴重建、滤波和导数处理。 |
| `--cutoff-hz FLOAT` | `15.0` | 低通滤波截止频率，单位 Hz。主要影响真实数据处理阶段。 |
| `--urdf-path PATH` | `models/05_urdf/urdf/05_urdf_temp.urdf` | 用于构建 Pinocchio 模型的 URDF 路径。 |
| `--config-path PATH` | `None` | 可选 YAML 配置文件路径。用于补充 active joints、base/ee link 等元信息。 |
| `--export-class-name NAME` | `sevendofDynamics` | 导出的 C++ 类名，同时决定生成的 `.h/.cpp` 文件名。 |
| `--gravity VALUE` | 代码默认 `"[9.81, 0, 0]"` | 运行时重力向量，表达在 base 坐标系下。可传预设关键字或逗号分隔三元组。 |
| `--payload-mode {none,lumped_last_link,external_wrench,augmented_link}` | `none` | payload 处理模式。`lumped_last_link` 把 payload 等效到末端连杆参数，`external_wrench` 作为独立外加项，`augmented_link` 当前主要是预留口。 |
| `--payload-mass FLOAT` | `0.0` | payload 质量，单位 kg。大于 0 时才会真正构造 payload 模型。 |
| `--payload-com CX CY CZ` | `None` | payload 质心位置，三维坐标。通常要求表达在 payload 参考连杆坐标系中。 |
| `--payload-reference-link NAME` | `None` | payload 挂接的参考连杆。默认会落到机器人末端执行器连杆。 |
| `--payload-com-frame NAME` | `None` | `payload-com` 所在坐标系名。未显式给出时默认等于 `payload-reference-link`。 |
| `--subtract-known-payload-gravity` | 关闭 | 如果已经知道 payload 质量和质心，则先从测量扭矩里扣除 payload 重力项，再辨识机器人本体参数。 |

### `--gravity` 可用格式

`--gravity` 支持两种写法：

1. 预设关键字

- `upright`：`[0, 0, -9.81]`
- `inverted`：`[0, 0, 9.81]`
- `wall_x` 或 `wall_x_neg`：`[-9.81, 0, 0]`
- `wall_x_pos`：`[9.81, 0, 0]`
- `wall_y` 或 `wall_y_neg`：`[0, -9.81, 0]`
- `wall_y_pos`：`[0, 9.81, 0]`

2. 直接传 base 坐标系下的三维向量

```bash
python run_pipeline.py --gravity "0,0,-9.81"
python run_pipeline.py --gravity "6.93,0,-6.93"
```

说明：

- 这里的重力是“运行时输入”，不是重新辨识条件。
- 只要参数模型是一致的，切换安装方式时只需要改 `--gravity`，不需要重新辨识本体参数。

### 常用运行组合

#### 1. 日常推荐：真实数据 + `ridge + base`

```bash
python run_pipeline.py \
  --data-source real \
  --real-torque-source sensed \
  --parameterization base \
  --solver-method ridge
```

适合：

- 工程上先得到稳定可用的结果
- 先关注 RMSE、泛化误差和 C++ 导出结果

#### 2. 物理一致性优先：真实数据 + `constrained + full`

```bash
python run_pipeline.py \
  --data-source real \
  --real-torque-source sensed \
  --parameterization full \
  --solver-method constrained
```

适合：

- 需要完整物理参数
- 后续要做 payload 等效、物理可行性检查、重力变向复用

#### 3. 指定侧装或倒装重力方向

```bash
python run_pipeline.py --gravity wall_x
python run_pipeline.py --gravity inverted
python run_pipeline.py --gravity "0,0,-9.81"
```

#### 4. 带固定 payload 做辨识前重力扣除

```bash
python run_pipeline.py \
  --data-source real \
  --parameterization full \
  --solver-method constrained \
  --payload-mode lumped_last_link \
  --payload-mass 1.25 \
  --payload-com 0.0 0.0 0.08 \
  --payload-reference-link tool0 \
  --subtract-known-payload-gravity
```

适合：

- 已知 payload 质量与质心
- 目标是辨识机器人本体无负载参数，而不是把 payload 污染进本体参数里

### 结果文件怎么看

`run_pipeline.py` 运行后，通常会输出或更新这些结果：

- `datasets/identified/theta_hat_*.json`
- `datasets/identified/evaluation_*_splits.json`
- `datasets/residual/compensation_result_*.json`
- `output/*.h`
- `output/*.cpp`

### `output` 代码导出时有哪些可选项

当前 `output/*.cpp` / `output/*.h` 的导出由 `run_pipeline.py` 统一触发，最常影响生成结果的是下面这些选项：

| 选项 | 是否影响导出的 `output` 代码 | 说明 |
| --- | --- | --- |
| `--export-class-name NAME` | 是 | 决定导出的 C++ 类名和文件名，例如 `sevendofDynamics.cpp/.h`。 |
| `--gravity VALUE` | 是 | 作为默认重力向量写进导出代码的初始值；运行时仍可通过 `setGravityVector()` 覆盖。 |
| `--parameterization {base,full}` | 是 | 决定被识别并写入导出代码的参数口径。`base` 更稳，`full` 更适合物理解释和 payload 等效。 |
| `--solver-method {ols,wls,ridge,constrained}` | 是 | 会影响最终识别到的参数，从而影响导出代码数值。 |
| `--payload-mode {none,lumped_last_link,external_wrench,augmented_link}` | 是 | 只对导出阶段真正支持 `none` 和 `lumped_last_link`。`external_wrench` 当前主要用于 Python 运行时，不直接导出进 C++。 |
| `--payload-mass FLOAT` + `--payload-com CX CY CZ` | 有条件影响 | 当 `--payload-mode lumped_last_link` 时，会把已知 payload 直接 lump 到末端参数后再导出，生成“默认带 payload”的代码。 |
| `--subtract-known-payload-gravity` | 否 | 这一步只影响辨识前的数据预处理，不会单独在导出代码里生成“扣除逻辑”。 |

关于 payload，需要区分两种使用方式：

- 导出时就已知 payload：
  用 `--payload-mode lumped_last_link --payload-mass --payload-com`，生成的 `output` 代码默认就带这个 payload。
- 导出时不知道 payload，运行时才知道：
  不需要重生成代码。当前生成版 `sevendofDynamics` 已重新支持通过 `calculateEstimateJointToqrues(..., parms, ...)` 这类接口，在运行时通过 RTOS 风格 `parms` 注入 payload 质量和质心。

关于非线性补偿：

- 主流程会训练线性残差补偿和 MLP 补偿，并把评估结果写入 `datasets/residual/compensation_result_*.json`。
- 但当前导出的 `output/*.cpp` 仍然默认是纯物理主模型导出，不包含非线性补偿项。
- 生成文件头部元数据里会看到：

```text
nonlinear_compensation: disabled_for_export
```

- 也就是说，现在可以“评估 MLP 有多大提升”，但不会自动把 MLP 一起固化到 `output` 代码里。

其中重点看：

- `parameterization`
- `solver_method`
- `gravity_config`
- `physical_sanity`
- `train_rmse / val_rmse / test_rmse`
- `possible_overfit`

如果你要直接比较两种辨识口径，不要手工跑两遍后再自己抄表，建议直接用：

```bash
python python/compare_solver_modes.py --data-source real --real-torque-source sensed
```

它会并排比较：

- `ridge + base`
- `constrained + full`

并输出：

- `train/val/test RMSE`
- `physical sanity`
- 每关节质量/惯量可行性
- 导出到 `output` 后的单样本扭矩对比

## 常见测试命令

下面这些命令是现在最常用的“打开方式”和测试口令，建议优先直接复制运行。

### 1. 运行主流程并生成最新 `output` 代码

```bash
conda run -n urdfly python run_pipeline.py --data-source real --export-class-name sevendofDynamics
```

如果你希望导出时就把已知 payload lump 进默认参数：

```bash
conda run -n urdfly python run_pipeline.py \
  --data-source real \
  --parameterization full \
  --solver-method constrained \
  --payload-mode lumped_last_link \
  --payload-mass 1.25 \
  --payload-com 0 0 0.08 \
  --payload-reference-link end_link \
  --export-class-name sevendofDynamics
```

### 2. 检查当前生成代码和 Python 参考实现是否一致

这个测试会重新走一遍参考识别流程，再把生成的 C++ 逐样本和 Python 对比。

```bash
conda run -n pinocchio_py python output/test/test_cpp_consistency.py --export-class-name sevendofDynamics --sample-limit 8
```

### 3. 测最终生成代码，不重生成，默认无 payload 覆盖

这个测试直接编译当前已经存在的 `output/sevendofDynamics.cpp`，验证“最终文件”本身。

```bash
conda run -n pinocchio_py python output/test/test_output.py --export-class-name sevendofDynamics --sample-limit 8
```

### 4. 测最终生成代码，运行时注入 payload 质量和质心

这个测试不重生成代码，而是通过 RTOS 风格 `parms` 在运行时把 payload 注入最终导出的 C++。

```bash
conda run -n pinocchio_py python output/test/test_output.py \
  --export-class-name sevendofDynamics \
  --sample-limit 8 \
  --payload-mass 1.25 \
  --payload-com 0 0 0.08
```

### 5. 只测“运行时 payload 覆盖”这条链路是否正确

如果你只关心“生成代码能不能在运行时接受 payload 输入”，可以直接跑这个更聚焦的测试：

```bash
conda run -n pinocchio_py python output/test/test_cpp_payload_override.py --export-class-name sevendofDynamics
```

### 6. 交互式单样本查询，不带 payload

如果你已经激活环境，推荐直接这样开：

```bash
conda activate pinocchio_py
python output/test/query_output_torque.py --export-class-name sevendofDynamics
```

如果你想继续用 `conda run`，为了保留交互式 stdin，建议：

```bash
conda run --no-capture-output -n pinocchio_py python output/test/query_output_torque.py --export-class-name sevendofDynamics
```

### 7. 交互式或命令行单样本查询，带 payload 运行时覆盖

命令行一次性传完：

```bash
conda run -n pinocchio_py python output/test/query_output_torque.py \
  --export-class-name sevendofDynamics \
  --gravity 9.81 0 0 \
  --q 0 0 0 0 0 0 0 \
  --dq 0 0 0 0 0 0 0 \
  --ddq 0 0 0 0 0 0 0 \
  --payload-mass 1.25 \
  --payload-com 0 0 0.08
```

或者进入交互模式后，在脚本提示里输入 payload 质量和质心。

### 8. 检查 payload 和重力运行时扩展是否正常

```bash
conda run -n urdfly python output/test/test_runtime_extensions.py
```

这个脚本会检查：

- gravity preset 是否正确
- payload 点质量转 10 维刚体参数是否符合平行轴定理
- `lumped_last_link` 和 `external_wrench` 在准静态条件下是否一致

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
