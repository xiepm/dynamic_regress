# 快速上手

这份文档按“**从 `git clone` 下来，到跑通真实数据辨识和可视化**”的顺序来写。  
如果你是第一次接手这个项目，建议就按下面命令一步一步照着执行。

## 1. clone 项目

```bash
git clone <你的仓库地址>
cd dynamic_regress
```

如果你已经在本地有这个目录，直接进入项目根目录即可：

```bash
cd /home/xpm/projects/dynamic7dof/dynamic_regress
```

## 2. 准备运行环境

推荐直接使用 `conda`。

### 方式 A：用 `environment.yml`

```bash
conda env create -f environment.yml
conda activate panda_id
```

### 方式 B：手动创建环境

```bash
conda create -n panda_id python=3.11 -y
conda activate panda_id
conda install -c conda-forge numpy pandas scipy pyyaml pinocchio -y
pip install scikit-learn matplotlib jupyter
```

### 如果你沿用当前项目已经验证过的环境

当前这套工程已经在 `urdfly` 环境里实际跑通过，所以也可以直接用：

```bash
conda activate urdfly
```

先确认解释器是不是你想用的环境：

```bash
which python
python -c "import sys; print(sys.executable)"
```

不要用系统自带的：

```bash
/usr/bin/python3
```

否则容易出现 `ModuleNotFoundError: No module named 'pandas'` 这类问题。

## 3. 项目里有哪些核心脚本

主流程入口是：

- [run_pipeline.py](/home/xpm/projects/dynamic7dof/dynamic_regress/run_pipeline.py)

主要模块有：

- [python/load_model.py](/home/xpm/projects/dynamic7dof/dynamic_regress/python/load_model.py)
- [python/process_measured_data.py](/home/xpm/projects/dynamic7dof/dynamic_regress/python/process_measured_data.py)
- [python/identify_parameters.py](/home/xpm/projects/dynamic7dof/dynamic_regress/python/identify_parameters.py)
- [python/residual_compensation.py](/home/xpm/projects/dynamic7dof/dynamic_regress/python/residual_compensation.py)
- [plot_identification_results.ipynb](/home/xpm/projects/dynamic7dof/dynamic_regress/plot_identification_results.ipynb)

你可以把它理解成：

1. `run_pipeline.py` 负责串起整条流程
2. `process_measured_data.py` 负责真实数据清洗和预处理
3. `identify_parameters.py` 负责动力学参数辨识
4. `residual_compensation.py` 负责线性补偿和 MLP 非线性补偿
5. `plot_identification_results.ipynb` 负责读最新结果并画图

## 4. 模型文件放哪里

当前默认使用的 URDF 是：

- [models/05_urdf/urdf/05_urdf_temp.urdf](/home/xpm/projects/dynamic7dof/dynamic_regress/models/05_urdf/urdf/05_urdf_temp.urdf)

如果你不改命令，主程序会默认加载它。

如果以后换了新的 URDF，可以显式指定：

```bash
python run_pipeline.py --urdf-path 你的urdf文件路径
```

如果还有额外 yaml 配置，也可以一起传：

```bash
python run_pipeline.py --urdf-path 你的urdf文件路径 --config-path 你的yaml路径
```

## 5. 真实数据应该放在哪里

真实数据统一放到：

- [datasets/real/raw](/home/xpm/projects/dynamic7dof/dynamic_regress/datasets/real/raw)

当前目录结构是：

```text
datasets/
├── synthetic/
│   ├── raw/
│   └── processed/
├── real/
│   ├── raw/
│   ├── normalized/
│   └── processed/
├── identified/
├── residual/
└── visualization/
```

这里要注意：

- `synthetic` 和 `real` 是两类并列的数据来源
- `raw / normalized / processed` 只是处理阶段
- 多个真实 CSV 可以一起放进 `datasets/real/raw/`
- 主程序会自动批量读取并合并

## 6. 真实数据格式和单位要求

当前真实 CSV 约定列名形式类似：

```text
timestamp
q1_pos, q1_vel, q1_acc, q1_sensedTorque, q1_cur
q2_pos, q2_vel, q2_acc, q2_sensedTorque, q2_cur
...
q7_pos, q7_vel, q7_acc, q7_sensedTorque, q7_cur
```

当前默认单位约定是：

- 位置：`deg`
- 速度：`deg/s`
- 加速度：`deg/s^2`
- 扭矩：`N·m`

进入辨识主链前，程序会自动转换成：

- `q -> rad`
- `dq -> rad/s`
- `ddq -> rad/s^2`
- `tau` 保持 `N·m`

也就是说，只要你的真实数据格式和这个约定一致，直接放进 `datasets/real/raw/` 就可以。

## 7. 最常用的运行命令

### 7.1 直接跑默认流程

```bash
python run_pipeline.py
```

这会使用默认 URDF；如果 `datasets/real/raw/` 里有真实 CSV，`auto` 模式会优先走真实数据。

### 7.2 明确指定跑真实数据

```bash
python run_pipeline.py --data-source real
```

### 7.3 明确指定真实数据 + 指定 URDF

```bash
python run_pipeline.py --data-source real --urdf-path models/05_urdf/urdf/05_urdf_temp.urdf
```

### 7.4 如果你不想先激活环境

```bash
conda run -n urdfly python run_pipeline.py --data-source real
```

## 8. 跑完后会生成什么

比较关键的输出有：

- [datasets/real/normalized](/home/xpm/projects/dynamic7dof/dynamic_regress/datasets/real/normalized)
- [datasets/real/processed](/home/xpm/projects/dynamic7dof/dynamic_regress/datasets/real/processed)
- [datasets/identified/theta_hat_real.json](/home/xpm/projects/dynamic7dof/dynamic_regress/datasets/identified/theta_hat_real.json)
- [datasets/identified/evaluation_real_splits.json](/home/xpm/projects/dynamic7dof/dynamic_regress/datasets/identified/evaluation_real_splits.json)
- [datasets/identified/stability_eval.json](/home/xpm/projects/dynamic7dof/dynamic_regress/datasets/identified/stability_eval.json)
- [datasets/residual/compensation_result_real.json](/home/xpm/projects/dynamic7dof/dynamic_regress/datasets/residual/compensation_result_real.json)
- [datasets/visualization/latest_results.json](/home/xpm/projects/dynamic7dof/dynamic_regress/datasets/visualization/latest_results.json)

其中最重要的一点是：

- `datasets/visualization/latest_results.json` 是给可视化 notebook 用的“最新结果出口”
- 每次重新跑 `run_pipeline.py`，它都会被**新结果覆盖**

## 9. 可视化应该怎么操作

可视化 notebook 是：

- [plot_identification_results.ipynb](/home/xpm/projects/dynamic7dof/dynamic_regress/plot_identification_results.ipynb)

推荐顺序：

1. 先跑主程序，生成最新结果
2. 再打开 notebook
3. 执行 `Run All`

最常用命令是：

```bash
python run_pipeline.py --data-source real
```

然后启动 notebook：

```bash
jupyter notebook
```

打开 `plot_identification_results.ipynb` 后：

- 第一格会显示当前结果摘要
- 包括结果时间、机器人模型、数据来源、读取的 JSON 路径、是否保存图片
- 后面的图都会直接读取 `datasets/visualization/latest_results.json`

### 一个非常重要的点

notebook **不会自动实时监听 JSON 文件变化**。  
所以如果你重新跑了一次主程序，想看到新图，必须：

- 重新执行 notebook 的单元
- 最稳妥的方法是直接 `Run All`

不过现在 notebook 已经做了增强：

- 即使你只点某一个图的单元，它也会自动补加载上下文
- 不一定非得从第一格开始执行

## 10. 图片默认保存吗

默认**不保存图片**，只在 notebook 里显示。

如果你想把图保存下来，需要在 notebook 第一格里把：

```python
SAVE_FIGURES = False
```

改成：

```python
SAVE_FIGURES = True
```

这样图片会统一覆盖保存到：

- [datasets/visualization/figures](/home/xpm/projects/dynamic7dof/dynamic_regress/datasets/visualization/figures)

也就是说：

- `False`：只显示，不落盘
- `True`：每次新结果覆盖旧图片，不会越存越乱

## 11. 一套建议你直接照抄的完整流程

如果你现在就是要跑真实数据并看图，可以直接按这个顺序：

### 第一步：进入项目并启用环境

```bash
cd /home/xpm/projects/dynamic7dof/dynamic_regress
conda activate urdfly
```

### 第二步：把真实数据放进目录

把你的 CSV 放到：

```text
datasets/real/raw/
```

### 第三步：运行主程序

```bash
python run_pipeline.py --data-source real
```

### 第四步：启动可视化

```bash
jupyter notebook
```

打开：

```text
plot_identification_results.ipynb
```

然后点击：

```text
Run All
```

### 第五步：如果你又重新跑了一次主程序

比如你改了模型、换了数据、调了参数，再执行：

```bash
python run_pipeline.py --data-source real
```

之后回到 notebook：

- 再执行一次 `Run All`
- 它就会读入最新覆盖后的 `datasets/visualization/latest_results.json`

## 12. 常见问题

### 12.1 报错 `ModuleNotFoundError: No module named 'pandas'`

通常是因为你用了系统 Python，而不是 conda 环境。

检查：

```bash
which python
python -c "import sys; print(sys.executable)"
```

### 12.2 notebook 打开后图还是旧的

通常不是数据流断了，而是：

- 你没有重新执行 notebook 单元
- Jupyter 还在显示上一次缓存的图像输出

解决方法：

- 重新执行第一格
- 或者直接 `Run All`

### 12.3 只点某一张图时报 `plt is not defined`

这个问题现在已经做了兼容；如果还遇到，通常说明 notebook 内核状态异常。  
最稳妥的方法仍然是：

- `Kernel -> Restart Kernel`
- 然后 `Run All`

## 13. 你最该关注的几个结果

建议优先看：

- `train / val / test RMSE`
- 每个关节各自的 `RMSE / MAE`
- `Linear` 和 `MLP` 的补偿改善率
- `stability_eval.json` 里的多随机种子稳定性

如果是工程上要快速判断结果好不好，最直接就是看：

1. `test` 是否明显比 `train` 差很多
2. 哪个关节误差最大
3. `MLP` 是否比线性补偿显著更好




验证：python3 output/test/query_output_torque.py --export-class-name sevendofDynamics --show-physical
