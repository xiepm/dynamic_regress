# Quick Start

## 1. Create the environment

Recommended:

```bash
conda env create -f environment.yml
conda activate panda_id
```

If you prefer to install manually:

```bash
conda create -n panda_id python=3.11 -y
conda activate panda_id
conda install -c conda-forge numpy pandas scipy pyyaml pinocchio -y
```

## 2. Run the pipeline

```bash
python run_pipeline.py
```

Recommended mode:

```bash
python run_pipeline.py --parameterization base
```

Comparison mode:

```bash
python run_pipeline.py --parameterization full
```

## 3. Windows + PyCharm

Best practice on Windows:

1. Create the conda environment in `Anaconda Prompt`
2. Open the project in PyCharm
3. Set interpreter:

```text
File -> Settings -> Project -> Python Interpreter
Add Interpreter -> Conda -> Existing environment
```

4. Point PyCharm to:

```text
C:\Users\你的用户名\miniconda3\envs\panda_id\python.exe
```

Do not install `pinocchio` one package at a time from the PyCharm UI if you can avoid it. Using the prebuilt conda environment is much more reliable.

## 4. Main outputs

After a successful run, check:

```text
datasets/golden/
datasets/measured/raw/
datasets/measured/processed/
datasets/identified/
datasets/residual/
```

Key files:

- `datasets/identified/theta_hat_synthetic.json`
- `datasets/identified/evaluation_synthetic.json`
- `datasets/residual/compensation_result.json`

## 5. Typical interpretation

- `base` is the recommended physically meaningful result
- `full` often runs but is numerically ill-conditioned
- focus on `rank`, `active_condition_number`, and `global_rmse`
