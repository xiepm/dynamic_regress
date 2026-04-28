from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Matplotlib may try to write caches under the user's home directory. In this
# workspace that can be read-only, so keep the notebook quiet and reproducible.
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-dynamic-regress')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/dynamic-regress-cache')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_DIR = PROJECT_ROOT / 'python'
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from identify_parameters import ParameterIdentifier
from load_model import URDFLoader


def _load_raw_prediction_context(payload: dict) -> dict:
    """
    加载 raw 全量数据，并用最新辨识参数回放预测力矩。

    目前主要面向真实数据场景：读取 `datasets/real/normalized/*_normalized.csv`，
    拼接成完整时序后，再使用最新的 identified parameters 计算每个关节的预测力矩。
    """
    data_source = payload.get('data_source')
    if data_source != 'real':
        return {
            'RAW_AVAILABLE': False,
            'RAW_REASON': f"raw visualization currently only supports real data, got data_source={data_source!r}.",
        }

    result_stem = payload.get('result_stem', 'real_sensed')
    identified_path = PROJECT_ROOT / 'datasets' / 'identified' / f'theta_hat_{result_stem}_latest.json'
    if not identified_path.exists():
        identified_path = PROJECT_ROOT / 'datasets' / 'identified' / f'theta_hat_{result_stem}.json'
    if not identified_path.exists():
        return {
            'RAW_AVAILABLE': False,
            'RAW_REASON': f'identified parameter file not found: {identified_path}',
        }

    normalized_dir = PROJECT_ROOT / 'datasets' / 'real' / 'normalized'
    normalized_paths = sorted(normalized_dir.glob('*_normalized.csv'))
    if not normalized_paths:
        return {
            'RAW_AVAILABLE': False,
            'RAW_REASON': f'no normalized raw files found under {normalized_dir}',
        }

    frames = []
    segment_ranges = []
    cursor = 0
    for path in normalized_paths:
        frame = pd.read_csv(path)
        frame = frame.copy()
        frame['source_file'] = path.name
        frame['raw_sample_index'] = np.arange(len(frame), dtype=int)
        frames.append(frame)
        next_cursor = cursor + len(frame)
        segment_ranges.append({
            'file': path.name,
            'start': int(cursor),
            'end': int(next_cursor),
        })
        cursor = next_cursor
    raw_df = pd.concat(frames, ignore_index=True)

    urdf_path = PROJECT_ROOT / 'models' / '05_urdf' / 'urdf' / '05_urdf_temp.urdf'
    gravity_vector = [9.81, 0.0, 0.0]
    try:
        robot_model = URDFLoader(str(urdf_path), gravity_vector=gravity_vector).build_robot_model()
    except Exception as exc:
        return {
            'RAW_AVAILABLE': False,
            'RAW_REASON': f'failed to build robot model for raw replay: {exc}',
        }

    theta_payload = json.loads(identified_path.read_text(encoding='utf-8'))
    parameterization = theta_payload.get('parameterization', 'base')
    identifier = ParameterIdentifier(robot_model, parameterization=parameterization)
    result = {
        'theta_hat': np.asarray(theta_payload['theta_hat'], dtype=float),
        'active_parameter_indices': np.asarray(theta_payload['active_parameter_indices'], dtype=int),
    }
    tau_pred = identifier.predict_torques(raw_df, result)
    tau_meas = np.column_stack([raw_df[f'tau_{joint_idx}'].to_numpy() for joint_idx in range(1, robot_model.num_joints + 1)])
    tau_err = tau_meas - tau_pred

    return {
        'RAW_AVAILABLE': True,
        'RAW_REASON': None,
        'RAW_DF': raw_df,
        'RAW_SAMPLE_INDEX': np.arange(len(raw_df), dtype=int),
        'RAW_SOURCE_FILES': [item['file'] for item in segment_ranges],
        'RAW_SEGMENT_RANGES': segment_ranges,
        'RAW_TAU_MEAS': tau_meas,
        'RAW_TAU_PRED': tau_pred,
        'RAW_TAU_ERR': tau_err,
    }


def load_plot_context(save_figures: bool = False) -> dict:
    """Load the latest pipeline results and return notebook plotting globals."""
    plt.rcParams.update({
        'figure.dpi': 120,
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
    })

    vis_file = Path('datasets') / 'visualization' / 'latest_results.json'
    if not vis_file.exists():
        raise FileNotFoundError(
            f'未找到最新可视化结果文件: {vis_file}。请先运行 run_pipeline.py 生成最新结果。'
        )

    payload = json.loads(vis_file.read_text(encoding='utf-8'))
    figure_dir = Path(payload['paths']['figure_dir'])
    if save_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    original_savefig = plt.savefig

    def savefig_latest(filename, *args, **kwargs):
        if not save_figures:
            return None
        target = figure_dir / Path(filename).name
        return original_savefig(target, *args, **kwargs)

    plt.savefig = savefig_latest

    raw_context = _load_raw_prediction_context(payload)

    return {
        'json': json,
        'Path': Path,
        'np': np,
        'pd': pd,
        'plt': plt,
        'mpatches': mpatches,
        'GridSpec': GridSpec,
        'VIS_FILE': vis_file,
        'SAVE_FIGURES': save_figures,
        'payload': payload,
        'FIGURE_DIR': figure_dir,
        'GENERATED_AT': payload.get('generated_at', 'unknown'),
        'RUN_TIMESTAMP': payload.get('run_timestamp', 'unknown'),
        'RESULT_STEM': payload.get('result_stem', 'unknown'),
        'ROBOT_NAME': payload.get('robot_name', 'unknown'),
        'DATA_SOURCE': payload.get('data_source', 'unknown'),
        'C_BLUE': '#3888DC',
        'C_ORANGE': '#EE872E',
        'C_GREEN': '#33A12B',
        'C_RED': '#D72626',
        'C_GRAY': '#8C8C8C',
        'C_PURPLE': '#9466BC',
        'JOINTS': payload['joints'],
        'N_JOINTS': len(payload['joints']),
        'STAGES': payload['stages'],
        'rmse_ident': np.array(payload['identification']['test']['joint_rmse']),
        'rmse_linear': np.array(payload['linear']['test']['joint_rmse']),
        'rmse_mlp': np.array(payload['mlp']['test']['joint_rmse']),
        'mae_ident': np.array(payload['identification']['test']['joint_mae']),
        'mae_linear': np.array(payload['linear']['test']['joint_mae']),
        'mae_mlp': np.array(payload['mlp']['test']['joint_mae']),
        'impr_linear': np.array(payload['linear']['test']['joint_improvement_percent']),
        'impr_mlp': np.array(payload['mlp']['test']['joint_improvement_percent']),
        'rmse_global': np.array(payload['global_pipeline']['rmse']),
        'mae_global': np.array(payload['global_pipeline']['mae']),
        'global_improvements': np.array(payload['global_pipeline']['improvement_percent']),
        'stability_rmse_mean': np.array(payload['stability']['rmse_mean']),
        'stability_rmse_std': np.array(payload['stability']['rmse_std']),
        **raw_context,
    }
