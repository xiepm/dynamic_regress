from __future__ import annotations

import json
import os
from pathlib import Path

# Matplotlib may try to write caches under the user's home directory. In this
# workspace that can be read-only, so keep the notebook quiet and reproducible.
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-dynamic-regress')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/dynamic-regress-cache')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


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

    return {
        'json': json,
        'Path': Path,
        'np': np,
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
    }
