"""
Post-identification helpers for the dynamics pipeline.

This module keeps evaluation, serialization, export, compensation, and summary
logic separate from the core identification stage.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from export_inverse_dynamics_code import export_identified_inverse_dynamics_cpp
from residual_compensation import MLPCompensator, ResidualAnalyzer, ResidualCompensator
from runtime_dynamics import GravityConfig, PayloadMode
import pipeline_identification as pipeline_id


def np_to_list(value):
    """Convert numpy arrays/scalars into JSON-friendly Python lists."""
    return np.asarray(value, dtype=float).tolist()


def _joint_metric_array(metric_map: dict[str, float]) -> list[float]:
    ordered_items = sorted(metric_map.items(), key=lambda item: int(item[0].split('_')[-1]))
    return [float(value) for _, value in ordered_items]


def write_visualization_payload(
    *,
    project_root: Path,
    robot_model,
    effective_data_source: str,
    result_stem: str,
    split_evaluation: dict[str, dict],
    compensation_eval: dict[str, dict],
    mlp_eval: dict[str, dict],
    stability_eval: dict,
) -> Path:
    """Write the notebook-facing visualization payload."""
    vis_dir = project_root / "datasets" / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = vis_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    identification_test = split_evaluation['test']
    linear_test = compensation_eval['test']
    mlp_test = mlp_eval['test']
    joint_count = len(identification_test['joint_rmse'])
    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'robot_name': robot_model.name,
        'data_source': effective_data_source,
        'result_stem': result_stem,
        'joints': [f'J{joint_idx}' for joint_idx in range(1, joint_count + 1)],
        'stages': ['Identification', '+ Linear comp.', '+ MLP comp.'],
        'paths': {
            'figure_dir': str(figure_dir),
            'latest_results_file': str(vis_dir / "latest_results.json"),
            'snapshot_results_file': str(vis_dir / f"results_{result_stem}.json"),
        },
        'identification': {
            'test': {
                'global_rmse': float(identification_test['global_rmse']),
                'global_mae': float(identification_test['global_mae']),
                'joint_rmse': _joint_metric_array(identification_test['joint_rmse']),
                'joint_mae': _joint_metric_array(identification_test['joint_mae']),
            },
        },
        'linear': {
            'test': {
                'global_rmse': float(linear_test['rmse']),
                'global_mae': float(linear_test['mae']),
                'improvement_percent': float(linear_test['improvement_percent']),
                'joint_rmse': _joint_metric_array(linear_test['joint_rmse']),
                'joint_mae': _joint_metric_array(linear_test['joint_mae']),
                'joint_improvement_percent': _joint_metric_array(linear_test['joint_improvement_percent']),
            },
        },
        'mlp': {
            'test': {
                'global_rmse': float(mlp_test['rmse']),
                'global_mae': float(mlp_test['mae']),
                'improvement_percent': float(mlp_test['improvement_percent']),
                'joint_rmse': _joint_metric_array(mlp_test['joint_rmse']),
                'joint_mae': _joint_metric_array(mlp_test['joint_mae']),
                'joint_improvement_percent': _joint_metric_array(mlp_test['joint_improvement_percent']),
            },
        },
        'stability': {
            'splits': ['Train', 'Val', 'Test'],
            'rmse_mean': [
                float(stability_eval['summary']['train']['mean']),
                float(stability_eval['summary']['val']['mean']),
                float(stability_eval['summary']['test']['mean']),
            ],
            'rmse_std': [
                float(stability_eval['summary']['train']['std']),
                float(stability_eval['summary']['val']['std']),
                float(stability_eval['summary']['test']['std']),
            ],
        },
        'global_pipeline': {
            'rmse': [
                float(identification_test['global_rmse']),
                float(linear_test['rmse']),
                float(mlp_test['rmse']),
            ],
            'mae': [
                float(identification_test['global_mae']),
                float(linear_test['mae']),
                float(mlp_test['mae']),
            ],
            'improvement_percent': [
                float(linear_test['improvement_percent']),
                float(mlp_test['improvement_percent']),
            ],
        },
    }

    snapshot_path = vis_dir / f"results_{result_stem}.json"
    with open(snapshot_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    output_path = vis_dir / "latest_results.json"
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(f"[Diag] Visualization payload updated: {output_path}")
    print(f"[Diag] Visualization timestamped snapshot saved: {snapshot_path}")
    return output_path


def evaluate_residual_compensation_by_split(
    analyzer,
    compensator,
    identifier,
    result: dict,
    splits: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """Evaluate the linear residual compensator on train / val / test."""
    residual_splits = {}
    for split_name, split_df in splits.items():
        residual_df = analyzer.compute_residuals(split_df, identifier, result)
        residual_splits[split_name] = analyzer.feature_engineering(residual_df)

    train_info = compensator.train_with_cross_validation(residual_splits['train'])
    return {
        'train_fit': train_info,
        'train': compensator.evaluate_compensator(residual_splits['train']),
        'val': compensator.evaluate_compensator(residual_splits['val']),
        'test': compensator.evaluate_compensator(residual_splits['test']),
    }


def run_stability_evaluation(
    *,
    df: pd.DataFrame,
    identifier,
    result: dict,
    split_dataframe_fn: Callable[..., dict[str, pd.DataFrame]],
    n_seeds: int = 5,
) -> dict:
    """Repeat train/val/test splitting across random seeds to assess stability."""
    seeds = [123, 42, 7, 2024, 999][:n_seeds]
    per_seed = []
    for seed in seeds:
        splits = split_dataframe_fn(df, seed=seed)
        seed_result = identifier.identify_parameters(
            splits['train'],
            method=result['method'].lower(),
            reference_parameters=result.get('reference_parameters'),
            ridge_lambda=result.get('ridge_lambda', 0.0) or 1e-4,
        )
        split_eval = pipeline_id.evaluate_identification_splits(identifier, seed_result, splits)
        per_seed.append({
            'seed': seed,
            'train_rmse': split_eval['train']['global_rmse'],
            'val_rmse': split_eval['val']['global_rmse'],
            'test_rmse': split_eval['test']['global_rmse'],
        })

    metrics = {}
    for split_name in ('train', 'val', 'test'):
        values = np.array([entry[f'{split_name}_rmse'] for entry in per_seed], dtype=float)
        metrics[split_name] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}

    print(f"\n[Diag] Stability evaluation ({len(seeds)} seeds):")
    print(f"[Diag]   Train RMSE: mean={metrics['train']['mean']:.3f}, std={metrics['train']['std']:.3f}")
    print(f"[Diag]   Val   RMSE: mean={metrics['val']['mean']:.3f}, std={metrics['val']['std']:.3f}")
    print(f"[Diag]   Test  RMSE: mean={metrics['test']['mean']:.3f}, std={metrics['test']['std']:.3f}")
    return {'seeds': seeds, 'per_seed': per_seed, 'summary': metrics}


def write_identification_artifacts(
    *,
    identified_dir: Path,
    result_stem: str,
    result_stem_base: str,
    robot_model,
    identification_bundle: dict,
    split_evaluation: dict[str, dict],
    overfitting_summary: dict,
) -> None:
    """Persist identification parameters and split-evaluation artifacts."""
    result = identification_bundle['result']
    context = identification_bundle['context']
    theta_payload = {
        'robot_name': robot_model.name,
        'identification_mode': context['identification_mode'],
        'parameterization': context['parameterization'],
        'data_source': context['data_source'],
        'real_torque_source': context['real_torque_source'],
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'rank': result['rank'],
        'full_regressor_rank': result['full_regressor_rank'],
        'base_parameter_count': result['base_parameter_count'],
        'condition_number': result['condition_number'],
        'active_condition_number': result['active_condition_number'],
        'active_parameter_indices': result['active_parameter_indices'].tolist(),
        'base_column_indices': result['base_column_indices'].tolist(),
        'base_transform_full_from_beta': np_to_list(result['base_transform_full_from_beta']),
        'theta_hat': np_to_list(result['theta_hat']),
        'theta_hat_full': np_to_list(result['theta_hat_full']),
        'pi_full_hat': np_to_list(result['pi_full_hat']),
        'beta_hat': np_to_list(result['beta_hat']),
        'parameter_vector': result['parameter_vector'].to_dict(),
        'gravity_config': result['gravity_config'],
        'physical_sanity': result['physical_sanity'],
        'reference_parameters': None if result['reference_parameters'] is None else np_to_list(result['reference_parameters']),
        'method': result['method'],
        'ridge_lambda': result['ridge_lambda'],
        'optimizer_success': result['optimizer_success'],
        'optimizer_status': result['optimizer_status'],
        'optimizer_iterations': result['optimizer_iterations'],
        'payload_mode': context['payload_mode'],
        'payload_model': context['payload_model'],
        'subtract_known_payload_gravity': context['subtract_known_payload_gravity'],
    }
    with open(identified_dir / f"theta_hat_{result_stem}.json", 'w', encoding='utf-8') as handle:
        json.dump(theta_payload, handle, indent=2)
    with open(identified_dir / f"theta_hat_{result_stem_base}_latest.json", 'w', encoding='utf-8') as handle:
        json.dump(theta_payload, handle, indent=2)

    evaluation_payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'split_metrics': split_evaluation,
        'overfitting_summary': overfitting_summary,
    }
    with open(identified_dir / f"evaluation_{result_stem}_splits.json", 'w', encoding='utf-8') as handle:
        json.dump(evaluation_payload, handle, indent=2)
    with open(identified_dir / f"evaluation_{result_stem_base}_latest_splits.json", 'w', encoding='utf-8') as handle:
        json.dump(evaluation_payload, handle, indent=2)


def run_post_identification_stage(
    *,
    project_root: Path,
    robot_model,
    df_processed: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    identification_bundle: dict,
    effective_data_source: str,
    real_torque_source: str,
    identification_mode: str,
    effective_gravity,
    effective_payload_mode: PayloadMode,
    payload_model,
    resolved_urdf_path: Path,
    parameterization: str,
    export_class_name: str,
    split_dataframe_fn: Callable[..., dict[str, pd.DataFrame]],
) -> dict:
    """Run evaluation/export/compensation/visualization consumers of one identified result."""
    identifier = identification_bundle['identifier']
    result = identification_bundle['result']

    split_evaluation = pipeline_id.evaluate_identification_splits(identifier, result, splits)
    overfitting_summary = pipeline_id.summarize_overfitting(split_evaluation)
    pipeline_id.print_per_joint_summary(
        "Identification per-joint metrics (test split):",
        split_evaluation['test'],
        ['joint_rmse', 'joint_mae'],
    )

    identified_dir = project_root / "datasets" / "identified"
    identified_dir.mkdir(parents=True, exist_ok=True)
    residual_dir = project_root / "datasets" / "residual"
    residual_dir.mkdir(parents=True, exist_ok=True)
    result_stem_base = f"real_{real_torque_source}" if effective_data_source == 'real' else 'synthetic'
    result_stem = result_stem_base

    write_identification_artifacts(
        identified_dir=identified_dir,
        result_stem=result_stem,
        result_stem_base=result_stem_base,
        robot_model=robot_model,
        identification_bundle=identification_bundle,
        split_evaluation=split_evaluation,
        overfitting_summary=overfitting_summary,
    )

    print("\n[STEP 7-8] Residual analysis and compensation...")
    print("-" * 78)
    analyzer = ResidualAnalyzer(robot_model)
    compensator = ResidualCompensator(num_joints=robot_model.num_joints)
    compensation_eval = evaluate_residual_compensation_by_split(analyzer, compensator, identifier, result, splits)
    mlp_compensator = MLPCompensator(num_joints=robot_model.num_joints)
    residual_splits = {}
    for split_name, split_df in splits.items():
        residual_df = analyzer.compute_residuals(split_df, identifier, result)
        residual_splits[split_name] = analyzer.feature_engineering(residual_df)
    mlp_train_info = mlp_compensator.train(residual_splits['train'])
    mlp_eval = {
        'train_fit': mlp_train_info,
        'train': mlp_compensator.evaluate(residual_splits['train']),
        'val': mlp_compensator.evaluate(residual_splits['val']),
        'test': mlp_compensator.evaluate(residual_splits['test']),
    }
    print("\nCompensator comparison (test set):")
    print(f"  Linear    MAE: {compensation_eval['test']['mae']:.3f} N·m  Improvement: {compensation_eval['test']['improvement_percent']:.1f}%")
    print(f"  MLP       MAE: {mlp_eval['test']['mae']:.3f} N·m  Improvement: {mlp_eval['test']['improvement_percent']:.1f}%")
    pipeline_id.print_per_joint_summary(
        "Linear compensator per-joint metrics (test split):",
        compensation_eval['test'],
        ['joint_rmse', 'joint_mae', 'joint_improvement_percent'],
    )
    pipeline_id.print_per_joint_summary(
        "MLP compensator per-joint metrics (test split):",
        mlp_eval['test'],
        ['joint_rmse', 'joint_mae', 'joint_improvement_percent'],
    )
    if mlp_eval['test']['improvement_percent'] - compensation_eval['test']['improvement_percent'] > 15.0:
        print("Suggestion: nonlinear compensation shows significant benefit, consider a deeper/wider MLP")

    generated_dynamics_paths = export_identified_inverse_dynamics_cpp(
        robot_model=robot_model,
        result=result,
        urdf_path=resolved_urdf_path,
        output_dir=project_root / "output",
        result_stem=result_stem,
        class_name=export_class_name,
        parameter_vector=result['parameter_vector'],
        gravity_config=GravityConfig.from_any(effective_gravity),
        payload_model=payload_model,
        payload_mode=effective_payload_mode,
        generation_metadata={
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'robot_name': robot_model.name,
            'data_source': effective_data_source,
            'real_torque_source': real_torque_source if effective_data_source == 'real' else None,
            'parameterization': parameterization,
            'identification_mode': identification_mode,
            'payload_mode': effective_payload_mode.value,
            'nonlinear_compensation': 'disabled_for_export',
        },
    )
    print("Generated identified inverse-dynamics code with explicit gravity decomposition:")
    for label, path in generated_dynamics_paths.items():
        print(f"  {label}: {path}")

    compensation_payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'linear': compensation_eval,
        'mlp': mlp_eval,
    }
    with open(residual_dir / f"compensation_result_{result_stem}.json", 'w', encoding='utf-8') as handle:
        json.dump(compensation_payload, handle, indent=2)
    with open(residual_dir / f"compensation_result_{result_stem_base}_latest.json", 'w', encoding='utf-8') as handle:
        json.dump(compensation_payload, handle, indent=2)

    stability_eval = run_stability_evaluation(
        df=df_processed,
        identifier=identifier,
        result=result,
        split_dataframe_fn=split_dataframe_fn,
        n_seeds=5,
    )
    with open(identified_dir / f"stability_eval_{result_stem}.json", 'w', encoding='utf-8') as handle:
        json.dump(stability_eval, handle, indent=2)
    with open(identified_dir / "stability_eval_latest.json", 'w', encoding='utf-8') as handle:
        json.dump(stability_eval, handle, indent=2)
    write_visualization_payload(
        project_root=project_root,
        robot_model=robot_model,
        effective_data_source=effective_data_source,
        result_stem=result_stem,
        split_evaluation=split_evaluation,
        compensation_eval=compensation_eval,
        mlp_eval=mlp_eval,
        stability_eval=stability_eval,
    )
    return {
        'split_evaluation': split_evaluation,
        'overfitting_summary': overfitting_summary,
        'compensation_eval': compensation_eval,
        'mlp_eval': mlp_eval,
        'stability_eval': stability_eval,
        'generated_dynamics_paths': generated_dynamics_paths,
        'result_stem': result_stem,
        'result_stem_base': result_stem_base,
    }


def print_pipeline_summary(
    *,
    robot_model,
    df_processed: pd.DataFrame,
    result: dict,
    post_bundle: dict,
) -> None:
    """Print the compact end-of-run pipeline summary."""
    split_evaluation = post_bundle['split_evaluation']
    overfitting_summary = post_bundle['overfitting_summary']
    compensation_eval = post_bundle['compensation_eval']
    mlp_eval = post_bundle['mlp_eval']
    generated_dynamics_paths = post_bundle['generated_dynamics_paths']

    print("\n" + "=" * 78)
    print(" PIPELINE COMPLETED")
    print("=" * 78)
    print(f"  Robot: {robot_model.name}")
    print(f"  Samples processed: {len(df_processed)}")
    print(f"  Full parameters: {result['num_parameters_full']}")
    print(f"  Active parameters: {result['num_parameters_active']}")
    print(f"  Rank: {result['full_regressor_rank']}")
    print(f"  Condition number: {result['condition_number']:.2e}")
    print(f"  Active condition number: {result['active_condition_number']:.2e}")
    print(f"  Train RMSE: {split_evaluation['train']['global_rmse']:.6f} N·m")
    print(f"  Val RMSE:   {split_evaluation['val']['global_rmse']:.6f} N·m")
    print(f"  Test RMSE:  {split_evaluation['test']['global_rmse']:.6f} N·m")
    print(f"  Possible overfit: {overfitting_summary['possible_overfit']}")
    print(f"  Linear compensation test improvement: {compensation_eval['test']['improvement_percent']:.2f}%")
    print(f"  MLP compensation test improvement: {mlp_eval['test']['improvement_percent']:.2f}%")
    print(f"  Generated dynamics code: {generated_dynamics_paths['project_cpp']}")
    print("=" * 78 + "\n")
