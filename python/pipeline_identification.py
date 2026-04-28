"""
Identification-stage helpers for the dynamics pipeline.

This module contains the pieces that belong to parameter identification itself,
separate from result serialization, compensation, export, and visualization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from identify_parameters import ParameterIdentifier
from runtime_dynamics import PayloadMode


def evaluate_identification_splits(
    identifier,
    result: dict,
    splits: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """Evaluate one identified parameter set on train / val / test splits."""
    split_metrics = {}
    for split_name, split_df in splits.items():
        print(f"\nEvaluating identification on {split_name} split...")
        split_metrics[split_name] = identifier.evaluate_identification(split_df, result)
    return split_metrics


def summarize_overfitting(split_metrics: dict[str, dict]) -> dict[str, float | bool]:
    """Produce a lightweight overfitting heuristic from split RMSEs."""
    train_rmse = split_metrics['train']['global_rmse']
    val_rmse = split_metrics['val']['global_rmse']
    test_rmse = split_metrics['test']['global_rmse']
    val_ratio = float(val_rmse / train_rmse) if train_rmse > 0 else np.inf
    test_ratio = float(test_rmse / train_rmse) if train_rmse > 0 else np.inf
    return {
        'train_rmse': float(train_rmse),
        'val_rmse': float(val_rmse),
        'test_rmse': float(test_rmse),
        'val_to_train_rmse_ratio': val_ratio,
        'test_to_train_rmse_ratio': test_ratio,
        'possible_overfit': bool(val_ratio > 1.25 or test_ratio > 1.25),
    }


def print_per_joint_summary(title: str, metrics: dict[str, dict], metric_keys: list[str]) -> None:
    """Print a compact per-joint metric summary."""
    print(f"\n{title}")
    joint_labels = []
    for metric_key in metric_keys:
        joint_labels.extend(metrics.get(metric_key, {}).keys())
    ordered_labels = sorted(set(joint_labels), key=lambda label: int(label.split('_')[-1]))
    for label in ordered_labels:
        joint_idx = int(label.split('_')[-1])
        parts = []
        for metric_key in metric_keys:
            value = metrics.get(metric_key, {}).get(label)
            if value is None:
                continue
            if 'improvement' in metric_key:
                parts.append(f"{metric_key}={value:.2f}%")
            else:
                parts.append(f"{metric_key}={value:.6f} N·m")
        if parts:
            print(f"  Joint {joint_idx}: " + ", ".join(parts))


def run_identification_stage(
    *,
    robot_model,
    splits: dict[str, pd.DataFrame],
    effective_data_source: str,
    parameterization: str,
    identification_mode: str,
    solver_method: str,
    real_torque_source: str,
    payload_model,
    effective_payload_mode: PayloadMode,
    subtract_known_payload_gravity: bool,
    ridge_lambda: float = 1e-4,
) -> dict[str, Any]:
    """
    Execute the core identification stage only.

    This stops at parameter estimation and keeps post-identification consumers
    in separate modules.
    """
    print("\n[STEP 6] Identifying parameters...")
    print("-" * 78)
    identifier = ParameterIdentifier(robot_model, parameterization=parameterization)
    reference_parameters = None if effective_data_source == 'real' else robot_model.full_parameter_vector()
    result = identifier.identify_parameters(
        splits['train'],
        method=solver_method,
        reference_parameters=reference_parameters,
        ridge_lambda=ridge_lambda,
    )
    return {
        'identifier': identifier,
        'result': result,
        'reference_parameters': reference_parameters,
        'context': {
            'parameterization': parameterization,
            'identification_mode': identification_mode,
            'data_source': effective_data_source,
            'real_torque_source': real_torque_source if effective_data_source == 'real' else None,
            'payload_mode': effective_payload_mode.value,
            'payload_model': None if payload_model is None else payload_model.to_dict(),
            'subtract_known_payload_gravity': bool(subtract_known_payload_gravity),
        },
    }
