#!/usr/bin/env python3
"""
Physically consistent Panda dynamics identification pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python"))

from generate_golden_data import GoldenDataGenerator
from identify_parameters import ParameterIdentifier
from load_model import URDFLoader, print_model_info
from process_measured_data import (
    MeasuredDataProcessor,
    compute_excitation_diagnostics,
    create_synthetic_measured_data,
)
from residual_compensation import ResidualAnalyzer, ResidualCompensator


SUPPORTED_PARAMETERIZATIONS = ('base', 'full')


def run_pipeline(
    robot_name: str = 'panda',
    identification_mode: str = 'rigid_body_friction',
    parameterization: str = 'base',
):
    if robot_name != 'panda':
        raise ValueError("Only robot_name='panda' is supported in this refactored pipeline.")
    if identification_mode != 'rigid_body_friction':
        raise ValueError("Only identification_mode='rigid_body_friction' is currently supported.")
    if parameterization not in SUPPORTED_PARAMETERIZATIONS:
        raise ValueError(f"Unsupported parameterization: {parameterization}")

    print("\n" + "=" * 78)
    print(" PHYSICALLY CONSISTENT PANDA DYNAMICS IDENTIFICATION PIPELINE")
    print("=" * 78 + "\n")
    print(f"Robot: {robot_name}")
    print(f"Identification mode: {identification_mode}")
    print(f"Parameterization: {parameterization}\n")

    print("[STEP 1] Loading Panda model...")
    print("-" * 78)
    urdf_path = project_root / "models" / "urdf" / "panda_arm_minimal.urdf"
    config_path = project_root / "models" / "configs" / "panda_config.yaml"
    loader = URDFLoader(str(urdf_path), str(config_path))
    robot_model = loader.build_robot_model()
    print_model_info(robot_model)

    print("[STEP 2] Generating golden data...")
    print("-" * 78)
    golden_dir = project_root / "datasets" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    generator = GoldenDataGenerator(robot_model)
    fixed_cases = generator.generate_fixed_cases(num_cases=12)
    random_cases = generator.generate_random_cases(num_cases=96, seed=42)
    trajectory_cases = generator.generate_trajectory_cases(num_points=120)
    generator.export_to_json(fixed_cases, str(golden_dir / "fixed_cases.json"))
    generator.export_to_json(random_cases, str(golden_dir / "random_cases_seed42.json"))
    generator.export_to_json(trajectory_cases, str(golden_dir / "trajectory_cases.json"))
    print(f"Generated {len(fixed_cases)} fixed, {len(random_cases)} random, {len(trajectory_cases)} trajectory cases")

    print("\n[STEP 5] Creating and processing synthetic measured data...")
    print("-" * 78)
    raw_dir = project_root / "datasets" / "measured" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "synthetic_panda_run01.csv"
    df_raw = create_synthetic_measured_data(robot_model, num_samples=1600, seed=2026)
    df_raw.to_csv(raw_path, index=False)
    print(f"Saved raw synthetic data to {raw_path}")

    processor = MeasuredDataProcessor(num_joints=robot_model.num_joints)
    df_synced = processor.synchronize_timestamps(df_raw, reference_freq=100.0)
    df_filtered = processor.apply_low_pass_filter(df_synced, cutoff_hz=15.0, sampling_freq=100.0)
    df_diff = processor.differentiate_position(df_filtered, sampling_freq=100.0)

    proc_dir = project_root / "datasets" / "measured" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    proc_path = proc_dir / "synthetic_panda_run01_proc.csv"
    df_processed = processor.clean_and_export(df_diff, str(proc_path))
    excitation = compute_excitation_diagnostics(df_processed, robot_model.num_joints)
    print("Excitation diagnostics:")
    for key, value in excitation.items():
        print(f"  {key}: {value:.6f}")

    print("\n[STEP 6] Identifying parameters...")
    print("-" * 78)
    identifier = ParameterIdentifier(robot_model, parameterization=parameterization)
    result = identifier.identify_parameters(df_processed, method='ols')
    evaluation = identifier.evaluate_identification(df_processed, result)

    identified_dir = project_root / "datasets" / "identified"
    identified_dir.mkdir(parents=True, exist_ok=True)
    with open(identified_dir / "theta_hat_synthetic.json", 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'robot_name': robot_model.name,
                'identification_mode': identification_mode,
                'parameterization': parameterization,
                'rank': result['rank'],
                'full_regressor_rank': result['full_regressor_rank'],
                'base_parameter_count': result['base_parameter_count'],
                'condition_number': result['condition_number'],
                'active_condition_number': result['active_condition_number'],
                'active_parameter_indices': result['active_parameter_indices'].tolist(),
                'theta_hat': np_to_list(result['theta_hat']),
                'theta_hat_full': np_to_list(result['theta_hat_full']),
                'theta_true': np_to_list(result['theta_true']),
            },
            handle,
            indent=2,
        )
    with open(identified_dir / "evaluation_synthetic.json", 'w', encoding='utf-8') as handle:
        json.dump(evaluation, handle, indent=2)

    print("\n[STEP 7-8] Residual analysis and compensation...")
    print("-" * 78)
    analyzer = ResidualAnalyzer(robot_model)
    datasets = analyzer.build_residual_dataset(df_processed, identifier, result, test_split=0.2, val_split=0.1)
    compensator = ResidualCompensator(num_joints=robot_model.num_joints)
    compensator.train_linear_compensator(datasets['train'])
    compensation_eval = compensator.evaluate_compensator(datasets['test'])

    residual_dir = project_root / "datasets" / "residual"
    residual_dir.mkdir(parents=True, exist_ok=True)
    with open(residual_dir / "compensation_result.json", 'w', encoding='utf-8') as handle:
        json.dump(compensation_eval, handle, indent=2)

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
    print(f"  Global torque RMSE: {evaluation['global_rmse']:.6f} N·m")
    print(f"  Residual compensation improvement: {compensation_eval['improvement_percent']:.2f}%")
    print("=" * 78 + "\n")

    return {
        'robot_name': robot_model.name,
        'parameterization': parameterization,
        'rank': result['full_regressor_rank'],
        'base_parameter_count': result['base_parameter_count'],
        'condition_number': result['condition_number'],
        'active_condition_number': result['active_condition_number'],
        'global_rmse': evaluation['global_rmse'],
        'residual_improvement': compensation_eval['improvement_percent'],
    }


def np_to_list(value):
    return np.asarray(value, dtype=float).tolist()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Panda dynamics identification pipeline.")
    parser.add_argument('--robot-name', default='panda', choices=['panda'])
    parser.add_argument('--identification-mode', default='rigid_body_friction', choices=['rigid_body_friction'])
    parser.add_argument('--parameterization', default='base', choices=SUPPORTED_PARAMETERIZATIONS)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        run_pipeline(
            robot_name=args.robot_name,
            identification_mode=args.identification_mode,
            parameterization=args.parameterization,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
