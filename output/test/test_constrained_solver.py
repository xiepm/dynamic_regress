#!/usr/bin/env python3
"""
Smoke test for the physically constrained identification solver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "python"))

from identify_parameters import ParameterIdentifier
from load_model import URDFLoader
from run_pipeline import DEFAULT_URDF_PATH, _prepare_synthetic_dataset, _split_dataframe_for_learning


def main() -> int:
    loader = URDFLoader(str(DEFAULT_URDF_PATH), gravity_vector=[9.81, 0.0, 0.0])
    robot_model = loader.build_robot_model()
    df = _prepare_synthetic_dataset(robot_model, 100.0, 15.0).head(24).copy()
    splits = _split_dataframe_for_learning(df)

    identifier = ParameterIdentifier(robot_model, parameterization='full')
    result = identifier.identify_parameters(
        splits['train'],
        method='constrained',
        reference_parameters=None,
        ridge_lambda=1e-4,
    )

    masses = np.asarray(result['pi_full_hat'][:10 * robot_model.num_joints:10], dtype=float)
    if not result['optimizer_success']:
        print("FAILED: constrained solver did not converge.")
        print(result['optimizer_status'])
        return 1
    if np.any(masses <= 0.0):
        print("FAILED: constrained solver returned non-positive link masses.")
        print(masses)
        return 1
    if not result['physical_sanity']['is_valid']:
        print("FAILED: constrained solver still violates physical sanity checks.")
        print(result['physical_sanity']['issues'])
        return 1

    evaluation = identifier.evaluate_identification(splits['test'], result)
    print("=" * 72)
    print("Constrained solver check")
    print("=" * 72)
    print(f"Optimizer status: {result['optimizer_status']}")
    print(f"Optimizer iterations: {result['optimizer_iterations']}")
    print(f"Masses: {masses.tolist()}")
    print(f"Test global RMSE: {evaluation['global_rmse']:.6f} N·m")
    print("PASSED: constrained solver produced physically valid parameters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
