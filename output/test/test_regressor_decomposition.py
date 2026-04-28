#!/usr/bin/env python3
"""
验证显式重力分解 regressor 与旧整体 regressor 在同一重力下数值一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "python"))

from identify_parameters import RegressorBuilder
from load_model import URDFLoader
from run_pipeline import (
    DEFAULT_CUTOFF_HZ,
    DEFAULT_GRAVITY_VECTOR,
    DEFAULT_REAL_DATA_DIR,
    DEFAULT_REAL_TORQUE_SOURCE,
    DEFAULT_SAMPLING_FREQ,
    DEFAULT_URDF_PATH,
    _prepare_real_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare gravity-decomposed regressor against the legacy direct regressor.")
    parser.add_argument("--real-data-dir", default=str(DEFAULT_REAL_DATA_DIR))
    parser.add_argument("--real-torque-source", default=DEFAULT_REAL_TORQUE_SOURCE)
    parser.add_argument("--sampling-freq", type=float, default=DEFAULT_SAMPLING_FREQ)
    parser.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ)
    parser.add_argument("--urdf-path", default=str(DEFAULT_URDF_PATH))
    parser.add_argument("--gravity", nargs=3, type=float, default=DEFAULT_GRAVITY_VECTOR)
    parser.add_argument("--sample-limit", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    loader = URDFLoader(args.urdf_path, gravity_vector=list(args.gravity))
    robot_model = loader.build_robot_model()
    df = _prepare_real_dataset(
        robot_model,
        Path(args.real_data_dir),
        args.sampling_freq,
        args.cutoff_hz,
        args.real_torque_source,
    )
    if args.sample_limit > 0:
        df = df.head(args.sample_limit).copy()

    builder = RegressorBuilder(robot_model)
    Phi_new, _ = builder.build_regressor_matrix(df)
    Phi_old, _ = builder.build_regressor_matrix_legacy(df)
    diff = Phi_new - Phi_old

    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    zero_gravity_df = df.copy()
    zero_gravity_df['gravity_x'] = 0.0
    zero_gravity_df['gravity_y'] = 0.0
    zero_gravity_df['gravity_z'] = 0.0
    zero_components = builder.build_regressor_components(zero_gravity_df)
    zero_phi = builder.assemble_regressor_from_components(zero_components)
    zero_rigid = (
        zero_components['Y_M']
        + zero_components['Y_C']
    )
    zero_gravity_diff = zero_phi[:, :builder.num_rigid_params] - zero_rigid
    zero_gravity_max_abs = float(np.max(np.abs(zero_gravity_diff)))

    print("=" * 72)
    print("Regressor decomposition check")
    print("=" * 72)
    print(f"Samples compared: {len(df)}")
    print(f"Legacy vs decomposed max abs diff: {max_abs:.12e}")
    print(f"Legacy vs decomposed RMSE:         {rmse:.12e}")
    print(f"Zero-gravity rigid-only max diff:  {zero_gravity_max_abs:.12e}")
    print("=" * 72)

    if max_abs > args.tolerance or zero_gravity_max_abs > args.tolerance:
        print("FAILED: decomposition does not match legacy regressor within tolerance.")
        return 1

    print("PASSED: decomposition matches legacy regressor within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
