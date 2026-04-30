#!/usr/bin/env python3
"""
Verify that generated RTOS-style dynamics code honors runtime `parms` overrides.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "python"))

from output.test.test_cpp_consistency import (  # type: ignore
    _build_reference_pipeline,
    _compile_harness,
    _run_harness,
    _write_harness_input,
)
from runtime_dynamics import GravityConfig, InertialParameterVector, RobotDynamicsModel
from run_pipeline import (
    DEFAULT_CUTOFF_HZ,
    DEFAULT_DATA_SOURCE,
    DEFAULT_EXPORT_CLASS_NAME,
    DEFAULT_GRAVITY_VECTOR,
    DEFAULT_IDENTIFICATION_MODE,
    DEFAULT_PARAMETERIZATION,
    DEFAULT_REAL_DATA_DIR,
    DEFAULT_REAL_TORQUE_SOURCE,
    DEFAULT_SAMPLING_FREQ,
    DEFAULT_URDF_PATH,
)


def _build_rtos_parms(parameter_vector: InertialParameterVector) -> np.ndarray:
    num_joints = parameter_vector.num_joints
    parms = np.zeros(num_joints * 13, dtype=float)
    theta = parameter_vector.to_theta_full(with_friction=True)
    rigid = theta[: 10 * num_joints]
    viscous = theta[10 * num_joints:11 * num_joints]
    coulomb = theta[11 * num_joints:12 * num_joints]
    for joint_idx in range(num_joints):
        phi = rigid[10 * joint_idx:10 * (joint_idx + 1)]
        offset = 13 * joint_idx
        parms[offset + 0] = phi[4]
        parms[offset + 1] = phi[5]
        parms[offset + 2] = phi[6]
        parms[offset + 3] = phi[7]
        parms[offset + 4] = phi[8]
        parms[offset + 5] = phi[9]
        parms[offset + 6] = phi[1]
        parms[offset + 7] = phi[2]
        parms[offset + 8] = phi[3]
        parms[offset + 9] = phi[0]
        parms[offset + 10] = 0.0
        parms[offset + 11] = viscous[joint_idx]
        parms[offset + 12] = coulomb[joint_idx]
    return parms


def _apply_payload_to_rtos_last_link(parms: np.ndarray, *, mass: float, com: np.ndarray) -> np.ndarray:
    updated = np.asarray(parms, dtype=float).copy()
    x, y, z = np.asarray(com, dtype=float).reshape(3)
    payload_delta = np.array(
        [
            mass * (y * y + z * z),
            -mass * x * y,
            -mass * x * z,
            mass * (x * x + z * z),
            -mass * y * z,
            mass * (x * x + y * y),
            mass * x,
            mass * y,
            mass * z,
            mass,
        ],
        dtype=float,
    )
    updated[-13:-3] += payload_delta
    return updated


def _rtos_parms_to_parameter_vector(parms: np.ndarray, num_joints: int) -> InertialParameterVector:
    rigid = np.zeros((num_joints, 10), dtype=float)
    viscous = np.zeros(num_joints, dtype=float)
    coulomb = np.zeros(num_joints, dtype=float)
    for joint_idx in range(num_joints):
        offset = 13 * joint_idx
        rigid[joint_idx, 0] = parms[offset + 9]
        rigid[joint_idx, 1] = parms[offset + 6]
        rigid[joint_idx, 2] = parms[offset + 7]
        rigid[joint_idx, 3] = parms[offset + 8]
        rigid[joint_idx, 4] = parms[offset + 0]
        rigid[joint_idx, 5] = parms[offset + 1]
        rigid[joint_idx, 6] = parms[offset + 2]
        rigid[joint_idx, 7] = parms[offset + 3]
        rigid[joint_idx, 8] = parms[offset + 4]
        rigid[joint_idx, 9] = parms[offset + 5]
        viscous[joint_idx] = parms[offset + 11]
        coulomb[joint_idx] = parms[offset + 12]
    return InertialParameterVector(rigid_parameters=rigid, viscous_friction=viscous, coulomb_friction=coulomb)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check runtime payload override through RTOS parms.")
    parser.add_argument("--data-source", default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--real-data-dir", default=str(DEFAULT_REAL_DATA_DIR))
    parser.add_argument("--real-torque-source", default=DEFAULT_REAL_TORQUE_SOURCE)
    parser.add_argument("--sampling-freq", type=float, default=DEFAULT_SAMPLING_FREQ)
    parser.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ)
    parser.add_argument("--urdf-path", default=str(DEFAULT_URDF_PATH))
    parser.add_argument("--gravity", nargs=3, type=float, default=DEFAULT_GRAVITY_VECTOR, metavar=("GX", "GY", "GZ"))
    parser.add_argument("--parameterization", default=DEFAULT_PARAMETERIZATION)
    parser.add_argument("--identification-mode", default=DEFAULT_IDENTIFICATION_MODE)
    parser.add_argument("--export-class-name", default=DEFAULT_EXPORT_CLASS_NAME)
    parser.add_argument("--payload-mass", type=float, default=1.25)
    parser.add_argument("--payload-com", nargs=3, type=float, default=[0.0, 0.0, 0.08], metavar=("CX", "CY", "CZ"))
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    pipeline = _build_reference_pipeline(
        data_source=args.data_source,
        real_data_dir=Path(args.real_data_dir),
        real_torque_source=args.real_torque_source,
        sampling_freq=args.sampling_freq,
        cutoff_hz=args.cutoff_hz,
        urdf_path=Path(args.urdf_path),
        gravity_vector=list(args.gravity),
        parameterization=args.parameterization,
        identification_mode=args.identification_mode,
        export_class_name=args.export_class_name,
    )

    sample_df = pipeline["splits"]["test"].reset_index(drop=True).head(1).copy()
    q = sample_df[[f"q_{joint_idx}" for joint_idx in range(1, 8)]].to_numpy(dtype=float)[0]
    dq = sample_df[[f"dq_{joint_idx}" for joint_idx in range(1, 8)]].to_numpy(dtype=float)[0]
    ddq = sample_df[[f"ddq_{joint_idx}" for joint_idx in range(1, 8)]].to_numpy(dtype=float)[0]

    base_parms = _build_rtos_parms(pipeline["result"]["parameter_vector"])
    overridden_parms = _apply_payload_to_rtos_last_link(
        base_parms,
        mass=float(args.payload_mass),
        com=np.asarray(args.payload_com, dtype=float),
    )

    runtime_model = RobotDynamicsModel(pipeline["robot_model"])
    expected_vector = _rtos_parms_to_parameter_vector(overridden_parms, pipeline["robot_model"].num_joints)
    expected_tau = runtime_model.compute_tau_robot(
        q,
        dq,
        ddq,
        expected_vector,
        gravity_config=GravityConfig.from_any(args.gravity),
    )

    with tempfile.TemporaryDirectory(prefix="identified_payload_override_", dir=str(project_root / "output" / "test")) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        input_path = temp_dir / "samples.txt"
        parms_path = temp_dir / "parms.txt"
        binary_path = temp_dir / "identified_dynamics_harness"
        _write_harness_input(sample_df, input_path)
        parms_path.write_text(" ".join(f"{value:.17g}" for value in overridden_parms), encoding="utf-8")
        _compile_harness(binary_path, Path(pipeline["generated"]["project_cpp"]))
        cpp_prediction = _run_harness(binary_path, input_path, list(args.gravity), parms_path)

    max_abs_error = float(np.max(np.abs(cpp_prediction[0] - expected_tau)))
    print("=" * 72)
    print("generated dynamics runtime-parms payload override check")
    print("=" * 72)
    print(f"Payload mass: {args.payload_mass}")
    print(f"Payload COM: {list(args.payload_com)}")
    print(f"Expected tau: {expected_tau.tolist()}")
    print(f"C++ tau:      {cpp_prediction[0].tolist()}")
    print(f"Max abs error: {max_abs_error:.12e}")
    print("=" * 72)

    if max_abs_error > args.tolerance:
        print(
            f"FAILED: max_abs_error={max_abs_error:.12e} exceeds tolerance "
            f"{args.tolerance:.12e}"
        )
        return 1

    print(f"PASSED: runtime parms payload override matches Python reference within {args.tolerance:.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
