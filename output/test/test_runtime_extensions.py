#!/usr/bin/env python3
"""
Smoke tests for the runtime dynamics extensions introduced for payload/gravity handling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "python"))

from load_model import URDFLoader
from run_pipeline import DEFAULT_URDF_PATH
from runtime_dynamics import (
    GravityConfig,
    InertialParameterVector,
    PayloadMode,
    PayloadModel,
    RobotDynamicsModel,
    ValidationTools,
)


def main() -> int:
    loader = URDFLoader(str(DEFAULT_URDF_PATH), gravity_vector=[9.81, 0.0, 0.0])
    robot_model = loader.build_robot_model()
    runtime_model = RobotDynamicsModel(robot_model)
    parameter_vector = InertialParameterVector.from_robot_model_prior(robot_model)

    if parameter_vector.last_link_offset() != 10 * (robot_model.num_joints - 1):
        print("FAILED: last_link_offset does not point to the final rigid-parameter block.")
        return 1

    gravity_upright = GravityConfig.from_preset("upright")
    gravity_side = GravityConfig.from_any([9.81, 0.0, 0.0])
    if not np.allclose(gravity_upright.gravity_vector_base, np.array([0.0, 0.0, -9.81])):
        print("FAILED: upright gravity preset is incorrect.")
        return 1
    if not np.allclose(gravity_side.gravity_vector_base, np.array([9.81, 0.0, 0.0])):
        print("FAILED: custom gravity vector parsing is incorrect.")
        return 1

    payload = PayloadModel(
        mass=1.8,
        com_position=np.array([0.02, -0.03, 0.11]),
        com_frame=robot_model.ee_link,
        reference_link=robot_model.ee_link,
        inertia_about_com=None,
        is_point_mass_approx=True,
    )
    delta_phi = payload.to_inertial_parameter_delta()
    expected_delta = np.array(
        [
            1.8,
            1.8 * 0.02,
            1.8 * -0.03,
            1.8 * 0.11,
            1.8 * ((-0.03) ** 2 + 0.11 ** 2),
            -1.8 * 0.02 * -0.03,
            -1.8 * 0.02 * 0.11,
            1.8 * (0.02 ** 2 + 0.11 ** 2),
            -1.8 * -0.03 * 0.11,
            1.8 * (0.02 ** 2 + (-0.03) ** 2),
        ],
        dtype=float,
    )
    if not np.allclose(delta_phi, expected_delta, atol=1e-12):
        print("FAILED: payload point-mass inertial delta does not match the parallel-axis formula.")
        return 1

    sanity = ValidationTools.parameter_sanity_check(parameter_vector)
    if "per_link" not in sanity or len(sanity["per_link"]) != robot_model.num_joints:
        print("FAILED: parameter sanity check summary is incomplete.")
        return 1

    q = np.array([0.15, -0.3, 0.25, -0.2, 0.1, -0.05, 0.08], dtype=float)
    dq = np.zeros(robot_model.num_joints, dtype=float)
    ddq = np.zeros(robot_model.num_joints, dtype=float)
    tau_lumped = runtime_model.compute_total_torque(
        q,
        dq,
        ddq,
        parameter_vector,
        gravity_config=gravity_side,
        payload_model=payload,
        payload_mode=PayloadMode.LUMPED_LAST_LINK,
    )
    tau_external = runtime_model.compute_total_torque(
        q,
        dq,
        ddq,
        parameter_vector,
        gravity_config=gravity_side,
        payload_model=payload,
        payload_mode=PayloadMode.EXTERNAL_WRENCH,
    )
    diff = tau_lumped - tau_external
    if float(np.max(np.abs(diff))) > 1e-6:
        print("FAILED: lumped and external payload gravity compensation diverge in the quasi-static case.")
        print(f"max abs diff = {float(np.max(np.abs(diff))):.6e}")
        return 1

    print("=" * 72)
    print("Runtime extension checks")
    print("=" * 72)
    print(f"Gravity preset check: {gravity_upright.gravity_vector_base.tolist()}")
    print(f"Payload delta max abs error: {float(np.max(np.abs(delta_phi - expected_delta))):.6e}")
    print(f"Lumped vs external static max diff: {float(np.max(np.abs(diff))):.6e}")
    print(f"Sanity valid: {sanity['is_valid']}")
    print("PASSED: runtime dynamics extensions behave as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
