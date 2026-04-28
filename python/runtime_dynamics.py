"""
面向运行时动力学计算的统一对象模型。

本模块把“重力方向配置、完整物理惯性参数、payload 模型、运行时动力学接口、
参数物理合理性检查”集中起来，目标是让辨识结果不仅能用于离线最小二乘拟合，
还能直接服务于后续实时动力学补偿与代码导出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional

import numpy as np

from load_model import RobotModel, parse_gravity_vector, pin


def _skew(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of a 3D vector."""
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _dynamic_params_to_inertia_matrix(dynamic_params: np.ndarray) -> np.ndarray:
    """Convert Pinocchio's 6 inertia dynamic parameters into a symmetric matrix."""
    ixx, ixy, ixz, iyy, iyz, izz = np.asarray(dynamic_params, dtype=float).reshape(6)
    return np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ],
        dtype=float,
    )


class PayloadMode(str, Enum):
    """Supported payload handling strategies."""

    NONE = "none"
    LUMPED_LAST_LINK = "lumped_last_link"
    EXTERNAL_WRENCH = "external_wrench"
    AUGMENTED_LINK = "augmented_link"

    @classmethod
    def from_value(cls, value: str | None) -> "PayloadMode":
        if value is None:
            return cls.NONE
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unsupported payload mode: {value!r}")


@dataclass
class GravityConfig:
    """
    Runtime gravity direction configuration expressed in the robot base frame.
    """

    gravity_vector_base: np.ndarray
    preset_name: str | None = None
    magnitude: float | None = None
    source_frame: str = "base"

    PRESETS: dict[str, np.ndarray] = field(
        default_factory=lambda: {
            "upright": np.array([0.0, 0.0, -9.81], dtype=float),
            "default": np.array([0.0, 0.0, -9.81], dtype=float),
            "inverted": np.array([0.0, 0.0, 9.81], dtype=float),
            "wall_x_neg": np.array([-9.81, 0.0, 0.0], dtype=float),
            "wall_x_pos": np.array([9.81, 0.0, 0.0], dtype=float),
            "wall_y_neg": np.array([0.0, -9.81, 0.0], dtype=float),
            "wall_y_pos": np.array([0.0, 9.81, 0.0], dtype=float),
        }
    )

    def __post_init__(self) -> None:
        self.gravity_vector_base = parse_gravity_vector(self.gravity_vector_base)
        if self.magnitude is None:
            self.magnitude = float(np.linalg.norm(self.gravity_vector_base))

    @classmethod
    def from_any(cls, value: str | Iterable[float] | np.ndarray | "GravityConfig" | None) -> "GravityConfig":
        if isinstance(value, cls):
            return cls(
                gravity_vector_base=np.asarray(value.gravity_vector_base, dtype=float),
                preset_name=value.preset_name,
                magnitude=value.magnitude,
                source_frame=value.source_frame,
            )
        if value is None:
            return cls.from_preset("default")
        if isinstance(value, str):
            normalized = value.strip().lower()
            presets = cls.PRESETS.fget(cls) if isinstance(cls.PRESETS, property) else None  # pragma: no cover
            del presets
            preset_map = {
                "upright": np.array([0.0, 0.0, -9.81], dtype=float),
                "default": np.array([0.0, 0.0, -9.81], dtype=float),
                "inverted": np.array([0.0, 0.0, 9.81], dtype=float),
                "wall_x_neg": np.array([-9.81, 0.0, 0.0], dtype=float),
                "wall_x_pos": np.array([9.81, 0.0, 0.0], dtype=float),
                "wall_y_neg": np.array([0.0, -9.81, 0.0], dtype=float),
                "wall_y_pos": np.array([0.0, 9.81, 0.0], dtype=float),
            }
            if normalized in preset_map:
                return cls(gravity_vector_base=preset_map[normalized], preset_name=normalized, source_frame="base")
        parsed = parse_gravity_vector(value)
        return cls(gravity_vector_base=parsed, preset_name=None, source_frame="base")

    @classmethod
    def from_preset(cls, preset_name: str) -> "GravityConfig":
        normalized = preset_name.strip().lower()
        preset_map = {
            "upright": np.array([0.0, 0.0, -9.81], dtype=float),
            "default": np.array([0.0, 0.0, -9.81], dtype=float),
            "inverted": np.array([0.0, 0.0, 9.81], dtype=float),
            "wall_x_neg": np.array([-9.81, 0.0, 0.0], dtype=float),
            "wall_x_pos": np.array([9.81, 0.0, 0.0], dtype=float),
            "wall_y_neg": np.array([0.0, -9.81, 0.0], dtype=float),
            "wall_y_pos": np.array([0.0, 9.81, 0.0], dtype=float),
        }
        if normalized not in preset_map:
            raise ValueError(f"Unsupported gravity preset: {preset_name!r}")
        return cls(gravity_vector_base=preset_map[normalized], preset_name=normalized, source_frame="base")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gravity_vector_base": np.asarray(self.gravity_vector_base, dtype=float).tolist(),
            "preset_name": self.preset_name,
            "magnitude": float(self.magnitude),
            "source_frame": self.source_frame,
        }


@dataclass
class PayloadModel:
    """
    Payload inertial model expressed relative to a reference link or tool frame.
    """

    mass: float
    com_position: np.ndarray
    com_frame: str
    reference_link: str
    inertia_about_com: np.ndarray | None = None
    is_point_mass_approx: bool = True

    def __post_init__(self) -> None:
        self.com_position = np.asarray(self.com_position, dtype=float).reshape(3)
        if self.inertia_about_com is not None:
            self.inertia_about_com = np.asarray(self.inertia_about_com, dtype=float).reshape(3, 3)
            self.is_point_mass_approx = False
        if self.mass < 0.0:
            raise ValueError("Payload mass must be non-negative.")

    def _resolve_frame_transform(self, frame_transform: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        if frame_transform is None:
            return np.eye(3, dtype=float), np.zeros(3, dtype=float)

        transform = np.asarray(frame_transform, dtype=float)
        if transform.shape == (4, 4):
            return transform[:3, :3], transform[:3, 3]
        if transform.shape == (3, 4):
            return transform[:3, :3], transform[:3, 3]
        raise ValueError(
            "frame_transform must be a 4x4 homogeneous transform or a 3x4 [R|t] matrix."
        )

    def resolve_com_in_reference_link(self, frame_transform: np.ndarray | None = None) -> np.ndarray:
        """
        Resolve the payload COM into the reference-link coordinates.
        """
        normalized_frame = self.com_frame.strip().lower()
        if normalized_frame in {"reference_link", "link", self.reference_link.lower()}:
            return self.com_position.copy()
        if normalized_frame == "tool":
            if frame_transform is None:
                raise ValueError(
                    "Payload COM is expressed in the tool frame; provide frame_transform "
                    "from tool to reference_link before lumping."
                )
            rotation, translation = self._resolve_frame_transform(frame_transform)
            return rotation @ self.com_position + translation
        raise ValueError(
            f"Unsupported payload com_frame={self.com_frame!r}. Expected the reference link name, "
            "'link', 'reference_link', or 'tool'."
        )

    def to_inertial_parameter_delta(self, frame_transform: np.ndarray | None = None) -> np.ndarray:
        """
        Convert payload mass/COM/inertia into a 10-parameter delta for a rigid link.
        """
        com_ref = self.resolve_com_in_reference_link(frame_transform)
        cx, cy, cz = com_ref
        shift_inertia = self.mass * (
            np.dot(com_ref, com_ref) * np.eye(3, dtype=float) - np.outer(com_ref, com_ref)
        )

        if self.inertia_about_com is None:
            inertia_ref = shift_inertia
        else:
            if self.com_frame.strip().lower() == "tool":
                rotation, _ = self._resolve_frame_transform(frame_transform)
                inertia_about_com_ref = rotation @ self.inertia_about_com @ rotation.T
            else:
                inertia_about_com_ref = self.inertia_about_com
            inertia_ref = inertia_about_com_ref + shift_inertia

        return np.array(
            [
                self.mass,
                self.mass * cx,
                self.mass * cy,
                self.mass * cz,
                inertia_ref[0, 0],
                inertia_ref[0, 1],
                inertia_ref[0, 2],
                inertia_ref[1, 1],
                inertia_ref[1, 2],
                inertia_ref[2, 2],
            ],
            dtype=float,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mass": float(self.mass),
            "com_position": self.com_position.tolist(),
            "com_frame": self.com_frame,
            "reference_link": self.reference_link,
            "inertia_about_com": None if self.inertia_about_com is None else self.inertia_about_com.tolist(),
            "is_point_mass_approx": bool(self.is_point_mass_approx),
        }


@dataclass
class InertialParameterVector:
    """
    Full physically interpretable inertial + friction parameter vector.
    """

    rigid_parameters: np.ndarray
    viscous_friction: np.ndarray | None = None
    coulomb_friction: np.ndarray | None = None
    extra_friction: np.ndarray | None = None

    def __post_init__(self) -> None:
        rigid = np.asarray(self.rigid_parameters, dtype=float)
        if rigid.ndim == 1:
            if rigid.size % 10 != 0:
                raise ValueError(f"Rigid parameter vector must be divisible by 10, got length {rigid.size}.")
            rigid = rigid.reshape(-1, 10)
        if rigid.ndim != 2 or rigid.shape[1] != 10:
            raise ValueError(f"Rigid parameter matrix must have shape (n, 10), got {rigid.shape}.")
        self.rigid_parameters = rigid
        num_joints = rigid.shape[0]

        if self.viscous_friction is None:
            self.viscous_friction = np.zeros(num_joints, dtype=float)
        else:
            self.viscous_friction = np.asarray(self.viscous_friction, dtype=float).reshape(num_joints)

        if self.coulomb_friction is None:
            self.coulomb_friction = np.zeros(num_joints, dtype=float)
        else:
            self.coulomb_friction = np.asarray(self.coulomb_friction, dtype=float).reshape(num_joints)

        if self.extra_friction is None:
            self.extra_friction = np.array([], dtype=float)
        else:
            self.extra_friction = np.asarray(self.extra_friction, dtype=float).reshape(-1)

    @property
    def num_joints(self) -> int:
        return int(self.rigid_parameters.shape[0])

    @classmethod
    def from_robot_model_prior(cls, robot_model: RobotModel) -> "InertialParameterVector":
        full_theta = np.asarray(robot_model.full_parameter_vector(), dtype=float)
        return cls.from_theta_full(full_theta, robot_model.num_joints)

    @classmethod
    def from_theta_full(cls, theta_full: np.ndarray, num_joints: int) -> "InertialParameterVector":
        theta = np.asarray(theta_full, dtype=float).reshape(-1)
        rigid_count = 10 * num_joints
        if theta.size < rigid_count:
            raise ValueError(
                f"theta_full length {theta.size} is smaller than rigid-parameter count {rigid_count}."
            )
        rigid = theta[:rigid_count].reshape(num_joints, 10)
        remainder = theta[rigid_count:]
        viscous = remainder[:num_joints] if remainder.size >= num_joints else np.zeros(num_joints, dtype=float)
        coulomb_start = num_joints
        coulomb_end = coulomb_start + num_joints
        coulomb = (
            remainder[coulomb_start:coulomb_end]
            if remainder.size >= coulomb_end
            else np.zeros(num_joints, dtype=float)
        )
        extra = remainder[coulomb_end:] if remainder.size > coulomb_end else np.array([], dtype=float)
        return cls(rigid_parameters=rigid, viscous_friction=viscous, coulomb_friction=coulomb, extra_friction=extra)

    @classmethod
    def from_identification_result(cls, result: Dict[str, Any], num_joints: int) -> "InertialParameterVector":
        theta_full = result.get("pi_full_hat", result.get("theta_hat_full"))
        if theta_full is None:
            raise KeyError("Identification result does not contain 'pi_full_hat' or 'theta_hat_full'.")
        return cls.from_theta_full(np.asarray(theta_full, dtype=float), num_joints)

    def copy(self) -> "InertialParameterVector":
        return InertialParameterVector(
            rigid_parameters=self.rigid_parameters.copy(),
            viscous_friction=self.viscous_friction.copy(),
            coulomb_friction=self.coulomb_friction.copy(),
            extra_friction=self.extra_friction.copy(),
        )

    def link_slice(self, link_index: int) -> slice:
        if not 0 <= link_index < self.num_joints:
            raise IndexError(f"link_index {link_index} out of range for {self.num_joints} joints.")
        start = 10 * link_index
        return slice(start, start + 10)

    def get_link_phi(self, link_index: int) -> np.ndarray:
        return self.rigid_parameters[link_index].copy()

    def set_link_phi(self, link_index: int, phi_i: np.ndarray) -> None:
        self.rigid_parameters[link_index] = np.asarray(phi_i, dtype=float).reshape(10)

    def add_to_link(self, link_index: int, delta_phi: np.ndarray) -> None:
        self.rigid_parameters[link_index] += np.asarray(delta_phi, dtype=float).reshape(10)

    def last_link_offset(self) -> int:
        return 10 * (self.num_joints - 1)

    def theta_rigid(self) -> np.ndarray:
        return self.rigid_parameters.reshape(-1)

    def to_theta_full(self, with_friction: bool = True) -> np.ndarray:
        theta = self.theta_rigid()
        if not with_friction:
            return theta
        return np.concatenate(
            [
                theta,
                np.asarray(self.viscous_friction, dtype=float),
                np.asarray(self.coulomb_friction, dtype=float),
                np.asarray(self.extra_friction, dtype=float),
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        links = []
        for link_idx, phi in enumerate(self.rigid_parameters):
            links.append(
                {
                    "link_index": int(link_idx),
                    "mass": float(phi[0]),
                    "first_moments": [float(value) for value in phi[1:4]],
                    "inertia_tensor_dynamic_params": [float(value) for value in phi[4:10]],
                }
            )
        return {
            "num_joints": self.num_joints,
            "links": links,
            "viscous_friction": np.asarray(self.viscous_friction, dtype=float).tolist(),
            "coulomb_friction": np.asarray(self.coulomb_friction, dtype=float).tolist(),
            "extra_friction": np.asarray(self.extra_friction, dtype=float).tolist(),
        }


class ValidationTools:
    """Numerical validation helpers for identified inertial parameters."""

    @staticmethod
    def parameter_sanity_check(parameter_vector: InertialParameterVector) -> Dict[str, Any]:
        per_link = []
        issues: list[str] = []

        for link_index in range(parameter_vector.num_joints):
            phi = parameter_vector.get_link_phi(link_index)
            mass = float(phi[0])
            inertia_matrix = _dynamic_params_to_inertia_matrix(phi[4:10])
            eigenvalues = np.linalg.eigvalsh(inertia_matrix)
            triangle_ok = (
                inertia_matrix[0, 0] <= inertia_matrix[1, 1] + inertia_matrix[2, 2] + 1e-9
                and inertia_matrix[1, 1] <= inertia_matrix[0, 0] + inertia_matrix[2, 2] + 1e-9
                and inertia_matrix[2, 2] <= inertia_matrix[0, 0] + inertia_matrix[1, 1] + 1e-9
            )

            link_issues = []
            if mass <= 0.0:
                link_issues.append("non_positive_mass")
            if np.min(eigenvalues) < -1e-9:
                link_issues.append("non_psd_inertia")
            if not triangle_ok:
                link_issues.append("triangle_inequality_violation")

            if link_issues:
                issues.extend([f"link_{link_index}: {issue}" for issue in link_issues])

            per_link.append(
                {
                    "link_index": int(link_index),
                    "mass": mass,
                    "principal_inertia_eigenvalues": eigenvalues.tolist(),
                    "triangle_inequality_ok": bool(triangle_ok),
                    "issues": link_issues,
                }
            )

        viscous = np.asarray(parameter_vector.viscous_friction, dtype=float)
        coulomb = np.asarray(parameter_vector.coulomb_friction, dtype=float)
        friction_summary = {
            "viscous_nonfinite_indices": np.where(~np.isfinite(viscous))[0].astype(int).tolist(),
            "coulomb_nonfinite_indices": np.where(~np.isfinite(coulomb))[0].astype(int).tolist(),
            "viscous_abs_max": float(np.max(np.abs(viscous))) if viscous.size else 0.0,
            "coulomb_abs_max": float(np.max(np.abs(coulomb))) if coulomb.size else 0.0,
        }
        if friction_summary["viscous_nonfinite_indices"]:
            issues.append("viscous_friction_contains_nonfinite")
        if friction_summary["coulomb_nonfinite_indices"]:
            issues.append("coulomb_friction_contains_nonfinite")

        return {
            "is_valid": not issues,
            "issues": issues,
            "per_link": per_link,
            "friction": friction_summary,
        }


class RobotDynamicsModel:
    """
    Unified runtime dynamics interface built on top of the loaded Pinocchio model.
    """

    def __init__(self, robot_model: RobotModel):
        if pin is None:
            raise RuntimeError("Pinocchio is required to build RobotDynamicsModel.")
        self.robot_model = robot_model
        self.num_joints = robot_model.num_joints
        self.num_rigid_params = 10 * self.num_joints
        self.gravity_config = GravityConfig.from_any(robot_model.gravity_vector)

    def set_gravity_config(self, gravity_config: GravityConfig | str | np.ndarray | list[float]) -> None:
        self.gravity_config = GravityConfig.from_any(gravity_config)
        self.robot_model.set_gravity(self.gravity_config.gravity_vector_base)

    def _rigid_regressor_with_conditions(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        gravity_vector: np.ndarray,
    ) -> np.ndarray:
        model = self.robot_model.pinocchio_model
        data = model.createData()
        previous_gravity = model.gravity.linear.copy()
        try:
            model.gravity.linear = np.asarray(gravity_vector, dtype=float).copy()
            return np.asarray(
                pin.computeJointTorqueRegressor(model, data, q, dq, ddq),
                dtype=float,
            ).reshape(self.num_joints, self.num_rigid_params)
        finally:
            model.gravity.linear = previous_gravity

    def compute_Y_M(self, q: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        zero = np.zeros(self.num_joints, dtype=float)
        return self._rigid_regressor_with_conditions(q, zero, ddq, np.zeros(3, dtype=float))

    def compute_Y_C(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        zero = np.zeros(self.num_joints, dtype=float)
        return self._rigid_regressor_with_conditions(q, dq, zero, np.zeros(3, dtype=float))

    def compute_Y_Gx(self, q: np.ndarray) -> np.ndarray:
        zero = np.zeros(self.num_joints, dtype=float)
        return self._rigid_regressor_with_conditions(q, zero, zero, np.array([1.0, 0.0, 0.0], dtype=float))

    def compute_Y_Gy(self, q: np.ndarray) -> np.ndarray:
        zero = np.zeros(self.num_joints, dtype=float)
        return self._rigid_regressor_with_conditions(q, zero, zero, np.array([0.0, 1.0, 0.0], dtype=float))

    def compute_Y_Gz(self, q: np.ndarray) -> np.ndarray:
        zero = np.zeros(self.num_joints, dtype=float)
        return self._rigid_regressor_with_conditions(q, zero, zero, np.array([0.0, 0.0, 1.0], dtype=float))

    def compute_Y_F(self, dq: np.ndarray, parameter_vector: InertialParameterVector | None = None) -> np.ndarray:
        del parameter_vector
        block = np.zeros((self.num_joints, 2 * self.num_joints), dtype=float)
        sign = np.where(dq > 1e-4, 1.0, np.where(dq < -1e-4, -1.0, 0.0))
        for joint_idx in range(self.num_joints):
            block[joint_idx, joint_idx] = float(dq[joint_idx])
            block[joint_idx, self.num_joints + joint_idx] = float(sign[joint_idx])
        return block

    def assemble_components(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        return {
            "Y_M": self.compute_Y_M(q, ddq),
            "Y_C": self.compute_Y_C(q, dq),
            "Y_Gx": self.compute_Y_Gx(q),
            "Y_Gy": self.compute_Y_Gy(q),
            "Y_Gz": self.compute_Y_Gz(q),
        }

    def compute_tau_robot(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        parameter_vector: InertialParameterVector,
        gravity_config: GravityConfig | str | np.ndarray | list[float] | None = None,
    ) -> np.ndarray:
        gravity = GravityConfig.from_any(gravity_config or self.gravity_config)
        components = self.assemble_components(q, dq, ddq)
        rigid_regressor = (
            components["Y_M"]
            + components["Y_C"]
            + gravity.gravity_vector_base[0] * components["Y_Gx"]
            + gravity.gravity_vector_base[1] * components["Y_Gy"]
            + gravity.gravity_vector_base[2] * components["Y_Gz"]
        )
        rigid_tau = rigid_regressor @ parameter_vector.theta_rigid()
        friction_block = self.compute_Y_F(dq, parameter_vector)
        friction_theta = np.concatenate(
            [
                np.asarray(parameter_vector.viscous_friction, dtype=float),
                np.asarray(parameter_vector.coulomb_friction, dtype=float),
            ]
        )
        friction_tau = friction_block @ friction_theta
        return rigid_tau + friction_tau

    def compute_tau_breakdown(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        parameter_vector: InertialParameterVector,
        gravity_config: GravityConfig | str | np.ndarray | list[float] | None = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-component regressors and torque contributions for one state.

        This is the runtime-facing diagnostic view used to compare a single
        sample against a real robot measurement.
        """
        gravity = GravityConfig.from_any(gravity_config or self.gravity_config)
        components = self.assemble_components(q, dq, ddq)
        rigid_theta = parameter_vector.theta_rigid()
        friction_theta = np.concatenate(
            [
                np.asarray(parameter_vector.viscous_friction, dtype=float),
                np.asarray(parameter_vector.coulomb_friction, dtype=float),
            ]
        )
        friction_block = self.compute_Y_F(dq, parameter_vector)

        tau_M = components["Y_M"] @ rigid_theta
        tau_C = components["Y_C"] @ rigid_theta
        tau_Gx_unit = components["Y_Gx"] @ rigid_theta
        tau_Gy_unit = components["Y_Gy"] @ rigid_theta
        tau_Gz_unit = components["Y_Gz"] @ rigid_theta
        tau_Gx = gravity.gravity_vector_base[0] * tau_Gx_unit
        tau_Gy = gravity.gravity_vector_base[1] * tau_Gy_unit
        tau_Gz = gravity.gravity_vector_base[2] * tau_Gz_unit
        tau_gravity = tau_Gx + tau_Gy + tau_Gz
        tau_friction = friction_block @ friction_theta
        tau_rigid = tau_M + tau_C + tau_gravity
        tau_total = tau_rigid + tau_friction

        return {
            "gravity_vector_base": gravity.gravity_vector_base.copy(),
            "Y_M": components["Y_M"],
            "Y_C": components["Y_C"],
            "Y_Gx": components["Y_Gx"],
            "Y_Gy": components["Y_Gy"],
            "Y_Gz": components["Y_Gz"],
            "Y_F": friction_block,
            "theta_rigid": rigid_theta,
            "theta_friction": friction_theta,
            "tau_M": tau_M,
            "tau_C": tau_C,
            "tau_Gx_unit": tau_Gx_unit,
            "tau_Gy_unit": tau_Gy_unit,
            "tau_Gz_unit": tau_Gz_unit,
            "tau_Gx": tau_Gx,
            "tau_Gy": tau_Gy,
            "tau_Gz": tau_Gz,
            "tau_gravity": tau_gravity,
            "tau_friction": tau_friction,
            "tau_rigid": tau_rigid,
            "tau_total": tau_total,
        }

    def apply_payload_lumped(
        self,
        parameter_vector: InertialParameterVector,
        payload_model: PayloadModel,
        frame_transform: np.ndarray | None = None,
        link_index: int | None = None,
    ) -> InertialParameterVector:
        updated = parameter_vector.copy()
        target_link = self.num_joints - 1 if link_index is None else int(link_index)
        updated.add_to_link(target_link, payload_model.to_inertial_parameter_delta(frame_transform=frame_transform))
        return updated

    def compute_payload_gravity_torque(
        self,
        q: np.ndarray,
        payload_model: PayloadModel,
        gravity_config: GravityConfig | str | np.ndarray | list[float] | None = None,
        frame_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        gravity = GravityConfig.from_any(gravity_config or self.gravity_config)
        payload_only = InertialParameterVector(
            rigid_parameters=np.zeros((self.num_joints, 10), dtype=float),
            viscous_friction=np.zeros(self.num_joints, dtype=float),
            coulomb_friction=np.zeros(self.num_joints, dtype=float),
        )
        payload_only.add_to_link(
            self.num_joints - 1,
            payload_model.to_inertial_parameter_delta(frame_transform=frame_transform),
        )
        zero = np.zeros(self.num_joints, dtype=float)
        return self.compute_tau_robot(
            q,
            zero,
            zero,
            payload_only,
            gravity_config=gravity,
        )

    def compute_total_torque(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        parameter_vector: InertialParameterVector,
        gravity_config: GravityConfig | str | np.ndarray | list[float] | None = None,
        payload_model: PayloadModel | None = None,
        payload_mode: PayloadMode | str = PayloadMode.NONE,
        frame_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        mode = PayloadMode.from_value(payload_mode if not isinstance(payload_mode, PayloadMode) else payload_mode.value)
        if payload_model is None or mode == PayloadMode.NONE:
            return self.compute_tau_robot(q, dq, ddq, parameter_vector, gravity_config=gravity_config)
        if mode == PayloadMode.LUMPED_LAST_LINK:
            loaded = self.apply_payload_lumped(parameter_vector, payload_model, frame_transform=frame_transform)
            return self.compute_tau_robot(q, dq, ddq, loaded, gravity_config=gravity_config)
        if mode == PayloadMode.EXTERNAL_WRENCH:
            tau_robot = self.compute_tau_robot(q, dq, ddq, parameter_vector, gravity_config=gravity_config)
            return tau_robot + self.compute_payload_gravity_torque(
                q,
                payload_model,
                gravity_config=gravity_config,
                frame_transform=frame_transform,
            )
        raise NotImplementedError(f"Payload mode {mode.value!r} is not implemented in v1 runtime dynamics.")
