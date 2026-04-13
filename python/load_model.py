"""
Step 1: Load Panda Model & Known Parameters
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml


def _import_pinocchio():
    """Import pinocchio while tolerating ROS PYTHONPATH pollution."""
    try:
        import pinocchio as pin_module
        return pin_module
    except Exception:
        ros_paths = [
            path for path in sys.path
            if 'ros/humble' in path and 'site-packages' in path
        ]
        if ros_paths:
            sys.path = [path for path in sys.path if path not in ros_paths]
        try:
            return importlib.import_module('pinocchio')
        except ImportError:
            print("Warning: pinocchio not installed. Install with: pip install pinocchio")
            return None


pin = _import_pinocchio()


@dataclass
class JointParameters:
    """Active-joint dynamic quantities extracted from the URDF."""
    name: str
    mass: float
    com: np.ndarray
    inertia_matrix: np.ndarray
    damping: float
    friction: float


@dataclass
class RobotModel:
    """Robot metadata and Pinocchio model used by the identification pipeline."""
    name: str
    num_joints: int
    joint_names: List[str]
    joint_params: List[JointParameters]
    pinocchio_model: object
    pinocchio_data: object
    joint_limits: Dict[str, Dict[str, float]]
    base_link: str
    ee_link: str
    description_source: str

    @property
    def damping(self) -> np.ndarray:
        return np.array([jp.damping for jp in self.joint_params], dtype=float)

    @property
    def friction(self) -> np.ndarray:
        return np.array([jp.friction for jp in self.joint_params], dtype=float)

    def inertial_parameter_vector(self) -> np.ndarray:
        """Concatenate Pinocchio dynamic parameters for all active joints."""
        params = []
        for joint_id in range(1, self.pinocchio_model.njoints):
            params.append(np.array(self.pinocchio_model.inertias[joint_id].toDynamicParameters(), dtype=float))
        return np.concatenate(params)

    def full_parameter_vector(self) -> np.ndarray:
        """Rigid-body parameters followed by viscous and Coulomb friction."""
        return np.concatenate([self.inertial_parameter_vector(), self.damping, self.friction])


class URDFLoader:
    """Load the Panda URDF and derive dynamic quantities from it."""

    def __init__(self, urdf_path: str, config_path: str):
        self.urdf_path = Path(urdf_path)
        self.config_path = Path(config_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    def load_config(self) -> Dict:
        with open(self.config_path, 'r', encoding='utf-8') as handle:
            return yaml.safe_load(handle)

    def load_pinocchio_model(self) -> Tuple[object, object]:
        if pin is None:
            raise ImportError("pinocchio is required. Install with: pip install pinocchio")
        model = pin.buildModelFromUrdf(str(self.urdf_path))
        data = model.createData()
        return model, data

    def build_robot_model(self) -> RobotModel:
        config = self.load_config()
        pin_model, pin_data = self.load_pinocchio_model()

        joint_names = list(config['active_joint_names'])
        joint_params: List[JointParameters] = []
        joint_limits: Dict[str, Dict[str, float]] = {}

        for local_index, joint_name in enumerate(joint_names, start=1):
            joint_id = pin_model.getJointId(joint_name)
            if joint_id == 0:
                raise ValueError(f"Joint {joint_name} not found in URDF")

            inertia = pin_model.inertias[joint_id]
            limit_index = local_index - 1
            joint_limits[joint_name] = {
                'lower': float(pin_model.lowerPositionLimit[limit_index]),
                'upper': float(pin_model.upperPositionLimit[limit_index]),
                'velocity': float(pin_model.velocityLimit[limit_index]),
                'effort': float(pin_model.effortLimit[limit_index]),
            }
            joint_params.append(
                JointParameters(
                    name=joint_name,
                    mass=float(inertia.mass),
                    com=np.array(inertia.lever, dtype=float),
                    inertia_matrix=np.array(inertia.inertia, dtype=float),
                    damping=float(pin_model.damping[limit_index]),
                    friction=float(pin_model.friction[limit_index]),
                )
            )

        return RobotModel(
            name=config['robot_name'],
            num_joints=len(joint_names),
            joint_names=joint_names,
            joint_params=joint_params,
            pinocchio_model=pin_model,
            pinocchio_data=pin_data,
            joint_limits=joint_limits,
            base_link=config['base_link'],
            ee_link=config['ee_link'],
            description_source=config.get('description_source', 'unknown'),
        )


def print_model_info(model: RobotModel):
    """Print the key dynamic quantities used for identification."""
    print(f"\n{'=' * 70}")
    print(f"Robot Model: {model.name}")
    print(f"Source: {model.description_source}")
    print(f"Active Joints: {model.num_joints}")
    print(f"Base Link: {model.base_link}")
    print(f"End Effector: {model.ee_link}")
    print(f"{'=' * 70}")

    for index, jp in enumerate(model.joint_params, start=1):
        diag_inertia = np.diag(jp.inertia_matrix)
        print(f"\n[Joint {index}] {jp.name}")
        print(f"  Mass: {jp.mass:.6f} kg")
        print(f"  CoM: {np.array2string(jp.com, precision=6)}")
        print(f"  Inertia diag: {np.array2string(diag_inertia, precision=6)}")
        print(f"  Damping: {jp.damping:.6f}")
        print(f"  Friction: {jp.friction:.6f}")

    print(f"\nFull parameter vector length: {len(model.full_parameter_vector())}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    urdf_path = project_root / "models" / "urdf" / "panda_arm_minimal.urdf"
    config_path = project_root / "models" / "configs" / "panda_config.yaml"
    loader = URDFLoader(str(urdf_path), str(config_path))
    model = loader.build_robot_model()
    print_model_info(model)
