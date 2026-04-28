"""
从 URDF 构建辨识流水线使用的机器人模型对象。

这个模块对应整条流水线的 Step 1，由 `run_pipeline.py` 最先调用。它负责读取
URDF（以及可选 yaml 配置），提取关节动力学先验、构建 Pinocchio model/data，
并把这些信息统一封装成 `RobotModel`，供下游 golden data 生成、参数辨识和残差补偿
模块共享。
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def _candidate_pinocchio_paths() -> List[Path]:
    """Return likely site-packages locations for the pinocchio_py conda env."""
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    home = Path.home()
    candidates = [
        home / 'miniconda3' / 'envs' / 'pinocchio_py' / 'lib' / py_ver / 'site-packages',
        home / 'anaconda3' / 'envs' / 'pinocchio_py' / 'lib' / py_ver / 'site-packages',
    ]

    env_prefix = Path(sys.prefix)
    if env_prefix.name == 'pinocchio_py':
        candidates.insert(0, env_prefix / 'lib' / py_ver / 'site-packages')

    return [path for path in candidates if path.exists()]


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
            for candidate in _candidate_pinocchio_paths():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                try:
                    return importlib.import_module('pinocchio')
                except ImportError:
                    continue

            print(
                "Warning: pinocchio not installed in the current interpreter and "
                "was not found under the 'pinocchio_py' conda environment. "
                "Install with: pip install pinocchio"
            )
            return None


pin = _import_pinocchio()


def parse_gravity_vector(
    value: Union[str, np.ndarray, list, None],
) -> np.ndarray:
    """
    把各种重力输入格式统一解析为 shape-(3,) 的浮点向量。

    Parameters
    ----------
    value : Union[str, np.ndarray, list, None]
        支持预设关键字、逗号分隔字符串、长度为 3 的数组/列表，或 `None`。

    Returns
    -------
    np.ndarray
        基坐标系下的重力向量，单位 m/s²。

    Raises
    ------
    ValueError
        当输入格式非法、长度不为 3 或关键字不受支持时抛出。
    """
    presets = {
        'upright': np.array([0.0, 0.0, -9.81], dtype=np.float64),
        'default': np.array([0.0, 0.0, -9.81], dtype=np.float64),
        'inverted': np.array([0.0, 0.0, 9.81], dtype=np.float64),
        'upside_down': np.array([0.0, 0.0, 9.81], dtype=np.float64),
        'wall_x': np.array([-9.81, 0.0, 0.0], dtype=np.float64),
        'wall_x_pos': np.array([9.81, 0.0, 0.0], dtype=np.float64),
        'wall_y': np.array([0.0, -9.81, 0.0], dtype=np.float64),
        'wall_y_pos': np.array([0.0, 9.81, 0.0], dtype=np.float64),
    }
    preset_help = ", ".join(sorted(presets.keys()))

    if value is None:
        return presets['default'].copy()

    if isinstance(value, (list, np.ndarray)):
        gravity_vector = np.asarray(value, dtype=np.float64)
        if gravity_vector.shape != (3,):
            raise ValueError(f"Gravity vector must have length 3, got shape {gravity_vector.shape}.")
        return gravity_vector

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in presets:
            return presets[normalized].copy()
        try:
            gravity_vector = np.asarray([float(part.strip()) for part in value.split(',')], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(
                f"Invalid gravity string '{value}'. Supported presets: {preset_help}"
            ) from exc
        if gravity_vector.shape != (3,):
            raise ValueError(
                f"Gravity vector string '{value}' must contain exactly 3 comma-separated values. "
                f"Supported presets: {preset_help}"
            )
        return gravity_vector

    raise ValueError(f"Unsupported gravity vector type: {type(value)!r}")


@dataclass
class JointParameters:
    """
    单个主动关节对应的动力学先验参数。

    这里保存的质量、质心、惯量、阻尼和摩擦，都是从 URDF / Pinocchio 模型里直接读出的
    “当前模型先验”，主要用于建模和初始化，而不应在真实数据场景里被误当成辨识真值。
    """
    name: str
    mass: float
    com: np.ndarray
    inertia_matrix: np.ndarray
    damping: float
    friction: float


@dataclass
class RobotModel:
    """
    流水线共享的机器人模型封装。

    这个类把 Pinocchio model/data、关节列表、关节限位、动力学先验和重力方向集中到一起，
    目的是让下游模块都通过同一个对象获取模型信息，而不是各自再去读 URDF。关键状态是
    `pinocchio_model`、`pinocchio_data` 和 `gravity_vector`；只要这里的重力向量被写入
    `pinocchio_model.gravity.linear`，所有 Pinocchio 动力学函数都会自动继承同一重力设定。
    """
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
    gravity_vector: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """在 dataclass 初始化后规范化重力向量并同步到 Pinocchio model。"""
        self.gravity_vector = parse_gravity_vector(self.gravity_vector)
        self._apply_gravity_to_pinocchio()

    def _apply_gravity_to_pinocchio(self) -> None:
        """把当前重力向量写回 `pinocchio_model.gravity.linear`。"""
        if pin is None or self.pinocchio_model is None:
            return
        self.pinocchio_model.gravity.linear = self.gravity_vector.copy()

    def set_gravity(self, gravity_vector: Union[str, np.ndarray, list]) -> None:
        """
        Update the gravity vector at runtime.

        Common presets:
        - 'upright' / 'default'
        - 'inverted' / 'upside_down'
        - 'wall_x', 'wall_x_pos'
        - 'wall_y', 'wall_y_pos'

        Example:
        - model.set_gravity('inverted')
        - model.set_gravity([0.0, -9.81, 0.0])

        Parameters
        ----------
        gravity_vector : Union[str, np.ndarray, list]
            新的重力向量配置，可以是关键字，也可以是三维向量。

        Returns
        -------
        None
            结果直接更新到当前模型对象和 Pinocchio model 中。
        """
        self.gravity_vector = parse_gravity_vector(gravity_vector)
        self._apply_gravity_to_pinocchio()
        print(f"Gravity vector updated: {self.gravity_vector}")

    @property
    def damping(self) -> np.ndarray:
        """按关节顺序返回粘性阻尼向量。"""
        return np.array([jp.damping for jp in self.joint_params], dtype=float)

    @property
    def friction(self) -> np.ndarray:
        """按关节顺序返回库仑摩擦先验向量。"""
        return np.array([jp.friction for jp in self.joint_params], dtype=float)

    def inertial_parameter_vector(self) -> np.ndarray:
        """
        按 Pinocchio 的动态参数顺序拼接所有主动关节的刚体参数。

        Returns
        -------
        np.ndarray
            所有关节刚体参数拼接后的长向量。
        """
        params = []
        for joint_id in range(1, self.pinocchio_model.njoints):
            params.append(np.array(self.pinocchio_model.inertias[joint_id].toDynamicParameters(), dtype=float))
        return np.concatenate(params)

    def full_parameter_vector(self) -> np.ndarray:
        """
        生成“刚体参数 + 阻尼 + 摩擦”的完整参数向量。

        Returns
        -------
        np.ndarray
            供 synthetic 参考值或调试用途使用的完整参数向量。
        """
        return np.concatenate([self.inertial_parameter_vector(), self.damping, self.friction])


class URDFLoader:
    """
    负责从 URDF / 可选配置文件构建 `RobotModel`。

    设计上把“文件读取、元信息推断、Pinocchio 构建、先验参数抽取”集中在同一个类里，
    是为了让主流程只需要关心“给一个路径，拿回一个可用模型”，而不用散落地处理文件系统、
    XML 解析和 Pinocchio 细节。
    """

    def __init__(
        self,
        urdf_path: str,
        config_path: str | None = None,
        gravity_vector: Union[str, np.ndarray, list, None] = None,
    ):
        """
        保存模型文件路径与重力配置，并做基本存在性校验。

        Parameters
        ----------
        urdf_path : str
            URDF 文件路径。
        config_path : str | None
            可选 yaml 配置路径。
        gravity_vector : Union[str, np.ndarray, list, None]
            基坐标系下的重力方向配置。

        Raises
        ------
        FileNotFoundError
            当 URDF 或配置文件不存在时抛出。
        """
        self.urdf_path = Path(urdf_path)
        self.config_path = Path(config_path) if config_path else None
        self.gravity_vector = parse_gravity_vector(gravity_vector)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        if self.config_path is not None and not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    def _infer_metadata_from_urdf(self) -> Dict:
        """
        当外部 yaml 配置不存在时，直接从 URDF 推断最基本的元信息。

        推断规则尽量保持朴素稳定：
        - robot_name: 取 `<robot name="...">`
        - active_joint_names: 取所有非 fixed 的 joint，保持 URDF 中出现顺序
        - base_link: 取“不是任何 joint child”的 link
        - ee_link:   取“不是任何 joint parent”的 link
        """
        root = ET.parse(self.urdf_path).getroot()
        robot_name = root.attrib.get('name', self.urdf_path.stem)

        all_links = [link.attrib['name'] for link in root.findall('link') if 'name' in link.attrib]
        movable_joints = []
        parent_links = set()
        child_links = set()

        for joint in root.findall('joint'):
            joint_name = joint.attrib.get('name')
            joint_type = joint.attrib.get('type', '')
            parent = joint.find('parent')
            child = joint.find('child')

            if parent is not None and 'link' in parent.attrib:
                parent_links.add(parent.attrib['link'])
            if child is not None and 'link' in child.attrib:
                child_links.add(child.attrib['link'])

            if joint_type != 'fixed' and joint_name:
                movable_joints.append(joint_name)

        base_candidates = [name for name in all_links if name not in child_links]
        ee_candidates = [name for name in all_links if name not in parent_links]

        return {
            'robot_name': robot_name,
            'active_joint_names': movable_joints,
            'base_link': base_candidates[0] if base_candidates else (all_links[0] if all_links else 'base_link'),
            'ee_link': ee_candidates[-1] if ee_candidates else (all_links[-1] if all_links else 'ee_link'),
            'description_source': str(self.urdf_path),
        }

    def load_config(self) -> Dict:
        """
        读取配置文件，并在必要时退回到 URDF 推断元信息。

        Returns
        -------
        Dict
            包含机器人名称、主动关节、基座/末端 link 等信息的配置字典。
        """
        inferred = self._infer_metadata_from_urdf()
        if self.config_path is None:
            return inferred

        with open(self.config_path, 'r', encoding='utf-8') as handle:
            loaded = yaml.safe_load(handle) or {}

        # 兼容一些“只有 joint 名单”的轻量 yaml。
        if 'controller_joint_names' in loaded and 'active_joint_names' not in loaded:
            loaded['active_joint_names'] = [
                name for name in loaded['controller_joint_names']
                if isinstance(name, str) and name.strip()
            ]

        merged = {**inferred, **loaded}
        return merged

    def load_pinocchio_model(self) -> Tuple[object, object]:
        """
        从 URDF 构建 Pinocchio model 和 data。

        Returns
        -------
        Tuple[object, object]
            `pin.Model` 与对应的 `pin.Data`。

        Raises
        ------
        ImportError
            当 pinocchio 不可用时抛出。
        """
        if pin is None:
            raise ImportError("pinocchio is required. Install with: pip install pinocchio")
        model = pin.buildModelFromUrdf(str(self.urdf_path))
        data = model.createData()
        return model, data

    def build_robot_model(self) -> RobotModel:
        """
        聚合配置、Pinocchio 模型和先验参数，构建统一的 `RobotModel`。

        Returns
        -------
        RobotModel
            可直接供整条辨识流水线复用的模型对象。

        Raises
        ------
        ValueError
            当配置里声明的关节在 URDF 中找不到时抛出。
        """
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
            gravity_vector=self.gravity_vector,
        )


def print_model_info(model: RobotModel):
    """
    打印辨识主流程最关心的模型摘要信息。

    Parameters
    ----------
    model : RobotModel
        已构建完成的机器人模型对象。

    Returns
    -------
    None
        结果直接打印到终端。
    """
    print(f"\n{'=' * 70}")
    print(f"Robot Model: {model.name}")
    print(f"Source: {model.description_source}")
    print(f"Active Joints: {model.num_joints}")
    print(f"Base Link: {model.base_link}")
    print(f"End Effector: {model.ee_link}")
    print(f"Gravity Vector: {model.gravity_vector}  (base frame, m/s²)")
    print(f"{'=' * 70}")
    print("The inertial / damping / friction values shown below are the current URDF prior values.")
    print("For real-robot identification, they should not be interpreted as identified ground truth.")

    for index, jp in enumerate(model.joint_params, start=1):
        diag_inertia = np.diag(jp.inertia_matrix)
        print(f"\n[Joint {index}] {jp.name}")
        print(f"  Mass: {jp.mass:.6f} kg")
        print(f"  CoM: {np.array2string(jp.com, precision=6)}")
        print(f"  Inertia diag: {np.array2string(diag_inertia, precision=6)}")
        print(f"  Damping: {jp.damping:.6f}")
        print(f"  Friction: {jp.friction:.6f}")

    print(f"\nFull parameter vector length: {len(model.full_parameter_vector())}")
    if np.allclose(model.damping, 0.0) and np.allclose(model.friction, 0.0):
        print("Note: all damping/friction values are currently 0 in the URDF.")
        print("      This is acceptable for a real robot model before friction-related parameters are identified.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    urdf_path = project_root / "models" / "05_urdf" / "urdf" / "05_urdf_temp.urdf"
    loader = URDFLoader(str(urdf_path))
    model = loader.build_robot_model()
    print_model_info(model)
