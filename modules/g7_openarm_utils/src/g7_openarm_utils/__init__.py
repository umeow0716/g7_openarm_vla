from .arm_limits import (
    ARM_JOINT_NAMES,
    ARM_LOWSTATE_MOTOR_INDICES,
    ARM_MOTOR16_INDICES,
    ARM_POSITION_LOWER_RAD,
    ARM_POSITION_UPPER_RAD,
    ARM_VELOCITY_LIMIT_RAD_S,
    position_limited_velocity_bounds,
)
from .gripper import (
    GRIPPER_COMMAND_RANGE,
    gripper_command_to_motor_position,
    gripper_command_velocity_to_motor_velocity,
    gripper_command_to_openness,
    gripper_motor_position_to_command,
    gripper_motor_velocity_to_command_velocity,
    gripper_openness_to_command,
)
from .idl import array_to_pose, pose_to_array
from .quat import (
    quat_conj,
    quat_from_yaw,
    quat_mul,
    quat_normalize,
    quat_rotate,
    quat_to_rotation_matrix,
    quat_yaw,
)


def load_hand_default_pose(model_path: str):
    # Keep MuJoCo optional for lowlevel/WBC users of this package.
    from .mujoco import load_hand_default_pose as _load_hand_default_pose

    return _load_hand_default_pose(model_path)


__all__ = [
    "ARM_JOINT_NAMES",
    "ARM_LOWSTATE_MOTOR_INDICES",
    "ARM_MOTOR16_INDICES",
    "ARM_POSITION_LOWER_RAD",
    "ARM_POSITION_UPPER_RAD",
    "ARM_VELOCITY_LIMIT_RAD_S",
    "GRIPPER_COMMAND_RANGE",
    "array_to_pose",
    "gripper_command_to_motor_position",
    "gripper_command_velocity_to_motor_velocity",
    "gripper_command_to_openness",
    "gripper_motor_position_to_command",
    "gripper_motor_velocity_to_command_velocity",
    "gripper_openness_to_command",
    "load_hand_default_pose",
    "pose_to_array",
    "position_limited_velocity_bounds",
    "quat_conj",
    "quat_from_yaw",
    "quat_mul",
    "quat_normalize",
    "quat_rotate",
    "quat_to_rotation_matrix",
    "quat_yaw",
]
