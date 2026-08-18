from .arm_limits import (
    ARM_POSITION_LOWER_BY_JOINT_NAME,
    ARM_POSITION_LOWER_RAD,
    ARM_POSITION_UPPER_BY_JOINT_NAME,
    ARM_POSITION_UPPER_RAD,
    ARM_VELOCITY_LIMIT_BY_JOINT_NAME,
    ARM_VELOCITY_LIMIT_RAD_S,
    arm_limit_arrays,
    position_limited_velocity_bounds,
)
from .joint_layout import (
    ACTUATED_MODEL_JOINT_NAMES,
    AMR_COMMAND_INDEX_BY_NAME,
    AMR_COMMAND_NAMES,
    ARM_COMMAND_INDEX_BY_NAME,
    ARM_COMMAND_MOTOR_NAMES,
    ARM_JOINT_NAMES,
    ARM_LOWSTATE_MOTOR_INDICES,
    ARM_MOTOR16_INDICES,
    ARM_MOTOR_NAMES,
    BASE_MOTOR_NAMES,
    BASE_STEER_MOTOR_NAMES,
    BASE_WHEEL_MOTOR_NAMES,
    FLOATING_BASE_CONFIG_NAMES,
    FLOATING_BASE_JOINT_NAME,
    FLOATING_BASE_VELOCITY_NAMES,
    LEFT_ARM_JOINT_NAMES,
    LEFT_ARM_MOTOR_NAMES,
    LEFT_GRIPPER_MOTOR_NAME,
    LEFT_HARDWARE_MOTOR_NAMES,
    MODEL_JOINTS_BY_MOTOR_NAME,
    MODEL_JOINT_TO_MOTOR_NAME,
    MOTOR_INDEX_BY_NAME,
    MOTOR_NAMES,
    PRIMARY_MODEL_JOINT_BY_MOTOR_NAME,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_ARM_MOTOR_NAMES,
    RIGHT_GRIPPER_MOTOR_NAME,
    RIGHT_HARDWARE_MOTOR_NAMES,
    UNITREE_HG_MOTOR_ARRAY_SIZE,
    amr_command_index,
    amr_command_values,
    arm_command_index,
    arm_command_indices,
    arm_command_values,
    motor_command,
    motor_index,
    motor_indices,
    motor_state_values,
)
from .gripper import (
    GRIPPER_COMMAND_RANGE,
    GRIPPER_MODEL_OPEN_DISTANCE_M,
    gripper_command_to_model_position,
    gripper_command_to_motor_position,
    gripper_command_velocity_to_model_velocity,
    gripper_command_velocity_to_motor_velocity,
    gripper_command_to_openness,
    gripper_model_position_to_command,
    gripper_motor_position_to_command,
    gripper_model_velocity_to_command_velocity,
    gripper_motor_velocity_to_command_velocity,
    gripper_openness_to_command,
)
from .quat import (
    quat_conj,
    quat_from_yaw,
    quat_mul,
    quat_normalize,
    quat_rotate,
    quat_to_rotation_matrix,
    quat_yaw,
)


def array_to_pose(array):
    # Keep CycloneDDS/Unitree IDL optional for pure layout/PinnZoo tooling.
    from .idl import array_to_pose as _array_to_pose

    return _array_to_pose(array)


def pose_to_array(pose):
    # Keep CycloneDDS/Unitree IDL optional for pure layout/PinnZoo tooling.
    from .idl import pose_to_array as _pose_to_array

    return _pose_to_array(pose)


def load_hand_default_pose(model_path: str):
    # Keep MuJoCo optional for lowlevel/WBC users of this package.
    from .mujoco import load_hand_default_pose as _load_hand_default_pose

    return _load_hand_default_pose(model_path)


__all__ = [
    "ACTUATED_MODEL_JOINT_NAMES",
    "AMR_COMMAND_INDEX_BY_NAME",
    "AMR_COMMAND_NAMES",
    "ARM_COMMAND_INDEX_BY_NAME",
    "ARM_COMMAND_MOTOR_NAMES",
    "ARM_MOTOR_NAMES",
    "BASE_MOTOR_NAMES",
    "BASE_STEER_MOTOR_NAMES",
    "BASE_WHEEL_MOTOR_NAMES",
    "FLOATING_BASE_CONFIG_NAMES",
    "FLOATING_BASE_JOINT_NAME",
    "FLOATING_BASE_VELOCITY_NAMES",
    "LEFT_ARM_JOINT_NAMES",
    "LEFT_ARM_MOTOR_NAMES",
    "LEFT_GRIPPER_MOTOR_NAME",
    "LEFT_HARDWARE_MOTOR_NAMES",
    "MODEL_JOINTS_BY_MOTOR_NAME",
    "MODEL_JOINT_TO_MOTOR_NAME",
    "MOTOR_INDEX_BY_NAME",
    "MOTOR_NAMES",
    "PRIMARY_MODEL_JOINT_BY_MOTOR_NAME",
    "RIGHT_ARM_JOINT_NAMES",
    "RIGHT_ARM_MOTOR_NAMES",
    "RIGHT_GRIPPER_MOTOR_NAME",
    "RIGHT_HARDWARE_MOTOR_NAMES",
    "UNITREE_HG_MOTOR_ARRAY_SIZE",
    "amr_command_index",
    "amr_command_values",
    "arm_command_index",
    "arm_command_indices",
    "arm_command_values",
    "motor_command",
    "motor_index",
    "motor_indices",
    "motor_state_values",
    "ARM_JOINT_NAMES",
    "ARM_LOWSTATE_MOTOR_INDICES",
    "ARM_MOTOR16_INDICES",
    "ARM_POSITION_LOWER_BY_JOINT_NAME",
    "ARM_POSITION_LOWER_RAD",
    "ARM_POSITION_UPPER_BY_JOINT_NAME",
    "ARM_POSITION_UPPER_RAD",
    "ARM_VELOCITY_LIMIT_BY_JOINT_NAME",
    "ARM_VELOCITY_LIMIT_RAD_S",
    "arm_limit_arrays",
    "GRIPPER_COMMAND_RANGE",
    "GRIPPER_MODEL_OPEN_DISTANCE_M",
    "array_to_pose",
    "gripper_command_to_model_position",
    "gripper_command_to_motor_position",
    "gripper_command_velocity_to_model_velocity",
    "gripper_command_velocity_to_motor_velocity",
    "gripper_command_to_openness",
    "gripper_model_position_to_command",
    "gripper_motor_position_to_command",
    "gripper_model_velocity_to_command_velocity",
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
