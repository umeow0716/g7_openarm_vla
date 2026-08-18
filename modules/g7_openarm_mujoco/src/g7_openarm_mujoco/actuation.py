from g7_openarm_config import GeneralConfig
from g7_openarm_utils import (
    BASE_MOTOR_NAMES,
    GRIPPER_MODEL_OPEN_DISTANCE_M,
    LEFT_HARDWARE_MOTOR_NAMES,
    RIGHT_HARDWARE_MOTOR_NAMES,
    gripper_command_to_model_position,
    gripper_command_velocity_to_model_velocity,
    gripper_model_position_to_command,
    gripper_model_velocity_to_command_velocity,
)

MUJOCO_GRIPPER_OPEN_DISTANCE_M = GRIPPER_MODEL_OPEN_DISTANCE_M

_BASE_MOTORS = frozenset(BASE_MOTOR_NAMES)
_LEFT_MOTORS = frozenset(LEFT_HARDWARE_MOTOR_NAMES)
_RIGHT_MOTORS = frozenset(RIGHT_HARDWARE_MOTOR_NAMES)


def motor_actuation_enabled(motor_name: str, general: GeneralConfig) -> bool:
    """Return actuation enable state by semantic motor group, never by array slot."""
    if motor_name in _BASE_MOTORS:
        return general.base_actuation_enabled
    if motor_name in _LEFT_MOTORS:
        return general.left_arm_actuation_enabled
    if motor_name in _RIGHT_MOTORS:
        return general.right_arm_actuation_enabled
    raise KeyError(f"Unknown logical motor name: {motor_name}")


# Backward-compatible MuJoCo-specific names. The conversion is model-space,
# not MuJoCo-order-specific, so PinnZoo can use the same functions.
gripper_command_to_mujoco_position = gripper_command_to_model_position
gripper_command_velocity_to_mujoco_velocity = gripper_command_velocity_to_model_velocity
mujoco_gripper_position_to_command = gripper_model_position_to_command
mujoco_gripper_velocity_to_command_velocity = gripper_model_velocity_to_command_velocity
