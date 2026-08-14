from g7_openarm_config import GeneralConfig
from g7_openarm_utils import GRIPPER_COMMAND_RANGE, gripper_openness_to_command

MUJOCO_GRIPPER_OPEN_DISTANCE_M = 0.045


def motor_actuation_enabled(index: int, general: GeneralConfig) -> bool:
    if not 0 <= index < 24:
        raise ValueError(f"motor index must be in [0, 23], got {index}")
    if index < 8:
        return general.base_actuation_enabled
    if index < 16:
        return general.left_arm_actuation_enabled
    return general.right_arm_actuation_enabled


def gripper_command_to_mujoco_position(command: float) -> float:
    openness = 1.0 - float(command) / GRIPPER_COMMAND_RANGE
    return openness * MUJOCO_GRIPPER_OPEN_DISTANCE_M


def gripper_command_velocity_to_mujoco_velocity(velocity: float) -> float:
    return -float(velocity) * MUJOCO_GRIPPER_OPEN_DISTANCE_M / GRIPPER_COMMAND_RANGE


def mujoco_gripper_position_to_command(position: float) -> float:
    openness = float(position) / MUJOCO_GRIPPER_OPEN_DISTANCE_M
    openness = min(max(openness, 0.0), 1.0)
    return gripper_openness_to_command(openness)


def mujoco_gripper_velocity_to_command_velocity(velocity: float) -> float:
    return -float(velocity) * GRIPPER_COMMAND_RANGE / MUJOCO_GRIPPER_OPEN_DISTANCE_M
