from __future__ import annotations

from g7_openarm_config import GeneralConfig
from g7_openarm_utils import (
    LEFT_GRIPPER_MOTOR_NAME,
    MODEL_JOINTS_BY_MOTOR_NAME,
    MOTOR_NAMES,
    RIGHT_GRIPPER_MOTOR_NAME,
    motor_command,
    motor_index,
)

from .actuation import (
    gripper_command_to_mujoco_position,
    gripper_command_velocity_to_mujoco_velocity,
    motor_actuation_enabled,
    mujoco_gripper_position_to_command,
    mujoco_gripper_velocity_to_command_velocity,
)

BASE_VISUALIZATION_Z_M = 0.160631
_GRIPPER_MOTOR_NAMES = frozenset(
    (LEFT_GRIPPER_MOTOR_NAME, RIGHT_GRIPPER_MOTOR_NAME)
)


def write_floating_base_qpos(layout, data, odom, *, z: float = BASE_VISUALIZATION_Z_M) -> None:
    """Write floating-base qpos through named q components."""
    values = {
        "x": odom.position.x,
        "y": odom.position.y,
        "z": z,
        "q_w": odom.quaternion.w,
        "q_x": odom.quaternion.x,
        "q_y": odom.quaternion.y,
        "q_z": odom.quaternion.z,
    }
    for name, value in values.items():
        data.qpos[layout.qpos_index(name)] = value


def write_lowstate_qpos(layout, data, lowstate) -> None:
    """Mirror Unitree-style LowState into MuJoCo by semantic motor/joint names."""
    for motor_name in MOTOR_NAMES:
        value = lowstate.motor_state[motor_index(motor_name)].q
        if motor_name in _GRIPPER_MOTOR_NAMES:
            value = gripper_command_to_mujoco_position(value)

        for joint_name in MODEL_JOINTS_BY_MOTOR_NAME[motor_name]:
            data.qpos[layout.qpos_index(joint_name)] = value


def write_lowstate_from_mujoco(layout, data, lowstate) -> None:
    """Copy MuJoCo q/v/actuator force into LowState using logical motor names."""
    for motor_name in MOTOR_NAMES:
        motor = layout.motor(motor_name)
        joint = motor.primary_joint
        position = data.qpos[joint.qpos_index]
        velocity = data.qvel[joint.qvel_index]

        if motor_name in _GRIPPER_MOTOR_NAMES:
            position = mujoco_gripper_position_to_command(position)
            velocity = mujoco_gripper_velocity_to_command_velocity(velocity)

        state = lowstate.motor_state[motor_index(motor_name)]
        state.q = position
        state.dq = velocity
        # Preserve the model's named jointactuatorfrc sensor behavior, including
        # any configured sensor noise, instead of bypassing it via qfrc_actuator.
        state.tau_est = data.sensordata[motor.torque_sensor_address]


def apply_lowcmd_to_mujoco(layout, data, lowcmd, general: GeneralConfig) -> None:
    """Apply LowCmd PD+feed-forward torques through name-resolved actuators."""
    for motor_name in MOTOR_NAMES:
        command = motor_command(lowcmd, motor_name)
        enabled = motor_actuation_enabled(motor_name, general)

        q_target = command.q
        dq_target = command.dq
        if motor_name in _GRIPPER_MOTOR_NAMES:
            q_target = gripper_command_to_mujoco_position(q_target)
            dq_target = gripper_command_velocity_to_mujoco_velocity(dq_target)

        for joint in layout.motor(motor_name).joints:
            if not enabled:
                data.ctrl[joint.actuator_id] = 0.0
                continue

            q_error = q_target - data.qpos[joint.qpos_index]
            dq_error = dq_target - data.qvel[joint.qvel_index]
            data.ctrl[joint.actuator_id] = (
                q_error * command.kp + dq_error * command.kd + command.tau
            )
