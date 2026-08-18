import atexit
import signal
import time
from typing import Any, TypedDict

import damiao_can as dc
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__IMUState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_
from unitree_sdk2py.utils.hz_sample import RecurrentThread

from g7_openarm_config import general_config
from g7_openarm_utils import (
    BASE_MOTOR_NAMES,
    BASE_WHEEL_MOTOR_NAMES,
    LEFT_GRIPPER_MOTOR_NAME,
    LEFT_HARDWARE_MOTOR_NAMES,
    RIGHT_GRIPPER_MOTOR_NAME,
    RIGHT_HARDWARE_MOTOR_NAMES,
    gripper_command_to_motor_position,
    gripper_command_velocity_to_motor_velocity,
    gripper_motor_position_to_command,
    gripper_motor_velocity_to_command_velocity,
    motor_command,
    motor_index,
)

from .config import config


BASE_CONFIG_INDEX_BY_NAME = {name: index for index, name in enumerate(BASE_MOTOR_NAMES)}
LEFT_HARDWARE_INDEX_BY_NAME = {
    name: index for index, name in enumerate(LEFT_HARDWARE_MOTOR_NAMES)
}
RIGHT_HARDWARE_INDEX_BY_NAME = {
    name: index for index, name in enumerate(RIGHT_HARDWARE_MOTOR_NAMES)
}


def _base_config_index(name: str) -> int:
    return BASE_CONFIG_INDEX_BY_NAME[name]


def _base_motor_id(name: str) -> int:
    return config.base_ids[_base_config_index(name)]


def _base_direction(name: str) -> float:
    return config.base_direction[_base_config_index(name)]


def _arm_direction(name: str, *, left: bool) -> float:
    index_by_name = LEFT_HARDWARE_INDEX_BY_NAME if left else RIGHT_HARDWARE_INDEX_BY_NAME
    directions = config.left_arm_direction if left else config.right_arm_direction
    return directions[index_by_name[name]]


class BusConfig(TypedDict):
    motor_types: list[Any]
    send_ids: list[int]
    recv_ids: list[int]
    control_modes: list[Any]


def _base_bus_config() -> BusConfig:
    control_modes: list[Any] = [dc.ControlMode.POS_VEL] * len(BASE_MOTOR_NAMES)
    for name in BASE_MOTOR_NAMES:
        motor_id = _base_motor_id(name)
        control_modes[motor_id - 1] = (
            dc.ControlMode.VEL if name in BASE_WHEEL_MOTOR_NAMES else dc.ControlMode.POS_VEL
        )

    return {
        "motor_types": [
            dc.MotorType.DM8009,
            dc.MotorType.DM6006,
            dc.MotorType.DM8009,
            dc.MotorType.DM6006,
            dc.MotorType.DM8009,
            dc.MotorType.DM6006,
            dc.MotorType.DM8009,
            dc.MotorType.DM6006,
        ],
        "send_ids": [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        "recv_ids": [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18],
        "control_modes": control_modes,
    }


def _arm_bus_config() -> BusConfig:
    return {
        "motor_types": [
            dc.MotorType.DM8009,
            dc.MotorType.DM8009,
            dc.MotorType.DM4340,
            dc.MotorType.DM4340,
            dc.MotorType.DM4310,
            dc.MotorType.DM4310,
            dc.MotorType.DM4310,
            dc.MotorType.DM4310,
        ],
        "send_ids": [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        "recv_ids": [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18],
        "control_modes": [dc.ControlMode.MIT] * 8,
    }


def build_bus_configs(*, base_enabled: bool, arms_enabled: bool = True) -> dict[str, BusConfig]:
    bus_configs: dict[str, BusConfig] = {}
    if base_enabled:
        bus_configs[config.base_can] = _base_bus_config()

    if arms_enabled:
        bus_configs[config.left_arm_can] = _arm_bus_config()
        bus_configs[config.right_arm_can] = _arm_bus_config()
    return bus_configs


def mapping_gripper(val: float, open_val: float, close_val: float) -> float:
    return gripper_command_to_motor_position(
        val,
        open_position=open_val,
        close_position=close_val,
    )


class HardwareNode:
    def __init__(self) -> None:
        self.base_enabled = general_config.base_actuation_enabled
        self.left_arm_command_enabled = general_config.left_arm_actuation_enabled
        self.right_arm_command_enabled = general_config.right_arm_actuation_enabled
        self.arms_enabled = general_config.arm_actuation_enabled
        self.bus_configs = build_bus_configs(
            base_enabled=self.base_enabled,
            arms_enabled=self.arms_enabled,
        )

        can_interfaces = list(self.bus_configs)
        self.group = dc.DamiaoCANGroup(can_interfaces=can_interfaces, enable_fd=config.can_fd)
        for can_interface, can_config in self.bus_configs.items():
            device = self.group.get_device(can_interface)
            device.init_motors(
                can_config["motor_types"],
                can_config["send_ids"],
                can_config["recv_ids"],
                can_config["control_modes"],
            )
            device.set_callback_mode_all(dc.CallbackMode.STATE)

            print(f"{can_interface}: expected responses = {device.expected_response_count()}")

        if not self.base_enabled:
            print(f"Control mode {general_config.control_mode.value}: skipping {config.base_can}")
        if not self.arms_enabled:
            print(
                f"Control mode {general_config.control_mode.value}: skipping "
                f"{config.left_arm_can} and {config.right_arm_can}"
            )
        else:
            if not self.left_arm_command_enabled:
                print(
                    f"Control mode {general_config.control_mode.value}: "
                    f"{config.left_arm_can} is state-only (no MIT commands)"
                )
            if not self.right_arm_command_enabled:
                print(
                    f"Control mode {general_config.control_mode.value}: "
                    f"{config.right_arm_can} is state-only (no MIT commands)"
                )

        self.group.enable_all()

        # Register cleanup immediately after enabling motors. Any later DDS or
        # thread-construction failure must still disable the hardware at exit.
        self._cleanup_done = False
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        atexit.register(self.cleanup)

        self.lowstate = unitree_hg_msg_dds__LowState_()
        self.lowstate_publisher = ChannelPublisher("rt/lowstate", LowState_)
        self.lowstate_publisher.Init()

        self.lowcmd = unitree_hg_msg_dds__LowCmd_()
        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_subscriber.Init(self.lowcmd_handler, 0)

        self.imustate = unitree_hg_msg_dds__IMUState_()
        self.imustate.quaternion[0] = 1.0
        self.lowstate.imu_state = self.imustate

        self.imustate_subscriber = ChannelSubscriber("rt/imustate", IMUState_)
        self.imustate_subscriber.Init(self.imustate_handler, 0)

        self.control_thread = RecurrentThread(
            name="control_thread",
            target=self.control_loop,
            interval=config.interval,
        )

        self.control_thread.Start()

    def cleanup(self) -> None:
        if self._cleanup_done:
            return

        self._cleanup_done = True
        self.group.disable_all()

    def _handle_sigterm(self, _signum: int, _frame: object) -> None:
        raise SystemExit(0)

    def lowcmd_handler(self, msg: LowCmd_) -> None:
        self.lowcmd = msg

    def imustate_handler(self, msg: IMUState_) -> None:
        self.imustate = msg
        self.lowstate.imu_state = self.imustate

    def _read_base_state(self) -> None:
        base = self.group.get_device(config.base_can)
        base_motors = base.get_motors()

        for name in BASE_MOTOR_NAMES:
            motor = base_motors[_base_motor_id(name) - 1]
            direction = _base_direction(name)
            state = self.lowstate.motor_state[motor_index(name)]
            state.q = motor.get_position() * direction
            state.dq = motor.get_velocity() * direction
            state.tau_est = motor.get_torque() * direction

    def _write_base_command(self) -> None:
        base = self.group.get_device(config.base_can)
        for name in BASE_MOTOR_NAMES:
            command = motor_command(self.lowcmd, name)
            direction = _base_direction(name)
            if name in BASE_WHEEL_MOTOR_NAMES:
                cmd = dc.VelParam(dq=command.dq * direction)
                base.vel_control_one(_base_motor_id(name) - 1, cmd)
            else:
                cmd = dc.PosVelParam(q=command.q * direction, dq=20.0)
                base.posvel_control_one(_base_motor_id(name) - 1, cmd)

    def control_loop(self) -> None:
        self.group.recv_all(8_000)

        if self.base_enabled:
            self._read_base_state()

        if self.arms_enabled:
            left_arm = self.group.get_device(config.left_arm_can)
            for name, motor in zip(LEFT_HARDWARE_MOTOR_NAMES, left_arm.get_motors(), strict=True):
                state = self.lowstate.motor_state[motor_index(name)]
                if name == LEFT_GRIPPER_MOTOR_NAME:
                    print(f'left_gripper: {motor.get_position():.3f}')
                    state.q = gripper_motor_position_to_command(
                        motor.get_position(),
                        open_position=config.left_gripper_open,
                        close_position=config.left_gripper_close,
                    )
                    state.dq = gripper_motor_velocity_to_command_velocity(
                        motor.get_velocity(),
                        open_position=config.left_gripper_open,
                        close_position=config.left_gripper_close,
                    )
                    state.tau_est = motor.get_torque()
                else:
                    direction = _arm_direction(name, left=True)
                    state.q = motor.get_position() * direction
                    state.dq = motor.get_velocity() * direction
                    state.tau_est = motor.get_torque() * direction

            right_arm = self.group.get_device(config.right_arm_can)
            for name, motor in zip(RIGHT_HARDWARE_MOTOR_NAMES, right_arm.get_motors(), strict=True):
                state = self.lowstate.motor_state[motor_index(name)]
                if name == RIGHT_GRIPPER_MOTOR_NAME:
                    print(f'right_gripper: {motor.get_position():.3f}')
                    state.q = gripper_motor_position_to_command(
                        motor.get_position(),
                        open_position=config.right_gripper_open,
                        close_position=config.right_gripper_close,
                    )
                    state.dq = gripper_motor_velocity_to_command_velocity(
                        motor.get_velocity(),
                        open_position=config.right_gripper_open,
                        close_position=config.right_gripper_close,
                    )
                    state.tau_est = motor.get_torque()
                else:
                    direction = _arm_direction(name, left=False)
                    state.q = motor.get_position() * direction
                    state.dq = motor.get_velocity() * direction
                    state.tau_est = motor.get_torque() * direction

        self.lowstate_publisher.Write(self.lowstate)

        if self.base_enabled:
            self._write_base_command()

        if self.left_arm_command_enabled:
            left_arm = self.group.get_device(config.left_arm_can)
            left_cmds = []
            for name in LEFT_HARDWARE_MOTOR_NAMES:
                command = motor_command(self.lowcmd, name)
                if name == LEFT_GRIPPER_MOTOR_NAME:
                    left_cmds.append(
                        dc.MITParam(
                            q=mapping_gripper(
                                command.q,
                                config.left_gripper_open,
                                config.left_gripper_close,
                            ),
                            dq=gripper_command_velocity_to_motor_velocity(
                                command.dq,
                                open_position=config.left_gripper_open,
                                close_position=config.left_gripper_close,
                            ),
                            kp=command.kp,
                            kd=command.kd,
                            tau=command.tau,
                        )
                    )
                else:
                    direction = _arm_direction(name, left=True)
                    left_cmds.append(
                        dc.MITParam(
                            q=command.q * direction,
                            dq=command.dq * direction,
                            kp=command.kp,
                            kd=command.kd,
                            tau=command.tau * direction,
                        )
                    )
            left_arm.mit_control_all(left_cmds)

        if self.right_arm_command_enabled:
            right_arm = self.group.get_device(config.right_arm_can)
            right_cmds = []
            for name in RIGHT_HARDWARE_MOTOR_NAMES:
                command = motor_command(self.lowcmd, name)
                if name == RIGHT_GRIPPER_MOTOR_NAME:
                    right_cmds.append(
                        dc.MITParam(
                            q=mapping_gripper(
                                command.q,
                                config.right_gripper_open,
                                config.right_gripper_close,
                            ),
                            dq=gripper_command_velocity_to_motor_velocity(
                                command.dq,
                                open_position=config.right_gripper_open,
                                close_position=config.right_gripper_close,
                            ),
                            kp=command.kp,
                            kd=command.kd,
                            tau=command.tau,
                        )
                    )
                else:
                    direction = _arm_direction(name, left=False)
                    right_cmds.append(
                        dc.MITParam(
                            q=command.q * direction,
                            dq=command.dq * direction,
                            kp=command.kp,
                            kd=command.kd,
                            tau=command.tau * direction,
                        )
                    )
            right_arm.mit_control_all(right_cmds)


def main() -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = HardwareNode()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
