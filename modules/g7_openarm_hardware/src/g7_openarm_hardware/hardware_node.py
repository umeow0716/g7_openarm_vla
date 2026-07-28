import atexit
import signal
import time
from typing import Any, TypedDict

import openarm_can as oa
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

from .config import config


class BusConfig(TypedDict):
    motor_types: list[Any]
    send_ids: list[int]
    recv_ids: list[int]
    control_modes: list[Any]


def _base_bus_config() -> BusConfig:
    control_modes: list[Any] = [oa.ControlMode.POS_VEL] * 8
    for logical_index, motor_id in enumerate(config.base_ids):
        control_modes[motor_id - 1] = (
            oa.ControlMode.VEL if logical_index % 2 else oa.ControlMode.POS_VEL
        )

    return {
        "motor_types": [
            oa.MotorType.DM8009,
            oa.MotorType.DM8009,
            oa.MotorType.DM8009,
            oa.MotorType.DM8009,
            oa.MotorType.DM6006,
            oa.MotorType.DM6006,
            oa.MotorType.DM6006,
            oa.MotorType.DM6006,
        ],
        "send_ids": [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        "recv_ids": [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18],
        "control_modes": control_modes,
    }


def _arm_bus_config() -> BusConfig:
    return {
        "motor_types": [
            oa.MotorType.DM8009,
            oa.MotorType.DM8009,
            oa.MotorType.DM4340,
            oa.MotorType.DM4340,
            oa.MotorType.DM4310,
            oa.MotorType.DM4310,
            oa.MotorType.DM4310,
            oa.MotorType.DM4310,
        ],
        "send_ids": [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        "recv_ids": [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18],
        "control_modes": [oa.ControlMode.MIT] * 8,
    }


def build_bus_configs(*, base_enabled: bool) -> dict[str, BusConfig]:
    bus_configs: dict[str, BusConfig] = {}
    if base_enabled:
        bus_configs[config.base_can] = _base_bus_config()

    bus_configs[config.left_arm_can] = _arm_bus_config()
    bus_configs[config.right_arm_can] = _arm_bus_config()
    return bus_configs


class HardwareNode:
    def __init__(self) -> None:
        self.base_enabled = general_config.base_enabled
        self.bus_configs = build_bus_configs(base_enabled=self.base_enabled)

        can_interfaces = list(self.bus_configs)
        self.group = oa.OpenArmGroup(can_interfaces=can_interfaces, enable_fd=config.can_fd)
        for can_interface, can_config in self.bus_configs.items():
            arm = self.group.get_openarm(can_interface)
            arm.init_arm_motors(
                can_config["motor_types"],
                can_config["send_ids"],
                can_config["recv_ids"],
                can_config["control_modes"],
            )
            arm.set_callback_mode_all(oa.CallbackMode.STATE)

            print(f"{can_interface}: expected responses = {arm.expected_response_count()}")

        if not self.base_enabled:
            print(f"Control mode {general_config.control_mode.value}: skipping {config.base_can}")

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
        base = self.group.get_openarm(config.base_can)
        base_motors = base.get_arm().get_motors()
        base.flush_rx()

        for i in range(8):
            motor = base_motors[config.base_ids[i] - 1]
            direction = config.base_direction[i]
            self.lowstate.motor_state[i].q = motor.get_position() * direction
            self.lowstate.motor_state[i].dq = motor.get_velocity() * direction
            self.lowstate.motor_state[i].tau_est = motor.get_torque() * direction

    def _write_base_command(self) -> None:
        base = self.group.get_openarm(config.base_can)
        for i in range(0, 8, 2):
            cmd = oa.PosVelParam(
                q=self.lowcmd.motor_cmd[i].q * config.base_direction[i],
                dq=20.0,
            )
            base.get_arm().posvel_control_one(config.base_ids[i] - 1, cmd)

        for i in range(1, 8, 2):
            cmd = oa.VelParam(dq=self.lowcmd.motor_cmd[i].dq * config.base_direction[i])
            base.get_arm().vel_control_one(config.base_ids[i] - 1, cmd)

    def control_loop(self) -> None:
        for _ in self.group.recv_wait_all(5000):
            pass

        if self.base_enabled:
            self._read_base_state()

        left_arm = self.group.get_openarm(config.left_arm_can)
        left_arm.flush_rx()
        for i, motor in enumerate(left_arm.get_arm().get_motors()):
            self.lowstate.motor_state[8 + i].q = motor.get_position() * config.left_arm_direction[i]
            self.lowstate.motor_state[8 + i].dq = (
                motor.get_velocity() * config.left_arm_direction[i]
            )
            self.lowstate.motor_state[8 + i].tau_est = (
                motor.get_torque() * config.left_arm_direction[i]
            )

        right_arm = self.group.get_openarm(config.right_arm_can)
        right_arm.flush_rx()
        for i, motor in enumerate(right_arm.get_arm().get_motors()):
            self.lowstate.motor_state[16 + i].q = (
                motor.get_position() * config.right_arm_direction[i]
            )
            self.lowstate.motor_state[16 + i].dq = (
                motor.get_velocity() * config.right_arm_direction[i]
            )
            self.lowstate.motor_state[16 + i].tau_est = (
                motor.get_torque() * config.right_arm_direction[i]
            )

        self.lowstate_publisher.Write(self.lowstate)

        if self.base_enabled:
            self._write_base_command()

        left_cmds = [
            oa.MITParam(
                q=self.lowcmd.motor_cmd[8 + i].q * config.left_arm_direction[i],
                dq=self.lowcmd.motor_cmd[8 + i].dq * config.left_arm_direction[i],
                kp=self.lowcmd.motor_cmd[8 + i].kp,
                kd=self.lowcmd.motor_cmd[8 + i].kd,
                tau=self.lowcmd.motor_cmd[8 + i].tau * config.left_arm_direction[i],
            )
            for i in range(8)
        ]
        right_cmds = [
            oa.MITParam(
                q=self.lowcmd.motor_cmd[16 + i].q * config.right_arm_direction[i],
                dq=self.lowcmd.motor_cmd[16 + i].dq * config.right_arm_direction[i],
                kp=self.lowcmd.motor_cmd[16 + i].kp,
                kd=self.lowcmd.motor_cmd[16 + i].kd,
                tau=self.lowcmd.motor_cmd[16 + i].tau * config.right_arm_direction[i],
            )
            for i in range(8)
        ]
        left_arm.get_arm().mit_control_all(left_cmds)
        right_arm.get_arm().mit_control_all(right_cmds)


def main() -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = HardwareNode()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
