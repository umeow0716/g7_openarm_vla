import time

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.hz_sample import RecurrentThread

from g7_openarm_config import ControlMode, general_config
from g7_openarm_idl import EETarget, Odom, VRJoy, WBCLowCmd, WBCLowCmd_default

from .config import config
from .control_layout import split_control_vector
from .ik_solver import G7OpenArmIKSolver
from .vr_joy_control import vr_joy_to_body_command


class Node:
    def __init__(self):
        self.ik_solver = (
            None
            if general_config.control_mode is ControlMode.BASE_ONLY
            else G7OpenArmIKSolver()
        )

        self.lowstate: LowState_ | None = None
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.lowstate_handler, 0)

        self.odom: Odom | None = None
        self.odom_subscriber = ChannelSubscriber("rt/odom", Odom)
        self.odom_subscriber.Init(self.odom_handler, 0)

        self.ee_target: EETarget | None = None
        self.ee_target_subscriber = ChannelSubscriber("rt/eetarget", EETarget)
        self.ee_target_subscriber.Init(self.ee_target_handler, 0)

        self.vr_joy: VRJoy | None = None
        self.vr_joy_received_at = 0.0
        self.vr_joy_subscriber = ChannelSubscriber("rt/vrjoy", VRJoy)
        self.vr_joy_subscriber.Init(self.vr_joy_handler, 0)

        self.wbc_lowcmd = WBCLowCmd_default()
        self.wbc_lowcmd_publisher = ChannelPublisher("rt/wbclowcmd", WBCLowCmd)
        self.wbc_lowcmd_publisher.Init()
        self.wbc_lowcmd_thread = RecurrentThread(
            name="wbc_lowcmd_thread",
            target=self.write_wbc_lowcmd,
            interval=config.interval,
        )
        self.wbc_lowcmd_thread.Start()

    def lowstate_handler(self, msg: LowState_):
        self.lowstate = msg

    def odom_handler(self, msg: Odom):
        self.odom = msg

    def ee_target_handler(self, msg: EETarget):
        self.ee_target = msg

    def vr_joy_handler(self, msg: VRJoy):
        self.vr_joy = msg
        self.vr_joy_received_at = time.monotonic()

    def write_wbc_lowcmd(self):
        joy_is_fresh = (
            self.vr_joy is not None and time.monotonic() - self.vr_joy_received_at <= 0.25
        )

        if general_config.control_mode is ControlMode.BASE_ONLY:
            amr_cmd = (
                vr_joy_to_body_command(self.vr_joy)
                if joy_is_fresh
                else np.zeros(3, dtype=np.float64)
            )
            openarm_cmd = np.zeros(16, dtype=np.float64)
        else:
            if self.lowstate is None or self.odom is None or self.ee_target is None:
                return

            assert self.ik_solver is not None
            u = self.ik_solver.solve_once(self.lowstate, self.odom, self.ee_target)
            amr_cmd, arm_cmd = split_control_vector(
                u,
                base_enabled=self.ik_solver.base_enabled,
            )

            openarm_cmd = np.concatenate(
                [
                    arm_cmd[:7],
                    [self.ee_target.left_gripper],
                    arm_cmd[7:14],
                    [self.ee_target.right_gripper],
                ],
                dtype=np.float64,
            )

            if general_config.control_mode is ControlMode.ARM_ONLY:
                amr_cmd = (
                    vr_joy_to_body_command(self.vr_joy)
                    if joy_is_fresh
                    else np.zeros(3, dtype=np.float64)
                )

        self.wbc_lowcmd.amr.data = amr_cmd.tolist()
        self.wbc_lowcmd.openarm.data = openarm_cmd.tolist()

        self.wbc_lowcmd_publisher.Write(self.wbc_lowcmd)


def main():
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = Node()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
