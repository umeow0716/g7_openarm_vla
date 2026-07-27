import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.hz_sample import RecurrentThread

from g7_openarm_idl import Odom, WBCLowCmd, WBCLowCmd_default

from .config import config
from .controller import Controller


class LowLevelNode:
    def __init__(self):
        self.base_enable = config.base_enabled

        self.wbc_lowcmd = WBCLowCmd_default()
        self.wbc_lowcmd_subscriber = ChannelSubscriber("rt/wbclowcmd", WBCLowCmd)
        self.wbc_lowcmd_subscriber.Init(self.wbc_lowcmd_handler, 0)

        self.lowstate: LowState_ | None = None
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.lowstate_handler, 0)

        self.odom: Odom | None = None
        self.odom_subscriber = ChannelSubscriber("rt/odom", Odom)
        self.odom_subscriber.Init(self.odom_handler, 0)

        self.lowcmd = unitree_hg_msg_dds__LowCmd_()
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.controller = Controller()

        self.write_lowcmd_thread = RecurrentThread(
            name="write_lowcmd",
            target=self.write_lowcmd,
            interval=config.interval,
        )
        self.write_lowcmd_thread.Start()

    def write_lowcmd(self):
        if self.lowstate is None or self.odom is None:
            return

        steer_pos_des, wheel_vel_des, q_des, dq_des, Kp, Kd, tau_ff = self.controller.update(
            lowstate=self.lowstate,
            odom=self.odom,
            amr_cmd=self.wbc_lowcmd.amr,
            openarm_cmd=self.wbc_lowcmd.openarm,
        )

        if self.base_enable:
            for i, motor in enumerate(self.lowcmd.motor_cmd[:8:2]):
                motor.q = steer_pos_des[i]
                motor.dq = 0.0
                motor.kp = 100.0
                motor.kd = 1.0
                motor.tau = 0.0

            for i, motor in enumerate(self.lowcmd.motor_cmd[1:8:2]):
                motor.dq = wheel_vel_des[i]
                motor.kp = 0.0
                motor.kd = 6.0
                motor.tau = 0.0

        for i, motor in enumerate(self.lowcmd.motor_cmd[8:24]):
            motor.q = q_des[i]
            motor.dq = dq_des[i]
            motor.kp = Kp[i]
            motor.kd = Kd[i]
            motor.tau = tau_ff[i]

        left_gripper = self.lowcmd.motor_cmd[15]
        left_gripper.q = self.wbc_lowcmd.openarm.data[7]
        left_gripper.dq = 0.0
        left_gripper.kp = 0.0
        left_gripper.kd = 0.0
        left_gripper.tau = 0.0

        right_gripper = self.lowcmd.motor_cmd[23]
        right_gripper.q = self.wbc_lowcmd.openarm.data[15]
        right_gripper.dq = 0.0
        right_gripper.kp = 0.0
        right_gripper.kd = 0.0
        right_gripper.tau = 0.0

        self.lowcmd_publisher.Write(self.lowcmd)

    def wbc_lowcmd_handler(self, msg: WBCLowCmd):
        self.wbc_lowcmd = msg

    def lowstate_handler(self, msg: LowState_):
        self.lowstate = msg

    def odom_handler(self, msg: Odom):
        self.odom = msg


def main():
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = LowLevelNode()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
