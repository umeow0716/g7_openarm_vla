import time

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.hz_sample import RecurrentThread

from g7_openarm_config import general_config
from g7_openarm_idl import Odom, WBCLowCmd, WBCLowCmd_default
from g7_openarm_utils import GRIPPER_COMMAND_RANGE

from .config import config
from .controller import Controller
from .initialization import INITIAL_KD, INITIAL_KP, ArmInitializer


class LowLevelNode:
    def __init__(self, *, initial: bool = True) -> None:
        self.base_enable = general_config.base_actuation_enabled
        self.left_arm_enable = general_config.left_arm_actuation_enabled
        self.right_arm_enable = general_config.right_arm_actuation_enabled
        self.arm_enable = general_config.arm_actuation_enabled

        self.initial_enable = initial and general_config.lowlevel_initial_allowed
        self.initializer = (
            ArmInitializer(
                general_config.initial_pos,
                target_gripper=general_config.initial_gripper,
                left_enabled=self.left_arm_enable,
                right_enabled=self.right_arm_enable,
            )
            if self.initial_enable
            else None
        )

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

    def _write_base_hold(self) -> None:
        if self.lowstate is None or not self.base_enable:
            return

        steer_pos_des, wheel_vel_des = self.controller.update_base(
            self.lowstate,
            WBCLowCmd_default().amr,
        )
        self._write_base_commands(steer_pos_des, wheel_vel_des)

    def _write_base_commands(self, steer_pos_des, wheel_vel_des) -> None:
        if not self.base_enable:
            return

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

    @staticmethod
    def _write_arm_joint_slice(
        lowcmd: LowCmd_,
        *,
        motor_offset: int,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        kp: np.ndarray | float,
        kd: np.ndarray | float,
        tau: np.ndarray | float,
    ) -> None:
        for i in range(7):
            motor = lowcmd.motor_cmd[motor_offset + i]
            motor.q = float(q_des[i])
            motor.dq = float(dq_des[i])
            motor.kp = float(kp[i] if isinstance(kp, np.ndarray) else kp)
            motor.kd = float(kd[i] if isinstance(kd, np.ndarray) else kd)
            motor.tau = float(tau[i] if isinstance(tau, np.ndarray) else tau)

    def _write_initial(self) -> bool:
        """Write one initialization tick. Return True while normal control must stay gated."""
        if not self.initial_enable or self.initializer is None:
            return False
        if self.lowstate is None:
            return True

        now = time.monotonic()
        if not self.initializer.started:
            q_start = np.array(
                [self.lowstate.motor_state[i].q for i in range(8, 24)],
                dtype=np.float64,
            )
            self.initializer.start(q_start, now=now)

        q_des, dq_des, done = self.initializer.sample(now=now)

        self._write_base_hold()
        if self.left_arm_enable:
            self._write_arm_joint_slice(
                self.lowcmd,
                motor_offset=8,
                q_des=q_des[:7],
                dq_des=dq_des[:7],
                kp=INITIAL_KP[:7],
                kd=INITIAL_KD[:7],
                tau=0.0,
            )
            left_gripper = self.lowcmd.motor_cmd[15]
            left_gripper.q = float(q_des[7])
            left_gripper.dq = float(dq_des[7])
            left_gripper.kp = INITIAL_KP[7]
            left_gripper.kd = INITIAL_KD[7]
            left_gripper.tau = 0.0

        if self.right_arm_enable:
            self._write_arm_joint_slice(
                self.lowcmd,
                motor_offset=16,
                q_des=q_des[8:15],
                dq_des=dq_des[8:15],
                kp=INITIAL_KP[:7],
                kd=INITIAL_KD[:7],
                tau=0.0,
            )
            right_gripper = self.lowcmd.motor_cmd[23]
            right_gripper.q = float(q_des[15])
            right_gripper.dq = float(dq_des[15])
            right_gripper.kp = INITIAL_KP[7]
            right_gripper.kd = INITIAL_KD[7]
            right_gripper.tau = 0.0
        self.lowcmd_publisher.Write(self.lowcmd)

        if done:
            self.initial_enable = False
            print("Low-level arm initialization complete")

        return True

    def _write_normal_arm_commands(
        self,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        tau_ff: np.ndarray,
    ) -> None:
        if self.left_arm_enable:
            self._write_arm_joint_slice(
                self.lowcmd,
                motor_offset=8,
                q_des=q_des[:7],
                dq_des=dq_des[:7],
                kp=kp[:7],
                kd=kd[:7],
                tau=tau_ff[:7],
            )

            left_gripper = self.lowcmd.motor_cmd[15]
            left_gripper.q = self.wbc_lowcmd.openarm.data[7] * GRIPPER_COMMAND_RANGE
            left_gripper.dq = 0.0
            left_gripper.kp = 20.0
            left_gripper.kd = 0.5
            left_gripper.tau = 0.0

        if self.right_arm_enable:
            self._write_arm_joint_slice(
                self.lowcmd,
                motor_offset=16,
                q_des=q_des[8:15],
                dq_des=dq_des[8:15],
                kp=kp[8:15],
                kd=kd[8:15],
                tau=tau_ff[8:15],
            )

            right_gripper = self.lowcmd.motor_cmd[23]
            right_gripper.q = self.wbc_lowcmd.openarm.data[15] * GRIPPER_COMMAND_RANGE
            right_gripper.dq = 0.0
            right_gripper.kp = 20.0
            right_gripper.kd = 0.5
            right_gripper.tau = 0.0

    def write_lowcmd(self) -> None:
        if self._write_initial():
            return

        if self.lowstate is None or self.odom is None:
            return

        if self.arm_enable:
            steer_pos_des, wheel_vel_des, q_des, dq_des, kp, kd, tau_ff = (
                self.controller.update(
                    lowstate=self.lowstate,
                    odom=self.odom,
                    amr_cmd=self.wbc_lowcmd.amr,
                    openarm_cmd=self.wbc_lowcmd.openarm,
                )
            )
        else:
            steer_pos_des, wheel_vel_des = self.controller.update_base(
                self.lowstate,
                self.wbc_lowcmd.amr,
            )

        self._write_base_commands(steer_pos_des, wheel_vel_des)

        if self.arm_enable:
            self._write_normal_arm_commands(q_des, dq_des, kp, kd, tau_ff)

        self.lowcmd_publisher.Write(self.lowcmd)

    def wbc_lowcmd_handler(self, msg: WBCLowCmd) -> None:
        self.wbc_lowcmd = msg

    def lowstate_handler(self, msg: LowState_) -> None:
        self.lowstate = msg

    def odom_handler(self, msg: Odom) -> None:
        self.odom = msg


def main(*, initial: bool = True) -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = LowLevelNode(initial=initial)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
