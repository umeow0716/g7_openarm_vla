import threading
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
from g7_openarm_utils import (
    ARM_COMMAND_MOTOR_NAMES,
    BASE_STEER_MOTOR_NAMES,
    BASE_WHEEL_MOTOR_NAMES,
    GRIPPER_COMMAND_RANGE,
    LEFT_ARM_MOTOR_NAMES,
    LEFT_GRIPPER_MOTOR_NAME,
    RIGHT_ARM_MOTOR_NAMES,
    RIGHT_GRIPPER_MOTOR_NAME,
    arm_command_index,
    arm_command_indices,
    motor_command,
    motor_state_values,
)

from .config import config
from .controller import Controller
from .initialization import ArmInitializer, initial_kd, initial_kp


LEFT_ARM_COMMAND_INDICES = arm_command_indices(list(LEFT_ARM_MOTOR_NAMES))
RIGHT_ARM_COMMAND_INDICES = arm_command_indices(list(RIGHT_ARM_MOTOR_NAMES))
LEFT_GRIPPER_COMMAND_INDEX = arm_command_index(LEFT_GRIPPER_MOTOR_NAME)
RIGHT_GRIPPER_COMMAND_INDEX = arm_command_index(RIGHT_GRIPPER_MOTOR_NAME)


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
        self.initial_plan_lock = threading.Lock()

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

        for name, q_des in zip(BASE_STEER_MOTOR_NAMES, steer_pos_des, strict=True):
            motor = motor_command(self.lowcmd, name)
            motor.q = float(q_des)
            motor.dq = 0.0
            motor.kp = 100.0
            motor.kd = 1.0
            motor.tau = 0.0

        for name, dq_des in zip(BASE_WHEEL_MOTOR_NAMES, wheel_vel_des, strict=True):
            motor = motor_command(self.lowcmd, name)
            motor.dq = float(dq_des)
            motor.kp = 0.0
            motor.kd = 6.0
            motor.tau = 0.0

    @staticmethod
    def _write_arm_joint_group(
        lowcmd: LowCmd_,
        *,
        motor_names: tuple[str, ...],
        q_des: np.ndarray,
        dq_des: np.ndarray,
        kp: np.ndarray | float,
        kd: np.ndarray | float,
        tau: np.ndarray | float,
    ) -> None:
        if len(motor_names) != len(q_des) or len(q_des) != len(dq_des):
            raise ValueError("arm command arrays must match motor_names length")
        for i, name in enumerate(motor_names):
            motor = motor_command(lowcmd, name)
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

        # The 5 s plan is created by lowstate_handler() from the first received
        # motor state. Until that happens, keep normal control gated.
        if not self.initializer.started:
            return True

        q_des, dq_des, done = self.initializer.sample(now=time.monotonic())

        self._write_base_hold()
        if self.left_arm_enable:
            self._write_arm_joint_group(
                self.lowcmd,
                motor_names=LEFT_ARM_MOTOR_NAMES,
                q_des=q_des[LEFT_ARM_COMMAND_INDICES],
                dq_des=dq_des[LEFT_ARM_COMMAND_INDICES],
                kp=initial_kp(LEFT_ARM_MOTOR_NAMES),
                kd=initial_kd(LEFT_ARM_MOTOR_NAMES),
                tau=0.0,
            )
            left_gripper = motor_command(self.lowcmd, LEFT_GRIPPER_MOTOR_NAME)
            left_gripper.q = float(q_des[LEFT_GRIPPER_COMMAND_INDEX])
            left_gripper.dq = float(dq_des[LEFT_GRIPPER_COMMAND_INDEX])
            left_gripper.kp = float(initial_kp((LEFT_GRIPPER_MOTOR_NAME,))[0])
            left_gripper.kd = float(initial_kd((LEFT_GRIPPER_MOTOR_NAME,))[0])
            left_gripper.tau = 0.0

        if self.right_arm_enable:
            self._write_arm_joint_group(
                self.lowcmd,
                motor_names=RIGHT_ARM_MOTOR_NAMES,
                q_des=q_des[RIGHT_ARM_COMMAND_INDICES],
                dq_des=dq_des[RIGHT_ARM_COMMAND_INDICES],
                kp=initial_kp(RIGHT_ARM_MOTOR_NAMES),
                kd=initial_kd(RIGHT_ARM_MOTOR_NAMES),
                tau=0.0,
            )
            right_gripper = motor_command(self.lowcmd, RIGHT_GRIPPER_MOTOR_NAME)
            right_gripper.q = float(q_des[RIGHT_GRIPPER_COMMAND_INDEX])
            right_gripper.dq = float(dq_des[RIGHT_GRIPPER_COMMAND_INDEX])
            right_gripper.kp = float(initial_kp((RIGHT_GRIPPER_MOTOR_NAME,))[0])
            right_gripper.kd = float(initial_kd((RIGHT_GRIPPER_MOTOR_NAME,))[0])
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
            self._write_arm_joint_group(
                self.lowcmd,
                motor_names=LEFT_ARM_MOTOR_NAMES,
                q_des=q_des[LEFT_ARM_COMMAND_INDICES],
                dq_des=dq_des[LEFT_ARM_COMMAND_INDICES],
                kp=kp[LEFT_ARM_COMMAND_INDICES],
                kd=kd[LEFT_ARM_COMMAND_INDICES],
                tau=tau_ff[LEFT_ARM_COMMAND_INDICES],
            )

            left_gripper = motor_command(self.lowcmd, LEFT_GRIPPER_MOTOR_NAME)
            left_gripper.q = (
                self.wbc_lowcmd.openarm.data[LEFT_GRIPPER_COMMAND_INDEX]
                * GRIPPER_COMMAND_RANGE
            )
            left_gripper.dq = 0.0
            left_gripper.kp = 20.0
            left_gripper.kd = 0.5
            left_gripper.tau = 0.0

        if self.right_arm_enable:
            self._write_arm_joint_group(
                self.lowcmd,
                motor_names=RIGHT_ARM_MOTOR_NAMES,
                q_des=q_des[RIGHT_ARM_COMMAND_INDICES],
                dq_des=dq_des[RIGHT_ARM_COMMAND_INDICES],
                kp=kp[RIGHT_ARM_COMMAND_INDICES],
                kd=kd[RIGHT_ARM_COMMAND_INDICES],
                tau=tau_ff[RIGHT_ARM_COMMAND_INDICES],
            )

            right_gripper = motor_command(self.lowcmd, RIGHT_GRIPPER_MOTOR_NAME)
            right_gripper.q = (
                self.wbc_lowcmd.openarm.data[RIGHT_GRIPPER_COMMAND_INDEX]
                * GRIPPER_COMMAND_RANGE
            )
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

        # Initialization is planned exactly once, from the first lowstate that
        # reaches this node. This makes the measured robot pose the trajectory
        # start instead of assuming any fixed joint-zero pose.
        if self.initial_enable and self.initializer is not None:
            with self.initial_plan_lock:
                if not self.initializer.started:
                    q_start = motor_state_values(msg, ARM_COMMAND_MOTOR_NAMES, "q")
                    self.initializer.plan_from_state(q_start, now=time.monotonic())
                    print("Low-level arm initialization planned from first lowstate")

    def odom_handler(self, msg: Odom) -> None:
        self.odom = msg


def main(*, initial: bool = True) -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = LowLevelNode(initial=initial)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
