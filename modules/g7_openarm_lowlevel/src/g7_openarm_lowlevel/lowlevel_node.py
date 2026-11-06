import threading
import time

import numpy as np
from unitree_sdk2py.idl.default import (
    HGLowCmd_,
    HGLowState_,
)

from g7_openarm_config import general_config
from g7_openarm_idl import Odom, WBCLowCmd, WBCLowCmd_default
from g7_openarm_utils.layout import ARM_JOINT, Joint, low_idx
from g7_openarm_utils.unitree import (
    ChannelFactoryInitialize,
    build_pub,
    build_sub,
    build_thread,
)

from .config import config
from .controller import Controller
from .initializer import InitialPoseSequencer, TcpReachMonitor


class LowLevelNode:
    def __init__(self, initial: bool = True):
        self.lowstate_ready = threading.Event()

        self.lowstate: HGLowState_ | None = None
        self.odom: Odom | None = None
        self.wbclowcmd: WBCLowCmd | None = None

        self.lowcmd_pub = build_pub("rt/lowcmd", HGLowCmd_)
        self.lowstate_sub = build_sub("rt/lowstate", HGLowState_, self.lowstate_handle)
        self.odom_sub = build_sub("rt/odom", Odom, self.odom_handle)
        self.wbclowcmd_sub = build_sub("rt/wbclowcmd", WBCLowCmd, self.wbclowcmd_handle)

        if not self.lowstate_ready.wait(timeout=5.0):
            raise RuntimeError("Timeout waiting for first lowstate message")
        assert self.lowstate is not None

        self.controller = Controller()
        self.controller.initialize(self.lowstate)

        # Until the arms reach general.initial_pos the lowlevel owns them and
        # ignores rt/wbclowcmd; afterwards it holds the pose until WBC speaks.
        start_q = self._arm_q(self.lowstate)
        # initial=False: skip the move and just hold the start-up pose until WBC speaks.
        self.sequencer = InitialPoseSequencer.plan(start_q, None if initial else start_q)
        self.reach_monitor = TcpReachMonitor(self.sequencer.goal)
        self.unlocked = not initial
        self.head_hold = [
            self.lowstate.motor_state[low_idx(Joint.HEAD_pitch)].q,
            self.lowstate.motor_state[low_idx(Joint.HEAD_yaw)].q,
        ]
        self.start_time: float | None = None
        self.last_warning = 0.0
        if initial:
            print(
                "lowlevel: moving arms to initial_pos over "
                f"{self.sequencer.duration:.1f}s before accepting WBC commands",
                flush=True,
            )

        self.control_thread = build_thread(config.hz, self.control_loop)

    @staticmethod
    def _arm_q(lowstate: HGLowState_) -> np.ndarray:
        return np.array([lowstate.motor_state[low_idx(j)].q for j in ARM_JOINT], dtype=np.float64)

    @staticmethod
    def _arm_dq(lowstate: HGLowState_) -> np.ndarray:
        return np.array([lowstate.motor_state[low_idx(j)].dq for j in ARM_JOINT], dtype=np.float64)

    def lowstate_handle(self, msg: HGLowState_):
        self.lowstate = msg
        self.lowstate_ready.set()

    def odom_handle(self, msg: Odom):
        self.odom = msg

    def wbclowcmd_handle(self, msg: WBCLowCmd):
        self.wbclowcmd = msg

    def _own_command(self, t: float) -> WBCLowCmd:
        """Lowlevel-owned command: arms follow the start-up path, rest holds still."""
        cmd = WBCLowCmd_default()
        q_ref = self.controller.reference_generator.q_ref
        cmd.dq_des = self.sequencer.dq_command(t, q_ref).tolist()
        cmd.head = list(self.head_hold)
        cmd.left_gripper = general_config.initial_gripper
        cmd.right_gripper = general_config.initial_gripper
        return cmd

    def control_loop(self):
        if self.lowcmd_pub is None:
            return
        lowstate, odom = self.lowstate, self.odom
        if lowstate is None or odom is None:
            return

        now = time.monotonic()
        if self.start_time is None:
            self.start_time = now
        t = now - self.start_time

        if not self.unlocked:
            if t < self.sequencer.duration:
                self.reach_monitor.reset()
            else:
                pos_err, ang_err = self.reach_monitor.errors(lowstate, odom)
                if self.reach_monitor.settled(now, pos_err, ang_err):
                    self.unlocked = True
                    print(
                        f"lowlevel: initial_pos reached after {t:.1f}s "
                        f"(TCP err {100 * pos_err:.1f} cm / {np.degrees(ang_err):.1f} deg), "
                        "WBC unlocked",
                        flush=True,
                    )
                elif t > self.sequencer.duration + 5.0 and now - self.last_warning > 1.0:
                    self.last_warning = now
                    print(
                        f"lowlevel: TCP still {100 * pos_err:.1f} cm / "
                        f"{np.degrees(ang_err):.1f} deg from initial_pos, WBC stays locked",
                        flush=True,
                    )

        wbclowcmd = self.wbclowcmd if self.unlocked and self.wbclowcmd is not None else None
        if wbclowcmd is None:
            wbclowcmd = self._own_command(t)

        lowcmd = self.controller.update(lowstate, odom, wbclowcmd)
        self.lowcmd_pub.Write(lowcmd)


def main(initial: bool = True):
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _node = LowLevelNode(initial=initial)  # keep the node alive
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
