import numpy as np
import time

from g7_openarm_config import general_config
from g7_openarm_idl import (
    EETarget,
    Odom,
    VRJoy,
    WBCLowCmd,
    WBCLowCmd_default,
)
from g7_openarm_utils import (
    ChannelFactoryInitialize,
    build_pub,
    build_sub,
    build_thread,
)
from unitree_sdk2py.idl.default import HGLowState_

from .config import config
from .ik_solver import G7OpenArmLQR


def vr_joy_to_body_command(joy: VRJoy):
    axes = np.clip(
        np.array([joy.ly * 0.5, -joy.lx * 0.5, -joy.rx * 0.5], dtype=np.float64),
        -0.3,
        0.3,
    )
    return axes


class WBCNode:
    def __init__(self, is_vr=False):
        self.is_vr = is_vr
        self.solver = G7OpenArmLQR()

        self.eetarget: EETarget | None = None
        self.lowstate: HGLowState_ | None = None
        self.odom: Odom | None = None
        self.vrjoy: VRJoy | None = None

        self.vr_joy_received_at = 0.0

        self.eetarget_sub  = build_sub("rt/eetarget", EETarget, self.eetarget_handler)
        self.lowstate_sub  = build_sub("rt/lowstate", HGLowState_, self.lowstate_handler)
        self.odom_sub      = build_sub("rt/odom", Odom, self.odom_handler)
        self.vrjoy_sub     = build_sub("rt/vrjoy", EETarget, self.vrjoy_handler)
        self.wbclowcmd_pub = build_pub("rt/wbclowcmd", WBCLowCmd)

        self.wbclowcmd_thread = build_thread(config.hz, self.control_loop)

    def eetarget_handler(self, msg: EETarget):
        self.eetarget = msg

    def lowstate_handler(self, msg: HGLowState_):
        self.lowstate = msg

    def odom_handler(self, msg: Odom):
        self.odom = msg

    def vrjoy_handler(self, msg: VRJoy):
        self.vrjoy = msg

    def control_loop(self):
        if self.wbclowcmd_pub is None:
            return
        if self.eetarget is None or self.lowstate is None or self.odom is None:
            return

        body_cmd = None
        if self.is_vr:
            if self.vrjoy is None:
                return
            joy_is_fresh = time.monotonic() - self.vr_joy_received_at <= 0.25
            if not joy_is_fresh:
                return
            body_cmd = vr_joy_to_body_command(self.vrjoy)

        cmd = WBCLowCmd_default()
        result = self.solver.solve_once(self.lowstate, self.odom, self.eetarget)
        cmd.mobile = result[:3].tolist()
        cmd.dq_des = result[3:].tolist()
        if body_cmd is not None:
            cmd.mobile = body_cmd.tolist()

        self.wbclowcmd_pub.Write(cmd)


def main(is_vr=False):
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    node = WBCNode(is_vr)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(is_vr=False)
