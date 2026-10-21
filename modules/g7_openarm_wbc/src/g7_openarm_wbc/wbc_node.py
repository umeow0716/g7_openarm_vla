import numpy as np
import time

from g7_openarm_config import general_config
from g7_openarm_idl import (
    Target,
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

from .base_filter import LowPassFilter
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
        self.base_lpf = LowPassFilter(
            sample_hz=config.hz, cutoff_hz=config.base_lpf_cutoff_hz
        )

        self.target: Target | None = None
        self.lowstate: HGLowState_ | None = None
        self.odom: Odom | None = None
        self.vrjoy: VRJoy | None = None

        self.vr_joy_received_at = 0.0

        self.target_sub  = build_sub("rt/target", Target, self.target_handler)
        self.lowstate_sub  = build_sub("rt/lowstate", HGLowState_, self.lowstate_handler)
        self.odom_sub      = build_sub("rt/odom", Odom, self.odom_handler)
        self.vrjoy_sub     = build_sub("rt/vrjoy", Target, self.vrjoy_handler)
        self.wbclowcmd_pub = build_pub("rt/wbclowcmd", WBCLowCmd)

        self.wbclowcmd_thread = build_thread(config.hz, self.control_loop)

    def target_handler(self, msg: Target):
        self.target = msg

    def lowstate_handler(self, msg: HGLowState_):
        self.lowstate = msg

    def odom_handler(self, msg: Odom):
        self.odom = msg

    def vrjoy_handler(self, msg: VRJoy):
        self.vrjoy = msg

    def control_loop(self):
        if self.wbclowcmd_pub is None:
            return
        if self.target is None or self.lowstate is None or self.odom is None:
            return

        body_cmd = None
        if self.is_vr:
            if self.vrjoy is None:
                return
            joy_is_fresh = time.monotonic() - self.vr_joy_received_at <= 0.25
            if not joy_is_fresh:
                return
            body_cmd = vr_joy_to_body_command(self.vrjoy)

        result = self.solver.solve_once(self.lowstate, self.odom, self.target)
        cmd = WBCLowCmd_default()
        cmd.dq_des = result[3:17].tolist()
        raw_base_cmd = body_cmd if body_cmd is not None else result[:3]
        cmd.mobile = self.base_lpf.update(raw_base_cmd).tolist()
        if general_config.head_enabled:
            cmd.head = result[-2:].tolist()

        self.wbclowcmd_pub.Write(cmd)


def main(is_vr=False):
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    node = WBCNode(is_vr)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(is_vr=False)
