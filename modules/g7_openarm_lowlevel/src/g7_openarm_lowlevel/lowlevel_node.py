import threading
import time

from g7_openarm_idl import Odom, WBCLowCmd
from g7_openarm_utils.unitree import (
    ChannelFactoryInitialize,
    build_pub,
    build_sub,
    build_thread,
)
from unitree_sdk2py.idl.default import (
    HGLowCmd_,
    HGLowState_,
)

from .config import config
from .controller import Controller

class LowLevelNode:
    def __init__(self):
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
        self.control_thread = build_thread(config.hz, self.control_loop)

    def lowstate_handle(self, msg: HGLowState_):
        self.lowstate = msg
        self.lowstate_ready.set()

    def odom_handle(self, msg: Odom):
        self.odom = msg

    def wbclowcmd_handle(self, msg: WBCLowCmd):
        self.wbclowcmd = msg

    def control_loop(self):
        if self.lowcmd_pub is None:
            return
        if self.lowstate is None or self.odom is None or self.wbclowcmd is None:
            return

        lowcmd = self.controller.update(self.lowstate, self.odom, self.wbclowcmd)
        self.lowcmd_pub.Write(lowcmd)


def main():
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    node = LowLevelNode()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
