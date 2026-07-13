import time
import numpy as np

from typing import Optional

from g7_openarm_idl import EETarget, Odom, WBCLowCmd, WBCLowCmd_default
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.utils.hz_sample import RecurrentThread
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from .ik_solver import G7OpenArmIKSolver


class Node:
    def __init__(self):
        self.ik_solver = G7OpenArmIKSolver()
        
        self.lowstate: Optional[LowState_] = None
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber = self.lowstate_subscriber.Init(self.lowstate_handler, 10)
        
        self.odom: Optional[Odom] = None
        self.odom_subscriber = ChannelSubscriber("rt/odom", Odom)
        self.odom_subscriber.Init(self.odom_handler, 10)
        
        self.ee_target: Optional[EETarget] = None
        self.ee_target_subscriber = ChannelSubscriber("rt/eetarget", EETarget)
        self.ee_target_subscriber.Init(self.ee_target_handler, 10)
        
        self.wbc_lowcmd = WBCLowCmd_default()
        self.wbc_lowcmd_publisher = ChannelPublisher("rt/wbclowcmd", WBCLowCmd)
        self.wbc_lowcmd_publisher.Init()
        self.wbc_lowcmd_thraed = RecurrentThread(
            name="wbc_lowcmd_thraed",
            target=self.write_wbc_lowcmd,
            interval=0.002,
        )
        self.wbc_lowcmd_thraed.Start()
    
    def lowstate_handler(self, msg: LowState_):
        self.lowstate = msg
 
    def odom_handler(self, msg: Odom):
        self.odom = msg

    def ee_target_handler(self, msg: EETarget):
        self.ee_target = msg
    
    def write_wbc_lowcmd(self):
        if self.lowstate is None or self.odom is None or self.ee_target is None:
            return

        u = self.ik_solver.solve_once(self.lowstate, self.odom, self.ee_target)
        openarm_cmd = np.concatenate([
            u[3:10],
            [0.0],
            u[10:17],
            [0.0],
        ], dtype=np.float64)
        
        self.wbc_lowcmd.amr.data = u[:3].tolist()
        self.wbc_lowcmd.openarm.data = openarm_cmd.tolist()

        self.wbc_lowcmd_publisher.Write(self.wbc_lowcmd)


def main():
    ChannelFactoryInitialize(0, "lo")
    node = Node()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()