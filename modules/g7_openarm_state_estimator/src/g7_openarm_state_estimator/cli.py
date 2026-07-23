import time
import numpy as np
import numpy.typing as npt

from g7_openarm_idl.odom import Odom, Odom_default

from typing import Optional

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.hz_sample import RecurrentThread

from .amr_ekf import AMREKF
from .config import config


def yaw_to_quat_wxyz(yaw: float) -> npt.NDArray[np.float64]:
    half = 0.5 * yaw

    return np.array([
        np.cos(half),  # w
        0.0,           # x
        0.0,           # y
        np.sin(half),  # z
    ], dtype=np.float64)

class OdomNode:
    def __init__(self):
        self.ekf = AMREKF()
        
        self.lowstate: Optional[LowState_] = None
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.lowstate_handler, 10)
        
        self.odom = Odom_default()
        self.odom_publisher = ChannelPublisher("rt/odom", Odom)
        self.odom_publisher.Init() 
        
        self.update_thread = RecurrentThread(
            name="update_thread",
            target=self.update_state,
            interval=config.interval,
        )
        self.update_thread.Start()
    
    def lowstate_handler(self, msg: LowState_):
        self.lowstate = msg

    def update_state(self, verbose=True):
        if self.lowstate is None:
            print("None", end="\r", flush=True)
            return
        
        x = self.ekf.update(lowstate=self.lowstate, dt=config.interval)
        
        self.odom.position.x = x.x
        self.odom.position.y = x.y
        self.odom.position.z = x.z
        self.odom.quaternion.w = x.quat[0]
        self.odom.quaternion.x = x.quat[1]
        self.odom.quaternion.y = x.quat[2]
        self.odom.quaternion.z = x.quat[3]
        self.odom.velocity.x = x.vx
        self.odom.velocity.y = x.vy
        self.odom.velocity.z = x.vz
        self.odom.angular_velocity.x = x.angular_velocity[0]
        self.odom.angular_velocity.y = x.angular_velocity[1]
        self.odom.angular_velocity.z = x.angular_velocity[2]
        self.odom.vdot.x = x.vdot[0]
        self.odom.vdot.y = x.vdot[1]
        self.odom.vdot.z = x.vdot[2]
        self.odom.angular_vdot.x = x.angular_vdot[0]
        self.odom.angular_vdot.y = x.angular_vdot[1]
        self.odom.angular_vdot.z = x.angular_vdot[2]
        
        self.odom_publisher.Write(self.odom)
        
        if verbose:
            print(f"{x.x:.3f}, {x.y:.3f}, {x.z:.3f}", end="\r", flush=True)
        

def main():
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = OdomNode()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
