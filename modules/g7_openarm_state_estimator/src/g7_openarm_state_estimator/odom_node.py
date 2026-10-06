import time

from g7_openarm_utils import ChannelFactoryInitialize, build_pub, build_sub
from unitree_sdk2py.idl.default import HGIMUState_, HGLowState_
from unitree_sdk2py.utils.hz_sample import RecurrentThread

from g7_openarm_idl.odom import Odom, Odom_default
from g7_openarm_utils.layout import BASE_ENABLED

from .amr_ekf import AMREKF
from .config import config


class OdomNode:
    def __init__(self):
        self.ekf = AMREKF()

        self.imustate: HGIMUState_ | None = None
        self.lowstate: HGLowState_ | None = None
        self.odom = Odom_default()
        self.static_odom = Odom_default()
        self.static_odom.position.x = 0.0
        self.static_odom.position.y = 0.0
        self.static_odom.position.z = 0.0
        self.static_odom.quaternion.w = 1.0
        self.static_odom.quaternion.x = 0.0
        self.static_odom.quaternion.y = 0.0
        self.static_odom.quaternion.z = 0.0
        self.static_odom.velocity.x = 0.0
        self.static_odom.velocity.y = 0.0
        self.static_odom.velocity.z = 0.0
        self.static_odom.angular_velocity.x = 0.0
        self.static_odom.angular_velocity.y = 0.0
        self.static_odom.angular_velocity.z = 0.0
        self.static_odom.vdot.x = 0.0
        self.static_odom.vdot.y = 0.0
        self.static_odom.vdot.z = 0.0
        self.static_odom.angular_vdot.x = 0.0
        self.static_odom.angular_vdot.y = 0.0
        self.static_odom.angular_vdot.z = 0.0
        
        self.imustate_sub = build_sub("rt/imustate", HGIMUState_, self.imustate_handler)
        self.lowstate_sub = build_sub("rt/lowstate", HGLowState_, self.lowstate_handler)
        self.odom_pub = build_pub("rt/odom", Odom)
        
        assert self.odom_pub is not None

        self.update_thread = RecurrentThread(
            name="update_thread",
            target=self.update_state,
            interval=config.interval,
        )
        self.update_thread.Start()

    def lowstate_handler(self, msg: HGLowState_):
        self.lowstate = msg

    def imustate_handler(self, msg: HGIMUState_):
        self.imustate = msg

    def update_state(self, verbose=True):
        if self.odom_pub is None:
            return

        if not BASE_ENABLED:
            self.odom_pub.Write(self.static_odom)
            return

        if self.lowstate is None or self.imustate is None:
            return

        x = self.ekf.update(self.imustate, self.lowstate)

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

        self.odom_pub.Write(self.odom)

        if verbose:
            print(f"{x.x:.3f}, {x.y:.3f}, {x.z:.3f}", end="\r", flush=True)


def main():
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = OdomNode()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
