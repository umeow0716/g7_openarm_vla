import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__IMUState_ as IMUState_default,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_

from .config import config


class VirtualIMUNode:
    """Publish a stationary, identity-orientation IMU state for base-less real runs."""

    def __init__(self) -> None:
        self.publisher = ChannelPublisher("rt/imustate", IMUState_)
        self.publisher.Init()

        self.state = IMUState_default()
        self.state.quaternion[0] = 1.0
        self.state.quaternion[1] = 0.0
        self.state.quaternion[2] = 0.0
        self.state.quaternion[3] = 0.0

        for index in range(3):
            self.state.gyroscope[index] = 0.0
            self.state.accelerometer[index] = 0.0
            self.state.rpy[index] = 0.0

    def publish(self) -> None:
        self.publisher.Write(self.state)

    def close(self) -> None:
        self.publisher.Close()


def main() -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)

    node = VirtualIMUNode()
    interval = 1.0 / config.imu_hz
    next_publish = time.monotonic()

    try:
        while True:
            node.publish()
            next_publish += interval
            time.sleep(max(0.0, next_publish - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        node.close()


if __name__ == "__main__":
    main()
