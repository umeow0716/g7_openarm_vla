import math
import threading
import time

import xspublic
from g7_openarm_utils.unitree import (
    ChannelFactoryInitialize,
    build_pub,
    build_thread,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__IMUState_ as IMUState_default,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_

from .config import config

DEG_TO_RAD = math.pi / 180.0


class IMUNode(xspublic.XsCallback):
    def __init__(self):
        super().__init__()

        self.state = IMUState_default()
        self.imustate_pub = build_pub("rt/imustate", IMUState_)

    def onLiveDataAvailable(self, dev, packet) -> None:
        acc = packet.calibrated_acc()
        gyr = packet.calibrated_gyr()
        quat = packet.orientation_quaternion()
        euler = packet.orientation_euler()

        self.state.accelerometer[0] = acc[0]
        self.state.accelerometer[1] = acc[1]
        self.state.accelerometer[2] = acc[2]

        self.state.gyroscope[0] = gyr[0]
        self.state.gyroscope[1] = gyr[1]
        self.state.gyroscope[2] = gyr[2]

        self.state.quaternion[0] = quat.w
        self.state.quaternion[1] = quat.x
        self.state.quaternion[2] = quat.y
        self.state.quaternion[3] = quat.z

        self.state.rpy[0] = euler.roll * DEG_TO_RAD
        self.state.rpy[1] = euler.pitch * DEG_TO_RAD
        self.state.rpy[2] = euler.yaw * DEG_TO_RAD

        if self.imustate_pub is not None:
            self.imustate_pub.Write(self.state)


def find_mti_port():
    for port in xspublic.XsScanner.scan_ports():
        device_id = port.device_id()

        if device_id.is_mti() or device_id.is_mtig():
            return port

    raise RuntimeError("No MTi device found.")


def configure_device(device) -> None:
    if not device.goto_config():
        raise RuntimeError(device.last_result_text())

    device.read_emts_and_device_configuration()

    output_configuration = [
        xspublic.XsOutputConfiguration(
            xspublic.Acceleration,
            config.imu_hz,
        ),
        xspublic.XsOutputConfiguration(
            xspublic.RateOfTurn,
            config.imu_hz,
        ),
        xspublic.XsOutputConfiguration(
            xspublic.Quaternion,
            config.imu_hz,
        ),
    ]

    if not device.set_output_configuration(output_configuration):
        raise RuntimeError(device.last_result_text())

    if not device.goto_measurement():
        raise RuntimeError(device.last_result_text())


class VirtualIMUNode:
    def __init__(self) -> None:
        self.state = IMUState_default()
        self.state.quaternion[0] = 1.0
        self.state.quaternion[1] = 0.0
        self.state.quaternion[2] = 0.0
        self.state.quaternion[3] = 0.0

        for index in range(3):
            self.state.gyroscope[index] = 0.0
            self.state.accelerometer[index] = 0.0
            self.state.rpy[index] = 0.0

        self.imustate_pub = build_pub("rt/imustate", IMUState_)
        self.control_thread = build_thread(config.imu_hz, self.control_loop)

    def control_loop(self) -> None:
        if self.imustate_pub is not None:
            self.imustate_pub.Write(self.state)


def main(virtual=False) -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)

    if not virtual:
        control = xspublic.XsControl()
        port = find_mti_port()

        if not control.open_port(
            port.port_name(),
            port.baud_rate(),
        ):
            raise RuntimeError("Could not open MTi port.")

        device = control.device(port.device_id())

        if device is None:
            control.close()
            raise RuntimeError("Could not get MTi device.")

        node = IMUNode()
        device.add_callback_handler(node)

        try:
            configure_device(device)
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            device.remove_callback_handler(node)
            node.publisher.Close()
            control.close_port(port.port_name())
            control.close()
    else:
        node = VirtualIMUNode()
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main(virtual=False)
