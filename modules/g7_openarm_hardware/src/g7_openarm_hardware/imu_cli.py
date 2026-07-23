import math
import threading

import xspublic

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
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

        self.publisher = ChannelPublisher(
            "rt/imustate",
            IMUState_,
        )
        self.publisher.Init()

        self.state = IMUState_default()

    def onLiveDataAvailable(self, dev, packet) -> None:
        acc = packet.calibrated_acc()
        gyr = packet.calibrated_gyr()
        quat = packet.orientation_quaternion()
        euler = packet.orientation_euler()

        state = self.state

        state.accelerometer[0] = acc[0]
        state.accelerometer[1] = acc[1]
        state.accelerometer[2] = acc[2]

        state.gyroscope[0] = gyr[0]
        state.gyroscope[1] = gyr[1]
        state.gyroscope[2] = gyr[2]

        state.quaternion[0] = quat.w
        state.quaternion[1] = quat.x
        state.quaternion[2] = quat.y
        state.quaternion[3] = quat.z

        state.rpy[0] = euler.roll * DEG_TO_RAD
        state.rpy[1] = euler.pitch * DEG_TO_RAD
        state.rpy[2] = euler.yaw * DEG_TO_RAD

        self.publisher.Write(state)


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


def main() -> None:
    ChannelFactoryInitialize(
        config.dds.domain_id,
        config.dds.interface,
    )

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


if __name__ == "__main__":
    main()
