import math
import threading
import traceback
from queue import Empty, Full, Queue
from typing import MutableSequence, cast

import xspublic

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__IMUState_ as IMUState_default,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_

from .config import config


def deg_to_rad(value: float) -> float:
    return float(value) * math.pi / 180.0


class CallbackHandler(xspublic.XsCallback):
    def __init__(self, max_buffer: int = 5):
        super().__init__()

        if max_buffer <= 0:
            raise ValueError("max_buffer must be greater than zero")

        self._queue: Queue[xspublic.XsDataPacket] = Queue(
            maxsize=max_buffer
        )

    def wait_next_packet(
        self,
        timeout: float | None = None,
    ) -> xspublic.XsDataPacket:
        return self._queue.get(timeout=timeout)

    def onLiveDataAvailable(self, dev, packet) -> None:
        try:
            copied_packet = xspublic.XsDataPacket(packet)

            try:
                self._queue.put_nowait(copied_packet)
                return
            except Full:
                pass

            try:
                self._queue.get_nowait()
            except Empty:
                pass

            try:
                self._queue.put_nowait(copied_packet)
            except Full:
                pass

        except Exception:
            traceback.print_exc()


class IMUNode:
    def __init__(self, callback_handler: CallbackHandler):
        self.callback_handler = callback_handler
        self.stop_event = threading.Event()

        self.imu_state_pub = ChannelPublisher(
            "rt/imustate",
            IMUState_,
        )
        self.imu_state_pub.Init()

        self.state = IMUState_default()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                packet = self.callback_handler.wait_next_packet(
                    timeout=0.5
                )
            except Empty:
                continue

            try:
                self.publish_packet(packet)
            except Exception:
                traceback.print_exc()

    def publish_packet(
        self,
        packet: xspublic.XsDataPacket,
    ) -> None:
        if packet.contains_calibrated_acc():
            acc = packet.calibrated_acc()

            accelerometer = cast(
                MutableSequence[float],
                self.state.accelerometer,
            )
            accelerometer[0] = float(acc[0])
            accelerometer[1] = float(acc[1])
            accelerometer[2] = float(acc[2])

        if packet.contains_calibrated_gyr():
            gyr = packet.calibrated_gyr()

            gyroscope = cast(
                MutableSequence[float],
                self.state.gyroscope,
            )
            gyroscope[0] = float(gyr[0])
            gyroscope[1] = float(gyr[1])
            gyroscope[2] = float(gyr[2])

        if packet.contains_orientation():
            quat = packet.orientation_quaternion()

            quaternion = cast(
                MutableSequence[float],
                self.state.quaternion,
            )
            quaternion[0] = float(quat.w)
            quaternion[1] = float(quat.x)
            quaternion[2] = float(quat.y)
            quaternion[3] = float(quat.z)

            euler = packet.orientation_euler()

            rpy = cast(
                MutableSequence[float],
                self.state.rpy,
            )
            rpy[0] = deg_to_rad(euler.roll)
            rpy[1] = deg_to_rad(euler.pitch)
            rpy[2] = deg_to_rad(euler.yaw)

        self.imu_state_pub.Write(self.state)


def find_mti_port():
    ports = xspublic.XsScanner.scan_ports()

    for port in ports:
        device_id = port.device_id()

        if device_id.is_mti() or device_id.is_mtig():
            return port

    return None


def main() -> None:
    ChannelFactoryInitialize(
        config.dds.domain_id,
        config.dds.interface,
    )

    control = xspublic.XsControl()

    mt_port = None
    device = None
    callback = None
    node = None

    try:
        mt_port = find_mti_port()

        if mt_port is None:
            raise RuntimeError(
                "No MTi device found. Aborting."
            )

        port_name = mt_port.port_name()
        baud_rate = mt_port.baud_rate()

        print(
            f"Opening MTi device: "
            f"port={port_name}, baudrate={baud_rate}"
        )

        if not control.open_port(port_name, baud_rate):
            raise RuntimeError(
                "Could not open port: "
                f"{control.last_result_text()}"
            )

        device = control.device(mt_port.device_id())

        if device is None:
            raise RuntimeError(
                "Could not obtain the XsDevice instance."
            )

        print(
            f"Device: {device.product_code()}  "
            f"ID: {device.device_id().to_string()} opened."
        )

        callback = CallbackHandler(max_buffer=5)
        device.add_callback_handler(callback)

        if not device.goto_config():
            raise RuntimeError(
                "Could not put device into configuration mode: "
                f"{device.last_result_text()}"
            )

        device.read_emts_and_device_configuration()

        imu_config = [
            xspublic.XsOutputConfiguration(
                xspublic.PacketCounter,
                0,
            ),
            xspublic.XsOutputConfiguration(
                xspublic.SampleTimeFine,
                0,
            ),
            xspublic.XsOutputConfiguration(
                xspublic.StatusWord,
                0,
            ),
            xspublic.XsOutputConfiguration(
                xspublic.Acceleration,
                100,
            ),
            xspublic.XsOutputConfiguration(
                xspublic.FreeAcceleration,
                100,
            ),
            xspublic.XsOutputConfiguration(
                xspublic.RateOfTurn,
                100,
            ),
            xspublic.XsOutputConfiguration(
                xspublic.MagneticField,
                100,
            ),
            xspublic.XsOutputConfiguration(
                xspublic.Quaternion,
                100,
            ),
        ]

        if not device.set_output_configuration(imu_config):
            raise RuntimeError(
                "Could not configure MTi output: "
                f"{device.last_result_text()}"
            )

        if not device.goto_measurement():
            raise RuntimeError(
                "Could not put device into measurement mode: "
                f"{device.last_result_text()}"
            )

        print("MTi entered measurement mode.")
        print("Publishing IMU data to rt/imustate.")
        print("Press Ctrl+C to stop.")

        node = IMUNode(callback)
        node.run()

    except KeyboardInterrupt:
        print("\nStopping IMU node...")

    finally:
        if node is not None:
            node.stop()

        if device is not None and callback is not None:
            try:
                device.remove_callback_handler(callback)
            except Exception:
                traceback.print_exc()

        if mt_port is not None:
            try:
                control.close_port(mt_port.port_name())
            except Exception:
                traceback.print_exc()

        try:
            control.close()
        except Exception:
            traceback.print_exc()

        print("IMU device closed.")


if __name__ == "__main__":
    main()