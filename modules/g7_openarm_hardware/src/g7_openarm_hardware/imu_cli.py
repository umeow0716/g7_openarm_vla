import math
import threading
import time
import traceback
from queue import Empty, Full, Queue
from typing import MutableSequence, cast

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


IMU_OUTPUT_HZ = config.imu_hz
QUEUE_SIZE = 1
STATS_INTERVAL_S = 2.0

# Xsens SampleTimeFine uses a 10 kHz clock.
SAMPLE_TIME_FINE_HZ = 10_000.0


def deg_to_rad(value: float) -> float:
    return float(value) * math.pi / 180.0


class CallbackHandler(xspublic.XsCallback):
    def __init__(self, max_buffer: int = QUEUE_SIZE):
        super().__init__()

        if max_buffer <= 0:
            raise ValueError("max_buffer must be greater than zero")

        self._queue: Queue[xspublic.XsDataPacket] = Queue(
            maxsize=max_buffer
        )
        self._stats_lock = threading.Lock()
        self._queue_dropped = 0

    def wait_next_packet(
        self,
        timeout: float | None = None,
    ) -> xspublic.XsDataPacket:
        return self._queue.get(timeout=timeout)

    def queue_dropped_total(self) -> int:
        with self._stats_lock:
            return self._queue_dropped

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
            else:
                with self._stats_lock:
                    self._queue_dropped += 1

            try:
                self._queue.put_nowait(copied_packet)
            except Full:
                with self._stats_lock:
                    self._queue_dropped += 1

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

        self._last_packet_counter: int | None = None
        self._last_sample_time_fine: int | None = None

        self._packet_gap_total = 0
        self._processed_count = 0
        self._sensor_tick_total = 0
        self._sensor_tick_samples = 0
        self._stats_started_at = time.monotonic()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                packet = self.callback_handler.wait_next_packet(
                    timeout=0.5
                )
            except Empty:
                self._print_statistics_if_due()
                continue

            try:
                self.publish_packet(packet)
            except Exception:
                traceback.print_exc()

    def publish_packet(
        self,
        packet: xspublic.XsDataPacket,
    ) -> None:
        self._update_packet_diagnostics(packet)

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

        self._processed_count += 1
        self._print_statistics_if_due()

    def _update_packet_diagnostics(
        self,
        packet: xspublic.XsDataPacket,
    ) -> None:
        if packet.contains_packet_counter():
            counter = int(packet.packet_counter())

            if self._last_packet_counter is not None:
                expected = (self._last_packet_counter + 1) & 0xFFFF
                gap = (counter - expected) & 0xFFFF

                # Ignore implausible backwards/reset jumps.
                if 0 < gap < 0x8000:
                    self._packet_gap_total += gap

            self._last_packet_counter = counter

        if packet.contains_sample_time_fine():
            sample_time = int(packet.sample_time_fine())

            if self._last_sample_time_fine is not None:
                delta_ticks = (
                    sample_time - self._last_sample_time_fine
                ) & 0xFFFFFFFF

                if 0 < delta_ticks < 0x80000000:
                    self._sensor_tick_total += delta_ticks
                    self._sensor_tick_samples += 1

            self._last_sample_time_fine = sample_time

    def _print_statistics_if_due(self) -> None:
        now = time.monotonic()
        elapsed = now - self._stats_started_at

        if elapsed < STATS_INTERVAL_S:
            return

        receive_rate = self._processed_count / elapsed

        if self._sensor_tick_samples > 0:
            average_ticks = (
                self._sensor_tick_total / self._sensor_tick_samples
            )
            average_sensor_dt_ms = (
                average_ticks / SAMPLE_TIME_FINE_HZ
            ) * 1000.0
            sensor_dt_text = f"{average_sensor_dt_ms:.3f} ms"
        else:
            sensor_dt_text = "n/a"

        print(
            f"IMU rate={receive_rate:.1f} Hz, "
            f"sensor_dt={sensor_dt_text}, "
            f"packet_gaps={self._packet_gap_total}, "
            f"queue_drops={self.callback_handler.queue_dropped_total()}"
        )

        self._processed_count = 0
        self._sensor_tick_total = 0
        self._sensor_tick_samples = 0
        self._stats_started_at = now


def find_mti_port():
    for port in xspublic.XsScanner.scan_ports():
        device_id = port.device_id()

        if device_id.is_mti() or device_id.is_mtig():
            return port

    return None


def build_output_configuration():
    return [
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
            IMU_OUTPUT_HZ,
        ),
        xspublic.XsOutputConfiguration(
            xspublic.RateOfTurn,
            IMU_OUTPUT_HZ,
        ),
        xspublic.XsOutputConfiguration(
            xspublic.Quaternion,
            IMU_OUTPUT_HZ,
        ),
    ]


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
            raise RuntimeError("No MTi device found. Aborting.")

        port_name = mt_port.port_name()
        baud_rate = mt_port.baud_rate()

        # The current xspublic binding may display high baud-rate enum values
        # as XsBaudRate.???, so do not rely on enum string formatting here.
        print(f"Opening MTi device: port={port_name}")

        if not control.open_port(port_name, baud_rate):
            raise RuntimeError("Could not open MTi serial port.")

        device = control.device(mt_port.device_id())

        if device is None:
            raise RuntimeError(
                "Could not obtain the XsDevice instance."
            )

        print(
            f"Device: {device.product_code()}  "
            f"ID: {device.device_id().to_string()} opened."
        )

        callback = CallbackHandler(max_buffer=QUEUE_SIZE)
        device.add_callback_handler(callback)

        if not device.goto_config():
            raise RuntimeError(
                "Could not put the device into configuration mode: "
                f"{device.last_result_text()}"
            )

        device.read_emts_and_device_configuration()

        output_configuration = build_output_configuration()

        if not device.set_output_configuration(
            output_configuration
        ):
            raise RuntimeError(
                "Could not configure MTi output: "
                f"{device.last_result_text()}"
            )

        if not device.goto_measurement():
            raise RuntimeError(
                "Could not put the device into measurement mode: "
                f"{device.last_result_text()}"
            )

        print(
            f"MTi measurement mode active: "
            f"{IMU_OUTPUT_HZ} Hz, latest-sample queue."
        )
        print("Publishing DDS topic: rt/imustate")
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
