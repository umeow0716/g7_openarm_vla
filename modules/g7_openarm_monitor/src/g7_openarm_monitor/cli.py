import time
import threading
import sys

from g7_openarm_idl import EETarget, Odom, WBCLowCmd
from typing import Any
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowState_, LowCmd_

from .config import config


class HzMonitor:
    def __init__(self, topic: str, msg_type: Any) -> None:
        self.topic    = topic
        self.msg_type = msg_type 
        self._count = 0
        self._sub   = ChannelSubscriber(self.topic, self.msg_type)
        self._sub.Init(self.callback, 10)
        self._start = time.monotonic()
        self._lock  = threading.Lock()
        self._hz    = 0.0

    def callback(self, _: Any) -> None:
        with self._lock:
            self._count += 1

    def read_and_reset(self) -> int:
        with self._lock:
            count = self._count
            self._count = 0
            return count

    def get_hz(self) -> float:
        elapsed = time.monotonic() - self._start
        if elapsed < 1.0:
            return self._hz
        count = self.read_and_reset()
        self._hz = count / elapsed
        self._start = time.monotonic()
        
        return self._hz


def main() -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    
    eetarget_monitor  = HzMonitor("rt/eetarget", EETarget)
    imustate_monitor  = HzMonitor("rt/imustate", IMUState_)
    lowstate_monitor  = HzMonitor("rt/lowstate", LowState_)
    lowcmd_monitor    = HzMonitor("rt/lowcmd", LowCmd_)
    odom_monitor      = HzMonitor("rt/odom", Odom)
    wbclowcmd_monitor = HzMonitor("rt/wbclowcmd", WBCLowCmd)
    
    monitors = [
        eetarget_monitor,
        imustate_monitor,
        lowstate_monitor,
        lowcmd_monitor,
        odom_monitor,
        wbclowcmd_monitor
    ]
    
    while True:
        sys.stdout.write("\033[H\033[2J\033[3J")

        for monitor in monitors:
            print(
                f"{monitor.topic}: "
                f"{monitor.get_hz():.2f} Hz"
            )

        sys.stdout.flush()
        time.sleep(config.interval)


if __name__ == '__main__':
    main()