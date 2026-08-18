import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

from g7_openarm_idl import WBCLowCmd
from g7_openarm_utils import amr_command_values


class SimpleMonitor:
    def __init__(self) -> None:
        self.topic = "rt/wbclowcmd"
        self.msg_type = WBCLowCmd
        self._sub = ChannelSubscriber(self.topic, self.msg_type)
        self._sub.Init(self.callback, 10)

    def callback(self, msg: WBCLowCmd) -> None:
        vx, vy, wz = amr_command_values(msg.amr)
        print(f"{vx:.3f} {vy:.3f} {wz:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        default="lo",
        help="DDS network interface, e.g. enp2s0",
    )
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.interface)

    SimpleMonitor()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
