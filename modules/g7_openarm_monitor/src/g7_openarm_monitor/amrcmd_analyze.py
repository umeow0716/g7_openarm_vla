import argparse
import time

from g7_openarm_idl import WBCLowCmd
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

class SimpleMonitor:
    def __init__(self) -> None:
        self.topic    = "rt/wbclowcmd"
        self.msg_type = WBCLowCmd 
        self._sub   = ChannelSubscriber(self.topic, self.msg_type)
        self._sub.Init(self.callback, 10)

    def callback(self, msg: WBCLowCmd) -> None:
        print(f"{msg.amr.data[0]:.3f} {msg.amr.data[1]:.3f} {msg.amr.data[2]:.3f}")


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


if __name__ == '__main__':
    main()