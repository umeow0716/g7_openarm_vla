from __future__ import annotations

import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.hz_sample import RecurrentThread

from g7_openarm_idl import EETarget, Odom

from .config import config
from .frame_transformer import FrameTransformer


class HommiInterfaceNode:
    def __init__(self) -> None:
        self.frame_transformer = FrameTransformer()

        # Initialize every callback-visible field before subscriber Init calls.
        self.lowstate: LowState_ | None = None
        self.odom: Odom | None = None
        self.eetarget: EETarget | None = None

        self.eetarget_publisher = ChannelPublisher("rt/eetarget", EETarget)
        self.eetarget_publisher.Init()

        self.hommi_subscriber = ChannelSubscriber("rt/hommi", EETarget)
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.odom_subscriber = ChannelSubscriber("rt/odom", Odom)

        self.hommi_subscriber.Init(self.hommi_handler, 0)
        self.lowstate_subscriber.Init(self.lowstate_handler, 0)
        self.odom_subscriber.Init(self.odom_handler, 0)

        self.eetarget_thread = RecurrentThread(
            name="eetarget_thread",
            interval=config.interval,
            target=self.write_eetarget,
        )
        self.eetarget_thread.Start()

    def hommi_handler(self, msg: EETarget) -> None:
        lowstate = self.lowstate
        odom = self.odom
        if lowstate is None or odom is None:
            return

        self.eetarget = self.frame_transformer.transfer(msg, lowstate, odom)

    def lowstate_handler(self, msg: LowState_) -> None:
        self.lowstate = msg

    def odom_handler(self, msg: Odom) -> None:
        self.odom = msg

    def write_eetarget(self) -> None:
        eetarget = self.eetarget
        if eetarget is None:
            return

        self.eetarget_publisher.Write(eetarget)


def main() -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = HommiInterfaceNode()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
