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
    def __init__(self):
        self.frame_transformer = FrameTransformer()

        self.hommi_subscriber = ChannelSubscriber("rt/hommi", EETarget)
        self.hommi_subscriber.Init(self.hommi_handler, 0)

        self.lowstate: LowState_ | None = None
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.lowstate_handler, 0)

        self.odom: Odom | None = None
        self.odom_subscriber = ChannelSubscriber("rt/odom", Odom)
        self.odom_subscriber.Init(self.odom_handler, 0)

        self.eetarget: EETarget | None = None
        self.eetarget_publisher = ChannelPublisher("rt/eetarget", Odom)
        self.eetarget_publisher.Init()

        self.eetarget_thread = RecurrentThread(
            name="eetarget_thread", interval=config.interval, target=self.write_eetarget
        )
        self.eetarget_thread.Start()

    def hommi_handler(self, msg: EETarget):
        if self.lowstate is None or self.odom is None:
            return

        self.eetarget = self.frame_transformer.transfer(msg, self.lowstate, self.odom)

    def lowstate_handler(self, msg: LowState_):
        self.lowstate = msg

    def odom_handler(self, msg: Odom):
        self.odom = msg

    def write_eetarget(self):
        if self.eetarget is None:
            return

        self.eetarget_publisher.Write(self.eetarget)


def main():
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = HommiInterfaceNode()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
