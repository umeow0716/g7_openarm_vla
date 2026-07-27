from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.utils.hz_sample import RecurrentThread

from g7_openarm_idl import EETarget
from g7_openarm_mujoco.resources import model_directory
from g7_openarm_utils.idl import array_to_pose, pose_to_array
from g7_openarm_utils.mujoco import load_hand_default_pose

from .config import config
from .pose_mapping import remap_pose_yaw_only, yaw_alignment_quaternion
from .udp_receiver import JsonUdpReceiver
from .udp_response import VRUDPResponse


class VRNode:
    def __init__(self) -> None:
        self.eetarget_publisher = ChannelPublisher("rt/eetarget", EETarget)
        self.eetarget_publisher.Init()

        self.udp_receiver = JsonUdpReceiver()
        message = self._wait_for_message()

        print("waiting 5 seconds for zero state pose...")
        time.sleep(5.0)
        message = self.udp_receiver.latest() or message

        with model_directory() as model_dir:
            hand_default_pose = load_hand_default_pose((model_dir / "scene.xml").as_posix())

        self.left_origin_pose = pose_to_array(hand_default_pose.left_target)
        self.right_origin_pose = pose_to_array(hand_default_pose.right_target)
        self.left_first_pose, self.right_first_pose = self._controller_poses(message)
        self.left_yaw_alignment = yaw_alignment_quaternion(
            self.left_first_pose, self.left_origin_pose
        )
        self.right_yaw_alignment = yaw_alignment_quaternion(
            self.right_first_pose, self.right_origin_pose
        )

        self.eetarget_thread = RecurrentThread(
            name="eetarget_thread",
            target=self.write_eetarget,
            interval=config.interval,
        )
        self.eetarget_thread.Start()

    def _wait_for_message(self) -> VRUDPResponse:
        while True:
            message = self.udp_receiver.latest()
            if message is not None:
                return message
            print("waiting...")
            time.sleep(1.0)

    @staticmethod
    def _controller_poses(message: VRUDPResponse) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return (
            message.left_controller.as_mujoco_pose(),
            message.right_controller.as_mujoco_pose(),
        )

    def write_eetarget(self) -> None:
        message = self.udp_receiver.latest()
        if message is None:
            return

        left_pose, right_pose = self._controller_poses(message)
        left_target = remap_pose_yaw_only(
            left_pose,
            self.left_first_pose,
            self.left_origin_pose,
            self.left_yaw_alignment,
        )
        right_target = remap_pose_yaw_only(
            right_pose,
            self.right_first_pose,
            self.right_origin_pose,
            self.right_yaw_alignment,
        )

        self.eetarget_publisher.Write(
            EETarget(array_to_pose(left_target), array_to_pose(right_target))
        )


def main() -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = VRNode()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
