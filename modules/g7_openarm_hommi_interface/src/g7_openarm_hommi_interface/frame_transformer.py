from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from g7_openarm_idl import EETarget
from g7_openarm_pinnzoo import PinnZooModel, kinematics
from g7_openarm_utils.idl import array_to_pose, pose_to_array
from g7_openarm_utils.quat import quat_mul, quat_normalize, quat_rotate

if TYPE_CHECKING:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    from g7_openarm_idl import Odom


def compose_pose_wxyz(
    parent_pose: npt.NDArray[np.float64],
    local_pose: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compose two poses:

        T_world_target = T_world_parent @ T_parent_target

    Pose format:
        [x, y, z, qw, qx, qy, qz]
    """
    parent_pose = np.asarray(parent_pose, dtype=np.float64)
    local_pose = np.asarray(local_pose, dtype=np.float64)

    parent_position = parent_pose[:3]
    parent_quaternion = quat_normalize(parent_pose[3:7])

    local_position = local_pose[:3]
    local_quaternion = quat_normalize(local_pose[3:7])

    world_position = parent_position + quat_rotate(parent_quaternion, local_position)

    world_quaternion = quat_mul(
        parent_quaternion,
        local_quaternion,
    )
    world_quaternion = quat_normalize(world_quaternion)

    return np.concatenate(
        [world_position, world_quaternion],
        dtype=np.float64,
    )


class FrameTransformer:
    def __init__(
        self,
        lib_path: str | None = None,
    ):
        self.model = PinnZooModel(lib_path)

    def transfer(
        self,
        hommi_frame: EETarget,
        lowstate: "LowState_",
        odom: "Odom",
    ) -> EETarget:
        x_lib = PinnZooModel.build_x_lib(lowstate, odom)
        kin = np.asarray(
            kinematics(self.model, x_lib),
            dtype=np.float64,
        )

        left_pose = kin[:7]

        left_pose_in_left_frame = np.asarray(
            pose_to_array(hommi_frame.left_target),
            dtype=np.float64,
        )

        right_pose_in_left_frame = np.asarray(
            pose_to_array(hommi_frame.right_target),
            dtype=np.float64,
        )

        left_world_pose = compose_pose_wxyz(
            parent_pose=left_pose,
            local_pose=left_pose_in_left_frame,
        )

        right_world_pose = compose_pose_wxyz(
            parent_pose=left_pose,
            local_pose=right_pose_in_left_frame,
        )

        world_pose_target = EETarget(
            array_to_pose(left_world_pose),
            array_to_pose(right_world_pose),
        )

        return world_pose_target
