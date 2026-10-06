from __future__ import annotations

import numpy as np
import numpy.typing as npt

from g7_openarm_utils.layout.enums import Dim
from g7_openarm_utils.quat import quat_to_rotation_matrix
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Pose_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_

from .odom import Odom


def pose_to_array(pose: Pose_) -> list[float]:
    position = pose.position
    orientation = pose.orientation
    return [
        position.x,
        position.y,
        position.z,
        orientation.w,
        orientation.x,
        orientation.y,
        orientation.z,
    ]


def array_to_pose(array: npt.ArrayLike) -> Pose_:
    pose_array = np.asarray(array, dtype=np.float64)
    if pose_array.shape != (7,):
        raise ValueError(f"Pose must have shape (7,), got {pose_array.shape}")

    pose = geometry_msgs_msg_dds__Pose_()
    pose.position.x = float(pose_array[0])
    pose.position.y = float(pose_array[1])
    pose.position.z = float(pose_array[2])
    pose.orientation.w = float(pose_array[3])
    pose.orientation.x = float(pose_array[4])
    pose.orientation.y = float(pose_array[5])
    pose.orientation.z = float(pose_array[6])
    return pose
