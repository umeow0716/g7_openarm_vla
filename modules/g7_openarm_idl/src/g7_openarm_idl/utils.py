import numpy as np
import numpy.typing as npt
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Pose_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_


def pose_to_array(pose: Pose_) -> npt.NDArray[np.float64]:
    pos = pose.position
    ori = pose.orientation
    return np.array([pos.x, pos.y, pos.z, ori.w, ori.x, ori.y, ori.z], dtype=np.float64)


def array_to_pose(arr: npt.NDArray[np.float64]) -> Pose_:
    pose = geometry_msgs_msg_dds__Pose_()

    pose.position.x = arr[0]
    pose.position.y = arr[1]
    pose.position.z = arr[2]
    pose.orientation.w = arr[3]
    pose.orientation.x = arr[4]
    pose.orientation.y = arr[5]
    pose.orientation.z = arr[6]

    return pose
