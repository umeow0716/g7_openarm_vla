import numpy as np
import numpy.typing as npt

from g7_openarm_idl.utils import pose_to_array, array_to_pose
from g7_openarm_pinnzoo import PinnZooModel, kinematics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from g7_openarm_idl import EETarget, Odom
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


DEFAULT_LIB_PATH = PinnZooModel.get_default_lib_path()


def quat_normalize_wxyz(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small")

    return q / norm

def quat_multiply_wxyz(
    q1: npt.NDArray[np.float64],
    q2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Quaternion multiplication:

        q_result = q1 ⊗ q2

    Both quaternions use [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_rotate_wxyz(
    q: npt.NDArray[np.float64],
    vector: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Rotate a 3D vector by quaternion q.

    q uses [w, x, y, z].
    """
    q = quat_normalize_wxyz(q)

    w = q[0]
    xyz = q[1:]

    # Equivalent to q * [0, vector] * conjugate(q),
    # but avoids constructing two extra quaternions.
    return (
        2.0 * np.dot(xyz, vector) * xyz
        + (w * w - np.dot(xyz, xyz)) * vector
        + 2.0 * w * np.cross(xyz, vector)
    )


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

    if parent_pose.shape != (7,):
        raise ValueError(
            f"parent_pose must have shape (7,), got {parent_pose.shape}"
        )

    if local_pose.shape != (7,):
        raise ValueError(
            f"local_pose must have shape (7,), got {local_pose.shape}"
        )

    parent_position = parent_pose[:3]
    parent_quaternion = quat_normalize_wxyz(parent_pose[3:7])

    local_position = local_pose[:3]
    local_quaternion = quat_normalize_wxyz(local_pose[3:7])

    world_position = (
        parent_position
        + quat_rotate_wxyz(parent_quaternion, local_position)
    )

    world_quaternion = quat_multiply_wxyz(
        parent_quaternion,
        local_quaternion,
    )
    world_quaternion = quat_normalize_wxyz(world_quaternion)

    return np.concatenate(
        [world_position, world_quaternion],
        dtype=np.float64,
    )


class FrameTransformer:
    def __init__(
        self,
        lib_path: str | None = None,
    ):
        self.lib_path = str(DEFAULT_LIB_PATH if lib_path is None else lib_path)
        self.model = PinnZooModel(self.lib_path)

    def transfer(
        self,
        hommi_frame: "EETarget",
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