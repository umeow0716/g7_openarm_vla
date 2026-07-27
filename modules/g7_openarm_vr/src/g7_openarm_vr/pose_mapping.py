from __future__ import annotations

import numpy as np
import numpy.typing as npt

from g7_openarm_utils import quat_from_yaw, quat_mul, quat_normalize, quat_rotate, quat_yaw

FloatArray = npt.NDArray[np.float64]


def yaw_alignment_quaternion(
    first_pose: npt.ArrayLike,
    origin_pose: npt.ArrayLike,
) -> FloatArray:
    """Return the world-Z-only rotation that aligns initial VR and robot yaw."""
    first = np.asarray(first_pose, dtype=np.float64)
    origin = np.asarray(origin_pose, dtype=np.float64)
    if first.shape != (7,) or origin.shape != (7,):
        raise ValueError("First and origin poses must both have shape (7,)")
    return quat_from_yaw(quat_yaw(origin[3:]) - quat_yaw(first[3:]))


def remap_pose_yaw_only(
    current_pose: npt.ArrayLike,
    first_pose: npt.ArrayLike,
    origin_pose: npt.ArrayLike,
    yaw_alignment: npt.ArrayLike | None = None,
) -> FloatArray:
    """Map a basis-converted VR pose to the robot origin using only yaw calibration.

    The fixed Unity-to-MuJoCo basis conversion must already have been applied.
    This function then rotates the controller translation delta by the session
    yaw alignment and applies that same world-Z rotation to its orientation.
    It does not align initial controller pitch or roll to the robot hand.
    """
    current = np.asarray(current_pose, dtype=np.float64)
    first = np.asarray(first_pose, dtype=np.float64)
    origin = np.asarray(origin_pose, dtype=np.float64)
    if current.shape != (7,) or first.shape != (7,) or origin.shape != (7,):
        raise ValueError("Current, first, and origin poses must all have shape (7,)")

    alignment = (
        yaw_alignment_quaternion(first, origin)
        if yaw_alignment is None
        else quat_normalize(yaw_alignment)
    )
    position = origin[:3] + quat_rotate(alignment, current[:3] - first[:3])
    orientation = quat_normalize(quat_mul(alignment, current[3:]))
    return np.concatenate((position, orientation), dtype=np.float64)
