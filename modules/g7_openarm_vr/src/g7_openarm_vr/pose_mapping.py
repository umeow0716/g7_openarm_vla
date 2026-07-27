from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from g7_openarm_utils import quat_conj, quat_mul, quat_normalize


def _as_pose(pose: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,):
        raise ValueError(f"{name} pose must have shape (7,), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} pose must contain only finite values")

    result = value.copy()
    result[3:] = quat_normalize(result[3:])
    return result


@dataclass(frozen=True, slots=True)
class RelativePoseMapper:
    """Map VR poses relative to a captured pose onto a robot origin pose.

    Translation remains expressed in the already-converted MuJoCo world axes:
    moving the controller forward/right/up therefore always changes robot
    +X/-Y/+Z respectively. Orientation follows the controller rotation relative
    to the captured pose, with the robot origin orientation as its zero pose.

    The two constants are precomputed at initialization:

    - ``position_offset = origin_position - first_position``
    - ``orientation_post_rotation = inverse(first_orientation) * origin_orientation``

    Each update then requires only one vector addition and one quaternion
    multiplication.
    """

    position_offset: npt.NDArray[np.float64]
    orientation_post_rotation: npt.NDArray[np.float64]

    @classmethod
    def from_poses(
        cls,
        first_pose: npt.ArrayLike,
        origin_pose: npt.ArrayLike,
    ) -> RelativePoseMapper:
        first = _as_pose(first_pose, name="First")
        origin = _as_pose(origin_pose, name="Origin")
        return cls(
            position_offset=origin[:3] - first[:3],
            orientation_post_rotation=quat_normalize(quat_mul(quat_conj(first[3:]), origin[3:])),
        )

    def map(self, current_pose: npt.ArrayLike) -> npt.NDArray[np.float64]:
        current = _as_pose(current_pose, name="Current")
        position = current[:3] + self.position_offset
        orientation = quat_normalize(quat_mul(current[3:], self.orientation_post_rotation))
        return np.concatenate((position, orientation), dtype=np.float64)
