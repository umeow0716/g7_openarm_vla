"""HoMMI point-cloud cropping (UmiTaskDataset ``crop_pointcloud``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def transform_points(transform: npt.ArrayLike, points: FloatArray) -> FloatArray:
    """Apply a 4x4 transform to ``[..., 3]`` points."""
    t = np.asarray(transform, dtype=np.float32)
    return (points @ t[:3, :3].T + t[:3, 3]).astype(np.float32, copy=False)


def pointmap_mask(
    pointmap: FloatArray,
    head: FloatArray,
    tcps: Sequence[FloatArray],
    *,
    depth_range: tuple[float, float],
    gripper_length: float,
) -> npt.NDArray[np.bool_]:
    """HoMMI ``crop_pointcloud``: valid depth range and not behind either gripper.

    ``pointmap`` is ``[H, W, 3]`` in the head optical frame; ``head`` and ``tcps``
    are world poses at the same instant.
    """
    depth = pointmap[..., 2]
    mask = (depth > depth_range[0]) & (depth < depth_range[1])
    for tcp in tcps:
        head_in_gripper = np.linalg.inv(tcp.astype(np.float64)) @ head.astype(np.float64)
        z = transform_points(head_in_gripper, pointmap)[..., 2]
        mask &= z > -gripper_length
    return mask


@dataclass(frozen=True, slots=True)
class PointCloudCrop:
    """HoMMI ``crop_pointcloud`` settings (from the checkpoint's dataset config)."""

    enabled: bool = True
    depth_range: tuple[float, float] = (0.1, 4.0)
    gripper_length: float = 0.13  # m behind each TCP along its approach axis

    def mask(
        self, pointmap: FloatArray, head: FloatArray, tcps: Sequence[FloatArray]
    ) -> npt.NDArray[np.bool_]:
        if not self.enabled:
            return np.ones(pointmap.shape[:-1], dtype=bool)
        return pointmap_mask(
            pointmap,
            head,
            tcps,
            depth_range=self.depth_range,
            gripper_length=self.gripper_length,
        )
