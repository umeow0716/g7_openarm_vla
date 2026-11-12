from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from .crop import PointCloudCrop, transform_points
from .geometry import relative_pose9
from .state import RobotSnapshot
from .vcam import HeadObservation

FloatArray = npt.NDArray[np.float32]

SIDES = ("left", "right")
HEAD_RGB_KEY = "camera_head_main_rgb"
HEAD_POINTMAP_KEY = "camera_head_pointmap"
HEAD_MASK_KEY = "camera_head_pointmap_mask"
WRIST_RGB_KEYS = {"left": "camera_left_main_rgb", "right": "camera_right_main_rgb"}


@dataclass(frozen=True, slots=True)
class ObservationSample:
    sequence: int
    timestamp: float
    head: HeadObservation
    wrist_rgb: Mapping[str, npt.NDArray[np.uint8]]  # side -> [S, S, 3]
    robot: RobotSnapshot
    head_robot: RobotSnapshot | None = None

    @property
    def head_state(self) -> RobotSnapshot:
        return self.robot if self.head_robot is None else self.head_robot


class ObservationHistory:
    def __init__(self, horizon: int) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self._horizon = int(horizon)
        self._samples: deque[ObservationSample] = deque(maxlen=horizon)
        self._condition = threading.Condition()

    def append(self, sample: ObservationSample) -> None:
        with self._condition:
            self._samples.append(sample)
            self._condition.notify_all()

    def wait_for_new(
        self,
        last_sequence: int,
        stop_event: threading.Event,
    ) -> tuple[ObservationSample, ...] | None:
        with self._condition:
            while not stop_event.is_set():
                if (
                    len(self._samples) == self._horizon
                    and self._samples[-1].sequence > last_sequence
                ):
                    return tuple(self._samples)
                self._condition.wait(timeout=0.1)
        return None

    def latest(self) -> ObservationSample | None:
        with self._condition:
            return self._samples[-1] if self._samples else None

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()


class Ego3dObservationBuilder:
    """Build HoMMI ``ego3d_lookat_policy`` observations (``reference_frame='left'``).

    Mirrors HoMMI's ``UmiTaskDataset`` / ``policy_server.get_real_umi_obs_dict``:
    every pose, the head pointmap and its mask are expressed relative to the left
    TCP at the newest observation step.
    """

    REFERENCE_SIDE = "left"

    def __init__(
        self,
        shape_meta: Mapping[str, Any],
        *,
        crop: PointCloudCrop,
        device: torch.device,
    ) -> None:
        obs_meta = shape_meta.get("obs")
        if not isinstance(obs_meta, Mapping):
            raise ValueError("model shape_meta is missing obs metadata")
        required = {
            HEAD_RGB_KEY,
            HEAD_POINTMAP_KEY,
            *WRIST_RGB_KEYS.values(),
            *(
                f"gripper_{s}_{k}"
                for s in SIDES
                for k in ("eef_pos", "eef_rot_axis_angle", "gripper_width")
            ),
        }
        missing = sorted(required - set(obs_meta))
        if missing:
            raise ValueError(f"model is not an ego3d policy; missing obs keys {missing}")

        horizons = {int(obs_meta[k].get("horizon", 1)) for k in required}
        if len(horizons) != 1:
            raise ValueError(f"all observations must share one horizon, got {horizons}")
        self.horizon = horizons.pop()

        self.head_size = int(obs_meta[HEAD_RGB_KEY]["shape"][1])
        if tuple(obs_meta[HEAD_POINTMAP_KEY]["shape"]) != (3, self.head_size, self.head_size):
            raise ValueError("head pointmap and head RGB must share one square resolution")
        wrist_sizes = {int(obs_meta[k]["shape"][1]) for k in WRIST_RGB_KEYS.values()}
        if len(wrist_sizes) != 1:
            raise ValueError("left/right wrist RGB must share one resolution")
        self.wrist_size = wrist_sizes.pop()

        self.crop = crop
        self._device = device

    def build_numpy(self, samples: Sequence[ObservationSample]) -> dict[str, np.ndarray]:
        if len(samples) != self.horizon:
            raise ValueError(f"expected {self.horizon} observation samples, got {len(samples)}")

        base = samples[-1].robot.tcp(self.REFERENCE_SIDE).astype(np.float64)
        base_inv = np.linalg.inv(base)

        out: dict[str, np.ndarray] = {}
        for side in SIDES:
            tcps = np.stack([s.robot.tcp(side) for s in samples], axis=0)
            pose9 = relative_pose9(base, tcps)
            out[f"gripper_{side}_eef_pos"] = pose9[:, :3]
            out[f"gripper_{side}_eef_rot_axis_angle"] = pose9[:, 3:9]
            out[f"gripper_{side}_gripper_width"] = np.asarray(
                [[s.robot.gripper_width(side)] for s in samples], dtype=np.float32
            )
            out[WRIST_RGB_KEYS[side]] = _to_chw_float(
                np.stack([s.wrist_rgb[side] for s in samples], axis=0)
            )

        points, masks = [], []
        for s in samples:
            robot = s.head_state
            mask = self.crop.mask(s.head.pointmap, robot.head, (robot.left_tcp, robot.right_tcp))
            head_in_base = base_inv @ robot.head.astype(np.float64)
            points.append(transform_points(head_in_base, s.head.pointmap))
            masks.append(mask)
        out[HEAD_POINTMAP_KEY] = np.stack(points, axis=0).transpose(0, 3, 1, 2)
        out[HEAD_MASK_KEY] = np.stack(masks, axis=0)[:, None].astype(np.float32)
        out[HEAD_RGB_KEY] = _to_chw_float(np.stack([s.head.rgb for s in samples], axis=0))
        return out

    def build(self, samples: Sequence[ObservationSample]) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(np.ascontiguousarray(value))
            .unsqueeze(0)
            .to(device=self._device, non_blocking=True)
            for key, value in self.build_numpy(samples).items()
        }


def _to_chw_float(images: npt.NDArray[np.uint8]) -> FloatArray:
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"expected uint8 [T,H,W,3], got {images.shape} {images.dtype}")
    return images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
