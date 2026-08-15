from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from hommi_train.dataset import relative_pose9

from .camera import RGBFrame
from .state import RobotSnapshot

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ObservationSample:
    sequence: int
    timestamp: float
    rgb: RGBFrame
    robot: RobotSnapshot


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

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()


def _preprocess_rgb_uint8(
    frames: torch.Tensor,
    *,
    image_size: int,
) -> torch.Tensor:
    """Match hommi_train's deterministic black-pad-to-square + area resize."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"expected [N,3,H,W], got {tuple(frames.shape)}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    height, width = int(frames.shape[-2]), int(frames.shape[-1])
    side = max(height, width)
    pad_height = side - height
    pad_width = side - width
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    x = F.pad(frames, (left, right, top, bottom), mode="constant", value=0)
    if side != image_size:
        x = F.interpolate(
            x.to(dtype=torch.float32),
            size=(image_size, image_size),
            mode="area",
        )
        x = x.round_().clamp_(0, 255).to(dtype=torch.uint8)
    elif x.dtype != torch.uint8:
        x = x.round().clamp(0, 255).to(dtype=torch.uint8)
    return x.contiguous()


class HommiObservationBuilder:
    """Build the canonical single-arm HoMMI [B,T,...] observation dictionary."""

    REQUIRED_LOW_DIM = (
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
    )

    def __init__(
        self,
        shape_meta: Mapping[str, Any],
        *,
        arm: str,
        device: torch.device,
    ) -> None:
        self._shape_meta = shape_meta
        self._arm = arm
        self._device = device

        obs_meta = shape_meta.get("obs")
        if not isinstance(obs_meta, Mapping):
            raise ValueError("model shape_meta is missing obs metadata")

        active = {
            key: value
            for key, value in obs_meta.items()
            if not value.get("ignore_by_policy", False)
        }
        rgb_keys = [
            key
            for key, value in active.items()
            if value.get("type", "low_dim") == "rgb"
        ]
        if len(rgb_keys) != 1:
            raise ValueError(
                "g7_openarm_hommi currently supports the canonical single-camera "
                f"single-arm model, got RGB keys {rgb_keys}"
            )
        self.rgb_key = rgb_keys[0]

        for key in self.REQUIRED_LOW_DIM:
            if key not in active:
                raise ValueError(f"model is missing required observation key {key!r}")

        horizons = {int(value.get("horizon", 1)) for value in active.values()}
        if len(horizons) != 1:
            raise ValueError(f"all active observations must share one horizon, got {horizons}")
        self.horizon = horizons.pop()
        if self.horizon != 2:
            raise ValueError(
                f"this deployment expects HoMMI obs horizon 2, model requires {self.horizon}"
            )

        rgb_shape = tuple(int(x) for x in active[self.rgb_key]["shape"])
        if len(rgb_shape) != 3 or rgb_shape[0] != 3 or rgb_shape[1] != rgb_shape[2]:
            raise ValueError(
                f"RGB model input must be square [3,H,W], got {rgb_shape}"
            )
        self.image_size = rgb_shape[1]

        action_meta = shape_meta.get("action")
        if not isinstance(action_meta, Mapping):
            raise ValueError("model shape_meta is missing action metadata")
        action_shape = tuple(int(x) for x in action_meta["shape"])
        if action_shape != (10,):
            raise ValueError(
                "g7_openarm_hommi currently expects the user's single-arm 10-D "
                f"model action, got {action_shape}"
            )

    def build(
        self,
        samples: Sequence[ObservationSample],
    ) -> dict[str, torch.Tensor]:
        if len(samples) != self.horizon:
            raise ValueError(
                f"expected {self.horizon} observation samples, got {len(samples)}"
            )

        latest_matrix = samples[-1].robot.matrix(self._arm)
        world_history = np.stack(
            [sample.robot.matrix(self._arm) for sample in samples],
            axis=0,
        )
        pose9 = relative_pose9(latest_matrix, world_history)
        gripper = np.asarray(
            [
                [sample.robot.gripper_openness(self._arm)]
                for sample in samples
            ],
            dtype=np.float32,
        )

        rgb_np = np.stack([sample.rgb for sample in samples], axis=0)
        rgb = torch.from_numpy(rgb_np).permute(0, 3, 1, 2)
        rgb = _preprocess_rgb_uint8(rgb, image_size=self.image_size)
        rgb = rgb.to(dtype=torch.float32).div_(255.0)

        tensors = {
            self.rgb_key: rgb,
            "robot0_eef_pos": torch.from_numpy(
                pose9[:, :3].astype(np.float32, copy=False)
            ),
            "robot0_eef_rot_axis_angle": torch.from_numpy(
                pose9[:, 3:9].astype(np.float32, copy=False)
            ),
            "robot0_gripper_width": torch.from_numpy(gripper),
        }
        return {
            key: value.unsqueeze(0).to(
                device=self._device,
                non_blocking=True,
            )
            for key, value in tensors.items()
        }
