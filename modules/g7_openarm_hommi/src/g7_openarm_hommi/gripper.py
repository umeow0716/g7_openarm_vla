from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GripperWidthRange:
    """Robot openness (0=closed, 1=open) <-> HoMMI gripper width.

    iPhUMI measures gripper width as the ArUco finger-tag separation in meters,
    so the training scale is not the robot's 0-1 openness. The closed/open
    extremes are taken from the checkpoint's gripper_width normalizer range,
    i.e. the model's normalized gripper value in [-1, 1] maps linearly onto the
    robot's openness in [0, 1].
    """

    closed: float
    opened: float

    def __post_init__(self) -> None:
        if not self.opened > self.closed:
            raise ValueError(f"invalid gripper width range {self.closed}..{self.opened}")

    def to_width(self, openness: float) -> float:
        return self.closed + float(np.clip(openness, 0.0, 1.0)) * (self.opened - self.closed)

    def to_openness(self, width: float) -> float:
        return float(np.clip((width - self.closed) / (self.opened - self.closed), 0.0, 1.0))
