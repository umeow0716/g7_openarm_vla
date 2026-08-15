from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from hommi_train.dataset import rotation_6d_to_matrix_umi

from g7_openarm_utils import GRIPPER_COMMAND_RANGE, gripper_openness_to_command

from .geometry import matrix_to_pose7_wxyz
from .state import RobotSnapshot

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class TargetPoint:
    execute_at: float
    left_pose7: tuple[float, ...]
    right_pose7: tuple[float, ...]
    left_gripper: float
    right_gripper: float


def _wbc_gripper_scalar(openness: float) -> float:
    """Convert HoMMI openness to the legacy WBC scalar consumed by lowlevel.

    HoMMI:   0 = closed, 1 = open.
    WBC path: scalar * GRIPPER_COMMAND_RANGE is the lowcmd coordinate, where
              0 = open and GRIPPER_COMMAND_RANGE = closed.
    """
    clipped = float(np.clip(openness, 0.0, 1.0))
    return gripper_openness_to_command(clipped) / GRIPPER_COMMAND_RANGE


def decode_action_chunk(
    action_chunk: npt.ArrayLike,
    *,
    reference: RobotSnapshot,
    arm: str,
    interval: float,
    obs_horizon: int,
) -> tuple[TargetPoint, ...]:
    """Decode every action against the SAME prediction-time FK reference frame."""
    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 10:
        raise ValueError(f"expected action chunk [N,10], got {actions.shape}")
    if interval <= 0.0:
        raise ValueError(f"interval must be positive, got {interval}")
    if obs_horizon < 1:
        raise ValueError(f"obs_horizon must be >= 1, got {obs_horizon}")

    base = reference.matrix(arm).astype(np.float32, copy=False)
    left_hold = tuple(float(x) for x in reference.left_pose7)
    right_hold = tuple(float(x) for x in reference.right_pose7)
    left_gripper_hold = _wbc_gripper_scalar(reference.left_gripper_openness)
    right_gripper_hold = _wbc_gripper_scalar(reference.right_gripper_openness)

    points: list[TargetPoint] = []
    first_offset_steps = obs_horizon - 1

    for index, action in enumerate(actions):
        relative = np.eye(4, dtype=np.float32)
        relative[:3, 3] = action[:3]
        relative[:3, :3] = rotation_6d_to_matrix_umi(action[3:9])
        target = base @ relative
        target_pose7 = tuple(float(x) for x in matrix_to_pose7_wxyz(target))
        model_gripper = _wbc_gripper_scalar(float(action[9]))

        left_pose7 = left_hold
        right_pose7 = right_hold
        left_gripper = left_gripper_hold
        right_gripper = right_gripper_hold

        if arm == "left":
            left_pose7 = target_pose7
            left_gripper = model_gripper
        elif arm == "right":
            right_pose7 = target_pose7
            right_gripper = model_gripper
        else:
            raise ValueError(f"unknown arm side {arm!r}")

        points.append(
            TargetPoint(
                execute_at=reference.timestamp
                + (first_offset_steps + index) * interval,
                left_pose7=left_pose7,
                right_pose7=right_pose7,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
            )
        )

    return tuple(points)


class AtomicTrajectory:
    """A generation-replaced time-indexed trajectory consumed by the 20 Hz sender."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._points: deque[TargetPoint] = deque()

    def replace(
        self,
        points: Sequence[TargetPoint],
        *,
        now: float,
        drop_expired: bool,
    ) -> int:
        if drop_expired:
            usable = [point for point in points if point.execute_at > now]
        else:
            # Preserve action order but restart it from the next publisher tick.
            usable = [
                TargetPoint(
                    execute_at=now + index * (
                        points[1].execute_at - points[0].execute_at
                        if len(points) > 1
                        else 0.0
                    ),
                    left_pose7=point.left_pose7,
                    right_pose7=point.right_pose7,
                    left_gripper=point.left_gripper,
                    right_gripper=point.right_gripper,
                )
                for index, point in enumerate(points)
            ]

        if not usable:
            return 0

        with self._lock:
            self._points = deque(usable)
        return len(usable)

    def pop_due(self, *, now: float) -> TargetPoint | None:
        latest: TargetPoint | None = None
        with self._lock:
            while self._points and self._points[0].execute_at <= now:
                latest = self._points.popleft()
        return latest

    def clear(self) -> None:
        with self._lock:
            self._points.clear()
