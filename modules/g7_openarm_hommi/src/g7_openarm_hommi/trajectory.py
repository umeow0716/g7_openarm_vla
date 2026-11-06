from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from .geometry import matrix_to_pose7_wxyz, rotation_6d_to_matrix_umi
from .gripper import GripperWidthRange
from .state import RobotSnapshot

FloatArray = npt.NDArray[np.float32]

# ego3d_lookat_policy action layout (23-D). HoMMI builds it from the sampler's
# robot order (head, left, right), with the head contributing only its look-at
# point. Verified against the checkpoint's action normalizer: dims 0-2 match the
# camera_head_lookatpoint statistics, 12 / 22 match the left / right
# gripper_width statistics, and 6 / 10 / 16 / 20 are the rot6d r00 / r11 terms
# (min > 0.3).
ACTION_DIM = 23
LOOKAT = slice(0, 3)
ARM_SLICES = {
    "left": (slice(3, 6), slice(6, 12), 12),
    "right": (slice(13, 16), slice(16, 22), 22),
}


@dataclass(frozen=True, slots=True)
class TargetPoint:
    execute_at: float
    left_pose7: tuple[float, ...]
    right_pose7: tuple[float, ...]
    left_gripper_openness: float
    right_gripper_openness: float
    look_at_point: tuple[float, float, float]


def snapshot_target(
    robot: RobotSnapshot,
    *,
    gripper_ranges: dict[str, GripperWidthRange],
    look_ahead: float = 0.5,
) -> TargetPoint:
    """Target that keeps the robot where it is (seed for the first chunk)."""
    look_at = robot.head[:3, 3] + look_ahead * robot.head[:3, 2]
    return TargetPoint(
        execute_at=robot.timestamp,
        left_pose7=tuple(float(x) for x in matrix_to_pose7_wxyz(robot.left_tcp)),
        right_pose7=tuple(float(x) for x in matrix_to_pose7_wxyz(robot.right_tcp)),
        left_gripper_openness=gripper_ranges["left"].to_openness(robot.left_gripper_width),
        right_gripper_openness=gripper_ranges["right"].to_openness(robot.right_gripper_width),
        look_at_point=(float(look_at[0]), float(look_at[1]), float(look_at[2])),
    )


def decode_action_chunk(
    action_chunk: npt.ArrayLike,
    *,
    reference: RobotSnapshot,
    gripper_ranges: dict[str, GripperWidthRange],
    interval: float,
    obs_horizon: int,
) -> tuple[TargetPoint, ...]:
    """Decode a ``[N, 23]`` chunk against the newest observation's left TCP.

    Poses: ``T_world = T_left_ref @ T_rel`` (HoMMI ``convert_pose_mat_rep(...,
    backward=True)``) in the robot's ``*_tcp`` frame. The look-at point is
    mapped from the left-TCP frame to world. Gripper widths are converted to
    robot openness with the checkpoint's per-side width range. Step ``i`` of
    the returned ``action`` (which starts at ``action_pred[obs_horizon - 1]``)
    executes at ``t_obs + (obs_horizon - 1 + i) * interval``.
    """
    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"expected action chunk [N,{ACTION_DIM}], got {actions.shape}")
    if interval <= 0.0:
        raise ValueError(f"interval must be positive, got {interval}")
    if obs_horizon < 1:
        raise ValueError(f"obs_horizon must be >= 1, got {obs_horizon}")

    base = reference.left_tcp.astype(np.float64)
    points: list[TargetPoint] = []
    for index, action in enumerate(actions):
        poses: dict[str, tuple[float, ...]] = {}
        openness: dict[str, float] = {}
        for side, (pos, rot, width) in ARM_SLICES.items():
            relative = np.eye(4, dtype=np.float64)
            relative[:3, 3] = action[pos]
            relative[:3, :3] = rotation_6d_to_matrix_umi(action[rot])
            world = (base @ relative).astype(np.float32)
            poses[side] = tuple(float(x) for x in matrix_to_pose7_wxyz(world))
            openness[side] = gripper_ranges[side].to_openness(float(action[width]))

        look_at = base[:3, :3] @ action[LOOKAT].astype(np.float64) + base[:3, 3]
        points.append(
            TargetPoint(
                execute_at=reference.timestamp + (obs_horizon - 1 + index) * interval,
                left_pose7=poses["left"],
                right_pose7=poses["right"],
                left_gripper_openness=openness["left"],
                right_gripper_openness=openness["right"],
                look_at_point=(float(look_at[0]), float(look_at[1]), float(look_at[2])),
            )
        )
    return tuple(points)


def _slerp_wxyz(q0: npt.ArrayLike, q1: npt.ArrayLike, alpha: float) -> np.ndarray:
    """Shortest-arc slerp between two wxyz quaternions (hemisphere-aligned)."""
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(a @ b)
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        out = a + alpha * (b - a)
    else:
        theta = np.arccos(dot)
        out = (np.sin((1.0 - alpha) * theta) * a + np.sin(alpha * theta) * b) / np.sin(theta)
    out /= np.linalg.norm(out)
    return out if out[0] >= 0.0 else -out


def _lerp(a: Sequence[float], b: Sequence[float], alpha: float) -> tuple[float, ...]:
    return tuple(float(x + alpha * (y - x)) for x, y in zip(a, b, strict=True))


def interpolate_target(p0: TargetPoint, p1: TargetPoint, now: float) -> TargetPoint:
    """Blend two timestamped targets: lerp positions/grippers/look-at, slerp rotations."""
    span = p1.execute_at - p0.execute_at
    alpha = 1.0 if span <= 0.0 else float(np.clip((now - p0.execute_at) / span, 0.0, 1.0))

    def pose(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
        return _lerp(a[:3], b[:3], alpha) + tuple(
            float(x) for x in _slerp_wxyz(a[3:], b[3:], alpha)
        )

    return TargetPoint(
        execute_at=now,
        left_pose7=pose(p0.left_pose7, p1.left_pose7),
        right_pose7=pose(p0.right_pose7, p1.right_pose7),
        left_gripper_openness=_lerp(
            (p0.left_gripper_openness,), (p1.left_gripper_openness,), alpha
        )[0],
        right_gripper_openness=_lerp(
            (p0.right_gripper_openness,), (p1.right_gripper_openness,), alpha
        )[0],
        look_at_point=_lerp(p0.look_at_point, p1.look_at_point, alpha),  # type: ignore[arg-type]
    )


class AtomicTrajectory:
    """Timestamped action chunk sampled (interpolated) by the publisher.

    The publisher runs at the WBC rate and calls :meth:`sample`, which linearly
    interpolates positions, gripper openness and the look-at point and slerps
    rotations between the two action steps bracketing ``now``.

    A new chunk replaces the old one *from the currently commanded target*: the
    target sampled at replacement time becomes the first knot, so chunk
    switches never jump. Steps whose time already passed are superseded by that
    knot; if the whole chunk is late, the robot blends to its final step within
    one action interval instead of freezing on the previous target.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._knots: list[TargetPoint] = []

    def replace(
        self,
        points: Sequence[TargetPoint],
        *,
        now: float,
        latency_compensation: bool,
    ) -> int:
        """Replace the trajectory with a new chunk; return how many steps were late."""
        if not points:
            raise ValueError("cannot replace the trajectory with an empty chunk")
        step = points[1].execute_at - points[0].execute_at if len(points) > 1 else 0.0
        if latency_compensation:
            chunk = list(points)
        else:
            # Re-time the chunk so its first step is reached one interval from now.
            chunk = [
                replace(point, execute_at=now + (index + 1) * step)
                for index, point in enumerate(points)
            ]
        future = [point for point in chunk if point.execute_at > now]
        expired = len(chunk) - len(future)
        if not future:
            future = [replace(chunk[-1], execute_at=now + max(step, 1e-3))]

        with self._lock:
            current = self._sample_locked(now)
            self._knots = ([current] if current is not None else []) + future
        return expired

    def _sample_locked(self, now: float) -> TargetPoint | None:
        knots = self._knots
        if not knots:
            return None
        if now <= knots[0].execute_at:
            return replace(knots[0], execute_at=now)
        if now >= knots[-1].execute_at:
            return replace(knots[-1], execute_at=now)
        for p0, p1 in pairwise(knots):
            if now <= p1.execute_at:
                return interpolate_target(p0, p1, now)
        return replace(knots[-1], execute_at=now)

    def sample(self, *, now: float) -> TargetPoint | None:
        """Return the interpolated target at ``now`` (``None`` before any chunk)."""
        with self._lock:
            return self._sample_locked(now)

    def hold(self, point: TargetPoint, *, now: float) -> None:
        """Replace the trajectory with a single held target."""
        with self._lock:
            self._knots = [replace(point, execute_at=now)]

    def freeze(self, *, now: float) -> None:
        """Hold the currently commanded target (no-op before any chunk)."""
        with self._lock:
            current = self._sample_locked(now)
            self._knots = [current] if current is not None else []

    def clear(self) -> None:
        with self._lock:
            self._knots = []
