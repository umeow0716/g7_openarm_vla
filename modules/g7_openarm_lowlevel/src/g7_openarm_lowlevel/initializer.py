from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from g7_openarm_config import general_config
from g7_openarm_pinnzoo import PinnZooModel, build_lib_x, kinematics, pinnzoo_q_idx
from g7_openarm_utils.layout import (
    ARM_JOINT,
    LEFT_ARM_ENABLED,
    RIGHT_ARM_ENABLED,
    KinBody,
)

if TYPE_CHECKING:
    from unitree_sdk2py.idl.default import HGLowState_

    from g7_openarm_idl import Odom

FloatArray = npt.NDArray[np.float64]

# Peak joint speed of the start-up path; the duration follows the largest move.
INITIAL_SPEED = 0.3  # rad/s
MIN_DURATION = 2.0  # s
# Feedback gain pulling the reference back onto the path.
PATH_GAIN = 2.0  # 1/s
# Unlock once every active TCP has stayed within these bounds of its
# initial_pos pose for SETTLE_TIME seconds without interruption.
TCP_POSITION_TOLERANCE = 0.03  # m
TCP_ANGLE_TOLERANCE = np.deg2rad(10.0)  # rad
SETTLE_TIME = 2.0  # s

ACTIVE_ARM_MASK = np.array([LEFT_ARM_ENABLED] * 7 + [RIGHT_ARM_ENABLED] * 7, dtype=bool)


def initial_arm_positions() -> FloatArray:
    """``general.initial_pos`` in ARM_JOINT order (L1..L7, R1..R7)."""
    return np.array([general_config.initial_pos[joint] for joint in ARM_JOINT], dtype=np.float64)


@dataclass
class InitialPoseSequencer:
    """Linear joint-space path from the start-up pose to ``general.initial_pos``.

    The path is turned into a joint velocity command (path velocity plus a
    small position feedback) so it still goes through the ReferenceGenerator's
    jerk / acceleration / velocity / position limits and torque governor.
    Inactive arms are not moved.
    """

    start: FloatArray
    goal: FloatArray
    duration: float

    @classmethod
    def plan(cls, start: npt.ArrayLike, goal: npt.ArrayLike | None = None) -> InitialPoseSequencer:
        q0 = np.asarray(start, dtype=np.float64)
        q1 = initial_arm_positions() if goal is None else np.asarray(goal, dtype=np.float64)
        if q0.shape != (14,) or q1.shape != (14,):
            raise ValueError(f"expected 14 arm joints, got {q0.shape} -> {q1.shape}")
        q1 = np.where(ACTIVE_ARM_MASK, q1, q0)
        distance = float(np.max(np.abs(q1 - q0))) if q0.size else 0.0
        return cls(start=q0, goal=q1, duration=max(MIN_DURATION, distance / INITIAL_SPEED))

    def path(self, t: float) -> tuple[FloatArray, FloatArray]:
        """Path position and velocity at ``t`` seconds after start."""
        s = float(np.clip(t / self.duration, 0.0, 1.0))
        q = self.start + s * (self.goal - self.start)
        dq = (self.goal - self.start) / self.duration if s < 1.0 else np.zeros(14)
        return q, dq

    def dq_command(self, t: float, q_ref: npt.ArrayLike) -> FloatArray:
        q_path, dq_path = self.path(t)
        dq = dq_path + PATH_GAIN * (q_path - np.asarray(q_ref, dtype=np.float64))
        return np.where(ACTIVE_ARM_MASK, dq, 0.0)


class TcpReachMonitor:
    """TCP pose error to the initial pose, with a continuous settle timer."""

    def __init__(self, goal_q: npt.ArrayLike) -> None:
        self._model = PinnZooModel()
        self._goal_q = np.asarray(goal_q, dtype=np.float64)
        self._q_idx = [pinnzoo_q_idx(joint) for joint in ARM_JOINT]
        size = self._model.kinematics_body_size
        slices = {
            KinBody.name_to_kin_body(name): slice(i * size, i * size + 7)
            for i, name in enumerate(self._model.kinematics_bodies)
        }
        self._tcp_slices = [
            slices[body]
            for body, active in (
                (KinBody.L_TCP, LEFT_ARM_ENABLED),
                (KinBody.R_TCP, RIGHT_ARM_ENABLED),
            )
            if active
        ]
        self._settled_since: float | None = None

    def errors(self, lowstate: HGLowState_, odom: Odom) -> tuple[float, float]:
        """Worst (position [m], angle [rad]) TCP error over the active arms."""
        x = np.asarray(build_lib_x(self._model, lowstate, odom), dtype=np.float64)
        x_goal = x.copy()
        x_goal[self._q_idx] = self._goal_q
        now = np.asarray(kinematics(self._model, x), dtype=np.float64)
        goal = np.asarray(kinematics(self._model, x_goal), dtype=np.float64)
        pos_err, ang_err = 0.0, 0.0
        for sl in self._tcp_slices:
            a, b = now[sl], goal[sl]  # [x, y, z, qw, qx, qy, qz]
            pos_err = max(pos_err, float(np.linalg.norm(a[:3] - b[:3])))
            qa, qb = a[3:] / np.linalg.norm(a[3:]), b[3:] / np.linalg.norm(b[3:])
            ang_err = max(ang_err, 2.0 * float(np.arccos(np.clip(abs(qa @ qb), 0.0, 1.0))))
        return pos_err, ang_err

    def reset(self) -> None:
        self._settled_since = None

    def settled(self, now: float, pos_err: float, ang_err: float) -> bool:
        """True once the errors stayed within tolerance for SETTLE_TIME seconds."""
        if pos_err < TCP_POSITION_TOLERANCE and ang_err < TCP_ANGLE_TOLERANCE:
            if self._settled_since is None:
                self._settled_since = now
            return now - self._settled_since >= SETTLE_TIME
        self._settled_since = None
        return False
