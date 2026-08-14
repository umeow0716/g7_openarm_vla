from __future__ import annotations

import numpy as np
import numpy.typing as npt

from g7_openarm_utils import gripper_openness_to_command

INITIAL_DURATION_S = 5.0
INITIAL_KP = np.array([200.0, 200.0, 100.0, 100.0, 50.0, 50.0, 20.0, 20.0], dtype=np.float64)
INITIAL_KD = np.array([2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)


class ArmInitializer:
    """Generate a time-based linear trajectory for enabled arms and grippers.

    Motor-vector order is [L1..L7, L_gripper, R1..R7, R_gripper].
    ``target_gripper`` uses normalized openness: 0.0=closed, 1.0=open.
    """

    def __init__(
        self,
        target_7: tuple[float, ...],
        *,
        target_gripper: float = 1.0,
        left_enabled: bool = True,
        right_enabled: bool = True,
        duration_s: float = INITIAL_DURATION_S,
    ) -> None:
        if len(target_7) != 7:
            raise ValueError(f"target_7 must contain 7 values, got {len(target_7)}")
        if type(left_enabled) is not bool or type(right_enabled) is not bool:
            raise ValueError("left_enabled and right_enabled must be bool")
        if not left_enabled and not right_enabled:
            raise ValueError("at least one arm must be enabled for initialization")
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError(f"duration_s must be finite and positive, got {duration_s}")

        target = np.asarray(target_7, dtype=np.float64)
        if not np.all(np.isfinite(target)):
            raise ValueError("target_7 contains non-finite values")

        gripper_command = gripper_openness_to_command(target_gripper)
        self.target_16 = np.concatenate(
            [target, [gripper_command], target, [gripper_command]],
            dtype=np.float64,
        )
        self.active_16 = np.array(
            [left_enabled] * 8 + [right_enabled] * 8,
            dtype=np.bool_,
        )
        self.duration_s = float(duration_s)
        self.start_time: float | None = None
        self.start_16: npt.NDArray[np.float64] | None = None
        self.effective_target_16: npt.NDArray[np.float64] | None = None

    @property
    def started(self) -> bool:
        return self.start_time is not None

    def start(self, q_16: npt.ArrayLike, *, now: float) -> None:
        q = np.asarray(q_16, dtype=np.float64)
        if q.shape != (16,):
            raise ValueError(f"q_16 must have shape (16,), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("q_16 contains non-finite values")
        if not np.isfinite(now):
            raise ValueError(f"now must be finite, got {now}")

        self.start_16 = q.copy()
        self.effective_target_16 = np.where(self.active_16, self.target_16, q)
        self.start_time = float(now)

    def sample(
        self,
        *,
        now: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool]:
        if (
            self.start_time is None
            or self.start_16 is None
            or self.effective_target_16 is None
        ):
            raise RuntimeError("ArmInitializer.start() must be called before sample()")

        elapsed = max(0.0, float(now) - self.start_time)
        alpha = min(elapsed / self.duration_s, 1.0)
        delta = self.effective_target_16 - self.start_16
        q_des = self.start_16 + delta * alpha

        if alpha < 1.0:
            dq_des = delta / self.duration_s
        else:
            dq_des = np.zeros((16,), dtype=np.float64)

        return q_des, dq_des, alpha >= 1.0
