from __future__ import annotations

from types import MappingProxyType
from typing import Iterable

import numpy as np
import numpy.typing as npt

from g7_openarm_utils import (
    ARM_COMMAND_MOTOR_NAMES,
    LEFT_ARM_MOTOR_NAMES,
    LEFT_GRIPPER_MOTOR_NAME,
    LEFT_HARDWARE_MOTOR_NAMES,
    RIGHT_ARM_MOTOR_NAMES,
    RIGHT_GRIPPER_MOTOR_NAME,
    RIGHT_HARDWARE_MOTOR_NAMES,
    gripper_openness_to_command,
)

INITIAL_DURATION_S = 5.0

_INITIAL_KP_BY_MOTOR_NAME = MappingProxyType(
    {
        **{
            name: (200.0 if int(name.split("_")[1]) <= 4 else 30.0)
            for name in LEFT_ARM_MOTOR_NAMES
        },
        LEFT_GRIPPER_MOTOR_NAME: 30.0,
        **{
            name: (200.0 if int(name.split("_")[1]) <= 4 else 30.0)
            for name in RIGHT_ARM_MOTOR_NAMES
        },
        RIGHT_GRIPPER_MOTOR_NAME: 30.0,
    }
)
_INITIAL_KD_BY_MOTOR_NAME = MappingProxyType(
    {
        **{
            name: (2.5 if int(name.split("_")[1]) <= 4 else 0.2)
            for name in LEFT_ARM_MOTOR_NAMES
        },
        LEFT_GRIPPER_MOTOR_NAME: 0.2,
        **{
            name: (2.5 if int(name.split("_")[1]) <= 4 else 0.2)
            for name in RIGHT_ARM_MOTOR_NAMES
        },
        RIGHT_GRIPPER_MOTOR_NAME: 0.2,
    }
)


def initial_kp(motor_names: Iterable[str]) -> npt.NDArray[np.float64]:
    return np.asarray([_INITIAL_KP_BY_MOTOR_NAME[name] for name in motor_names], dtype=np.float64)


def initial_kd(motor_names: Iterable[str]) -> npt.NDArray[np.float64]:
    return np.asarray([_INITIAL_KD_BY_MOTOR_NAME[name] for name in motor_names], dtype=np.float64)


# Compatibility views for callers/tests that still need one-side arrays.
INITIAL_KP = initial_kp(LEFT_HARDWARE_MOTOR_NAMES)
INITIAL_KD = initial_kd(LEFT_HARDWARE_MOTOR_NAMES)
INITIAL_KP.setflags(write=False)
INITIAL_KD.setflags(write=False)


class ArmInitializer:
    """Generate a time-based linear trajectory in the named arm-command layout."""

    def __init__(
        self,
        target_7: tuple[float, ...],
        *,
        target_gripper: float = 1.0,
        left_enabled: bool = True,
        right_enabled: bool = True,
        duration_s: float = INITIAL_DURATION_S,
    ) -> None:
        if len(target_7) != len(LEFT_ARM_MOTOR_NAMES):
            raise ValueError(
                f"target_7 must contain {len(LEFT_ARM_MOTOR_NAMES)} values, got {len(target_7)}"
            )
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
        target_by_name = {
            **dict(zip(LEFT_ARM_MOTOR_NAMES, target, strict=True)),
            LEFT_GRIPPER_MOTOR_NAME: gripper_command,
            **dict(zip(RIGHT_ARM_MOTOR_NAMES, target, strict=True)),
            RIGHT_GRIPPER_MOTOR_NAME: gripper_command,
        }
        self.target_16 = np.asarray(
            [target_by_name[name] for name in ARM_COMMAND_MOTOR_NAMES],
            dtype=np.float64,
        )

        active_by_name = {
            **{name: left_enabled for name in LEFT_HARDWARE_MOTOR_NAMES},
            **{name: right_enabled for name in RIGHT_HARDWARE_MOTOR_NAMES},
        }
        self.active_16 = np.asarray(
            [active_by_name[name] for name in ARM_COMMAND_MOTOR_NAMES],
            dtype=np.bool_,
        )
        self.duration_s = float(duration_s)
        self.start_time: float | None = None
        self.start_16: npt.NDArray[np.float64] | None = None
        self.effective_target_16: npt.NDArray[np.float64] | None = None

    @property
    def started(self) -> bool:
        return self.start_time is not None

    def plan_from_state(self, q_16: npt.ArrayLike, *, now: float) -> None:
        q = np.asarray(q_16, dtype=np.float64)
        expected_shape = (len(ARM_COMMAND_MOTOR_NAMES),)
        if q.shape != expected_shape:
            raise ValueError(f"q_16 must have shape {expected_shape}, got {q.shape}")
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
            raise RuntimeError("ArmInitializer.plan_from_state() must be called before sample()")

        elapsed = max(0.0, float(now) - self.start_time)
        alpha = min(elapsed / self.duration_s, 1.0)
        delta = self.effective_target_16 - self.start_16
        q_des = self.start_16 + delta * alpha

        if alpha < 1.0:
            dq_des = delta / self.duration_s
        else:
            dq_des = np.zeros((len(ARM_COMMAND_MOTOR_NAMES),), dtype=np.float64)

        return q_des, dq_des, alpha >= 1.0
