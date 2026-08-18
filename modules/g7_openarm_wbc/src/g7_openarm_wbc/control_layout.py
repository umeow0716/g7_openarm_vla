from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from g7_openarm_config import ControlMode
from g7_openarm_utils import (
    AMR_COMMAND_NAMES,
    ARM_COMMAND_MOTOR_NAMES,
    ARM_MOTOR_NAMES,
    LEFT_ARM_MOTOR_NAMES,
    LEFT_GRIPPER_MOTOR_NAME,
    RIGHT_ARM_MOTOR_NAMES,
    RIGHT_GRIPPER_MOTOR_NAME,
    arm_command_index,
)

BASE_CONTROL_NAMES = AMR_COMMAND_NAMES
ARM_CONTROL_NAMES = ARM_MOTOR_NAMES
BASE_CONTROL_SIZE = len(BASE_CONTROL_NAMES)
ARM_CONTROL_SIZE = len(ARM_CONTROL_NAMES)
SINGLE_ARM_CONTROL_SIZE = len(LEFT_ARM_MOTOR_NAMES)

_CONTROL_NAMES_BY_BASE_ENABLED = MappingProxyType(
    {
        False: ARM_CONTROL_NAMES,
        True: BASE_CONTROL_NAMES + ARM_CONTROL_NAMES,
    }
)
_CONTROL_INDEX_BY_BASE_ENABLED = MappingProxyType(
    {
        base_enabled: MappingProxyType(
            {name: index for index, name in enumerate(names)}
        )
        for base_enabled, names in _CONTROL_NAMES_BY_BASE_ENABLED.items()
    }
)


def control_names(*, base_enabled: bool) -> tuple[str, ...]:
    return _CONTROL_NAMES_BY_BASE_ENABLED[base_enabled]


def control_index(name: str, *, base_enabled: bool) -> int:
    try:
        return _CONTROL_INDEX_BY_BASE_ENABLED[base_enabled][name]
    except KeyError as exc:
        raise KeyError(f"unknown WBC control name {name!r}") from exc


@lru_cache(maxsize=None)
def _cached_control_indices(
    names: tuple[str, ...],
    base_enabled: bool,
) -> npt.NDArray[np.intp]:
    indices = np.asarray(
        [control_index(name, base_enabled=base_enabled) for name in names],
        dtype=np.intp,
    )
    indices.setflags(write=False)
    return indices


def control_indices(
    names: tuple[str, ...],
    *,
    base_enabled: bool,
) -> npt.NDArray[np.intp]:
    """Resolve semantic controls once and reuse immutable integer indices."""
    return _cached_control_indices(tuple(names), base_enabled)


ARM_CONTROL_INDEX_BY_NAME = MappingProxyType(
    {name: index for index, name in enumerate(ARM_CONTROL_NAMES)}
)


def tracked_arms(control_mode: ControlMode) -> tuple[bool, bool]:
    """Return whether the WBC should track the left and right EE targets."""
    track_left = control_mode not in (
        ControlMode.BASE_ONLY,
        ControlMode.RIGHT_ARM,
        ControlMode.RIGHT_ARM_ONLY,
    )
    track_right = control_mode not in (
        ControlMode.BASE_ONLY,
        ControlMode.LEFT_ARM,
        ControlMode.LEFT_ARM_ONLY,
    )
    return track_left, track_right


def zero_inactive_arm_controls(
    u: npt.NDArray[np.float64],
    *,
    base_enabled: bool,
    track_left: bool,
    track_right: bool,
) -> None:
    """Force non-tracked arm joint velocities to zero in-place by motor name."""
    expected_size = control_size(base_enabled=base_enabled)
    if u.shape != (expected_size,):
        raise ValueError(f"Expected control vector shape ({expected_size},), got {u.shape}")

    if not track_left:
        u[control_indices(LEFT_ARM_MOTOR_NAMES, base_enabled=base_enabled)] = 0.0
    if not track_right:
        u[control_indices(RIGHT_ARM_MOTOR_NAMES, base_enabled=base_enabled)] = 0.0


def control_size(*, base_enabled: bool) -> int:
    return len(control_names(base_enabled=base_enabled))


def split_control_vector(
    u: npt.NDArray[np.float64],
    *,
    base_enabled: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Split WBC output without relying on positional slices."""
    expected_size = control_size(base_enabled=base_enabled)
    if u.shape != (expected_size,):
        raise ValueError(f"Expected control vector shape ({expected_size},), got {u.shape}")

    if base_enabled:
        base = u[control_indices(BASE_CONTROL_NAMES, base_enabled=True)].copy()
    else:
        base = np.zeros(BASE_CONTROL_SIZE, dtype=np.float64)

    arms = u[control_indices(ARM_CONTROL_NAMES, base_enabled=base_enabled)].copy()
    return base, arms


def openarm_command_from_arm_control(
    arm_u: npt.NDArray[np.float64],
    *,
    left_gripper_openness: float,
    right_gripper_openness: float,
) -> npt.NDArray[np.float64]:
    """Build the fixed 16-slot OpenArmCmd wire array from semantic motor names.

    The IDL wire format is necessarily index-based, but this is the only WBC
    boundary that converts the 14 named arm controls plus two named grippers
    into that transport representation. Gripper q is normalized openness:
    0=closed, 1=open.
    """
    if arm_u.shape != (ARM_CONTROL_SIZE,):
        raise ValueError(f"Expected arm control shape ({ARM_CONTROL_SIZE},), got {arm_u.shape}")

    gripper_values = {
        LEFT_GRIPPER_MOTOR_NAME: float(left_gripper_openness),
        RIGHT_GRIPPER_MOTOR_NAME: float(right_gripper_openness),
    }
    for name, openness in gripper_values.items():
        if not np.isfinite(openness) or not 0.0 <= openness <= 1.0:
            raise ValueError(
                f"{name} openness must be finite and in [0, 1], got {openness}"
            )

    output = np.zeros(len(ARM_COMMAND_MOTOR_NAMES), dtype=np.float64)
    for motor_name in ARM_CONTROL_NAMES:
        output[arm_command_index(motor_name)] = arm_u[ARM_CONTROL_INDEX_BY_NAME[motor_name]]
    output[arm_command_index(LEFT_GRIPPER_MOTOR_NAME)] = gripper_values[LEFT_GRIPPER_MOTOR_NAME]
    output[arm_command_index(RIGHT_GRIPPER_MOTOR_NAME)] = gripper_values[RIGHT_GRIPPER_MOTOR_NAME]
    return output
