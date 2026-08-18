from __future__ import annotations

from types import MappingProxyType
from typing import Iterable

import numpy as np
import numpy.typing as npt

from .joint_layout import (
    ARM_JOINT_NAMES,
    ARM_MOTOR_NAMES,
)

# Safety limits are keyed by semantic motor name. The exported arrays below
# are compatibility views generated in ARM_MOTOR_NAMES order; no caller needs
# to know the URDF/LowState/PinnZoo integer order to select a limit.
_ARM_LIMITS_BY_MOTOR_NAME = MappingProxyType(
    {
        "L_1": (-1.3962629999999998, 3.490659, 16.754666),
        "L_2": (-0.17453267320510335, 3.3161253267948965, 16.754666),
        "L_3": (-np.pi / 6.0, np.pi / 3.0, 5.445426),
        "L_4": (-2.443461, 0.0, 5.445426),
        "L_5": (-1.570796, 1.570796, 20.943946),
        "L_6": (-0.785398, 0.785398, 20.943946),
        "L_7": (-1.570796, 1.570796, 20.943946),
        "R_1": (-1.396263, 3.490659, 16.754666),
        "R_2": (-3.3161253267948965, 0.17453267320510335, 16.754666),
        "R_3": (-np.pi / 3.0, np.pi / 6.0, 5.445426),
        "R_4": (-2.443461, 0.0, 5.445426),
        "R_5": (-1.570796, 1.570796, 20.943946),
        "R_6": (-0.785398, 0.785398, 20.943946),
        "R_7": (-1.570796, 1.570796, 20.943946),
    }
)


def arm_limit_arrays(
    motor_names: Iterable[str] = ARM_MOTOR_NAMES,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    names = tuple(motor_names)
    try:
        limits = [_ARM_LIMITS_BY_MOTOR_NAME[name] for name in names]
    except KeyError as exc:
        raise KeyError(f"no arm safety limits defined for motor {exc.args[0]!r}") from exc
    values = np.asarray(limits, dtype=np.float64)
    if values.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    return values[:, 0], values[:, 1], values[:, 2]


ARM_POSITION_LOWER_RAD, ARM_POSITION_UPPER_RAD, ARM_VELOCITY_LIMIT_RAD_S = arm_limit_arrays()
for _array in (
    ARM_POSITION_LOWER_RAD,
    ARM_POSITION_UPPER_RAD,
    ARM_VELOCITY_LIMIT_RAD_S,
):
    _array.setflags(write=False)

# Joint-name views are useful when comparing directly with URDF/PinnZoo names.
ARM_POSITION_LOWER_BY_JOINT_NAME = MappingProxyType(
    {
        joint: float(_ARM_LIMITS_BY_MOTOR_NAME[motor][0])
        for motor, joint in zip(ARM_MOTOR_NAMES, ARM_JOINT_NAMES, strict=True)
    }
)
ARM_POSITION_UPPER_BY_JOINT_NAME = MappingProxyType(
    {
        joint: float(_ARM_LIMITS_BY_MOTOR_NAME[motor][1])
        for motor, joint in zip(ARM_MOTOR_NAMES, ARM_JOINT_NAMES, strict=True)
    }
)
ARM_VELOCITY_LIMIT_BY_JOINT_NAME = MappingProxyType(
    {
        joint: float(_ARM_LIMITS_BY_MOTOR_NAME[motor][2])
        for motor, joint in zip(ARM_MOTOR_NAMES, ARM_JOINT_NAMES, strict=True)
    }
)


def position_limited_velocity_bounds(
    position: npt.ArrayLike,
    velocity_limit: npt.ArrayLike,
    dt: float,
    *,
    motor_names: Iterable[str] = ARM_MOTOR_NAMES,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return named-joint velocity bounds respecting rate and position limits.

    ``position`` and ``velocity_limit`` are interpreted in the exact order of
    ``motor_names``. When a measured joint is already outside its configured
    range, the bounds allow only motion back toward the valid interval.
    """
    names = tuple(motor_names)
    q = np.asarray(position, dtype=np.float64)
    vmax = np.asarray(velocity_limit, dtype=np.float64)
    position_lower, position_upper, _ = arm_limit_arrays(names)

    expected_shape = (len(names),)
    if q.shape != expected_shape:
        raise ValueError(f"position must have shape {expected_shape}, got {q.shape}")
    if vmax.shape != expected_shape:
        raise ValueError(f"velocity_limit must have shape {expected_shape}, got {vmax.shape}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if np.any(~np.isfinite(q)):
        raise ValueError("position contains non-finite values")
    if np.any(~np.isfinite(vmax)) or np.any(vmax < 0.0):
        raise ValueError("velocity_limit must contain finite non-negative values")

    lower = np.maximum(-vmax, (position_lower - q) / dt)
    upper = np.minimum(vmax, (position_upper - q) / dt)

    below = q < position_lower
    above = q > position_upper

    lower[below] = 0.0
    upper[below] = vmax[below]
    lower[above] = -vmax[above]
    upper[above] = 0.0

    return lower, upper
