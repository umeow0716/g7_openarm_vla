from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Arm ordering used by WBC/PinnZoo:
#   [L1, L2, L3, L4, L5, L6, L7, R1, R2, R3, R4, R5, R6, R7]
ARM_JOINT_NAMES = (
    "L_1_joint",
    "L_2_joint",
    "L_3_joint",
    "L_4_joint",
    "L_5_joint",
    "L_6_joint",
    "L_7_joint",
    "R_1_joint",
    "R_2_joint",
    "R_3_joint",
    "R_4_joint",
    "R_5_joint",
    "R_6_joint",
    "R_7_joint",
)

# Position limits are copied from model/urdf/g7_openarm.urdf, except both J3
# limits are intentionally narrowed from +/-pi/2 to +/-pi/4.
ARM_POSITION_LOWER_RAD = np.array(
    [
        -1.3962629999999998,
        -0.17453267320510335,
        -np.pi / 6.0,
        -2.443461,
        -1.570796,
        -0.785398,
        -1.570796,
        -1.396263,
        -3.3161253267948965,
        -np.pi / 3.0,
        -2.443461,
        -1.570796,
        -0.785398,
        -1.570796,
    ],
    dtype=np.float64,
)
ARM_POSITION_UPPER_RAD = np.array(
    [
        3.490659,
        3.3161253267948965,
        np.pi / 3.0,
        0.0,
        1.570796,
        0.785398,
        1.570796,
        3.490659,
        0.17453267320510335,
        np.pi / 6.0,
        0.0,
        1.570796,
        0.785398,
        1.570796,
    ],
    dtype=np.float64,
)
ARM_VELOCITY_LIMIT_RAD_S = np.array(
    [
        16.754666,
        16.754666,
        5.445426,
        5.445426,
        20.943946,
        20.943946,
        20.943946,
        16.754666,
        16.754666,
        5.445426,
        5.445426,
        20.943946,
        20.943946,
        20.943946,
    ],
    dtype=np.float64,
)

# Indices for extracting the 14 actuated arm joints from the two layouts used
# in this project. Gripper slots are intentionally excluded.
ARM_LOWSTATE_MOTOR_INDICES = np.array(
    [*range(8, 15), *range(16, 23)],
    dtype=np.intp,
)
ARM_MOTOR16_INDICES = np.array(
    [*range(0, 7), *range(8, 15)],
    dtype=np.intp,
)

for _array in (
    ARM_POSITION_LOWER_RAD,
    ARM_POSITION_UPPER_RAD,
    ARM_VELOCITY_LIMIT_RAD_S,
    ARM_LOWSTATE_MOTOR_INDICES,
    ARM_MOTOR16_INDICES,
):
    _array.setflags(write=False)


def position_limited_velocity_bounds(
    position: npt.ArrayLike,
    velocity_limit: npt.ArrayLike,
    dt: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return velocity bounds that respect rate and one-step position limits.

    When a measured joint is already outside its configured range, the bounds
    allow only motion back toward the valid interval. This avoids requesting an
    impossible one-tick jump while still preventing motion farther outward.
    """
    q = np.asarray(position, dtype=np.float64)
    vmax = np.asarray(velocity_limit, dtype=np.float64)

    expected_shape = ARM_POSITION_LOWER_RAD.shape
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

    lower = np.maximum(-vmax, (ARM_POSITION_LOWER_RAD - q) / dt)
    upper = np.minimum(vmax, (ARM_POSITION_UPPER_RAD - q) / dt)

    below = q < ARM_POSITION_LOWER_RAD
    above = q > ARM_POSITION_UPPER_RAD

    lower[below] = 0.0
    upper[below] = vmax[below]
    lower[above] = -vmax[above]
    upper[above] = 0.0

    return lower, upper
