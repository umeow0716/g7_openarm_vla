import numpy as np
import numpy.typing as npt

from g7_openarm_config import ControlMode

BASE_CONTROL_SIZE = 3
ARM_CONTROL_SIZE = 14
SINGLE_ARM_CONTROL_SIZE = 7


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
    """Force non-tracked arm joint velocities to zero in-place."""
    expected_size = control_size(base_enabled=base_enabled)
    if u.shape != (expected_size,):
        raise ValueError(f"Expected control vector shape ({expected_size},), got {u.shape}")

    arm_offset = BASE_CONTROL_SIZE if base_enabled else 0
    if not track_left:
        u[arm_offset : arm_offset + SINGLE_ARM_CONTROL_SIZE] = 0.0
    if not track_right:
        right_start = arm_offset + SINGLE_ARM_CONTROL_SIZE
        u[right_start : right_start + SINGLE_ARM_CONTROL_SIZE] = 0.0


def control_size(*, base_enabled: bool) -> int:
    return ARM_CONTROL_SIZE + (BASE_CONTROL_SIZE if base_enabled else 0)


def split_control_vector(
    u: npt.NDArray[np.float64],
    *,
    base_enabled: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    expected_size = control_size(base_enabled=base_enabled)
    if u.shape != (expected_size,):
        raise ValueError(f"Expected control vector shape ({expected_size},), got {u.shape}")

    if base_enabled:
        return u[:BASE_CONTROL_SIZE], u[BASE_CONTROL_SIZE:]

    return np.zeros(BASE_CONTROL_SIZE, dtype=np.float64), u
