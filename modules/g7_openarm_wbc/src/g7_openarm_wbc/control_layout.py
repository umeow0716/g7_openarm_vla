import numpy as np
import numpy.typing as npt

BASE_CONTROL_SIZE = 3
ARM_CONTROL_SIZE = 14


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
