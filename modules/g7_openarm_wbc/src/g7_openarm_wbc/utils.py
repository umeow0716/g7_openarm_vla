import numpy as np
import numpy.typing as npt


def quat_residual(
    state_quat: npt.NDArray[np.float64],
    target_quat: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Shortest-arc, unitless quaternion residual in the common world frame."""
    state = state_quat / np.linalg.norm(state_quat)
    target = target_quat / np.linalg.norm(target_quat)
    if state @ target < 0.0:
        target = -target
    return state - target


def quat_residual_jacobian(
    Jq: npt.NDArray[np.float64],
    state_quat: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Derivative of the normalized current quaternion with respect to IK inputs."""
    norm = np.linalg.norm(state_quat)
    state = state_quat / norm
    return (Jq - np.outer(state, state @ Jq)) / norm
