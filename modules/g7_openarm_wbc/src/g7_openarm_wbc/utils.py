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


def look_at_pitch_yaw(
    head_pos: npt.NDArray[np.float64],
    head_quat: npt.NDArray[np.float64],
    look_at: npt.NDArray[np.float64],
):
    q = np.asarray(head_quat, dtype=np.float64)
    q = q / np.linalg.norm(q)
    u = -q[1:]
    p = np.asarray(look_at, dtype=np.float64) - np.asarray(head_pos, dtype=np.float64)
    t = 2.0 * np.cross(u, p)
    x, y, z = p + q[0] * t + np.cross(u, t)
    return float(np.arctan2(-z, np.hypot(x, y))), float(np.arctan2(y, x))
