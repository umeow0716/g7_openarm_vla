from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def quat_normalize(q: npt.ArrayLike) -> FloatArray:
    quaternion = np.asarray(q, dtype=np.float64)

    if quaternion.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {quaternion.shape}")
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("Quaternion must contain only finite values")

    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small")

    return quaternion / norm


def quat_mul(q1: npt.ArrayLike, q2: npt.ArrayLike) -> FloatArray:
    left = np.asarray(q1, dtype=np.float64)
    right = np.asarray(q2, dtype=np.float64)

    if left.shape != (4,) or right.shape != (4,):
        raise ValueError(
            f"Quaternion operands must both have shape (4,), got {left.shape} and {right.shape}"
        )

    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conj(q: npt.ArrayLike) -> FloatArray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_to_rotation_matrix(q: npt.ArrayLike) -> FloatArray:
    """Return the body-to-world rotation matrix for a wxyz quaternion."""
    w, x, y, z = quat_normalize(q)

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def quat_rotate(q: npt.ArrayLike, vector: npt.ArrayLike) -> FloatArray:
    """Rotate a 3D vector with a wxyz body-to-world quaternion."""
    vector_array = np.asarray(vector, dtype=np.float64)

    if vector_array.shape != (3,):
        raise ValueError(f"Vector must have shape (3,), got {vector_array.shape}")

    return quat_to_rotation_matrix(q) @ vector_array


def quat_from_yaw(yaw: float) -> FloatArray:
    """Return a wxyz quaternion containing only a world-Z yaw rotation."""
    half_yaw = 0.5 * float(yaw)
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)


def quat_yaw(q: npt.ArrayLike) -> float:
    """Extract world-Z yaw from a wxyz body-to-world quaternion."""
    w, x, y, z = quat_normalize(q)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
