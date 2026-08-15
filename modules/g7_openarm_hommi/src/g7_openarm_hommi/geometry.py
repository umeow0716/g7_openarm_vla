from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def matrix_to_pose7_wxyz(matrix: npt.ArrayLike) -> FloatArray:
    """Convert a rigid 4x4 transform to [x, y, z, qw, qx, qy, qz]."""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"matrix must have shape (4, 4), got {transform.shape}")

    rotation = transform[:3, :3]
    trace = float(np.trace(rotation))

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = 2.0 * np.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 0.0))
            if s < 1e-12:
                raise ValueError("rotation matrix is numerically singular")
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = 2.0 * np.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 0.0))
            if s < 1e-12:
                raise ValueError("rotation matrix is numerically singular")
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 0.0))
            if s < 1e-12:
                raise ValueError("rotation matrix is numerically singular")
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s

    quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("rotation matrix produced a zero quaternion")
    quaternion /= norm

    # q and -q encode the same orientation. Prefer a stable hemisphere to avoid
    # unnecessary sign flips in DDS targets/logs.
    if quaternion[0] < 0.0:
        quaternion = -quaternion

    return np.concatenate(
        (transform[:3, 3], quaternion),
        dtype=np.float64,
    ).astype(np.float32)
