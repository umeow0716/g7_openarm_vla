import numpy as np
import numpy.typing as npt


def quat_normalize(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small")

    return q / norm


def quat_mul(
    q1: npt.NDArray[np.float64],
    q2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conj(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
