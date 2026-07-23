import numpy as np
import numpy.typing as npt


def wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def yaw_to_quat_wxyz(yaw: np.float64 | float) -> npt.NDArray[np.float64]:
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def normalize_quat_wxyz(
    q: npt.NDArray[np.float64],
    eps: float = 1e-12,
) -> npt.NDArray[np.float64]:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)

    if (not np.isfinite(n)) or n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    return q / n


def quat_mul(
    q1: npt.NDArray[np.float64],
    q2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Hamilton product for MuJoCo-style quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_conj(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Quaternion conjugate for MuJoCo-style quaternions [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_err_current_to_target(
    state_quat: npt.NDArray[np.float64],
    target_quat: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    state_quat = normalize_quat_wxyz(state_quat)
    target_quat = normalize_quat_wxyz(target_quat)
    return quat_mul(target_quat, quat_conj(state_quat))


def ori_err_from_qerr(
    q_err_raw: npt.NDArray[np.float64],
    eps: float = 1e-8,
) -> npt.NDArray[np.float64]:
    q_err = normalize_quat_wxyz(q_err_raw)

    if q_err[0] < 0.0:
        q_err = -q_err

    w = np.clip(q_err[0], -1.0, 1.0)
    v = q_err[1:4]
    v_norm = np.linalg.norm(v)

    if v_norm < eps:
        return 2.0 * v

    angle = 2.0 * np.arctan2(v_norm, w)
    return angle * v / v_norm


def ori_err_quat(
    state_quat: npt.NDArray[np.float64],
    target_quat: npt.NDArray[np.float64],
    eps: float = 1e-8,
) -> npt.NDArray[np.float64]:
    q_err = quat_err_current_to_target(state_quat, target_quat)
    return ori_err_from_qerr(q_err, eps=eps)


def qerr_jac_wrt_current_quat(
    target_quat: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    target_quat = normalize_quat_wxyz(target_quat)
    J = np.zeros((4, 4), dtype=np.float64)

    for i in range(4):
        basis = np.zeros(4, dtype=np.float64)
        basis[i] = 1.0
        J[:, i] = quat_mul(target_quat, quat_conj(basis))

    return J


def ori_err_and_jac_wrt_qerr(
    q_err_raw: npt.NDArray[np.float64],
    eps: float = 1e-8,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    q_err_raw = normalize_quat_wxyz(q_err_raw)
    sign = -1.0 if q_err_raw[0] < 0.0 else 1.0
    q_err = sign * q_err_raw

    w = float(np.clip(q_err[0], -1.0, 1.0))
    v = q_err[1:4]

    vx, vy, vz = v
    vn2 = float(v @ v)
    vn = float(np.sqrt(vn2))

    J = np.zeros((3, 4), dtype=np.float64)

    if vn < eps:
        ori_err = 2.0 * v
        J[:, 1:4] = 2.0 * np.eye(3)
        return ori_err, sign * J

    angle = 2.0 * np.arctan2(vn, w)
    ori_err = angle * v / vn

    denom = w * w + vn2
    f = angle / vn
    df_dw = -2.0 / denom
    df_dvn = 2.0 * w / (vn * denom) - angle / vn2
    c = df_dvn / vn

    J[0, 0] = df_dw * vx
    J[1, 0] = df_dw * vy
    J[2, 0] = df_dw * vz

    J[0, 1] = f + c * vx * vx
    J[0, 2] = c * vx * vy
    J[0, 3] = c * vx * vz

    J[1, 1] = c * vy * vx
    J[1, 2] = f + c * vy * vy
    J[1, 3] = c * vy * vz

    J[2, 1] = c * vz * vx
    J[2, 2] = c * vz * vy
    J[2, 3] = f + c * vz * vz

    return ori_err, sign * J


def ori_err_jac_wrt_current_quat(
    state_quat: npt.NDArray[np.float64],
    target_quat: npt.NDArray[np.float64],
    eps: float = 1e-8,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    state_quat = normalize_quat_wxyz(state_quat)
    target_quat = normalize_quat_wxyz(target_quat)
    q_err = quat_err_current_to_target(state_quat, target_quat)
    ori_err, J_err_qerr = ori_err_and_jac_wrt_qerr(q_err, eps=eps)
    J_qerr_q = qerr_jac_wrt_current_quat(target_quat)
    return ori_err, J_err_qerr @ J_qerr_q


def quat_jac_to_ori_err_jac(
    Jq: npt.NDArray[np.float64],
    state_quat: npt.NDArray[np.float64],
    target_quat: npt.NDArray[np.float64],
    eps: float = 1e-8,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ori_err, J_err_q = ori_err_jac_wrt_current_quat(
        state_quat=state_quat,
        target_quat=target_quat,
        eps=eps,
    )
    Jr = J_err_q @ Jq

    if not np.all(np.isfinite(Jr)):
        Jr = np.zeros((3, Jq.shape[1]), dtype=np.float64)

    return ori_err, Jr
