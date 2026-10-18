import numpy as np
import osqp
from scipy import sparse


class TorqueGovernor:
    """14-axis jerk QP; torque demand is soft-constrained, jerk is hard-constrained."""

    def __init__(self, dt, jerk_max, accel_max, vel_max, pos_min, pos_max, torque_max):
        self.dt = dt
        self.jerk_max = jerk_max
        self.accel_max = accel_max
        self.vel_max = vel_max
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.torque_max = torque_max
        n = 14
        eye = sparse.eye(n, format="csc")
        full = sparse.csc_matrix(np.ones((n, n)))
        matrix = sparse.bmat([
            [eye, None],
            [None, eye],
            [full, -eye],
            [-full, -eye],
        ], format="csc")
        self._ax = matrix.data.copy()
        self._positive = np.empty((n, n), dtype=int)
        self._negative = np.empty((n, n), dtype=int)
        for col in range(n):
            start, end = matrix.indptr[col:col + 2] # type: ignore
            rows = matrix.indices[start:end] # type: ignore
            self._positive[:, col] = start + np.searchsorted(rows, np.arange(2 * n, 3 * n))
            self._negative[:, col] = start + np.searchsorted(rows, np.arange(3 * n, 4 * n))

        self._qp = osqp.OSQP()
        self._qp.setup(
            P=sparse.diags(np.r_[np.ones(n), np.full(n, 100.0)], format="csc"),
            q=np.zeros(2 * n), A=matrix,
            l=np.r_[np.full(n, -1.0), np.zeros(n), np.full(2 * n, -np.inf)],
            u=np.r_[np.ones(n), np.full(n, np.inf), np.ones(2 * n)],
            verbose=False, warm_starting=True, max_iter=200,
            eps_abs=1e-4, eps_rel=1e-4,
        )

    def solve(self, jerk_nom, q, dq, ddq, q_meas, dq_meas, M, tau_bias, omega, zeta):
        dt = self.dt
        du = dt * self.jerk_max
        dv = 0.5 * dt**2 * self.jerk_max
        dpos = 0.25 * dt**3 * self.jerk_max

        v_base = dq + dt * ddq
        q_base = q + dt * dq + 0.5 * dt**2 * ddq

        low = np.maximum(-1.0, (-self.accel_max - ddq) / du)
        high = np.minimum(1.0, (self.accel_max - ddq) / du)
        # Enforce reachable velocity and position bounds. If a boundary is already
        # unreachable under jerk limits, preserve the generator's existing clipping.
        for lo, hi in (
            ((-self.vel_max - v_base) / dv, (self.vel_max - v_base) / dv),
            ((self.pos_min - q_base) / dpos, (self.pos_max - q_base) / dpos),
        ):
            next_low = np.maximum(low, lo)
            next_high = np.minimum(high, hi)
            feasible = next_low <= next_high
            low = np.where(feasible, next_low, low)
            high = np.where(feasible, next_high, high)

        kp = omega**2
        kd = 2.0 * zeta * omega
        change = du + kp * dpos + kd * dv
        torque_0 = tau_bias + M @ (ddq + kp * (q_base - q_meas) + kd * (v_base - dq_meas))
        limit = self.torque_max  # Planning bound; not a physical torque clamp.
        coupling = (M * change[None, :]) / limit[:, None]
        offset = torque_0 / limit

        ax = self._ax.copy()
        ax[self._positive] = coupling
        ax[self._negative] = -coupling
        self._qp.update(
            q=np.r_[-np.clip(jerk_nom / self.jerk_max, -1.0, 1.0), np.zeros(14)],
            l=np.r_[low, np.zeros(14), np.full(28, -np.inf)],
            u=np.r_[high, np.full(14, np.inf), 1.0 - offset, 1.0 + offset],
            Ax=ax,
        )
        result = self._qp.solve()
        if result.info.status_val not in (1, 2) or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Torque governor QP: {result.info.status}")
        return self.jerk_max * np.clip(result.x[:14], low, high)
