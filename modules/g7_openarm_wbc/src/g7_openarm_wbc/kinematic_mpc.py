"""Kinematic MPC that follows the original bounded IK policy and plans deceleration."""

import numpy as np
import osqp
from scipy import sparse


class KinematicMPC:
    def __init__(self, nu, velocity_limit, acceleration_limit, task_limit, dt=0.01, horizon=20):
        self.nu = nu
        self.dt = dt
        self.horizon = horizon
        self.nx = nu * horizon
        self.velocity_limit = np.asarray(velocity_limit, dtype=np.float64)
        self.acceleration_limit = np.asarray(acceleration_limit, dtype=np.float64)
        self.task_limit = np.asarray(task_limit, dtype=np.float64)
        self.nt = len(self.task_limit)
        self._failed = False

        nx, n, N = self.nx, nu, horizon
        eye = sparse.eye(nx, format="csc")
        # z = [x_1 ... x_N, u_0 ... u_{N-1}], x_0 = 0.
        D = sparse.diags([np.ones(N), -np.ones(N - 1)], [0, -1],
                         shape=(N, N), format="csc")
        A_dyn = sparse.hstack([
            sparse.kron(D, sparse.eye(n), format="csc"), -dt * eye,
        ], format="csc")
        A_vel = sparse.hstack([sparse.csc_matrix((nx, nx)), eye], format="csc")
        A_task_u = sparse.kron(
            sparse.eye(N, format="csc"),
            sparse.csc_matrix(np.ones((self.nt, n))), format="csc",
        )
        A_task = sparse.hstack([
            sparse.csc_matrix((N * self.nt, nx)), A_task_u,
        ], format="csc")
        # Constrain future planned acceleration, without reintroducing prev_u.
        A_rate = sparse.hstack([
            sparse.csc_matrix(((N - 1) * n, nx)),
            sparse.kron(D[1:, :], sparse.eye(n), format="csc"),
        ], format="csc")
        self.A = sparse.vstack([A_dyn, A_vel, A_task, A_rate], format="csc")
        row = self.A.indices
        col = np.repeat(np.arange(2 * nx), np.diff(self.A.indptr))
        self._task_entries = (row >= 2 * nx) & (row < 2 * nx + N * self.nt)
        self._task_rows = (row[self._task_entries] - 2 * nx) % max(self.nt, 1)
        self._task_cols = col[self._task_entries] % n
        self._fixed_A = self.A.data.copy()

        # Stage cost: 0.5 * ||u_k + F x_k - u_ik||_W^2.
        # u_ik is exactly the output of the preceding bounded two-part IK;
        # F is its local position-dependent feedback gain.
        dense = sparse.csc_matrix(np.ones((n, n)))
        P_x = sparse.triu(sparse.kron(sparse.eye(N), dense, format="csc"), format="csc")
        P_u = sparse.triu(sparse.kron(sparse.eye(N), dense, format="csc"), format="csc")
        # x_k only participates in cost for k >= 1, paired with u_k.
        P_xu = sparse.kron(
            sparse.diags(np.ones(N - 1), 1, shape=(N, N), format="csc"),
            dense, format="csc",
        )
        self.P = sparse.bmat([
            [P_x, P_xu],
            [None, P_u],
        ], format="csc")
        prow = self.P.indices
        pcol = np.repeat(np.arange(2 * nx), np.diff(self.P.indptr))
        self._xx = (pcol < nx)
        self._xu = (pcol >= nx) & (prow < nx)
        self._uu = prow >= nx
        self._xx_i = prow[self._xx] % n
        self._xx_j = pcol[self._xx] % n
        self._xx_last = (prow[self._xx] // n) == (N - 1)
        self._xu_i = prow[self._xu] % n
        self._xu_j = pcol[self._xu] % n
        self._uu_i = (prow[self._uu] - nx) % n
        self._uu_j = (pcol[self._uu] - nx) % n
        # OSQP setup must start with a convex Hessian, not the nonconvex
        # all-ones placeholder used to retain the sparse update pattern.
        self.P.data[:] = np.where(prow == pcol, 1.0, 0.0)

        bound = np.tile(self.velocity_limit, N)
        task_bound = np.tile(self.task_limit, N)
        rate_bound = np.tile(self.acceleration_limit * dt, N - 1)
        self._lower = np.r_[np.zeros(nx), -bound, -task_bound, -rate_bound]
        self._upper = np.r_[np.zeros(nx), bound, task_bound, rate_bound]
        self._qp = osqp.OSQP()
        self._qp.setup(
            P=self.P, q=np.zeros(2 * nx), A=self.A,
            l=self._lower, u=self._upper,
            verbose=False, warm_starting=True, polishing=False,
            eps_abs=1e-4, eps_rel=1e-4, max_iter=400,
        )

    def solve(self, u_ik, F, W, J_task):
        W = 0.5 * (W + W.T)
        Fx = F.T @ W
        Hx = Fx @ F
        target = W @ u_ik
        x_target = F.T @ target

        Px = np.empty_like(self.P.data)
        Px[self._xx] = Hx[self._xx_i, self._xx_j]
        # The terminal displacement has no stage cost; tiny regularization.
        xx = Px[self._xx]
        xx[self._xx_last] = 1e-8 * (self._xx_i[self._xx_last] == self._xx_j[self._xx_last])
        Px[self._xx] = xx
        Px[self._xu] = Fx[self._xu_i, self._xu_j]
        Px[self._uu] = W[self._uu_i, self._uu_j]

        q = np.r_[
            np.tile(-x_target, self.horizon - 1), np.zeros(self.nu),
            np.tile(-target, self.horizon),
        ]
        Ax = self._fixed_A.copy()
        if self.nt:
            Ax[self._task_entries] = J_task[self._task_rows, self._task_cols]
        try:
            self._qp.update(Px=Px, Ax=Ax, q=q)
            result = self._qp.solve()
            ok = (
                result.info.status_val in (1, 2)
                and result.x is not None
                and np.all(np.isfinite(result.x))
            )
        except Exception as exc:
            ok = False
            reason = str(exc)
        else:
            reason = result.info.status
        if not ok:
            if not self._failed:
                print(f"[WBC MPC] QP failed: {reason}; falling back to bounded IK")
            self._failed = True
            return None
        self._failed = False
        return result.x[self.nx:self.nx + self.nu].copy()
