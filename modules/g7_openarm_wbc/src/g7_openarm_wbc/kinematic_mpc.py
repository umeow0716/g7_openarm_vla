"""Linearized pose-cost MPC with independent joint-speed penalty."""

import numpy as np
import osqp
from scipy import sparse


class KinematicMPC:
    def __init__(self, nu, velocity_limit, acceleration_limit, task_limit,
                 control_weight, dt=0.01, horizon=20):
        self.nu = nu
        self.dt = dt
        self.horizon = horizon
        self.nx = nu * horizon
        self.velocity_limit = np.asarray(velocity_limit, dtype=np.float64)
        self.acceleration_limit = np.asarray(acceleration_limit, dtype=np.float64)
        self.task_limit = np.asarray(task_limit, dtype=np.float64)
        self.control_weight = np.asarray(control_weight, dtype=np.float64)
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

        # Cost: sum_k 0.5*x_k.T*H_pose*x_k + g_pose.T*x_k
        #            + 0.5*u_k.T*diag(control_weight)*u_k.
        # x_k is displacement from measured state; true pose weights set H/g.
        P_x = sparse.triu(sparse.kron(
            sparse.eye(N), sparse.csc_matrix(np.ones((n, n))), format="csc"
        ), format="csc")
        P_u = sparse.diags(np.tile(self.control_weight, N), format="csc")
        self.P = sparse.block_diag([P_x, P_u], format="csc")
        prow = self.P.indices
        pcol = np.repeat(np.arange(2 * nx), np.diff(self.P.indptr))
        self._state_entries = pcol < nx
        self._state_rows = prow[self._state_entries] % n
        self._state_cols = pcol[self._state_entries] % n
        self._terminal_entries = self._state_entries & (pcol // n == N - 1)
        self._terminal_rows = prow[self._terminal_entries] % n
        self._terminal_cols = pcol[self._terminal_entries] % n
        self._fixed_P = self.P.data.copy()
        # Start at a strictly convex, numerically valid Hessian.
        self.P.data[self._state_entries] = (
            self._state_rows == self._state_cols
        ).astype(np.float64)

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

    def solve(self, H, g, J_task, H_terminal=None, g_terminal=None):
        H = 0.5 * (H + H.T)
        Px = self._fixed_P.copy()
        Px[self._state_entries] = H[self._state_rows, self._state_cols]
        if H_terminal is not None:
            H_terminal = 0.5 * (H_terminal + H_terminal.T)
            Px[self._terminal_entries] = H_terminal[
                self._terminal_rows, self._terminal_cols
            ]
        q_state = np.tile(g, self.horizon)
        if g_terminal is not None:
            q_state[-self.nu:] = g_terminal
        Ax = self._fixed_A.copy()
        if self.nt:
            Ax[self._task_entries] = J_task[self._task_rows, self._task_cols]
        try:
            self._qp.update(
                Px=Px, Ax=Ax,
                q=np.r_[q_state, np.zeros(self.nx)],
            )
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
