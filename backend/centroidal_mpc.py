import numpy as np
import osqp
from scipy import sparse


class CentroidalMPC:
    def __init__(self, dt=0.025, horizon_steps=24, com_height=0.85, gravity=9.81):
        self.dt = dt
        self.N = horizon_steps
        self.h = com_height
        self.g = gravity
        self.w = np.sqrt(gravity / com_height)
        self.T_ss = 0.30
        self.T_ds = 0.05
        self.foot_margin = 0.005

        self._build_model()
        self._build_qp()

    def _build_model(self):
        wT = self.w * self.dt
        ch = np.cosh(wT)
        sh = np.sinh(wT)

        self.A = np.array([
            [ch, sh / self.w, 0, 0],
            [self.w * sh, ch, 0, 0],
            [0, 0, ch, sh / self.w],
            [0, 0, self.w * sh, ch],
        ])

        self.B = np.array([
            [1 - ch, 0],
            [-self.w * sh, 0],
            [0, 1 - ch],
            [0, -self.w * sh],
        ])

    def _build_qp(self):
        n = self.N
        nx = 4
        nu = 2

        self.Q_vel = 10.0
        R_zmp = 1e-3
        R_dzmp = 5e-3
        self.Q_term = 100.0

        running_cost = sparse.kron(
            sparse.eye(n), sparse.diags([0, self.Q_vel, 0, self.Q_vel])
        )
        terminal_cost = sparse.diags([0, self.Q_term, 0, self.Q_term])
        control_cost = (
            sparse.kron(sparse.eye(n), R_zmp * sparse.eye(2))
            + sparse.kron(self._D2(n), R_dzmp * sparse.eye(2))
        )
        P = sparse.block_diag([running_cost, terminal_cost, control_cost], format="csc")
        P = P + 1e-8 * sparse.eye(P.shape[0])

        A_eq_list = []
        b_eq_list = []

        for k in range(n):
            row = nx * k
            A_k = sparse.lil_matrix((nx, nx * (n + 1) + nu * n))
            A_k[:, row:row + nx] = sparse.eye(nx)
            A_k[:, row + nx:row + 2 * nx] = -self.A
            A_k[:, nx * (n + 1) + nu * k:nx * (n + 1) + nu * (k + 1)] = -self.B
            A_eq_list.append(A_k)
            b_eq_list.append(np.zeros(nx))

        A_eq = sparse.vstack(A_eq_list, format="csc") if A_eq_list else sparse.csc_matrix((0, nx * (n + 1) + nu * n))
        b_eq = np.concatenate(b_eq_list) if b_eq_list else np.array([])

        self.nx = nx
        self.nu = nu
        self.P = P
        self.A_eq = A_eq
        self.b_eq = b_eq

        self._x0_idx = slice(0, nx)
        self._vel_idx = [1, 3]
        self._zmp_start = nx * (n + 1)

    def _D2(self, n):
        D = np.zeros((n, n))
        for i in range(1, n):
            D[i, i] = 1
            D[i, i - 1] = -1
        D[0, 0] = 1
        return D

    def solve(self, com_pos, com_vel, vx_des, vy_des, left_foot_pos, right_foot_pos, step_number=0):
        n = self.N
        nx, nu = self.nx, self.nu

        x0 = np.array([com_pos[0], com_vel[0], com_pos[1], com_vel[1]])
        v_ref = np.array([vx_des, vy_des])

        q = np.zeros(nx * (n + 1) + nu * n)
        for k in range(n):
            for vi in [0, 1]:
                idx = nx * k + self._vel_idx[vi]
                q[idx] = -2 * self.Q_vel * v_ref[vi]
        for vi in [0, 1]:
            idx = nx * n + self._vel_idx[vi]
            q[idx] = -2 * self.Q_term * v_ref[vi]

        A_ineq_list = []
        lb_list = []
        ub_list = []

        lo = np.array([0, 0])
        hi = np.array([0, 0])
        for k in range(n):
            phase = self._get_phase(step_number + k)
            lb, ub = self._get_zmp_bounds(phase, left_foot_pos, right_foot_pos)
            row = sparse.lil_matrix((2, nx * (n + 1) + nu * n))
            row[:, self._zmp_start + nu * k:self._zmp_start + nu * (k + 1)] = sparse.eye(2)
            A_ineq_list.append(row)
            lb_list.append(lb)
            ub_list.append(ub)

        A_ineq = sparse.vstack(A_ineq_list, format="csc") if A_ineq_list else sparse.csc_matrix((0, nx * (n + 1) + nu * n))
        l_ineq = np.concatenate(lb_list) if lb_list else np.array([])
        u_ineq = np.concatenate(ub_list) if ub_list else np.array([])

        x0_row = sparse.lil_matrix((nx, nx * (n + 1) + nu * n))
        x0_row[:, :nx] = sparse.eye(nx)
        A_eq_full = sparse.vstack([x0_row, self.A_eq], format="csc")
        b_eq_full = np.concatenate([x0, self.b_eq])

        A_total = sparse.vstack([A_eq_full, A_ineq], format="csc")
        l_total = np.concatenate([b_eq_full, l_ineq])
        u_total = np.concatenate([b_eq_full, u_ineq])

        try:
            prob = osqp.OSQP()
            prob.setup(P=self.P, q=q, A=A_total, l=l_total, u=u_total,
                       verbose=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=400, warm_start=True)
            res = prob.solve()

            if res.info.status != "solved":
                return self._current_solution if hasattr(self, '_current_solution') else (
                    np.zeros((4, n + 1)), np.zeros((2, n))
                )

            full = res.x
            X = full[:nx * (n + 1)].reshape((nx, n + 1), order="F")
            U = full[self._zmp_start:].reshape((nu, n), order="F")

            self._current_solution = (X, U)
            self._prob = prob
            return X, U
        except Exception:
            return self._current_solution if hasattr(self, '_current_solution') else (
                np.zeros((4, n + 1)), np.zeros((2, n))
            )

    def _get_phase(self, step_idx):
        t = step_idx * self.dt
        cycle_time = 2 * (self.T_ss + self.T_ds)
        t_mod = t % cycle_time
        if t_mod < self.T_ss:
            return "right_swing"
        elif t_mod < self.T_ss + self.T_ds:
            return "double"
        elif t_mod < self.T_ss + self.T_ds + self.T_ss:
            return "left_swing"
        else:
            return "double"

    def _get_zmp_bounds(self, phase, left_foot, right_foot):
        fl = self.foot_margin
        if phase == "right_swing":
            x_lb = left_foot[0] - 0.03 + fl
            x_ub = left_foot[0] + 0.03 - fl
            y_lb = left_foot[1] - 0.03 + fl
            y_ub = left_foot[1] + 0.03 - fl
        elif phase == "left_swing":
            x_lb = right_foot[0] - 0.03 + fl
            x_ub = right_foot[0] + 0.03 - fl
            y_lb = right_foot[1] - 0.03 + fl
            y_ub = right_foot[1] + 0.03 - fl
        else:
            fx_min = min(left_foot[0], right_foot[0])
            fx_max = max(left_foot[0], right_foot[0])
            fy_min = min(left_foot[1], right_foot[1])
            fy_max = max(left_foot[1], right_foot[1])
            if fx_max - fx_min < 0.08:
                fx_min -= 0.04
                fx_max += 0.04
            x_lb = fx_min - 0.03 + fl
            x_ub = fx_max + 0.03 - fl
            y_lb = fy_min - 0.03 + fl
            y_ub = fy_max + 0.03 - fl

        return np.array([x_lb, y_lb]), np.array([x_ub, y_ub])
