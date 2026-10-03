"""Habitat thermal model: lumped cabin node + 1D heat equation through the wall.

States x = [T_cabin, T_wall_1 (inner), ..., T_wall_N (outer)].
Continuous form:  C dx/dt = K x + b_amb * T_amb + e0 * Q
  Q = heat into cabin (heater + internal gains from electrical loads), W.
Discretized with backward Euler (unconditionally stable at dt = 1 h):
  x_{k+1} = Ad x_k + Bq Q_k + Ba T_amb_k
The map is LINEAR in Q, so it drops straight into the MPC linear program.
"""
import numpy as np
from .config import SOL_HOURS, ThermalConfig


def ambient_temp(t_h, cfg: ThermalConfig):
    local = np.mod(t_h, SOL_HOURS)
    # coldest at ambient_min_hour, warmest half a sol later
    phase = 2 * np.pi * (local - cfg.ambient_min_hour) / SOL_HOURS
    return cfg.ambient_mean - cfg.ambient_amp * np.cos(phase)


class ThermalModel:
    def __init__(self, cfg: ThermalConfig, dt_h: float):
        self.cfg = cfg
        N = cfg.n_wall_nodes
        A = cfg.wall_area_m2
        dx = cfg.wall_thickness_m / (N - 1)
        G_in = cfg.h_in * A
        G = cfg.wall_k * A / dx
        G_out = cfg.h_out * A
        c_node = cfg.wall_rho_c * dx * A

        n = N + 1
        K = np.zeros((n, n))
        Cdiag = np.full(n, c_node)
        Cdiag[0] = cfg.cabin_C
        # cabin <-> inner wall
        K[0, 0] -= G_in; K[0, 1] += G_in
        K[1, 1] -= G_in; K[1, 0] += G_in
        # wall conduction
        for i in range(1, N):
            K[i, i] -= G; K[i, i + 1] += G
            K[i + 1, i + 1] -= G; K[i + 1, i] += G
        # outer wall <-> ambient
        K[N, N] -= G_out
        b_amb = np.zeros(n); b_amb[N] = G_out
        e0 = np.zeros(n); e0[0] = 1.0

        dt = dt_h * 3600.0
        M = np.diag(Cdiag) - dt * K
        Minv = np.linalg.inv(M)
        self.Ad = Minv @ np.diag(Cdiag)
        self.Bq = Minv @ (dt * e0)
        self.Ba = Minv @ (dt * b_amb)
        self.n = n
        self.UA = 1.0 / (1 / G_in + (N - 1) / G + 1 / G_out)  # overall W/K, for sanity checks

    def initial_state(self, T_cabin, T_amb):
        """Linear steady-ish profile through the wall."""
        x = np.empty(self.n)
        x[0] = T_cabin
        x[1:] = np.linspace(T_cabin - 1.0, T_amb + 1.0, self.n - 1)
        return x

    def step(self, x, Q_w, T_amb):
        return self.Ad @ x + self.Bq * Q_w + self.Ba * T_amb
