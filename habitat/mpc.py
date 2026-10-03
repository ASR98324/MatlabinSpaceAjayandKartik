"""Thermal-aware model predictive controller (receding-horizon linear program).

Each hour:
  1. Forecast solar for the next H hours with the Fourier model (past data only).
  2. Solve an LP choosing heater power, essential/deferrable loads, and battery
     flows, subject to: power balance, battery dynamics, the heat-equation
     thermal model (linear in heat input), and the cabin comfort band.
  3. Apply only the first hour's decision, then re-plan next hour.

The cabin + walls are treated as an energy store: the optimizer can pre-heat
toward T_max with surplus solar and coast down toward T_min later.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .config import SOL_HOURS
from .controllers import RuleBasedController
from .forecast import fourier_forecast
from .thermal import ambient_temp


class MPCController:
    name = "mpc"

    def __init__(self, scen, thermal_model, horizon_h=36, reserve_frac=0.15,
                 w_defer=1.0, w_ess=5.0, w_term=0.8, w_comfort=0.3, temp_margin=0.5, fit_sols=2.0):
        self.s, self.th = scen, thermal_model
        self.H = int(horizon_h / scen.dt_h)
        self.reserve_frac, self.temp_margin, self.fit_sols = reserve_frac, temp_margin, fit_sols
        self.w_defer, self.w_ess, self.w_term, self.w_comfort = w_defer, w_ess, w_term, w_comfort
        self.fallback = RuleBasedController(scen)
        self.last_plan = None
        # Precompute cabin rows of Ad^k, Ad^k Bq, Ad^k Ba for k = 0..H
        Ad, Bq, Ba = thermal_model.Ad, thermal_model.Bq, thermal_model.Ba
        n, H = thermal_model.n, self.H
        self.row = np.zeros((H + 1, n)); self.mq = np.zeros(H + 1); self.ma = np.zeros(H + 1)
        P = np.eye(n)
        for k in range(H + 1):
            self.row[k] = P[0]; self.mq[k] = (P @ Bq)[0]; self.ma[k] = (P @ Ba)[0]
            P = Ad @ P

    def decide(self, obs):
        s, H, dt = self.s, self.H, self.s.dt_h
        if obs["t_h"] < SOL_HOURS:  # need one sol of history before forecasting
            return self.fallback.decide(obs)
        L, th, bc = s.loads, s.thermal, s.battery

        t_f = obs["t_h"] + dt * np.arange(H)
        P_fc = fourier_forecast(obs["gen_hist_t"], obs["gen_hist_kw"], t_f, fit_sols=self.fit_sols)
        T_amb = ambient_temp(t_f, th)
        local = np.mod(t_f, SOL_HOURS)
        window = ((L.deferrable_hours[0] <= local) & (local < L.deferrable_hours[1])).astype(float)

        # variable layout
        names = ["heat", "d", "e", "c", "dis", "curt", "dump", "cs", "sT", "sE", "sC"]
        idx = {nm: np.arange(i * H, (i + 1) * H) for i, nm in enumerate(names)}
        iE = np.arange(len(names) * H, len(names) * H + H + 1)
        nv = iE[-1] + 1
        g = L.internal_gain_frac
        cap = obs["cap_kwh"]

        bounds = [(0, None)] * nv
        for h in range(H):
            bounds[idx["heat"][h]] = (0, L.heater_max_kw)
            bounds[idx["d"][h]] = (0, window[h])
            bounds[idx["e"][h]] = (0, 1)
            bounds[idx["cs"][h]] = (0, L.critical_kw)
        for h in range(H + 1):
            bounds[iE[h]] = (0, cap)
        bounds[iE[0]] = (obs["E_kwh"], obs["E_kwh"])

        # equalities: power balance (H) + battery dynamics (H)
        Aeq = lil_matrix((2 * H, nv)); beq = np.zeros(2 * H)
        for h in range(H):
            r = h
            Aeq[r, idx["dis"][h]] = 1; Aeq[r, idx["c"][h]] = -1; Aeq[r, idx["curt"][h]] = -1
            Aeq[r, idx["e"][h]] = -L.essential_kw * dt; Aeq[r, idx["d"][h]] = -L.deferrable_kw * dt
            Aeq[r, idx["heat"][h]] = -dt; Aeq[r, idx["cs"][h]] = dt
            beq[r] = L.critical_kw * dt - P_fc[h] * dt
            r = H + h
            Aeq[r, iE[h + 1]] = 1; Aeq[r, iE[h]] = -1
            Aeq[r, idx["c"][h]] = -bc.eff_charge; Aeq[r, idx["dis"][h]] = 1 / bc.eff_discharge

        # inequalities: cabin temp lower/upper (2H) + reserve (H)
        x0 = obs["T_state"]
        Aub = lil_matrix((4 * H, nv)); bub = np.zeros(4 * H)
        Tmin = th.T_min + self.temp_margin
        reserve = self.reserve_frac * cap
        for h in range(H):
            # T[h+1] = free[h] + sum_{j<=h} mq[h-j] * Q[j],  Q in W
            free = self.row[h + 1] @ x0 + sum(self.ma[h - j] * T_amb[j] for j in range(h + 1)) \
                + sum(self.mq[h - j] * 1000 * g * L.critical_kw for j in range(h + 1))
            coef = {}
            for j in range(h + 1):
                m = self.mq[h - j] * 1000
                coef.setdefault(("heat", j), 0); coef[("heat", j)] += m
                coef.setdefault(("e", j), 0); coef[("e", j)] += m * g * L.essential_kw
                coef.setdefault(("d", j), 0); coef[("d", j)] += m * g * L.deferrable_kw
                coef.setdefault(("dump", j), 0); coef[("dump", j)] -= m
                coef.setdefault(("cs", j), 0); coef[("cs", j)] -= m * g
            # lower: -(free + coef.x) - sT <= -Tmin
            for (nm, j), v in coef.items():
                Aub[h, idx[nm][j]] = -v
                Aub[H + h, idx[nm][j]] = v
            Aub[h, idx["sT"][h]] = -1
            bub[h] = -(Tmin - free)
            bub[H + h] = th.T_max - free
            # reserve: -E[h+1] - sE <= -reserve
            Aub[2 * H + h, iE[h + 1]] = -1; Aub[2 * H + h, idx["sE"][h]] = -1
            bub[2 * H + h] = -reserve
            # soft comfort: sC >= T_set - T[h+1]  (crew prefers T_set when energy allows)
            for (nm, j), v in coef.items():
                Aub[3 * H + h, idx[nm][j]] = -v
            Aub[3 * H + h, idx["sC"][h]] = -1
            bub[3 * H + h] = -(th.T_set - free)

        c = np.zeros(nv)
        c[idx["d"]] = -self.w_defer * L.deferrable_kw * dt
        c[idx["e"]] = -self.w_ess * L.essential_kw * dt
        c[idx["cs"]] = 1e4
        c[idx["sT"]] = 500.0
        c[idx["sE"]] = 20.0
        c[idx["sC"]] = self.w_comfort
        c[idx["heat"]] = 0.01
        c[idx["dump"]] = 0.01
        c[iE[-1]] = -self.w_term

        res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq,
                      bounds=bounds, method="highs")
        if res.status != 0:
            return self.fallback.decide(obs)
        z = res.x
        self.last_plan = {nm: z[idx[nm]] for nm in names} | {"E": z[iE], "P_fc": P_fc}
        return {"heater_kw": float(z[idx["heat"][0]]),
                "essential_on": bool(z[idx["e"][0]] > 0.5),
                "deferrable_on": bool(z[idx["d"][0]] > 0.5)}
