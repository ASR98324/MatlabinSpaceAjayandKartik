"""Thermal-aware model predictive controller (receding-horizon linear program).

Each hour:
  1. Forecast solar for the next H hours with the Fourier model (past data only).
  2. Solve an LP choosing heater power, essential/deferrable loads, and battery
     flows, subject to: power balance, battery dynamics, the heat-equation
     thermal model (linear in heat input), and the cabin comfort band.
  3. Apply only the first hour's decision, then re-plan next hour.

The cabin + walls act as an energy store: the optimizer can pre-heat with
surplus solar and coast down toward T_min later.

Constraint matrices depend only on fixed parameters, so they are built once;
each hour only the right-hand sides and bounds change.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .config import SOL_HOURS
from .controllers import RuleBasedController
from .forecast import fourier_forecast
from .thermal import ambient_temp

NAMES = ["heat", "d", "e", "c", "dis", "curt", "dump", "cs", "sT", "sE", "sC"]


class MPCController:
    name = "mpc"

    def __init__(self, scen, thermal_model, horizon_h=36, reserve_frac=0.15,
                 w_defer=1.0, w_ess=5.0, w_term=0.8, w_comfort=0.3, temp_margin=0.5, fit_sols=2.0):
        self.s, self.th = scen, thermal_model
        self.H = H = int(horizon_h / scen.dt_h)
        self.reserve_frac, self.temp_margin, self.fit_sols = reserve_frac, temp_margin, fit_sols
        self.fallback = RuleBasedController(scen)
        self.last_plan = None
        L, bc, dt = scen.loads, scen.battery, scen.dt_h
        g = L.internal_gain_frac

        # Cabin rows of Ad^k, and impulse responses to heat (Bq) and ambient (Ba)
        Ad, Bq, Ba = thermal_model.Ad, thermal_model.Bq, thermal_model.Ba
        n = thermal_model.n
        self.row = np.zeros((H + 1, n)); mq = np.zeros(H + 1); ma = np.zeros(H + 1)
        P = np.eye(n)
        for k in range(H + 1):
            self.row[k] = P[0]; mq[k] = (P @ Bq)[0]; ma[k] = (P @ Ba)[0]
            P = Ad @ P
        ii, jj = np.tril_indices(H)
        Tq = np.zeros((H, H)); Tq[ii, jj] = mq[ii - jj]   # T[h+1] response to Q[j], j <= h
        Ta = np.zeros((H, H)); Ta[ii, jj] = ma[ii - jj]
        self.Ta = Ta
        self.crit_heat = Tq.sum(axis=1) * 1000 * g * L.critical_kw  # critical-load waste heat

        # Variable layout
        self.idx = {nm: np.arange(i * H, (i + 1) * H) for i, nm in enumerate(NAMES)}
        self.iE = np.arange(len(NAMES) * H, len(NAMES) * H + H + 1)
        self.nv = nv = self.iE[-1] + 1
        idx, iE = self.idx, self.iE
        I = sp.identity(H, format="csr")

        def block(rows, entries):
            M = sp.lil_matrix((rows, nv))
            for nm, mat in entries:
                cols = idx[nm] if nm in idx else nm
                M[:, cols] = mat
            return M

        # Equalities: power balance, battery dynamics
        Apow = block(H, [("dis", I), ("c", -I), ("curt", -I), ("e", -L.essential_kw * dt * I),
                         ("d", -L.deferrable_kw * dt * I), ("heat", -dt * I), ("cs", dt * I)])
        Abat = sp.lil_matrix((H, nv))
        Abat[np.arange(H), iE[1:]] = 1; Abat[np.arange(H), iE[:-1]] = -1
        Abat[:, idx["c"]] = -bc.eff_charge * I; Abat[:, idx["dis"]] = I / bc.eff_discharge
        self.Aeq = sp.vstack([Apow, Abat]).tocsr()

        # Cabin temperature as linear function of decisions: T = free + Tth @ z
        K = 1000 * Tq
        Tth = block(H, [("heat", K), ("e", K * g * L.essential_kw), ("d", K * g * L.deferrable_kw),
                        ("dump", -K), ("cs", -K * g)]).tocsr()
        sT = block(H, [("sT", I)]).tocsr(); sC = block(H, [("sC", I)]).tocsr()
        Ares = sp.lil_matrix((H, nv)); Ares[np.arange(H), iE[1:]] = -1; Ares[:, idx["sE"]] = -I
        self.Aub = sp.vstack([-Tth - sT, Tth, Ares.tocsr(), -Tth - sC]).tocsr()

        c = np.zeros(nv)
        c[idx["d"]] = -w_defer * L.deferrable_kw * dt
        c[idx["e"]] = -w_ess * L.essential_kw * dt
        c[idx["cs"]] = 1e4; c[idx["sT"]] = 500.0; c[idx["sE"]] = 20.0; c[idx["sC"]] = w_comfort
        c[idx["heat"]] = 0.01; c[idx["dump"]] = 0.01
        c[iE[-1]] = -w_term
        self.c = c

    def decide(self, obs):
        s, H, dt = self.s, self.H, self.s.dt_h
        if obs["t_h"] < SOL_HOURS:  # need one sol of history before forecasting
            return self.fallback.decide(obs)
        L, th = s.loads, s.thermal
        idx, iE = self.idx, self.iE

        t_f = obs["t_h"] + dt * np.arange(H)
        P_fc = fourier_forecast(obs["gen_hist_t"], obs["gen_hist_kw"], t_f, fit_sols=self.fit_sols)
        T_amb = ambient_temp(t_f, th)
        local = np.mod(t_f, SOL_HOURS)
        window = ((L.deferrable_hours[0] <= local) & (local < L.deferrable_hours[1])).astype(float)
        cap = obs["cap_kwh"]

        lb = np.zeros(self.nv); ub = np.full(self.nv, np.inf)
        ub[idx["heat"]] = L.heater_max_kw; ub[idx["d"]] = window; ub[idx["e"]] = 1.0
        ub[idx["cs"]] = L.critical_kw
        ub[iE] = cap; lb[iE[0]] = ub[iE[0]] = obs["E_kwh"]

        beq = np.concatenate([L.critical_kw * dt - P_fc * dt, np.zeros(H)])
        free = self.row[1:] @ obs["T_state"] + self.Ta @ T_amb + self.crit_heat
        bub = np.concatenate([-(th.T_min + self.temp_margin - free), th.T_max - free,
                              np.full(H, -self.reserve_frac * cap), -(th.T_set - free)])

        res = linprog(self.c, A_ub=self.Aub, b_ub=bub, A_eq=self.Aeq, b_eq=beq,
                      bounds=np.column_stack([lb, ub]), method="highs")
        if res.status != 0:
            return self.fallback.decide(obs)
        z = res.x
        self.last_plan = {nm: z[idx[nm]] for nm in NAMES} | {"E": z[iE], "P_fc": P_fc}
        return {"heater_kw": float(z[idx["heat"][0]]),
                "essential_on": bool(z[idx["e"][0]] > 0.5),
                "deferrable_on": bool(z[idx["d"][0]] > 0.5)}
