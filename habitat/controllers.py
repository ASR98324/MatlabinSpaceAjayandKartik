"""Controllers. Each sees only what a real habitat could measure:

obs = {t_h, sol, local_h, soc, E_kwh, cap_kwh, T_cabin, T_amb,
       gen_hist_t, gen_hist_kw, storm_flag, deferrable_window}
Returns {heater_kw, essential_on, deferrable_on}. Critical loads are always on.
"""
import numpy as np


def p_heater(T, T_set, cfg_loads, kp=2.0):
    return float(np.clip(kp * (T_set - T), 0.0, cfg_loads.heater_max_kw))


class NaiveController:
    """Runs everything, holds 21 C. The 'do nothing smart' baseline."""
    name = "naive"

    def __init__(self, scen):
        self.s = scen

    def decide(self, obs):
        return {"heater_kw": p_heater(obs["T_cabin"], self.s.thermal.T_set, self.s.loads),
                "essential_on": True, "deferrable_on": True}


class RuleBasedController:
    """Priority shedding on SOC thresholds. The REAL baseline to beat."""
    name = "rule_based"

    def __init__(self, scen, defer_soc=0.5, essential_soc=0.15, eco_soc=0.3, eco_setpoint=18.5):
        self.s = scen
        self.defer_soc, self.essential_soc = defer_soc, essential_soc
        self.eco_soc, self.eco_setpoint = eco_soc, eco_setpoint

    def decide(self, obs):
        soc = obs["soc"]
        T_set = self.eco_setpoint if soc < self.eco_soc else self.s.thermal.T_set
        return {"heater_kw": p_heater(obs["T_cabin"], T_set, self.s.loads),
                "essential_on": soc > self.essential_soc,
                "deferrable_on": soc > self.defer_soc}


class MPCController:
    """TODO: thermal-aware model predictive control (linear program).

    Every step, over horizon H (e.g. 36 h):
      forecast P_solar[0..H) with fourier_forecast (past data only)
      decision vars per hour: heater_kw[h] in [0, max], defer[h] in [0, 1],
                              ess[h] in [0, 1], batt energy E[h], cabin/wall temps x[h]
      dynamics:   x[h+1] = Ad x[h] + Bq * (1000*heater[h] + gain_frac*1000*loads[h]) + Ba*T_amb[h]
                  E[h+1] = E[h] + charge/discharge (split into + and - parts, LP-friendly)
      constraints: critical load always served; T_min <= T_cabin[h] <= T_max;
                   E[h] >= reserve; ess[h] = 1 unless forced
      objective:  maximize deferrable work + w_ess*essential - w_slack*(temp/energy slack)
    Solve with scipy.optimize.linprog (HiGHS); apply only hour 0 (receding horizon).
    The pre-heat-then-coast behavior should emerge on its own from the thermal dynamics.

    Until implemented, falls back to the rule-based controller so the pipeline runs.
    """
    name = "mpc"

    def __init__(self, scen, thermal_model=None, horizon_h=36):
        self.s = scen
        self.thermal = thermal_model
        self.H = horizon_h
        self.fallback = RuleBasedController(scen)

    def decide(self, obs):
        return self.fallback.decide(obs)
