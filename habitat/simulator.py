"""Closed-loop habitat simulation.

Physics enforces priority: if supply runs short, loads lose power in the order
deferrable -> heater -> essential -> critical, no matter what the controller asked.
"""
import numpy as np

from .battery import BatteryPack
from .config import SOL_HOURS, Scenario
from .forecast import StormDetector, fourier_forecast
from .solar import solar_power_kw, tau_profile
from .thermal import ThermalModel, ambient_temp

PRIORITY = ["critical", "essential", "heater", "deferrable"]


def simulate(scen: Scenario, make_controller):
    rng = np.random.default_rng(scen.seed)
    dt = scen.dt_h
    n_steps = int(scen.sols * SOL_HOURS / dt)
    t = np.arange(n_steps) * dt
    tau = tau_profile(t, scen.storm)
    gen = solar_power_kw(t, tau, scen.solar, rng)
    T_amb = ambient_temp(t, scen.thermal)

    thermal = ThermalModel(scen.thermal, dt)
    batt = BatteryPack(scen.battery, scen.failure, scen.data_dir)
    ctrl = make_controller(scen, thermal)
    detector = StormDetector()
    x = thermal.initial_state(scen.thermal.T_set, scen.thermal.ambient_mean)

    log = {k: np.zeros(n_steps) for k in
           ["soc", "E", "cap", "T_cabin", "curtailed", "storm_flag"] +
           [f"req_{p}" for p in PRIORITY] + [f"srv_{p}" for p in PRIORITY]}
    storm_flag = False
    sol_idx_prev = 0
    sol_start = 0

    for k in range(n_steps):
        sol = t[k] / SOL_HOURS
        # end of a sol: score yesterday's forecast vs actual (autonomous storm detection)
        if int(sol) != sol_idx_prev:
            past = slice(sol_start, k)
            if sol_start >= int(3 * SOL_HOURS / dt):
                hist = slice(0, sol_start)
                fc = fourier_forecast(t[hist], gen[hist], t[past])
                storm_flag = detector.update(gen[past].sum() * dt, fc.sum() * dt)
            sol_idx_prev, sol_start = int(sol), k
        batt.update_capacity(sol)
        local = np.mod(t[k], SOL_HOURS)
        window = scen.loads.deferrable_hours[0] <= local < scen.loads.deferrable_hours[1]

        obs = dict(t_h=t[k], sol=sol, local_h=local, soc=batt.soc, E_kwh=batt.E,
                   cap_kwh=batt.capacity_kwh, T_cabin=x[0], T_state=x.copy(), T_amb=T_amb[k],
                   gen_hist_t=t[:k], gen_hist_kw=gen[:k], storm_flag=storm_flag,
                   deferrable_window=window)
        a = ctrl.decide(obs)

        req = {"critical": scen.loads.critical_kw,
               "essential": scen.loads.essential_kw if a["essential_on"] else 0.0,
               "heater": float(np.clip(a["heater_kw"], 0, scen.loads.heater_max_kw)),
               "deferrable": scen.loads.deferrable_kw if (a["deferrable_on"] and window) else 0.0}

        # serve in priority order from solar + battery
        avail = gen[k] * dt + batt.max_discharge_kwh()
        srv = {}
        for p in PRIORITY:
            e = min(req[p] * dt, avail)
            srv[p] = e / dt
            avail -= e
        used = sum(srv.values()) * dt
        log["curtailed"][k] = batt.apply(gen[k] * dt - used)

        Q = 1000 * (srv["heater"] + scen.loads.internal_gain_frac *
                    (srv["critical"] + srv["essential"] + srv["deferrable"]))
        x = thermal.step(x, Q, T_amb[k])
        # Passive radiator louvers dump excess heat above T_max at no power cost (assumption).
        x[0] = min(x[0], scen.thermal.T_max)

        log["soc"][k], log["E"][k], log["cap"][k] = batt.soc, batt.E, batt.capacity_kwh
        log["T_cabin"][k], log["storm_flag"][k] = x[0], storm_flag
        for p in PRIORITY:
            log[f"req_{p}"][k], log[f"srv_{p}"][k] = req[p], srv[p]

    log.update(t=t, tau=tau, gen=gen, T_amb=T_amb, controller=ctrl.name)
    return log


def metrics(log, scen: Scenario):
    dt = scen.dt_h
    th = scen.thermal
    crit_short = log["req_critical"] - log["srv_critical"] > 1e-6
    window = np.array([scen.loads.deferrable_hours[0] <= np.mod(tt, SOL_HOURS) < scen.loads.deferrable_hours[1]
                       for tt in log["t"]])
    defer_possible = window.sum() * scen.loads.deferrable_kw * dt
    return {
        "critical_outage_h": float(crit_short.sum() * dt),
        "hours_below_T_min": float((log["T_cabin"] < th.T_min).sum() * dt),
        "min_cabin_C": float(log["T_cabin"].min()),
        "min_soc": float(log["soc"].min()),
        "science_done_frac": float(log["srv_deferrable"].sum() * dt / defer_possible),
        "essential_uptime_frac": float((log["srv_essential"] >= scen.loads.essential_kw - 1e-6).mean()),
    }


def plot_run(logs, scen: Scenario, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sols = logs[0]["t"] / SOL_HOURS
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    ax[0].plot(sols, logs[0]["gen"], color="orange", lw=0.8, label="solar (kW)")
    a0 = ax[0].twinx(); a0.plot(sols, logs[0]["tau"], color="brown", lw=1.5, label="dust tau")
    ax[0].set_ylabel("solar kW"); a0.set_ylabel("tau")
    for lg in logs:
        ax[1].plot(sols, lg["soc"], label=lg["controller"])
        ax[2].plot(sols, lg["T_cabin"], label=lg["controller"])
        ax[3].plot(sols, lg["srv_critical"] + lg["srv_essential"] + lg["srv_deferrable"], lw=0.7, label=lg["controller"])
    ax[1].set_ylabel("battery SOC"); ax[1].legend()
    ax[2].axhspan(scen.thermal.T_min, scen.thermal.T_max, color="green", alpha=0.08)
    ax[2].set_ylabel("cabin C"); ax[2].legend()
    ax[3].set_ylabel("non-heater load kW"); ax[3].set_xlabel("sol"); ax[3].legend()
    for a in ax:
        if scen.failure.string_fail_sol is not None:
            a.axvline(scen.failure.string_fail_sol, color="red", ls="--", lw=1)
    ax[0].set_title("Mars habitat: dust storm + battery string failure (red dashed)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
