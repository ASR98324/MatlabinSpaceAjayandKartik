"""Extra evidence for the judges.

    python run_analysis.py severity              # from results/monte_carlo.csv, instant
    python run_analysis.py detector --n 200      # storm detector + forecast accuracy, ~1 min
    python run_analysis.py sensitivity --n 30    # parameter robustness, a few minutes
    python run_analysis.py all                   # everything
"""
import argparse
import copy
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from habitat.config import SOL_HOURS
from habitat.forecast import fourier_forecast
from habitat.simulator import metrics, simulate
from run_monte_carlo import make_mpc, make_naive, make_rule, random_scenario

os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------- 1. severity
def severity():
    with open("results/monte_carlo.csv") as f:
        rows = list(csv.DictReader(f))
    bins = [("mild (tau < 3)", 0, 3.0), ("moderate (3-3.75)", 3.0, 3.75), ("severe (> 3.75)", 3.75, 99)]
    out = []
    print("\nZero life-support outage rate / worst-case outage (h), by storm severity")
    print(f"{'storm':<20}{'missions':>10}" + "".join(f"{c:>22}" for c in ["naive", "rule_based", "mpc"]))
    for name, lo, hi in bins:
        sel = [r for r in rows if lo <= float(r["tau_peak"]) < hi]
        n = len(sel) // 3
        line = f"{name:<20}{n:>10}"
        rec = {"storm": name, "missions": n}
        for c in ["naive", "rule_based", "mpc"]:
            crit = np.array([float(r["critical_outage_h"]) for r in sel if r["controller"] == c])
            z, worst = ((crit == 0).mean(), crit.max()) if len(crit) else (np.nan, np.nan)
            line += f"{f'{z:.0%} / {worst:.0f}h':>22}"
            rec[f"{c}_zero_outage"] = round(float(z), 3); rec[f"{c}_worst_h"] = float(worst)
        print(line); out.append(rec)
    with open("results/severity_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("saved results/severity_table.csv")


# ---------------------------------------------------------------- 2. detector
def _detector_one(i, sols, data_dir):
    import warnings; warnings.filterwarnings("ignore")
    s = random_scenario(i, sols, data_dir)
    lg = simulate(s, make_naive)  # detector and forecast don't depend on the controller
    t, gen, flag = lg["t"], lg["gen"], lg["storm_flag"]
    sol_of = (t // SOL_HOURS).astype(int)
    n_sols = sol_of.max()
    # flag value set at the start of sol k reflects the judgement on sol k-1
    flag_by_sol = {k - 1: bool(flag[sol_of == k][0]) for k in range(1, n_sols + 1)}
    start = s.storm.start_sol
    first_storm_sol = int(np.ceil(start))
    flagged_after = [k for k, f in flag_by_sol.items() if f and k >= int(np.floor(start))]
    delay = (flagged_after[0] + 1 - start) if flagged_after else np.nan  # sols from onset to alarm
    false_alarms = sum(1 for k, f in flag_by_sol.items() if f and k < int(np.floor(start)))
    clear_sols = [k for k in flag_by_sol if 3 <= k < int(np.floor(start))]
    # day-ahead forecast error on clear sols (daily energy)
    errs = []
    for k in clear_sols:
        hist = sol_of < k; day = sol_of == k
        fc = fourier_forecast(t[hist], gen[hist], t[day])
        errs.append(abs(fc.sum() - gen[day].sum()) / gen[day].sum())
    return {"run": i, "tau_peak": s.storm.tau_peak, "delay_sols": delay, "false_alarms": false_alarms,
            "clear_sols": len(clear_sols), "forecast_mape": float(np.mean(errs)) if errs else np.nan}


def detector(n, sols, data_dir, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_detector_one, range(n), [sols] * n, [data_dir] * n))
    d = np.array([r["delay_sols"] for r in rows]); fa = sum(r["false_alarms"] for r in rows)
    cs = sum(r["clear_sols"] for r in rows); mape = np.nanmean([r["forecast_mape"] for r in rows])
    print(f"\nStorm detector over {n} missions (no hand-set threshold)")
    print(f"  storms detected:          {np.isfinite(d).mean():.0%}")
    print(f"  median detection delay:   {np.nanmedian(d):.1f} sols after onset (90th pct {np.nanpercentile(d, 90):.1f})")
    print(f"  false alarms:             {fa} on {cs} clear sols ({fa / max(cs, 1):.1%})")
    print(f"  Fourier day-ahead forecast error on clear sols: {mape:.1%} (daily energy)")
    with open("results/detector_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("saved results/detector_eval.csv")


# ---------------------------------------------------------------- 3. sensitivity
VARIATIONS = [
    ("baseline", None, None),
    ("thin wall 7 cm", "thermal.wall_thickness_m", 0.07),
    ("thick wall 15 cm", "thermal.wall_thickness_m", 0.15),
    ("half cabin heat capacity", "thermal.cabin_C", 1.5e7),
    ("double cabin heat capacity", "thermal.cabin_C", 6.0e7),
    ("small battery 200 kWh", "battery.nominal_kwh", 200.0),
    ("big battery 400 kWh", "battery.nominal_kwh", 400.0),
    ("worse insulation k=0.045", "thermal.wall_k", 0.045),
]


def _sens_one(args):
    import warnings; warnings.filterwarnings("ignore")
    vi, i, sols, data_dir = args
    name, path, val = VARIATIONS[vi]
    s = random_scenario(i, sols, data_dir)
    if path:
        obj, attr = path.split(".")
        setattr(getattr(s, obj), attr, val)
    out = {}
    for cname, mk in [("rule_based", make_rule), ("mpc", make_mpc)]:
        m = metrics(simulate(s, mk), s)
        out[cname] = (m["critical_outage_h"], m["hours_below_T_min"], m["science_done_frac"])
    return vi, out


def sensitivity(n, sols, data_dir, workers):
    jobs = [(vi, i, sols, data_dir) for vi in range(len(VARIATIONS)) for i in range(n)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_sens_one, jobs))
    print(f"\nSensitivity: {n} random missions per setting, controllers NOT retuned")
    hdr = ["setting", "MPC better", "MPC worse", "rule outage h", "mpc outage h", "rule cold h", "mpc cold h"]
    print(f"{hdr[0]:<28}" + "".join(f"{h:>15}" for h in hdr[1:]))
    rows = []
    for vi, (name, _, _) in enumerate(VARIATIONS):
        rs = [o for v, o in res if v == vi]
        better = sum(1 for o in rs if o["mpc"][:2] < o["rule_based"][:2])
        worse = sum(1 for o in rs if o["mpc"][:2] > o["rule_based"][:2])
        r = {"setting": name, "mpc_better": better, "mpc_worse": worse,
             "rule_outage_h": np.mean([o["rule_based"][0] for o in rs]), "mpc_outage_h": np.mean([o["mpc"][0] for o in rs]),
             "rule_cold_h": np.mean([o["rule_based"][1] for o in rs]), "mpc_cold_h": np.mean([o["mpc"][1] for o in rs])}
        rows.append(r)
        print(f"{name:<28}{better:>15}{worse:>15}{r['rule_outage_h']:>15.1f}{r['mpc_outage_h']:>15.1f}"
              f"{r['rule_cold_h']:>15.1f}{r['mpc_cold_h']:>15.1f}")
    with open("results/sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("saved results/sensitivity.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["severity", "detector", "sensitivity", "all"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--sols", type=int, default=40)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    if a.what in ("severity", "all"):
        severity()
    if a.what in ("detector", "all"):
        detector(a.n or 200, a.sols, a.data_dir, a.workers)
    if a.what in ("sensitivity", "all"):
        sensitivity(a.n or 30, a.sols, a.data_dir, a.workers)


if __name__ == "__main__":
    main()
