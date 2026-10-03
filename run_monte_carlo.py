"""Randomized-scenario evaluation: the autonomy proof.

Every controller runs with FROZEN settings on N random missions (storm timing and
severity, battery failure time and size, which NASA cell, how aged it starts).
No retuning between runs.

    python run_monte_carlo.py --n 200            # full run, all CPU cores
    python run_monte_carlo.py --n 8 --sols 30    # quick smoke test
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from habitat.config import Scenario
from habitat.controllers import NaiveController, RuleBasedController
from habitat.mpc import MPCController
from habitat.simulator import metrics, simulate

CELLS = ["B0005", "B0006", "B0018"]  # cells that reach end of life; B0007 is censored


def make_naive(s, th): return NaiveController(s)
def make_rule(s, th): return RuleBasedController(s)
def make_mpc(s, th): return MPCController(s, th)


CONTROLLERS = {"naive": make_naive, "rule_based": make_rule, "mpc": make_mpc}


def random_scenario(i, sols, data_dir):
    rng = np.random.default_rng(1000 + i)
    s = Scenario(seed=1000 + i, sols=sols, data_dir=data_dir)
    s.storm.start_sol = rng.uniform(4, 12)
    s.storm.tau_peak = rng.uniform(2.0, 4.5)  # upper end is likely unsurvivable for any controller
    s.storm.ramp_sols = rng.uniform(2, 6)
    s.storm.plateau_sols = rng.uniform(2, 10)
    s.storm.decay_sols = rng.uniform(10, 30)
    if rng.random() < 0.8:
        s.failure.string_fail_sol = rng.uniform(3, sols - 5)
        s.failure.string_fail_frac = rng.uniform(0.1, 0.35)
    else:
        s.failure.string_fail_sol = None
    s.battery.cell_id = CELLS[rng.integers(len(CELLS))]
    s.battery.start_cycle = int(rng.integers(10, 110))
    return s


def run_one(i, sols, data_dir):
    import warnings
    warnings.filterwarnings("ignore")
    s = random_scenario(i, sols, data_dir)
    rows = []
    for name, mk in CONTROLLERS.items():
        m = metrics(simulate(s, mk), s)
        rows.append({"run": i, "controller": name, "tau_peak": s.storm.tau_peak,
                     "fail_sol": s.failure.string_fail_sol if s.failure.string_fail_sol is not None else np.nan,
                     "fail_frac": s.failure.string_fail_frac if s.failure.string_fail_sol is not None else 0.0,
                     "cell": s.battery.cell_id, "start_cycle": s.battery.start_cycle, **m})
    return rows


def summarize(rows):
    import csv
    os.makedirs("results", exist_ok=True)
    keys = list(rows[0].keys())
    with open("results/monte_carlo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    by = {c: [r for r in rows if r["controller"] == c] for c in CONTROLLERS}
    n = len(by["mpc"])
    print(f"\n{n} randomized missions, frozen settings\n")
    hdr = ["controller", "zero_crit_outage", "zero_cold_hours", "mean_crit_h", "mean_cold_h", "mean_science", "mean_essential"]
    print("".join(f"{h:>18}" for h in hdr))
    for c, rs in by.items():
        crit = np.array([r["critical_outage_h"] for r in rs]); cold = np.array([r["hours_below_T_min"] for r in rs])
        sci = np.array([r["science_done_frac"] for r in rs]); ess = np.array([r["essential_uptime_frac"] for r in rs])
        vals = [c, f"{(crit == 0).mean():.0%}", f"{(cold == 0).mean():.0%}", f"{crit.mean():.1f}", f"{cold.mean():.1f}",
                f"{sci.mean():.0%}", f"{ess.mean():.0%}"]
        print("".join(f"{v:>18}" for v in vals))

    # paired comparison: same mission, MPC vs rule-based
    rb = {r["run"]: r for r in by["rule_based"]}
    better = sum(1 for r in by["mpc"] if (r["critical_outage_h"], r["hours_below_T_min"]) <
                 (rb[r["run"]]["critical_outage_h"], rb[r["run"]]["hours_below_T_min"]))
    worse = sum(1 for r in by["mpc"] if (r["critical_outage_h"], r["hours_below_T_min"]) >
                (rb[r["run"]]["critical_outage_h"], rb[r["run"]]["hours_below_T_min"]))
    print(f"\nMPC vs rule-based on the same mission (safety: outage, then cold hours): "
          f"better {better}, worse {worse}, tied {n - better - worse}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    names = list(CONTROLLERS)
    for a, key, title in [(ax[0], "critical_outage_h", "Life-support outage (h)"),
                          (ax[1], "hours_below_T_min", "Hours below 18 C")]:
        a.boxplot([[r[key] for r in by[c]] for c in names], tick_labels=names)
        a.set_title(title)
    for c in names:
        tp = [r["tau_peak"] for r in by[c]]; crit = [r["critical_outage_h"] for r in by[c]]
        ax[2].scatter(tp, crit, s=12, alpha=0.6, label=c)
    ax[2].set_xlabel("storm peak tau"); ax[2].set_ylabel("life-support outage (h)")
    ax[2].set_title("Outage vs storm severity"); ax[2].legend()
    fig.suptitle(f"{n} randomized missions, frozen controller settings")
    fig.tight_layout(); fig.savefig("results/monte_carlo.png", dpi=130); plt.close(fig)
    print("saved results/monte_carlo.csv and results/monte_carlo.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sols", type=int, default=40)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    t0 = time.time(); rows = []; done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, i, args.sols, args.data_dir) for i in range(args.n)]
        for f in as_completed(futs):
            rows += f.result(); done += 1
            if done % max(1, args.n // 20) == 0 or done == args.n:
                el = time.time() - t0
                print(f"{done}/{args.n} missions, {el:.0f}s elapsed, ~{el / done * (args.n - done):.0f}s left", flush=True)
    rows.sort(key=lambda r: (r["run"], r["controller"]))
    summarize(rows)


if __name__ == "__main__":
    main()
