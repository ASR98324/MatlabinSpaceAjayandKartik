"""Run the default scenario with every controller, print metrics, save the plot.

    python run_scenario.py                 # default storm (tau peak 6)
    python run_scenario.py --tau-peak 10.8 # full 2018-style storm
"""
import argparse
import os

from habitat.config import Scenario
from habitat.controllers import MPCController, NaiveController, RuleBasedController
from habitat.simulator import metrics, plot_run, simulate

CONTROLLERS = {
    "naive": lambda s, th: NaiveController(s),
    "rule_based": lambda s, th: RuleBasedController(s),
    "mpc": lambda s, th: MPCController(s, th),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-peak", type=float, default=None)
    ap.add_argument("--sols", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    scen = Scenario(seed=args.seed, data_dir=args.data_dir)
    if args.tau_peak is not None:
        scen.storm.tau_peak = args.tau_peak
    if args.sols is not None:
        scen.sols = args.sols

    logs = [simulate(scen, mk) for mk in CONTROLLERS.values()]
    keys = list(metrics(logs[0], scen).keys())
    print(f"{'controller':<12}" + "".join(f"{k:>22}" for k in keys))
    for lg in logs:
        m = metrics(lg, scen)
        print(f"{lg['controller']:<12}" + "".join(f"{m[k]:>22.3f}" for k in keys))
    os.makedirs("results", exist_ok=True)
    plot_run(logs, scen, "results/timeline.png")
    print("saved results/timeline.png")


if __name__ == "__main__":
    main()
