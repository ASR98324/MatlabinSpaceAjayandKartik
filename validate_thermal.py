"""Independent verification of the controller's thermal model (habitat/thermal.py).

Test 1, wall: thermal.py (backward Euler, dt = 1 h, 10 nodes, the exact settings the
  MPC uses) vs an independent Crank-Nicolson solver (heat1d.py, fine grid and step).
  Cabin side held at 21 C, outside steps from -60 C to -95 C (sunset cold snap).
Test 2, cabin: heaters off, does the 1 h step predict cabin cooling accurately?
  Compared with thermal.py run at a 36 s step (time-converged reference).

    python validate_thermal.py
"""
import copy

import numpy as np

from habitat.config import Scenario
from habitat.thermal import ThermalModel
from heat1d import solve as cn_solve  # teammate's independent solver

T_IN, T_OUT0, T_OUT1, HOURS = 21.0, -60.0, -95.0, 24


def wall_test(th_cfg):
    L, k, rho_c = th_cfg.wall_thickness_m, th_cfg.wall_k, th_cfg.wall_rho_c
    alpha = k / rho_c
    # thermal.py configured to the same problem: fixed cabin (huge C), near-perfect contact
    cfg = copy.deepcopy(th_cfg)
    cfg.cabin_C, cfg.h_in, cfg.h_out = 1e18, 1e7, 1e7
    m = ThermalModel(cfg, 1.0)
    x_nodes = np.linspace(0, L, cfg.n_wall_nodes)
    x = np.concatenate([[T_IN], T_IN + (T_OUT0 - T_IN) * x_nodes / L])
    prof_bE = [x[1:].copy()]
    for _ in range(HOURS):
        x = m.step(x, 0.0, T_OUT1)
        prof_bE.append(x[1:].copy())
    prof_bE = np.array(prof_bE)

    Nf = 201
    xf = np.linspace(0, L, Nf)
    res = cn_solve(L, Nf, alpha, HOURS * 3600.0, method="cn", dt=5.0,
                   bc_types=("dirichlet", "dirichlet"), bc_values=(T_IN, T_OUT1),
                   T_init=T_IN + (T_OUT0 - T_IN) * xf / L, save_every=720)  # every hour
    prof_cn = np.array([np.interp(x_nodes, xf, T) for T in res["T"]])

    # heat leaving the cabin through the inner surface, W/m^2
    q_bE = k * (prof_bE[:, 0] - prof_bE[:, 1]) / (x_nodes[1])
    q_cn = k * (res["T"][:, 0] - res["T"][:, 1]) / (xf[1])
    return x_nodes, prof_bE, prof_cn, q_bE, q_cn


def cabin_test(scen, hours=48):
    def run(dt_h):
        m = ThermalModel(scen.thermal, dt_h)
        x = m.initial_state(scen.thermal.T_set, scen.thermal.ambient_mean)
        Q = 1000 * scen.loads.internal_gain_frac * scen.loads.critical_kw  # heaters off
        n = int(round(hours / dt_h)); out = [x[0]]
        for _ in range(n):
            x = m.step(x, Q, scen.thermal.ambient_mean); out.append(x[0])
        return np.arange(n + 1) * dt_h, np.array(out)
    return run(1.0), run(0.01)


def main():
    scen = Scenario()
    x_nodes, pb, pc, qb, qc = wall_test(scen.thermal)
    (t1, T1), (tr, Tr) = cabin_test(scen)
    T_ref_on_1h = np.interp(t1, tr, Tr)

    hrs = np.arange(HOURS + 1)
    print("Test 1  wall profile, thermal.py (dt=1h) vs Crank-Nicolson (dt=5s, 201 nodes)")
    print(f"  max node error after 1 h:  {np.abs(pb[1] - pc[1]).max():.2f} C")
    print(f"  max node error after 6 h:  {np.abs(pb[6] - pc[6]).max():.2f} C")
    print(f"  max node error after 24 h: {np.abs(pb[-1] - pc[-1]).max():.3f} C")
    print(f"  steady inner heat flux: {qb[-1]:.1f} vs {qc[-1]:.1f} W/m^2 (first-difference estimates)")
    print("Test 2  cabin cool-down, heaters off, 48 h")
    print(f"  cabin drop: {T1[0] - T1[-1]:.2f} C (dt=1h) vs {Tr[0] - Tr[-1]:.2f} C (reference)")
    print(f"  max cabin temperature error: {np.abs(T1 - T_ref_on_1h).max():.3f} C")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for h, c in [(1, "C0"), (3, "C1"), (6, "C2"), (24, "C3")]:
        ax[0].plot(x_nodes * 100, pc[h], "-", color=c, label=f"CN, {h} h")
        ax[0].plot(x_nodes * 100, pb[h], "o", color=c, mfc="none", label=f"thermal.py, {h} h")
    ax[0].set_xlabel("depth into wall (cm)"); ax[0].set_ylabel("C")
    ax[0].set_title("Wall profile after outside cold snap"); ax[0].legend(fontsize=7, ncol=2)
    ax[1].plot(hrs, qc, "-", label="Crank-Nicolson"); ax[1].plot(hrs, qb, "o", mfc="none", label="thermal.py (dt=1h)")
    ax[1].set_xlabel("hours"); ax[1].set_ylabel("W/m^2"); ax[1].set_title("Heat flux out of cabin"); ax[1].legend()
    ax[2].plot(tr, Tr, "-", label="reference (dt=36 s)"); ax[2].plot(t1, T1, "o", ms=3, mfc="none", label="thermal.py (dt=1h)")
    ax[2].set_xlabel("hours"); ax[2].set_ylabel("cabin C"); ax[2].set_title("Cabin cool-down, heaters off"); ax[2].legend()
    fig.suptitle("Verification of the controller's thermal model")
    fig.tight_layout(); fig.savefig("results/thermal_validation.png", dpi=130); plt.close(fig)
    print("saved results/thermal_validation.png")


if __name__ == "__main__":
    main()
