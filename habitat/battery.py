"""Habitat battery pack whose capacity follows a REAL NASA cell's fade curve."""
import glob
import os
import warnings

import numpy as np

from .config import BatteryConfig, FailureConfig


def load_nasa_capacity(cell_id: str, data_dir: str) -> np.ndarray:
    """Discharge capacity (Ah) per cycle for one NASA cell, e.g. 'B0005'."""
    hits = glob.glob(os.path.join(data_dir, "**", f"{cell_id}.mat"), recursive=True)
    if not hits:
        warnings.warn(f"{cell_id}.mat not found under '{data_dir}'. Using SYNTHETIC fade. "
                      "Do not report results from this mode.")
        cyc = np.arange(170)
        return 1.86 - 0.0032 * cyc - 0.15 * (cyc / 170) ** 3
    from scipy.io import loadmat
    cycles = loadmat(hits[0], simplify_cells=True)[cell_id]["cycle"]
    return np.array([c["data"]["Capacity"] for c in cycles if c["type"] == "discharge"], dtype=float)


class BatteryPack:
    def __init__(self, cfg: BatteryConfig, fail: FailureConfig, data_dir: str):
        self.cfg = cfg
        self.fail = fail
        self.cap_ah = load_nasa_capacity(cfg.cell_id, data_dir)
        self.capacity_kwh = self.capacity_at_sol(0.0)
        self.E = cfg.init_soc * self.capacity_kwh
        self.failed = False

    def health_at_sol(self, sol: float) -> float:
        """Capacity fraction from the NASA curve (1 cycle per sol). Holds last value past the data."""
        idx = self.cfg.start_cycle + sol * self.cfg.cycles_per_sol
        idx = np.clip(idx, 0, len(self.cap_ah) - 1)
        return float(np.interp(idx, np.arange(len(self.cap_ah)), self.cap_ah) / self.cfg.rated_ah)

    def capacity_at_sol(self, sol: float) -> float:
        cap = self.cfg.nominal_kwh * self.health_at_sol(sol)
        if self.fail.string_fail_sol is not None and sol >= self.fail.string_fail_sol:
            cap *= 1.0 - self.fail.string_fail_frac
        return cap

    def update_capacity(self, sol: float):
        self.capacity_kwh = self.capacity_at_sol(sol)
        self.E = min(self.E, self.capacity_kwh)  # energy in a dead string is lost
        self.failed = self.fail.string_fail_sol is not None and sol >= self.fail.string_fail_sol

    @property
    def soc(self) -> float:
        return self.E / self.capacity_kwh if self.capacity_kwh > 0 else 0.0

    def max_discharge_kwh(self) -> float:
        return self.E * self.cfg.eff_discharge

    def apply(self, net_kwh: float) -> float:
        """net_kwh > 0 charges, < 0 discharges (delivered energy). Returns curtailed surplus."""
        if net_kwh >= 0:
            room = self.capacity_kwh - self.E
            stored = min(net_kwh * self.cfg.eff_charge, room)
            self.E += stored
            return net_kwh - stored / self.cfg.eff_charge
        self.E = max(0.0, self.E + net_kwh / self.cfg.eff_discharge)
        return 0.0
