"""Mars solar generation with dust-storm attenuation.

Generated from physics (sun angle x dust attenuation x noise), NOT from a
Fourier series, so the Fourier forecaster in forecast.py is not circular.
"""
import numpy as np
from .config import SOL_HOURS, SolarConfig, StormConfig


def tau_profile(t_h: np.ndarray, storm: StormConfig) -> np.ndarray:
    """Dust optical depth over time: clear -> ramp -> plateau -> exponential-ish decay."""
    sol = np.asarray(t_h) / SOL_HOURS
    tau = np.full_like(sol, storm.tau_clear, dtype=float)
    s0 = storm.start_sol
    s1 = s0 + storm.ramp_sols
    s2 = s1 + storm.plateau_sols
    ramp = (sol >= s0) & (sol < s1)
    tau[ramp] = storm.tau_clear + (storm.tau_peak - storm.tau_clear) * (sol[ramp] - s0) / storm.ramp_sols
    tau[(sol >= s1) & (sol < s2)] = storm.tau_peak
    dec = sol >= s2
    tau[dec] = storm.tau_clear + (storm.tau_peak - storm.tau_clear) * np.exp(-(sol[dec] - s2) / (storm.decay_sols / 3))
    return tau


def cos_zenith(t_h: np.ndarray) -> np.ndarray:
    """Equatorial, equinox approximation: sun overhead at local noon."""
    local = np.mod(t_h, SOL_HOURS)
    hour_angle = 2 * np.pi * (local - SOL_HOURS / 2) / SOL_HOURS
    return np.clip(np.cos(hour_angle), 0.0, None)


def solar_power_kw(t_h, tau, cfg: SolarConfig, rng: np.random.Generator | None = None):
    mu = cos_zenith(t_h)
    p = cfg.solar_const_w_m2 * mu * np.exp(-cfg.k_tau * tau) * cfg.array_area_m2 * cfg.panel_eff / 1000.0
    if rng is not None and cfg.noise_std > 0:
        p = p * np.exp(rng.normal(0, cfg.noise_std, size=np.shape(p)))
    return p + cfg.baseload_kw
