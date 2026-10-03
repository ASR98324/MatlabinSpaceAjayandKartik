"""Fourier (harmonic regression) solar forecast + autonomous storm detector.

The forecaster only ever sees PAST measured generation. It never sees tau or
the true storm schedule.
"""
import numpy as np
from .config import SOL_HOURS


def _design(t_h, n_harm, period):
    w = 2 * np.pi * np.asarray(t_h)[:, None] / period
    k = np.arange(1, n_harm + 1)[None, :]
    return np.hstack([np.ones((len(t_h), 1)), np.cos(k * w), np.sin(k * w)])


def fourier_forecast(t_hist, p_hist, t_future, n_harm=6, fit_sols=3.0, period=SOL_HOURS):
    """Fit sum of sol-period harmonics to the last `fit_sols` of history, extrapolate."""
    t_hist = np.asarray(t_hist); p_hist = np.asarray(p_hist)
    if len(t_hist) < 12:
        return np.zeros(len(t_future))
    mask = t_hist >= t_hist[-1] - fit_sols * period
    X = _design(t_hist[mask], n_harm, period)
    coef, *_ = np.linalg.lstsq(X, p_hist[mask], rcond=None)
    return np.clip(_design(np.asarray(t_future), n_harm, period) @ coef, 0.0, None)


class StormDetector:
    """Flags a storm when a sol's actual energy falls well below its forecast.

    The threshold is learned from the forecaster's own past errors (median - k*MAD
    of the log ratio), so there is no hand-set cutoff.
    """

    def __init__(self, k_mad=4.0, min_history=3, floor=0.05):
        self.k_mad = k_mad
        self.min_history = min_history
        self.floor = floor  # minimum log-ratio drop (~5%) to avoid triggering on noise
        self.log_ratios = []
        self.flags = []

    def update(self, actual_kwh, forecast_kwh):
        if forecast_kwh <= 1e-6:
            return False
        r = np.log(max(actual_kwh, 1e-6) / forecast_kwh)
        hist = np.array([x for x, f in zip(self.log_ratios, self.flags) if not f])  # nominal sols only
        storm = False
        if len(hist) >= self.min_history:
            med = np.median(hist)
            mad = np.median(np.abs(hist - med)) * 1.4826 + 1e-9
            storm = r < med - max(self.k_mad * mad, self.floor)
        self.log_ratios.append(r)
        self.flags.append(storm)
        return storm
