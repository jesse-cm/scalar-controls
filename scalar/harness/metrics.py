"""The fixed metric set of the verification loop (docs/verification-loop-plan.md, phase 0).

Every definition here is taken from the code that produced the numbers in
docs/learned-estimation-and-control.md, not re-derived:

    knowledge median / p95      np.median / np.percentile(.., 95) of the MEKF-vs-truth angle [deg]
                                over the scoring window, split by the sunlit mask
    pointing mean / p95         truth-vs-target angle [deg] over the last orbit (scripts' ``last`` slice)
    settle time                 ClosedLoopResult.settle_time() [h]; inf = never settled
    saturation fraction         fraction of steps with any channel at >= 0.999 m_max (tune.py)
    coil duty per channel       mean |m_j| / m_max_j                      (rl_tune.evaluate)
    coil power                  sum_j duty_j * i_max_j * 3.3 V  [W]       (rl_tune.evaluate)
    max rate                    max |omega| [deg/s]
    divergence flag             max pointing error in the last orbit > 90 deg (rl_tune.evaluate)

A metric that does not apply to a run stays NaN and is written to the ledger as
an empty cell. Quantities outside the fixed set (sun-vector error, sysid matrix
error, ...) go to ``Metrics.extra``; they never reach the ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Metrics", "FIXED_METRICS", "COIL_VOLTAGE", "knowledge_stats",
           "pointing_stats", "actuation_stats", "sunlit_mask", "channel_dipoles"]

COIL_VOLTAGE = 3.3   # [V] the S6 H-bridge rail rl_tune.evaluate uses for coil power

FIXED_METRICS = (
    "know_med_all", "know_p95_all", "know_med_lit", "know_p95_lit", "know_med_ecl", "know_p95_ecl",
    "point_mean", "point_p95", "settle_h", "sat_frac",
    "duty_ch0", "duty_ch1", "duty_ch2", "coil_power_w", "max_rate_deg_s", "diverged",
)


@dataclass
class Metrics:
    """One (spec, seed) result. Units: deg, h, W, deg/s; fractions in 0..1; NaN = not applicable."""

    know_med_all: float = np.nan
    know_p95_all: float = np.nan
    know_med_lit: float = np.nan
    know_p95_lit: float = np.nan
    know_med_ecl: float = np.nan
    know_p95_ecl: float = np.nan
    point_mean: float = np.nan
    point_p95: float = np.nan
    settle_h: float = np.nan
    sat_frac: float = np.nan
    duty_ch0: float = np.nan
    duty_ch1: float = np.nan
    duty_ch2: float = np.nan
    coil_power_w: float = np.nan
    max_rate_deg_s: float = np.nan
    diverged: float = np.nan
    extra: dict = field(default_factory=dict)   # experiment-specific numbers, not ledger columns
    wall_s: float = np.nan

    def fixed(self):
        return {k: float(getattr(self, k)) for k in FIXED_METRICS}


def _med_p95(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return np.nan, np.nan
    return float(np.median(x)), float(np.percentile(x, 95))


def knowledge_stats(err_deg, lit):
    """Six knowledge numbers from an error series and its sunlit mask (same window)."""
    err_deg = np.asarray(err_deg, float)
    lit = np.asarray(lit, bool)
    out = {}
    out["know_med_all"], out["know_p95_all"] = _med_p95(err_deg)
    out["know_med_lit"], out["know_p95_lit"] = _med_p95(err_deg[lit])
    out["know_med_ecl"], out["know_p95_ecl"] = _med_p95(err_deg[~lit])
    return out


def pointing_stats(result, n_last, settle_thresh_deg=10.0, settle_hold=0.9, diverge_deg=90.0):
    """Pointing mean/p95 and divergence over the last ``n_last`` samples, settle time over the run."""
    pe = result.pointing_err_deg[-n_last:]
    return {"point_mean": float(np.mean(pe)),
            "point_p95": float(np.percentile(pe, 95)),
            "settle_h": float(result.settle_time(settle_thresh_deg, settle_hold)) / 3600.0,
            "diverged": float(np.max(pe) > diverge_deg)}


def channel_dipoles(dipole, actuators):
    """Per-channel signed dipole (N, n_ch) of a body-frame dipole history."""
    axes = np.array([tq.axis for tq in actuators.torquers])
    return np.asarray(dipole, float) @ axes.T


def actuation_stats(result, actuators, i_max, sl=slice(None)):
    """Saturation fraction, per-channel duty, coil power and max rate over the slice ``sl``."""
    m_max = actuators.m_max_vector
    mc = np.abs(channel_dipoles(result.dipole[sl], actuators))
    duty = np.mean(mc / m_max, axis=0)
    out = {"sat_frac": float(np.mean(np.any(mc >= 0.999 * m_max, axis=1))),
           "coil_power_w": float(np.sum(duty * np.asarray(i_max, float) * COIL_VOLTAGE)),
           "max_rate_deg_s": float(np.max(result.rate_deg_s[sl]))}
    for j, d in enumerate(duty):
        out[f"duty_ch{j}"] = float(d)
    return out


def sunlit_mask(orbit, jd0, t):
    """Sunlit flag at each sample time, by the same test tune.py applies inside its loop."""
    from ..environment.sun import eclipse_factor, sun_direction_eci
    return np.array([eclipse_factor(orbit.position(tk), sun_direction_eci(jd0 + tk / 86400.0)) > 0
                     for tk in t])
