"""Monte-Carlo comparison of attitude-determination estimators.

Runs many randomized scenarios (random initial attitude, random tumble axis and
rate, independent noise seed) and scores estimator variants on the *same*
scenarios, so differences are the estimators' and not the draw's. Aggregates the
post-convergence attitude-knowledge error against the 10-15 deg budget and ranks
the variants.

Variants compared by default:
    MEKF             full filter (TRIAD init, gyro coast through eclipse)
    MEKF cold-start  same filter, identity init + large P (no TRIAD)
    TRIAD-only       gyro-free algebraic two-vector fix, holds in eclipse
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attitude import rand_quat
from .constants import DEG
from .estimator.algebraic import run_triad
from .estimator.mekf import MEKF, run_mekf
from .estimator.ukf import USQUE, run_ukf
from .scenario import Scenario

__all__ = ["VariantStats", "randomized_scenario", "default_runners",
           "run_monte_carlo", "rank", "REALISM_LEVELS", "run_realism_sweep"]


@dataclass
class VariantStats:
    """Per-run error metrics (deg) for one estimator variant across the MC set."""

    name: str
    run_median: np.ndarray     # (R,) per-run median error
    run_p95: np.ndarray        # (R,) per-run 95th percentile
    run_max: np.ndarray        # (R,) per-run max
    run_eclipse_median: np.ndarray  # (R,) per-run median error in eclipse

    def overall_median(self):
        return float(np.median(self.run_median))

    def overall_p95(self):
        return float(np.median(self.run_p95))

    def worst(self):
        return float(np.max(self.run_max))

    def budget_met_fraction(self, budget=15.0):
        """Fraction of runs whose 95th-percentile error stays within budget."""
        return float(np.mean(self.run_p95 < budget))


def randomized_scenario(seed, dt=1.0, duration=None, albedo_coeff=0.0,
                        field_model_error_deg=0.0):
    """A scenario with random initial attitude + tumble and its own noise seed.

    ``albedo_coeff`` and ``field_model_error_deg`` inject the systematic error
    sources (0 => truth == model, the validation baseline).
    """
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    omega0 = rng.uniform(1.0, 8.0) * DEG * axis      # 1-8 deg/s tumble
    q0 = rand_quat(rng)
    sc = Scenario(dt=dt, seed=int(seed), q0=q0, omega0=omega0,
                  albedo_coeff=albedo_coeff,
                  field_model_error_deg=field_model_error_deg)
    if duration is not None:
        sc.duration = duration
    return sc


# Systematic-error levels for the realism sweep: (label, albedo_coeff, field_deg).
REALISM_LEVELS = [
    ("ideal (validation)", 0.0, 0.0),
    ("moderate", 0.3, 5.0),
    ("worst-case", 0.5, 8.0),
]


def run_realism_sweep(levels=None, n_runs=30, seed0=0, dt=1.0, warmup_frac=0.2):
    """Run the MEKF over randomized scenarios at several systematic-error levels.

    Returns {level_label: VariantStats}, quantifying how the two dominant
    unmodeled error sources (albedo, IGRF-vs-dipole) degrade accuracy.
    """
    levels = REALISM_LEVELS if levels is None else levels
    out = {}
    for name, alb, fld in levels:
        acc = {"median": [], "p95": [], "max": [], "ecl": []}
        for i in range(n_runs):
            res = randomized_scenario(seed0 + i, dt=dt, albedo_coeff=alb,
                                      field_model_error_deg=fld).run()
            w = int(len(res) * warmup_frac)
            err = run_mekf(res).err_deg
            conv = err[w:]
            acc["median"].append(np.median(conv))
            acc["p95"].append(np.percentile(conv, 95))
            acc["max"].append(conv.max())
            ecl = err[~res.lit]
            acc["ecl"].append(np.median(ecl) if ecl.size else np.nan)
        out[name] = VariantStats(name, np.array(acc["median"]),
                                 np.array(acc["p95"]), np.array(acc["max"]),
                                 np.array(acc["ecl"]))
    return out


def default_runners():
    """Name -> callable(SimResult) -> err_deg array, for the default comparison."""
    return {
        "MEKF": lambda res: run_mekf(res, MEKF(), init_triad=True).err_deg,
        "USQUE (UKF)": lambda res: run_ukf(res, USQUE(), init_triad=True).err_deg,
        "MEKF cold-start": lambda res: run_mekf(res, MEKF(), init_triad=False).err_deg,
        "TRIAD-only": lambda res: run_triad(res)[1],
    }


def run_monte_carlo(runners=None, n_runs=30, seed0=0, dt=1.0, duration=None,
                    warmup_frac=0.2):
    """Run the MC set and return {name: VariantStats}.

    ``warmup_frac`` of each run is discarded before scoring so the initial
    convergence transient doesn't dominate the statistics.
    """
    runners = default_runners() if runners is None else runners
    acc = {name: {"median": [], "p95": [], "max": [], "ecl": []}
           for name in runners}

    for i in range(n_runs):
        res = randomized_scenario(seed0 + i, dt=dt, duration=duration).run()
        w = int(len(res) * warmup_frac)
        eclipse_mask = ~res.lit
        for name, fn in runners.items():
            err = np.asarray(fn(res), dtype=float)
            conv = err[w:]
            acc[name]["median"].append(np.median(conv))
            acc[name]["p95"].append(np.percentile(conv, 95))
            acc[name]["max"].append(conv.max())
            ecl = err[eclipse_mask]
            acc[name]["ecl"].append(np.median(ecl) if ecl.size else np.nan)

    return {name: VariantStats(name,
                               np.array(a["median"]), np.array(a["p95"]),
                               np.array(a["max"]), np.array(a["ecl"]))
            for name, a in acc.items()}


def rank(stats, budget=15.0):
    """Return variants sorted best-first by overall median error."""
    return sorted(stats.values(), key=lambda s: s.overall_median())
