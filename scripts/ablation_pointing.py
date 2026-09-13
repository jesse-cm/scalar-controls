#!/usr/bin/env python3
"""Ablation of the fine-pointing error: who owns the 7.4 deg?

    python3 scripts/ablation_pointing.py [--seeds 100,101,102] [--workers N]

The tuned three-mode controller closes the loop on an MEKF estimate at 1 Hz.
Fine pointing is 7.4 deg mean / 18 deg p95 against a 5 deg criterion, on one
seed. This script splits that number two ways, over three seeds:

MODE A (one-at-a-time)  start from the baseline B and remove ONE factor.
MODE C (cumulative)     start from the ideal I (F1..F6 all removed) and add
                        factors back in expected-size order, ending at B.

Comparing the two answers the only question that matters: do the factors add
up, or do they interact?

FACTORS
    F1  gyro bias random walk      bias_rw=0 AND filter sigma_u -> 1e-9
    F2  gyro white noise           rate_noise_density=0
    F3  magnetometer white noise   noise_std=0
    F4  mag residual distortion    ideal Magnetometer(), no calibration pass
    F5  sun-sensor noise           SunSensorArray(noise_frac=0.0)
    F6  residual dipole            m_res=0
    F7  control step (1 Hz)        SKIPPED — cannot be toggled without editing
                                   scalar/control/tune.py, which is off-limits.
    F8  controller on truth        use_truth=True; the bound with the whole
                                   estimator side removed. Not a factor in the
                                   build-up, a reference line.

WHY THE sigma_u MONKEYPATCH
``simulate_closed_loop_estimated`` constructs its ``MEKF(n_iter=3)`` itself, so
the filter's assumed bias random walk is not reachable through the call. When
F1 zeroes the *plant's* bias RW the filter must be told, or the comparison is
between a quiet gyro and a filter still expecting a noisy one. The patch below
only setdefault()s sigma_u, so it cannot override an explicit argument, and it
is restored in a finally block.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import quat_from_rotvec  # noqa: E402
from scalar.constants import DEG  # noqa: E402
from scalar.control.rl_tune import (build_controller, calibration_pass,  # noqa: E402
                                    default_plant, load_params)
from scalar.control.tune import (actuators_from_matrix,  # noqa: E402
                                 simulate_closed_loop_estimated)
from scalar.dynamics import RigidBody  # noqa: E402
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.environment.sun import eclipse_factor, sun_direction_eci  # noqa: E402
from scalar.sensors.gyro import Gyro  # noqa: E402
from scalar.sensors.magnetometer import Magnetometer  # noqa: E402
from scalar.sensors.sun_sensor import SunSensorArray  # noqa: E402
from scalar.timeutil import julian_date  # noqa: E402

PARAMS_PATH = "results/lqr_tuned_3mode.json"
OUT_CSV = "results/ablation_pointing.csv"
SEEDS = (100, 101, 102)
ALL_FACTORS = ("F1", "F2", "F3", "F4", "F5", "F6")
SIGMA_U_OFF = 1e-9      # filter bias-RW when the plant's bias RW is zeroed

# --- the runs -------------------------------------------------------------
# Each config is (name, frozenset of REMOVED factors, use_truth).
MODE_A = [("B", frozenset(), False)]
MODE_A += [(f"B-{f}", frozenset({f}), False) for f in ALL_FACTORS]
MODE_A += [("F8 (truth)", frozenset(), True)]

_ALL = frozenset(ALL_FACTORS)
MODE_C = [
    ("I", _ALL, False),
    ("I+F1", _ALL - {"F1"}, False),
    ("I+F1+F4", _ALL - {"F1", "F4"}, False),
    ("I+F1+F4+F6", _ALL - {"F1", "F4", "F6"}, False),
    ("I+F1+F4+F6+F2+F3+F5 (=B)", frozenset(), False),
]


def _build(seed, removed, use_truth):
    """Assemble the one episode described by ``removed``; return the kwargs."""
    p, _ = load_params(PARAMS_PATH)
    plant = default_plant(mag_calibrated=True)
    orbit = CircularOrbit(inclination=plant.inclination_deg * DEG)
    jd0 = julian_date(plant.epoch)
    acts = actuators_from_matrix(plant.M, plant.i_max)

    # --- F4: residual magnetometer distortion after calibration -------------
    # Removing it means an ideal sensor (bias 0, soft-iron I) and therefore no
    # calibration pass at all; keeping it means the real bias/soft-iron plus
    # the ellipsoid fit, exactly as the baseline runs it.
    noise_std = 0.0 if "F3" in removed else Magnetometer.noise_std
    if "F4" in removed:
        mag = Magnetometer(noise_std=noise_std)
        cal = None
    else:
        mag = Magnetometer(noise_std=noise_std, bias=plant.mag_bias,
                           soft_iron=plant.mag_soft_iron)
        # The calibration pass sees the same sensor the pointing episode does.
        cal, _ = calibration_pass(plant, mag, orbit, jd0, seed)

    # --- F1 / F2: the gyro --------------------------------------------------
    gyro = Gyro(rate_noise_density=0.0 if "F2" in removed else Gyro.rate_noise_density,
                bias_rw=0.0 if "F1" in removed else Gyro.bias_rw)

    sun = SunSensorArray(noise_frac=0.0) if "F5" in removed else SunSensorArray()
    m_res = np.zeros(3) if "F6" in removed else plant.m_res

    ctrl = build_controller(p, plant, acts, orbit, jd0)
    ctrl.mode = 3
    return dict(
        body=RigidBody(plant.inertia), actuators=acts, orbit=orbit, jd0=jd0,
        q0=quat_from_rotvec(np.deg2rad(20) * np.array([0.6, 0.8, 0.0])),
        omega0=np.deg2rad([0.1, -0.05, 0.08]), n_orbits=3, seed=seed,
        controller=ctrl, m_res=m_res, magnetometer=mag, mag_calibration=cal,
        gyro=gyro, sun_sensor=sun, use_truth=use_truth,
    )


def _simulate(kw, match_sigma_u):
    """Run the episode, matching the filter's sigma_u when F1 is removed."""
    from scalar.estimator import mekf as mekf_mod
    if not match_sigma_u:
        return simulate_closed_loop_estimated(**kw)
    orig = mekf_mod.MEKF.__init__

    def patched(self, *a, **k):
        k.setdefault("sigma_u", SIGMA_U_OFF)
        orig(self, *a, **k)

    mekf_mod.MEKF.__init__ = patched
    try:
        return simulate_closed_loop_estimated(**kw)
    finally:
        mekf_mod.MEKF.__init__ = orig


def _metrics(r, orbit, jd0):
    """Last-orbit pointing/knowledge metrics plus the sunlit/eclipse split."""
    n_last = int(orbit.period)
    last = slice(-n_last, None)
    pe = r.pointing_err_deg[last]
    ke = r.knowledge_err_deg[last]
    t = r.t[last]
    lit = np.array([eclipse_factor(orbit.position(ti),
                                   sun_direction_eci(jd0 + ti / 86400.0)) > 0
                    for ti in t])
    return dict(
        point_mean=float(np.mean(pe)),
        point_p95=float(np.percentile(pe, 95)),
        know_med=float(np.median(ke)),
        lit_mean=float(np.mean(pe[lit])) if lit.any() else float("nan"),
        ecl_mean=float(np.mean(pe[~lit])) if (~lit).any() else float("nan"),
    )


def _worker(job):
    """Module-level worker so the spawn context can pickle it."""
    key, seed, removed, use_truth = job
    kw = _build(seed, removed, use_truth)
    orbit, jd0 = kw["orbit"], kw["jd0"]
    r = _simulate(kw, match_sigma_u=("F1" in removed))
    return key, _metrics(r, orbit, jd0)


# --- reporting ------------------------------------------------------------
COLS = ("point_mean", "point_p95", "know_med", "lit_mean", "ecl_mean")


def _agg(rows):
    """mean and spread (max-min) across seeds, per column."""
    out = {}
    for c in COLS:
        v = np.array([r[c] for r in rows], float)
        out[c] = (float(np.mean(v)), float(np.ptp(v)))
    return out


def _table(title, configs, results, seeds):
    print(f"\n{title}")
    hdr = f"{'config':<26}" + "".join(f"{c:>19}" for c in COLS)
    print(hdr)
    print("-" * len(hdr))
    for name, removed, use_truth in configs:
        rows = [results[(name, s)] for s in seeds]
        a = _agg(rows)
        line = f"{name:<26}"
        for c in COLS:
            m, sp = a[c]
            line += f"{m:>12.2f} +-{sp:>5.2f}"
        print(line)
    print(f"(mean over seeds {list(seeds)}; +- is the max-min spread across them)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)   # load_params / the csv path are repo-relative

    jobs, seen = [], {}
    for mode, configs in (("A", MODE_A), ("C", MODE_C)):
        for (name, removed, use_truth), seed in itertools.product(configs, seeds):
            sig = (removed, use_truth, seed)
            if sig in seen:      # B appears in both tables; simulate it once
                continue
            seen[sig] = (name, seed)
            jobs.append(((name, seed), seed, removed, use_truth))

    n_rows = sum(len(c) for c in (MODE_A, MODE_C)) * len(seeds)
    print(f"{len(jobs)} simulations ({n_rows} table rows; duplicate baselines reused)")
    workers = args.workers or max(1, min(len(jobs), mp.cpu_count() - 1))
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            done = dict(pool.map(_worker, jobs))
    else:
        done = dict(_worker(j) for j in jobs)

    results = {}
    for mode, configs in (("A", MODE_A), ("C", MODE_C)):
        for (name, removed, use_truth), seed in itertools.product(configs, seeds):
            results[(name, seed)] = done[seen[(removed, use_truth, seed)]]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "config", "seed", *COLS])
        for mode, configs in (("A", MODE_A), ("C", MODE_C)):
            for (name, removed, use_truth), seed in itertools.product(configs, seeds):
                m = results[(name, seed)]
                w.writerow([mode, name, seed, *[f"{m[c]:.4f}" for c in COLS]])

    _table("MODE A — one-at-a-time removal from the baseline B", MODE_A, results, seeds)
    _table("MODE C — cumulative build-up from the ideal I", MODE_C, results, seeds)
    print("\nF7 (control step) SKIPPED. It is not a factor of the same kind: the "
          "256-bin periodic-LQR schedule, the quantiser and the MEKF tuning are all "
          "built against the 1 Hz loop, so changing dt changes the controller as "
          "well as the sampling, and doing it honestly means editing "
          "scalar/control/tune.py — out of scope here.")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
