#!/usr/bin/env python3
"""Config B with the real sun sensor (7x VEML6031X00) instead of the ideal cube.

    PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
    $PY scripts/veml6031_baseline.py [--seeds 100,...,114] [--configs a|b]

Reuses scripts/estimator_levers.run_episode unchanged: the sun unit is
swapped by replacing estimator_levers._units() inside each worker with a
logging wrapper around either the old cube (SunSensorArray, reproduces B) or
scalar/sensors/veml6031.VEML6031Array. The wrapper consumes the shared rng
exactly like the cube, so gyro/magnetometer noise streams match B per seed.

Extra metrics, last orbit (same window as the harness), 1 Hz sun samples:
  clip_frac   sunlit samples with >= 1 sensor at 65535
  sun_err_med / sun_err_p95   angle(s_hat, true body sun) [deg], sunlit+valid
  sun_valid_frac              sunlit samples that gave a sun vector
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import estimator_levers as EL  # noqa: E402
from scalar.constants import DEG  # noqa: E402
from scalar.control.rl_tune import default_plant  # noqa: E402
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.sensors.sun_sensor import SunSensorArray  # noqa: E402
from scalar.sensors.veml6031 import COUNT_MAX, VEML6031Array  # noqa: E402

OUT_JSON = "results/veml6031_baseline.json"
SEEDS = tuple(range(100, 115))


def C(name, unit, albedo=0.0):
    return name, EL.C(name, unit=unit, albedo=albedo)[1]


CONFIGS = dict([
    C("B", "cube"),
    C("VEML 25ms", "veml25"),
    C("VEML 6.25ms", "veml6.25"),
    C("VEML 25ms clip-drop", "veml25-drop"),
    C("VEML 25ms clip-fill", "veml25-fill"),
    C("B a0.3", "cube", 0.3),
    C("VEML 25ms a0.3", "veml25", 0.3),
    C("VEML 6.25ms a0.3", "veml6.25", 0.3),
])


class Logged:
    """Delegating wrapper that records per-call truth and output."""

    def __init__(self, inner, is_veml):
        self.inner, self.is_veml = inner, is_veml
        self.lit, self.clip, self.err, self.valid = [], [], [], []
        self._s = None

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def measure(self, sun_dir_body, irradiance, lit, rng, *a, **kw):
        c = self.inner.measure(sun_dir_body, irradiance, lit, rng, *a, **kw)
        self._s = np.asarray(sun_dir_body, float)
        self.lit.append(bool(lit))
        self.clip.append(bool(self.is_veml and np.any(c >= COUNT_MAX)))
        return c

    def reconstruct(self, c, irradiance):
        s_hat, ok = self.inner.reconstruct(c, irradiance)
        self.valid.append(bool(ok))
        self.err.append(float(np.degrees(np.arccos(np.clip(s_hat @ self._s, -1, 1)))) if ok else np.nan)
        return s_hat, ok


def make_unit(unit, seed):
    base = SunSensorArray()                     # config B's ideal unit
    if unit == "cube":
        return Logged(base, False)
    it = 25 if unit.startswith("veml25") else 6.25
    mode = unit.split("-")[1] if "-" in unit else None
    aux = np.random.default_rng([seed, 6031])
    return Logged(VEML6031Array(integration_ms=it, base=base, clip_mode=mode, aux_rng=aux), True)


def _worker(job):
    name, cfg, seed = job
    t0 = time.time()
    unit = make_unit(cfg["unit"], seed)
    EL._units = lambda: {cfg["unit"]: unit}
    r = EL.run_episode(seed, cfg)
    plant = default_plant(mag_calibrated=True)
    period = int(CircularOrbit(inclination=plant.inclination_deg * DEG).period)
    lit = np.array(unit.lit[-period:]); clip = np.array(unit.clip[-period:])
    err = np.array(unit.err[-period:]); val = np.array(unit.valid[-period:])
    e = err[lit & val]
    r.update(clip_frac=float(clip[lit].mean()), sun_valid_frac=float(val[lit].mean()),
             sun_err_med=float(np.median(e)) if e.size else None,
             sun_err_p95=float(np.percentile(e, 95)) if e.size else None,
             n_sunlit=int(lit.sum()), wall_s=time.time() - t0)
    return name, seed, r


KEYS = ("point_mean", "point_p95", "know_med", "sun_err_med", "sun_err_p95", "clip_frac")


def summarize(out, names, seeds):
    ref = {"B": "B", "B a0.3": "B a0.3"}
    summ = {}
    for nm in names:
        ref_nm = "B a0.3" if nm.endswith("a0.3") else "B"
        S = out["runs"][nm]["seeds"]; R = out["runs"][ref_nm]["seeds"]
        d = {"ref": ref_nm}
        for k in KEYS + ("sun_valid_frac",):
            x = np.array([S[str(s)][k] for s in seeds], float)
            y = np.array([R[str(s)][k] for s in seeds], float)
            se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v)))  # noqa: E731
            d[k] = dict(mean=float(np.mean(x)), se=se(x),
                        delta=float(np.mean(x - y)), delta_se=se(x - y))
        summ[nm] = d
    del ref
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    os.chdir(ROOT)
    seeds = [int(s) for s in args.seeds.split(",")]
    names = [s for s in args.configs.split("|") if s] if args.configs else list(CONFIGS)
    jobs = [(nm, CONFIGS[nm], s) for nm in names for s in seeds]
    print(f"{len(jobs)} episodes", flush=True)
    workers = args.workers or max(1, min(len(jobs), mp.cpu_count() - 1))
    with mp.get_context("spawn").Pool(workers) as pool:
        done = list(pool.imap_unordered(_worker, jobs))
    out = {"meta": dict(params=EL.PARAMS_PATH, seeds=seeds, harness="scripts/estimator_levers.run_episode",
                        sensor="scalar/sensors/veml6031.py; gain x0.5, PD 1/4; 25 ms 0.8704 lx/ct, "
                               "6.25 ms 3.4816 lx/ct; 133,300 lx at 1 AU; B noise 2% of 133,300 lx; "
                               "unit gains; dark 4 ct; -Z pair averaged",
                        metrics="last of 3 orbits; sun metrics over sunlit 1 Hz samples; "
                                "summary mean, se = std/sqrt(n), delta = paired vs B (same albedo)"),
           "runs": {}}
    for nm, s, r in done:
        out["runs"].setdefault(nm, {"config": CONFIGS[nm], "seeds": {}})
        out["runs"][nm]["seeds"][str(s)] = r
    if "B" in names:
        out["summary"] = summarize(out, [n for n in names if ("a0.3" not in n) or "B a0.3" in names], seeds)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    for nm, d in out.get("summary", {}).items():
        print(f"{nm:<22}" + "  ".join(f"{k} {d[k]['mean']:.3f}±{d[k]['se']:.3f} (Δ{d[k]['delta']:+.3f}±{d[k]['delta_se']:.3f})"
                                       for k in KEYS))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
