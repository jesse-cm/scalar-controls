#!/usr/bin/env python3
"""Does an integral state on attitude error (fine-mode LQI) close the gyro-bias rate path?

    python3 scripts/lqi_fine.py [--Ti 200,500,1000,2000,5000,10000] [--workers N]

Baseline = results/lqr_tuned_3mode.json, ThreeModeController (ablation config B).
LQI = the same nine parameters, the same acquisition schedule, the fine slot
replaced by scalar.control.lqi.PeriodicLQI built from the same fine Bryson
weights plus ONE new weight: the integral time scale T_i (Q_z = 1/(theta_max_fine T_i)^2).

Stages
  0  baseline reproduction with rl_tune.evaluate, held-out 100-102 (expect 7.40/6.73/9.34)
  1  T_i grid on train seeds 0-1, scored by rl_tune.reward on the seed-mean metrics
  2  held-out 100-102 nominal + the +-10 % coil/inertia variants of verify_controller.py,
     baseline and LQI, eight THRESHOLDS
  3  isolated rate path: true attitude + estimated rate (mechanism_gyro_bias.run_logged,
     seed 100), baseline (expect 3.65) vs LQI; plus a constant-fake-rate control case
     and the pointing error split into mean offset (DC) and fluctuation (AC)

Both controllers go through rl_tune.evaluate unchanged; LQI is injected by
swapping rl_tune.build_controller inside the worker process.
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

from scalar.control import rl_tune  # noqa: E402
from scalar.control.lqi import LQIThreeModeController, build_lqi_fine  # noqa: E402

PARAMS_PATH = "results/lqr_tuned_3mode.json"
OUT_PATH = "results/lqi_fine.json"
TRAIN = (0, 1)
HELD_OUT = (100, 101, 102)
GATE_DEG = 10.0
Z_CLAMP_FACTOR = 1.0
VARIANTS = {"nominal": dict(),
            "coils-10_inertia+10": dict(scale_M=0.9, scale_inertia=1.1),
            "coils+10_inertia-10": dict(scale_M=1.1, scale_inertia=0.9)}
REPORT = ("fine_point_mean", "fine_point_p95", "fine_knowledge_med", "fine_power_w", "fine_sat_frac")


def make_lqi_builder(T_i):
    def build(params, plant, acts, orbit, jd0):
        th, wm, rho, enter, exit_, k_damp, fth, fwm, frho = params
        lqr = rl_tune._schedule(plant, acts, orbit, jd0, th, wm, rho)
        fine = build_lqi_fine(plant.inertia, acts, orbit, jd0, fth, fwm, frho, T_i,
                              gate_deg=GATE_DEG, z_clamp_factor=Z_CLAMP_FACTOR)
        return LQIThreeModeController(lqr, k_damp=k_damp, enter_deg_s=enter,
                                      exit_deg_s=max(exit_, enter + 0.2), fine=fine)
    return build


def _eval_job(job):
    """job = (key, T_i or None, variant, seed)."""
    key, T_i, variant, seed = job
    os.chdir(ROOT)
    params, _ = rl_tune.load_params(PARAMS_PATH)
    plant = rl_tune.default_plant(mag_calibrated=True, **VARIANTS[variant])
    orig = rl_tune.build_controller
    if T_i is not None:
        rl_tune.build_controller = make_lqi_builder(T_i)
    try:
        m = rl_tune.evaluate(params, plant, seed)
    finally:
        rl_tune.build_controller = orig
    return key, m


class _FakeRate:
    """Adds a constant body-frame fake rate to what the controller sees."""
    def __init__(self, inner, c):
        self.inner, self.c = inner, c

    @property
    def q_target(self):
        return self.inner.q_target

    def dipole(self, t, q, w, B):
        return self.inner.dipole(t, q, np.asarray(w) + self.c, B)


CONST_DB_DEG_S = 0.009 * np.array([0.6, -0.48, 0.64])   # |db| = the median bias error, seed 100


def _true_att_job(job):
    """Isolated rate path: controller sees q_true and g - b_hat. Seed 100.

    feed "const_db": perfect gyro (bias RW 0, sigma_u 1e-9), true attitude, and a
    CONSTANT fake rate of the median |db| -- the case an integrator must remove.
    Also returns the last-orbit pointing error split into its mean vector (DC)
    and the mean deviation from it (AC), and the 1/e correlation time of db.
    """
    key, T_i, feed = job
    os.chdir(ROOT)
    import mechanism_gyro_bias as mg
    from scalar.attitude import quat_from_rotvec
    from scalar.constants import DEG
    from scalar.control.tune import actuators_from_matrix
    from scalar.dynamics import RigidBody
    from scalar.environment.orbit import CircularOrbit
    from scalar.sensors.magnetometer import Magnetometer
    from scalar.timeutil import julian_date
    params, _ = rl_tune.load_params(PARAMS_PATH)
    plant = rl_tune.default_plant(mag_calibrated=True)
    body = RigidBody(plant.inertia)
    orbit = CircularOrbit(inclination=plant.inclination_deg * DEG)
    jd0 = julian_date(plant.epoch)
    acts = actuators_from_matrix(plant.M, plant.i_max)
    mag = Magnetometer(bias=plant.mag_bias, soft_iron=plant.mag_soft_iron)
    cal, _ = rl_tune.calibration_pass(plant, mag, orbit, jd0, 100)
    builder = rl_tune.build_controller if T_i is None else make_lqi_builder(T_i)
    ctrl = builder(params, plant, acts, orbit, jd0)
    ctrl.mode = 3
    q0 = quat_from_rotvec(np.deg2rad(20) * np.array([0.6, 0.8, 0.0]))
    w0 = np.deg2rad([0.1, -0.05, 0.08])
    if feed == "const_db":
        r = mg.run_logged(body, acts, orbit, jd0, _FakeRate(ctrl, np.deg2rad(CONST_DB_DEG_S)), q0, w0, 100,
                          plant.m_res, mag, cal, bias_rw=0.0, sigma_u=1e-9, feed="true_att")
    else:
        r = mg.run_logged(body, acts, orbit, jd0, ctrl, q0, w0, 100, plant.m_res, mag, cal, feed=feed)
    last = slice(-int(orbit.period), None)
    pe = r["pointing_err_deg"][last]
    pv = np.degrees(r["pe_vec"][last])
    db = np.degrees(r["db"][last])
    tau = []
    for i in range(3):
        x = db[:, i] - db[:, i].mean()
        c = np.correlate(x, x, "full")[len(x) - 1:]
        tau.append(int(np.argmax(c / c[0] < np.exp(-1))) if c[0] > 0 else None)
    out = dict(point_mean=float(np.mean(pe)), point_p95=float(np.percentile(pe, 95)),
               know_med=float(np.median(r["knowledge_err_deg"][last])),
               sat_frac=float(r["sat_frac"]),
               err_dc_deg=float(np.linalg.norm(pv.mean(0))),
               err_ac_deg=float(np.mean(np.linalg.norm(pv - pv.mean(0), axis=1))),
               db_corr_time_s=tau)
    if T_i is not None:
        out["z_final_deg_s"] = np.degrees(ctrl.fine.z).tolist()
    return key, out


def _agg(ms):
    agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    for k in ("acq_diverged", "fine_diverged"):
        agg[k] = float(max(m[k] for m in ms))
    return agg


def _row(m):
    return "  ".join(f"{k.replace('fine_', '')}={m[k]:.3f}" for k in REPORT)


def _crit(m):
    return {n: dict(value=v, threshold=t, passed=p) for n, v, t, p in rl_tune.criteria(m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ti", default="200,500,1000,2000,5000,10000")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    os.chdir(ROOT)
    grid = [float(x) for x in args.Ti.split(",")]
    workers = args.workers or max(1, mp.cpu_count() - 1)
    out = {"setup": {"params_path": PARAMS_PATH, "train_seeds": list(TRAIN), "held_out_seeds": list(HELD_OUT),
                     "gate_deg": GATE_DEG, "z_clamp": "z_clamp_factor * theta_max_fine * T_i",
                     "z_clamp_factor": Z_CLAMP_FACTOR, "Ti_grid_s": grid,
                     "integral_weight": "Q_z = I3 / (theta_max_fine * T_i)^2",
                     "thresholds": rl_tune.THRESHOLDS}}
    pool = mp.get_context("spawn").Pool(workers)
    t0 = time.time()

    # --- stage 0 + 1 together: baseline held-out, baseline train, T_i grid train ---
    jobs = [(("base", "nominal", s), None, "nominal", s) for s in HELD_OUT + TRAIN]
    jobs += [(("lqi", T, "nominal", s), T, "nominal", s) for T in grid for s in TRAIN]
    res = dict(pool.map(_eval_job, jobs))
    print(f"stage 0/1 done ({time.time() - t0:.0f} s)")
    print("\nBASELINE reproduction (expect fine_point_mean 7.40 / 6.73 / 9.34)")
    for s in HELD_OUT:
        print(f"  seed {s}: {_row(res[('base', 'nominal', s)])}")
    out["baseline_reproduction"] = {str(s): res[("base", "nominal", s)]["fine_point_mean"] for s in HELD_OUT}

    print("\nT_i GRID, train seeds 0-1 (seed-mean metrics; reward = rl_tune.reward)")
    base_tr = _agg([res[("base", "nominal", s)] for s in TRAIN])
    print(f"  baseline       reward {rl_tune.reward(base_tr):+.3f}  {_row(base_tr)}  "
          f"settle={base_tr['acq_settle_h']:.2f}h")
    grid_rows = {"baseline": dict(reward=rl_tune.reward(base_tr), metrics=base_tr)}
    best = None
    for T in grid:
        a = _agg([res[("lqi", T, "nominal", s)] for s in TRAIN])
        rw = rl_tune.reward(a)
        grid_rows[f"{T:g}"] = dict(reward=rw, metrics=a,
                                   per_seed={str(s): res[("lqi", T, "nominal", s)] for s in TRAIN})
        print(f"  T_i={T:7g} s  reward {rw:+.3f}  {_row(a)}  settle={a['acq_settle_h']:.2f}h "
              f"div={a['acq_diverged']:.0f}/{a['fine_diverged']:.0f}")
        if best is None or rw > best[0]:
            best = (rw, T)
    T_best = best[1]
    print(f"  chosen T_i = {T_best:g} s")
    out["train_grid"] = grid_rows
    out["chosen_Ti_s"] = T_best

    # --- stage 2 + 3: held-out + variants, and the isolated rate path ---
    jobs = [(("lqi", T_best, "nominal", s), T_best, "nominal", s) for s in HELD_OUT]
    for v in list(VARIANTS)[1:]:
        jobs += [(("base", v, 100), None, v, 100), (("lqi", T_best, v, 100), T_best, v, 100)]
    ta_jobs = [(("base", "true_att"), None, "true_att"), (("lqi", "true_att"), T_best, "true_att"),
               (("base", "estimated"), None, "estimated"), (("lqi", "estimated"), T_best, "estimated"),
               (("base", "const_db"), None, "const_db")]
    ta_jobs += [((f"lqi_Ti{T:g}", f), T, f) for T in grid if T >= 1000 for f in ("true_att", "const_db")
                if not (T == T_best and f == "true_att")]
    r_ta = pool.map_async(_true_att_job, ta_jobs)
    res.update(dict(pool.map(_eval_job, jobs)))
    ta = dict(r_ta.get())
    pool.close()
    print(f"\nstage 2/3 done ({time.time() - t0:.0f} s)")

    out["held_out"] = {}
    print("\nHELD-OUT + VARIANTS  (baseline | LQI)")
    cases = [("nominal", s) for s in HELD_OUT] + [(v, 100) for v in list(VARIANTS)[1:]]
    for v, s in cases:
        b, l = res[("base", v, s)], res[("lqi", T_best, v, s)]
        cb, cl = _crit(b), _crit(l)
        out["held_out"][f"{v}/{s}"] = {"baseline": {"metrics": b, "criteria": cb, "all_pass": all(c["passed"] for c in cb.values())},
                                       "lqi": {"metrics": l, "criteria": cl, "all_pass": all(c["passed"] for c in cl.values())}}
        print(f"  {v:22s} seed {s}")
        print(f"    base: {_row(b)}")
        print(f"    lqi : {_row(l)}")
        print("    " + "  ".join(f"{n}={cb[n]['value']:.2f}{'' if cb[n]['passed'] else '!'}"
                                 f"->{cl[n]['value']:.2f}{'' if cl[n]['passed'] else '!'}" for n in cb))

    out["isolated_rate_path_seed100"] = {f"{k[0]}/{k[1]}": v for k, v in ta.items()}
    print("\nISOLATED RATE PATH, seed 100 (run_logged; expect baseline true_att 3.65, estimated 7.40)")
    for k, v in ta.items():
        print(f"  {k[0]:12s} {k[1]:9s}: pointing mean {v['point_mean']:.3f}  p95 {v['point_p95']:.3f}  "
              f"DC {v['err_dc_deg']:.2f}  AC {v['err_ac_deg']:.2f}  know_med {v['know_med']:.3f}  "
              f"sat {v['sat_frac']:.4f}  db_tau {v['db_corr_time_s']}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_PATH} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
