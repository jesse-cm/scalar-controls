#!/usr/bin/env python3
"""Estimator-side levers on the gyro-bias "attitude-estimate path".

    PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3   # has torch
    $PY scripts/estimator_levers.py [--configs a,b,...] [--seeds 100,101,102]

Baseline = config B of scripts/ablation_pointing.py (tuned 3-mode controller,
results/lqr_tuned_3mode.json, calibrated magnetometer, residual dipole,
fine pointing from 20 deg, 3 orbits, last-orbit metrics). The controller and
its 1 Hz update are never changed. Levers, one at a time:

  L1  MLP-B sun front end (scalar/learning/sun_mlp.py), flight-like sun unit,
      albedo 0.3. Trained on dataset seed 0, unit seed 1 (disjoint from the
      episode seeds 100-102).
  L2  MEKF sigma_u (filter bias-RW) x 0.3 / 1 / 3 / 10, plant stays 1e-5.
  L3  Sensor rate 2 / 4 Hz (gyro, magnetometer, sun each selectable),
      controller still 1 Hz, without and with a measure-then-torque duty cycle.

DUTY CYCLE (not modelled in scalar/control/tune.py; added here)
  Within each 1 s control period the coils are OFF for the first (1-d) s and
  ON for the last d s ("measure, then torque"). The magnetometer may only be
  read at sample instants inside the OFF window (t + j/f < t + 1-d). The
  control dipole is computed at t and applied only in the ON window, so the
  average control torque is d x the commanded one ("authority x duty"). The
  residual dipole acts all the time. Gyro and sun are not affected by coils.

LOOP (copy of tune.simulate_closed_loop_estimated with a sample grid inside
each control second; at 1 Hz and no duty it reproduces that function exactly)
  * plant: RK4 on omega, exp-map on q, piecewise-constant torque per segment
    (segments split at sample instants and at the coil switch). Torque per
    segment uses B_body at the control instant t, as the baseline does.
  * MEKF propagates at every grid instant with the latest gyro sample (same
    "newest sample over the previous interval" convention as the baseline),
    updates on mag / sun when that sensor is sampled.
  * per-sample white noise: gyro scales with 1/sqrt(dt) (density), mag and sun
    keep their per-sample sigma (each read is an independent sample).
  * metrics at the 1 Hz control instants, last orbit:
      point = angle(q_true(t+1), target)                      (baseline def)
      know  = angle(mekf.q at t, q_true(t+1))                 (baseline def)
      know_aligned = angle(mekf.q at t, q_true(t))
      dbias = |b_hat(t) - b_true of the latest gyro sample|   [deg/s]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import (attitude_error_angle, propagate,  # noqa: E402
                             quat_from_rotvec, to_body)
from scalar.constants import DEG  # noqa: E402
from scalar.control.rl_tune import (build_controller, calibration_pass,  # noqa: E402
                                    default_plant, load_params)
from scalar.control.tune import actuators_from_matrix  # noqa: E402
from scalar.dynamics import RigidBody  # noqa: E402
from scalar.environment.magfield import magnetic_field_eci  # noqa: E402
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.environment.sun import (eclipse_factor, solar_irradiance,  # noqa: E402
                                    sun_direction_eci)
from scalar.estimator.mekf import MEKF  # noqa: E402
from scalar.sensors.gyro import Gyro  # noqa: E402
from scalar.sensors.magnetometer import Magnetometer  # noqa: E402
from scalar.sensors.sun_sensor import SunSensorArray  # noqa: E402
from scalar.timeutil import julian_date  # noqa: E402

PARAMS_PATH = "results/lqr_tuned_3mode.json"
OUT_JSON = "results/estimator_levers.json"
SEEDS = (100, 101, 102)
SIGMA_U_TRUE = 1.0e-5
DUTY = 0.6            # coil ON fraction of each control second (doc range 0.6-1.0)
MLP_DATA_SEED = 0     # train_sun_mlp dataset/torch seed
MLP_UNIT_SEED = 1     # flight_like_sensor unit (same unit as section 5)


def C(name, **kw):
    base = dict(front="analytic", unit="ideal", albedo=0.0, su_scale=1.0,
                gyro_hz=1, mag_hz=1, sun_hz=1, duty=None)
    base.update(kw)
    return name, base


CONFIGS = dict([
    C("B"),
    # L1 sun front end
    C("L1-ref analytic flight-unit a0.3", unit="flight", albedo=0.3),
    C("L1 MLP-B flight-unit a0.3", front="mlp", unit="flight", albedo=0.3),
    C("L1 MLP-B ideal-unit a0", front="mlp", unit="ideal", albedo=0.0),
    # L2 filter bias RW
    C("L2 su x0.3", su_scale=0.3),
    C("L2 su x3", su_scale=3.0),
    C("L2 su x10", su_scale=10.0),
    # L3 sensor rate, no duty cycle
    C("L3 all 2Hz", gyro_hz=2, mag_hz=2, sun_hz=2),
    C("L3 all 4Hz", gyro_hz=4, mag_hz=4, sun_hz=4),
    C("L3 gyro 4Hz", gyro_hz=4),
    C("L3 mag 4Hz", mag_hz=4),
    C("L3 sun 4Hz", sun_hz=4),
    # L3 with measure-then-torque duty cycle
    C("D B duty", duty=DUTY),
    C("D all 2Hz", gyro_hz=2, mag_hz=2, sun_hz=2, duty=DUTY),
    C("D all 4Hz", gyro_hz=4, mag_hz=4, sun_hz=4, duty=DUTY),
    C("D gyro 4Hz", gyro_hz=4, duty=DUTY),
    C("D mag 4Hz", mag_hz=4, duty=DUTY),
    C("D sun 4Hz", sun_hz=4, duty=DUTY),
])

# The MLP is trained once in the parent and shipped to workers.
_MLP = {}


def _units():
    from scalar.learning.sun_mlp import flight_like_sensor
    return {"ideal": SunSensorArray(),
            "flight": flight_like_sensor(np.random.default_rng(MLP_UNIT_SEED))}


def train_mlps(needed_units):
    from scalar.learning.sun_mlp import train_sun_mlp
    units = _units()
    out = {}
    for u in needed_units:
        t0 = time.time()
        mlp, err = train_sun_mlp(units[u], extra=True, epochs=15, seed=MLP_DATA_SEED)
        out[u] = mlp
        print(f"  MLP-B for {u} unit: val median {np.median(err):.2f} deg, "
              f"p95 {np.percentile(err, 95):.2f} deg ({time.time() - t0:.0f} s)", flush=True)
    return out


def _rk4(body, q, w, tau, h):
    k1 = body.omega_dot(w, tau); k2 = body.omega_dot(w + 0.5*h*k1, tau)
    k3 = body.omega_dot(w + 0.5*h*k2, tau); k4 = body.omega_dot(w + h*k3, tau)
    w_next = w + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return propagate(q, 0.5*(w + w_next), h), w_next


def run_episode(seed, cfg, mlp=None, n_orbits=3):
    p, _ = load_params(PARAMS_PATH)
    plant = default_plant(mag_calibrated=True)
    orbit = CircularOrbit(inclination=plant.inclination_deg * DEG)
    jd0 = julian_date(plant.epoch)
    acts = actuators_from_matrix(plant.M, plant.i_max)
    mag = Magnetometer(bias=plant.mag_bias, soft_iron=plant.mag_soft_iron)
    cal, _ = calibration_pass(plant, mag, orbit, jd0, seed)
    ctrl = build_controller(p, plant, acts, orbit, jd0)
    ctrl.mode = 3
    body = RigidBody(plant.inertia)
    sun = _units()[cfg["unit"]]
    if cfg["front"] == "mlp":
        def front(c, E, nb, ds):
            return mlp.reconstruct(c, E, sun, np.concatenate([nb, [ds]]))
    else:
        front = None

    rng = np.random.default_rng(seed)
    gyro = Gyro(); gyro.reset()
    mekf = MEKF(n_iter=3, sigma_u=SIGMA_U_TRUE * cfg["su_scale"])
    m_res = plant.m_res
    albedo = cfg["albedo"]
    duty = cfg["duty"]

    f = max(cfg["gyro_hz"], cfg["mag_hz"], cfg["sun_hz"])
    for k_ in ("gyro_hz", "mag_hz", "sun_hz"):
        assert f % cfg[k_] == 0
    dts = 1.0 / f
    g_every, m_every, s_every = (f // cfg["gyro_hz"], f // cfg["mag_hz"], f // cfg["sun_hz"])
    off_end = 1.0 - duty if duty is not None else 1.0   # coil-off window [0, off_end)
    # piecewise segments inside one control second: sample instants + coil switch
    cuts = sorted(set([j * dts for j in range(f)] + [1.0] +
                      ([off_end] if duty is not None and 0 < off_end < 1 else [])))

    q = quat_from_rotvec(np.deg2rad(20) * np.array([0.6, 0.8, 0.0]))
    w = np.deg2rad([0.1, -0.05, 0.08])
    n = int(n_orbits * orbit.period)
    pe = np.empty(n); ke = np.empty(n); ka = np.empty(n); db = np.empty(n)
    n_mag_reads = 0
    started = False
    g = None; b_true = None; Bm_last = None
    for k in range(n):
        t = float(k)
        q_ctrl_inst = None
        # ---- sample grid inside this control second, interleaved with plant ----
        seg_i = 0
        tau_on = tau_off = None
        for j in range(f):
            ts = t + j * dts
            jd = jd0 + ts / 86400.0
            r = orbit.position(ts)
            B_eci = magnetic_field_eci(r, jd)
            s_eci = sun_direction_eci(jd)
            E = solar_irradiance(jd)
            lit = eclipse_factor(r, s_eci) > 0
            r_hat = r / np.linalg.norm(r)
            new_g = (j % g_every == 0)
            if new_g:
                b_true = gyro.bias.copy()
                g = gyro.measure(w, 1.0 / cfg["gyro_hz"], rng)
            mag_ok = (j % m_every == 0) and (j * dts < off_end - 1e-9)
            if mag_ok:
                Bm = mag.measure(B_eci, q, rng)
                Bm = cal.apply(Bm)
                Bm_last = Bm
                n_mag_reads += 1
            sun_now = (j % s_every == 0)
            valid = False
            if sun_now:
                dayside = max(0.0, float(r_hat @ s_eci))
                currents = sun.measure(to_body(q, s_eci), E, lit, rng, to_body(q, -r_hat),
                                       albedo * E * dayside)
                if front is None:
                    s_hat, valid = sun.reconstruct(currents, E)
                else:
                    s_hat, valid = front(currents, E, to_body(mekf.q, -r_hat), dayside)
            # ---- estimator ----
            if k > 0 or j > 0:
                mekf.propagate(g, dts)
            if not started and valid:
                mekf.init_triad(s_eci, s_hat, B_eci, Bm_last); started = True
            if mag_ok:
                mekf.update_mag(B_eci, Bm_last)
            if valid:
                mekf.update_sun(s_eci, s_hat)
            # ---- controller at the 1 Hz instant ----
            if j == 0:
                q_ctrl_inst = q.copy()
                m = ctrl.dipole(t, mekf.q, g - mekf.b, Bm_last)
                B_body_t = to_body(q, B_eci)
                tau_on = np.cross(m + m_res, B_body_t)
                tau_off = np.cross(m_res, B_body_t)
                q_est_t = mekf.q.copy()
                db[k] = np.degrees(np.linalg.norm(mekf.b - b_true))
            # ---- plant up to the next sample instant ----
            t_next = (j + 1) * dts
            while seg_i < len(cuts) - 1 and cuts[seg_i] < t_next - 1e-12:
                a, bnd = cuts[seg_i], cuts[seg_i + 1]
                tau = tau_on if (duty is None or a >= off_end - 1e-12) else tau_off
                q, w = _rk4(body, q, w, tau, bnd - a)
                seg_i += 1
        pe[k] = np.degrees(attitude_error_angle(q, ctrl.q_target))
        ke[k] = np.degrees(attitude_error_angle(q_est_t, q))
        ka[k] = np.degrees(attitude_error_angle(q_est_t, q_ctrl_inst))
    last = slice(-int(orbit.period), None)
    return dict(point_mean=float(np.mean(pe[last])), point_p95=float(np.percentile(pe[last], 95)),
                know_med=float(np.median(ke[last])), know_aligned_med=float(np.median(ka[last])),
                know_p95=float(np.percentile(ke[last], 95)),
                dbias_med_dps=float(np.median(db[last])), dbias_p95_dps=float(np.percentile(db[last], 95)),
                mag_reads_per_s=n_mag_reads / n)


def _worker(job):
    name, cfg, seed, mlp = job
    t0 = time.time()
    r = run_episode(seed, cfg, mlp)
    r["wall_s"] = time.time() - t0
    return name, seed, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--combo", action="append", default=[],
                    help='extra config as JSON: {"name":..., <cfg keys>}')
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    seeds = [int(s) for s in args.seeds.split(",")]
    cfgs = dict(CONFIGS)
    for js in args.combo:
        d = json.loads(js); nm = d.pop("name")
        cfgs[nm] = C(nm, **d)[1]
    names = [s for s in args.configs.split("|") if s] if args.configs else list(CONFIGS)
    names += [json.loads(js)["name"] for js in args.combo]
    names = list(dict.fromkeys(names))

    units = sorted({cfgs[nm]["unit"] for nm in names if cfgs[nm]["front"] == "mlp"})
    mlps = train_mlps(units) if units else {}
    jobs = [(nm, cfgs[nm], s, mlps.get(cfgs[nm]["unit"]) if cfgs[nm]["front"] == "mlp" else None)
            for nm in names for s in seeds]
    # longest first
    jobs.sort(key=lambda j: -max(j[1]["gyro_hz"], j[1]["mag_hz"], j[1]["sun_hz"]))
    print(f"{len(jobs)} episodes", flush=True)
    workers = args.workers or max(1, min(len(jobs), mp.cpu_count() - 1))
    with mp.get_context("spawn").Pool(workers) as pool:
        done = list(pool.imap_unordered(_worker, jobs))

    try:
        with open(OUT_JSON) as fh:
            out = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        out = {"meta": {}, "runs": {}}
    for nm, s, r in done:
        out["runs"].setdefault(nm, {"config": cfgs[nm], "seeds": {}})
        out["runs"][nm]["config"] = cfgs[nm]
        out["runs"][nm]["seeds"][str(s)] = r
    # the seeds the file holds (it keeps runs from earlier invocations), not the default --seeds
    held = {nm: sorted(int(sd) for sd in run["seeds"]) for nm, run in out["runs"].items()}
    all_seeds = sorted(set().union(*held.values())) if held else []
    out["meta"] = dict(params=PARAMS_PATH, seeds=all_seeds, sigma_u_true=SIGMA_U_TRUE,
                       duty=DUTY, mlp_data_seed=MLP_DATA_SEED, mlp_unit_seed=MLP_UNIT_SEED,
                       metrics="last of 3 orbits at the 1 Hz control instants; see script docstring",
                       duty_model="coils OFF first (1-d) s, ON last d s of each control second; "
                                  "mag read only in OFF window; not modelled in scalar/ (added here)")
    partial = {nm: sd for nm, sd in held.items() if sd != all_seeds}
    if partial:
        out["meta"]["seeds_partial_runs"] = partial     # runs that lack some of meta.seeds
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"\n{'config':<34}{'seed':>5}{'point':>8}{'p95':>8}{'know':>7}{'dbias':>8}{'mag/s':>6}")
    for nm in names:
        for s in seeds:
            r = out["runs"][nm]["seeds"][str(s)]
            print(f"{nm:<34}{s:>5}{r['point_mean']:>8.2f}{r['point_p95']:>8.2f}{r['know_med']:>7.2f}"
                  f"{r['dbias_med_dps']:>8.4f}{r['mag_reads_per_s']:>6.1f}")
        pm = [out["runs"][nm]["seeds"][str(s)]["point_mean"] for s in seeds]
        print(f"{'  mean / spread':<39}{np.mean(pm):>8.2f} +-{np.ptp(pm):.2f}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
