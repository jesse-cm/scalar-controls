"""Reward-driven tuning of the LQR weights and mode thresholds (cross-entropy method).

WHAT IS TUNED (6 parameters, log-scaled where noted)
    theta_max [deg]   Bryson attitude scale           log  2 .. 45
    omega_max [deg/s] Bryson rate scale               log  0.05 .. 2
    rho               input-cost multiplier           log  0.01 .. 10
    enter   [deg/s]   rate below which pointing mode engages     0.1 .. 1.0
    exit    [deg/s]   rate above which damping resumes           1.0 .. 3.0
    k_damp            rate-damping gain m = k (w x B)   log  1e3 .. 1e5
The 256-bin schedule and the plant are fixed; the periodic Riccati schedule is
recomputed for every candidate.

WHY CEM AND NOT A POLICY NETWORK
Six numbers, episodes that cost tens of seconds, and a need to read the
answer afterwards. CEM samples a population, scores each by the same reward,
refits a Gaussian to the elites, repeats. Nothing is hidden in a network.

THE CRITERIA ARE THE CONTRACT
``criteria()`` turns one evaluation into named (value, threshold, pass)
rows. The reward is a smooth function of the same rows so the search has a
gradient, but verification never looks at the reward: it looks at the rows on
seeds the tuner did not train on (``scripts/verify_controller.py``).
"""
from __future__ import annotations

import datetime as dt
import json
import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np

from ..attitude import IDENTITY_QUAT, quat_from_rotvec
from ..constants import DEG
from ..dynamics import RigidBody
from ..environment.orbit import CircularOrbit
from ..sensors.magnetometer import Magnetometer
from ..sysid.magnetorquer_id import s3_coil_matrix
from ..timeutil import julian_date
from .periodic_lqr import (PeriodicLQR, ThreeModeController, TwoModeController, bryson_weights,
                           periodic_lqr_schedule, quantize_schedule)
from .tune import actuators_from_matrix, simulate_closed_loop_estimated

__all__ = ["PARAM_NAMES", "BOUNDS", "LOG_SCALE", "clip_params", "evaluate",
           "criteria", "reward", "cem_tune", "Plant", "default_plant"]

PARAM_NAMES = ["theta_max", "omega_max", "rho", "enter", "exit", "k_damp",
               "fine_theta_max", "fine_omega_max", "fine_rho"]
BOUNDS = np.array([[2.0, 45.0], [0.05, 2.0], [0.01, 10.0], [0.1, 1.0], [1.0, 3.0], [1e3, 1e5],
                   [1.0, 45.0], [0.05, 5.0], [0.01, 10.0]])
LOG_SCALE = np.array([True, True, True, False, False, True, True, True, True])

# --- verifiable criteria (thresholds written before any tuning run) ---------
THRESHOLDS = {
    "acq_settle_h":      6.0,    # pointing < 10 deg within 6 h from 120 deg / 1 deg/s
    "acq_diverged":      0.0,    # last-orbit pointing never > 90 deg  (0 = pass)
    "fine_point_mean":   5.0,    # deg, last orbit, MEKF in the loop
    "fine_point_p95":   15.0,    # deg
    "fine_max_rate":     0.5,    # deg/s after settling
    "fine_sat_frac":     0.2,    # fraction of steps with any channel saturated
    "fine_power_w":      0.3,    # mean coil power, W
    "fine_diverged":     0.0,
}


@dataclass
class Plant:
    """Everything the tuner is NOT allowed to change."""
    inertia: np.ndarray
    M: np.ndarray
    i_max: np.ndarray
    inclination_deg: float = 98.61
    epoch: dt.datetime = field(default_factory=lambda: dt.datetime(2026, 9, 11, 12))
    m_res: np.ndarray = field(default_factory=lambda: np.array([2e-3, -1e-3, 0.5e-3]))
    mag_bias: np.ndarray = field(default_factory=lambda: np.array([1.5e-6, -1.0e-6, 0.8e-6]))
    mag_soft_iron: np.ndarray = field(default_factory=lambda: np.eye(3) + np.array(
        [[0.05, 0.02, 0.0], [0.0, -0.03, 0.02], [0.01, 0.0, 0.04]]))
    mag_calibrated: bool = False  # run the ellipsoid calibration pass before pointing
    scale_M: float = 1.0          # robustness variants scale the true plant, not the design
    scale_inertia: float = 1.0


def default_plant(**kw):
    M, i_max = s3_coil_matrix()
    return Plant(RigidBody().inertia, M, i_max, **kw)


def clip_params(p):
    return np.clip(np.asarray(p, float), BOUNDS[:, 0], BOUNDS[:, 1])


def _to_z(p):
    """Parameters -> unconstrained search coordinates in [0,1]."""
    p = clip_params(p)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    z = np.where(LOG_SCALE, (np.log(p) - np.log(lo)) / (np.log(hi) - np.log(lo)), (p - lo) / (hi - lo))
    return z


def _from_z(z):
    z = np.clip(z, 0.0, 1.0)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    return np.where(LOG_SCALE, np.exp(np.log(lo) + z * (np.log(hi) - np.log(lo))), lo + z * (hi - lo))


def calibration_pass(plant, mag, orbit, jd0, seed, dt=1.0):
    """One orbit of free tumble with the real magnetometer -> ellipsoid fit.

    This is the flight sequence: detumble, one orbit of raw magnetometer
    logging against |B_model| along the orbit, fit, and only then allow
    pointing mode. The fit is attitude-independent, so it needs no estimate.
    Returns (MagCalibration, direction error deg after calibration, median).
    """
    from ..calibration.mag_onorbit import fit_magnetometer_ellipsoid
    from ..dynamics import simulate_free_tumble
    from ..environment.magfield import magnetic_field_eci
    from ..attitude import to_body
    rng = np.random.default_rng(seed + 7)
    body = RigidBody(plant.inertia * plant.scale_inertia)
    n = int(orbit.period / dt)
    tl = simulate_free_tumble(body, IDENTITY_QUAT, np.deg2rad([1.0, -0.6, 0.8]), dt, n)
    raw = np.empty((n + 1, 3)); F = np.empty(n + 1); true = np.empty((n + 1, 3))
    for k, t in enumerate(tl.t):
        B_eci = magnetic_field_eci(orbit.position(t), jd0 + t / 86400.0)
        raw[k] = mag.measure(B_eci, tl.q[k], rng)
        F[k] = np.linalg.norm(B_eci)
        true[k] = to_body(tl.q[k], B_eci)
    cal = fit_magnetometer_ellipsoid(raw, F)
    c = cal.apply(raw)
    err = np.degrees(np.arccos(np.clip(np.sum(c * true, 1) / np.linalg.norm(c, axis=1) / np.linalg.norm(true, axis=1), -1, 1)))
    return cal, float(np.median(err))


def _schedule(plant, acts, orbit, jd0, th, wm, rho):
    Q, R = bryson_weights(th * DEG, wm * DEG, acts.m_max_vector, rho)
    K, _ = periodic_lqr_schedule(plant.inertia, orbit, jd0, IDENTITY_QUAT, 1.0, Q, R, n_orbits_iter=3)
    return PeriodicLQR(quantize_schedule(K, 256), orbit.period, acts, IDENTITY_QUAT)


def build_controller(params, plant, acts, orbit, jd0):
    th, wm, rho, enter, exit_, k_damp, fth, fwm, frho = params
    lqr = _schedule(plant, acts, orbit, jd0, th, wm, rho)
    fine = _schedule(plant, acts, orbit, jd0, fth, fwm, frho)
    return ThreeModeController(lqr, k_damp=k_damp, enter_deg_s=enter, exit_deg_s=max(exit_, enter + 0.2),
                               fine=fine)


def evaluate(params, plant, seed, n_orbits_acq=4, n_orbits_fine=3):
    """Run the acquisition and fine-pointing episodes; return raw metrics."""
    params = clip_params(params)
    body = RigidBody(plant.inertia * plant.scale_inertia)
    acts_design = actuators_from_matrix(plant.M, plant.i_max)
    acts_true = actuators_from_matrix(plant.M * plant.scale_M, plant.i_max)
    orbit = CircularOrbit(inclination=plant.inclination_deg * DEG)
    jd0 = julian_date(plant.epoch)
    P = orbit.period
    mag = Magnetometer(bias=plant.mag_bias, soft_iron=plant.mag_soft_iron)
    out = {}
    cal = None
    if plant.mag_calibrated:
        cal, out["cal_dir_err_med"] = calibration_pass(plant, mag, orbit, jd0, seed)

    ctrl = build_controller(params, plant, acts_design, orbit, jd0)
    ctrl.lqr.actuators = acts_true; ctrl.fine.actuators = acts_true
    r = simulate_closed_loop_estimated(body, acts_true, orbit, jd0, q0=np.array([0.5, 0.5, 0.5, 0.5]),
                                       omega0=np.deg2rad([1.0, -0.6, 0.8]), n_orbits=n_orbits_acq, seed=seed,
                                       controller=ctrl, m_res=plant.m_res, magnetometer=mag,
                                       mag_calibration=cal)
    last = slice(-int(P), None)
    out["acq_settle_h"] = r.settle_time() / 3600.0
    out["acq_diverged"] = float(np.max(r.pointing_err_deg[last]) > 90.0)
    out["acq_point_mean"] = float(np.mean(r.pointing_err_deg[last]))

    ctrl = build_controller(params, plant, acts_design, orbit, jd0)
    ctrl.lqr.actuators = acts_true; ctrl.fine.actuators = acts_true
    ctrl.mode = 3
    q0 = quat_from_rotvec(np.deg2rad(20) * np.array([0.6, 0.8, 0.0]))
    r = simulate_closed_loop_estimated(body, acts_true, orbit, jd0, q0=q0, omega0=np.deg2rad([0.1, -0.05, 0.08]),
                                       n_orbits=n_orbits_fine, seed=seed, controller=ctrl,
                                       m_res=plant.m_res, magnetometer=mag, mag_calibration=cal)
    pe = r.pointing_err_deg[last]
    duty = np.mean(np.abs(r.dipole[last]) / acts_true.m_max_vector, axis=0)
    out["fine_point_mean"] = float(np.mean(pe))
    out["fine_point_p95"] = float(np.percentile(pe, 95))
    out["fine_max_rate"] = float(np.max(r.rate_deg_s[last]))
    out["fine_sat_frac"] = float(np.mean(np.any(np.abs(r.dipole[last]) >= 0.999 * acts_true.m_max_vector, axis=1)))
    out["fine_power_w"] = float(np.sum(duty * plant.i_max * 3.3))
    out["fine_diverged"] = float(np.max(pe) > 90.0)
    out["fine_knowledge_med"] = float(np.median(r.knowledge_err_deg[last]))
    return out


def criteria(metrics):
    """[(name, value, threshold, passed)] — every threshold is 'value <= threshold'."""
    rows = []
    for name, thr in THRESHOLDS.items():
        v = metrics[name]
        rows.append((name, v, thr, bool(v <= thr)))
    return rows


def reward(metrics):
    """Smooth stand-in for the criteria: sum of clipped margins, divergence penalised hard."""
    r = 0.0
    for name, thr in THRESHOLDS.items():
        v = metrics[name]
        if name.endswith("diverged"):
            r -= 5.0 * v
        else:
            r += float(np.clip((thr - v) / thr, -2.0, 1.0))
    return r


def _worker(args):
    z, plant, seeds = args
    p = _from_z(z)
    ms = [evaluate(p, plant, s) for s in seeds]
    agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    for k in ("acq_diverged", "fine_diverged"):
        agg[k] = float(max(m[k] for m in ms))
    return reward(agg), agg


def cem_tune(plant, train_seeds=(0, 1), pop=12, elites=4, generations=5, init=None,
             sigma0=0.25, workers=None, rng_seed=0, log=print):
    """Cross-entropy method over the 6 parameters. Returns (best_params, history)."""
    rng = np.random.default_rng(rng_seed)
    mu = _to_z(init) if init is not None else np.full(len(PARAM_NAMES), 0.5)
    sigma = np.full(len(PARAM_NAMES), sigma0)
    history = []
    best = (-np.inf, None, None)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers or max(1, mp.cpu_count() - 1)) as pool:
        for g in range(generations):
            Z = np.clip(mu + sigma * rng.standard_normal((pop, len(mu))), 0, 1)
            if g == 0 and init is not None:
                Z[0] = mu
            results = pool.map(_worker, [(z, plant, train_seeds) for z in Z])
            scores = np.array([r for r, _ in results])
            order = np.argsort(-scores)
            mu = Z[order[:elites]].mean(axis=0)
            sigma = Z[order[:elites]].std(axis=0) + 0.02
            if scores[order[0]] > best[0]:
                best = (scores[order[0]], _from_z(Z[order[0]]), results[order[0]][1])
            history.append({"gen": g, "best": float(scores[order[0]]), "mean": float(scores.mean()),
                            "params": dict(zip(PARAM_NAMES, map(float, _from_z(mu))))})
            log(f"gen {g}: best {scores[order[0]]:+.2f}  mean {scores.mean():+.2f}  "
                f"mu={ {k: round(v, 3) for k, v in history[-1]['params'].items()} }")
    return best[1], best[2], history


def save_params(path, params, metrics=None, extra=None):
    with open(path, "w") as f:
        json.dump({"params": dict(zip(PARAM_NAMES, map(float, params))),
                   "train_metrics": metrics, **(extra or {})}, f, indent=2)


def load_params(path):
    with open(path) as f:
        d = json.load(f)
    prm = dict(d["params"])
    for k, src in (("fine_theta_max", "theta_max"), ("fine_omega_max", "omega_max"), ("fine_rho", "rho")):
        prm.setdefault(k, prm[src])
    return np.array([prm[k] for k in PARAM_NAMES]), d
