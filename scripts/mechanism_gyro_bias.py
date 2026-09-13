#!/usr/bin/env python3
"""Causal test: is fine-pointing error gyro-bias error x the controller's rate gain?

    python3 scripts/mechanism_gyro_bias.py

CLAIM UNDER TEST
    The fine-mode dipole is  m = -K(phase) @ [ -aer(q, target) ; omega_meas - b_hat ].
    A bias-estimation error  db = b_hat - b_true  enters the second block as a
    fake rate  -db, and the attitude offset that cancels it is, per axis,
        aer(q, target) ~= -(K_omega / K_theta) * db      [rad]
    i.e. a "seconds of offset per rad/s of fake rate" gain ratio r.

WHAT IS RUN (all seed 100 unless noted; metrics over the last of 3 orbits)
    T1  r from ctrl.fine.K (256 phase bins) -> predicted pointing time series
        |r (*) db(t)| vs the observed pointing mean.
    T2  Interventions on the two feedback paths, same seeds/noise draws:
          true-rate   : controller gets omega_true but the ESTIMATED attitude
                        (cuts only the bias-error path)
          true-att    : controller gets q_true but the ESTIMATED rate
                        (keeps only the bias-error path)
    T3  Direction: angle between the pointing-error rotvec and r (*) db(t),
        seeds 100 and 101.
    T4  Size of |db| (median, p95) for the default gyro and a perfect gyro.

Nothing under scalar/ is touched: the closed loop below is a copy of
``scalar.control.tune.simulate_closed_loop_estimated`` with extra logging
(mekf.b, the true gyro bias, the estimate, the dipole) and the two path
interventions. Keep it in sync with that function if it changes.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import (IDENTITY_QUAT, attitude_error_angle,  # noqa: E402
                             attitude_error_rotvec, propagate, quat_from_rotvec, to_body)
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
OUT_PATH = "results/mechanism_gyro_bias.json"
N_ORBITS = 3
DT = 1.0


# --------------------------------------------------------------------------
# instrumented copy of simulate_closed_loop_estimated (tune.py:88)
# --------------------------------------------------------------------------
def run_logged(body, actuators, orbit, jd0, ctrl, q0, omega0, seed, m_res,
               magnetometer, mag_calibration, bias_rw=1.0e-5, sigma_u=1.0e-5,
               feed="estimated", n_orbits=N_ORBITS, dt=DT):
    """Truth -> sensors -> MEKF -> controller -> coils, logging the bias error.

    ``feed`` selects what the controller is shown:
        "estimated" : (mekf.q, g - mekf.b)          -- flight / baseline
        "true_rate" : (mekf.q, omega_true)          -- cut the bias-error path
        "true_att"  : (q_true,  g - mekf.b)         -- keep only that path
    The field is always the calibrated magnetometer reading, as in flight.
    """
    assert feed in ("estimated", "true_rate", "true_att")
    rng = np.random.default_rng(seed)
    gyro = Gyro(bias_rw=bias_rw)
    mag = magnetometer
    sun = SunSensorArray()
    gyro.reset()
    mekf = MEKF(n_iter=3, sigma_u=sigma_u)
    n = int(n_orbits * orbit.period / dt)
    q, w = np.asarray(q0, float), np.asarray(omega0, float)
    t_arr = np.arange(n) * dt
    pe = np.empty(n); ke = np.empty(n); rate = np.empty(n)
    mh = np.empty((n, 3))
    db = np.empty((n, 3))          # b_hat - b_true at the measurement instant
    b_true_h = np.empty((n, 3)); b_hat_h = np.empty((n, 3))
    pe_vec = np.empty((n, 3))      # pointing-error rotvec, truth vs target
    w_true_h = np.empty((n, 3))
    n_sat = 0
    started = False
    m_max = actuators.m_max_vector
    for k in range(n):
        t = t_arr[k]
        jd = jd0 + t / 86400.0
        r = orbit.position(t)
        B_eci = magnetic_field_eci(r, jd)
        s_eci = sun_direction_eci(jd)
        E = solar_irradiance(jd)
        lit = eclipse_factor(r, s_eci) > 0
        r_hat = r / np.linalg.norm(r)
        # --- sensors ---
        b_true = gyro.bias.copy()          # bias USED by this sample (measure advances it)
        g = gyro.measure(w, dt, rng)
        Bm = mag.measure(B_eci, q, rng)
        if mag_calibration is not None:
            Bm = mag_calibration.apply(Bm)
        dayside = max(0.0, float(r_hat @ s_eci))
        currents = sun.measure(to_body(q, s_eci), E, lit, rng, to_body(q, -r_hat), 0.0)
        s_hat, valid = sun.reconstruct(currents, E)
        # --- estimator ---
        if k > 0:
            mekf.propagate(g, dt)
        if not started and valid:
            mekf.init_triad(s_eci, s_hat, B_eci, Bm); started = True
        mekf.update_mag(B_eci, Bm)
        if valid:
            mekf.update_sun(s_eci, s_hat)
        # --- controller on the estimate, field from the magnetometer ---
        if feed == "estimated":
            q_c, w_c = mekf.q, g - mekf.b
        elif feed == "true_rate":
            q_c, w_c = mekf.q, w
        else:                                     # true_att
            q_c, w_c = q, g - mekf.b
        m = ctrl.dipole(t, q_c, w_c, Bm)
        n_sat += int(np.any(np.abs([m @ tq.axis for tq in actuators.torquers]) >= 0.999 * m_max))
        m_dist = np.zeros(3) if m_res is None else np.asarray(m_res, float)
        tau = np.cross(m + m_dist, to_body(q, B_eci))
        # --- plant (RK4 on omega, exp-map on q, zero-order-hold torque) ---
        k1 = body.omega_dot(w, tau); k2 = body.omega_dot(w + 0.5*dt*k1, tau)
        k3 = body.omega_dot(w + 0.5*dt*k2, tau); k4 = body.omega_dot(w + dt*k3, tau)
        w_next = w + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        q = propagate(q, 0.5*(w + w_next), dt); w = w_next
        pe[k] = np.degrees(attitude_error_angle(q, ctrl.q_target))
        ke[k] = np.degrees(attitude_error_angle(mekf.q, q))
        rate[k] = np.degrees(np.linalg.norm(w)); mh[k] = m
        pe_vec[k] = attitude_error_rotvec(q, ctrl.q_target)
        db[k] = mekf.b - b_true
        b_true_h[k] = b_true; b_hat_h[k] = mekf.b
        w_true_h[k] = w
    return dict(t=t_arr, pointing_err_deg=pe, knowledge_err_deg=ke, rate_deg_s=rate,
                dipole=mh, db=db, b_true=b_true_h, b_hat=b_hat_h, pe_vec=pe_vec,
                w_true=w_true_h, sat_frac=n_sat / n)


# --------------------------------------------------------------------------
def gain_ratios(K):
    """Per-bin per-axis rate-gain / attitude-gain column-norm ratio [s].

    K has shape (n_bins, 3, 6); column i is the attitude channel for axis i and
    column 3+i the rate channel. r_i = |K[:, 3+i]| / |K[:, i]|, i.e. the radians
    of attitude offset that balance one rad/s of fake rate on that axis.
    """
    K = np.asarray(K, float)
    num = np.linalg.norm(K[:, :, 3:], axis=1)     # (n_bins, 3)
    den = np.linalg.norm(K[:, :, :3], axis=1)     # (n_bins, 3)
    return num / den


def bin_index(t, period, n_bins):
    return (np.floor((t % period) / period * n_bins).astype(int)) % n_bins


def full_offset_prediction(K, t, db, period):
    """Exact static-balance prediction  aer = -K_theta^-1 K_omega db  [rad]."""
    idx = bin_index(t, period, K.shape[0])
    out = np.empty((len(t), 3))
    inv = np.array([np.linalg.solve(K[b, :, :3], K[b, :, 3:]) for b in range(K.shape[0])])
    for k in range(len(t)):
        out[k] = -inv[idx[k]] @ db[k]
    return out


def stats(x):
    return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), max=float(np.max(x)))


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    np.set_printoptions(precision=4, suppress=True)

    params, _ = load_params(PARAMS_PATH)
    plant = default_plant(mag_calibrated=True)
    body = RigidBody(plant.inertia * plant.scale_inertia)
    orbit = CircularOrbit(inclination=plant.inclination_deg * DEG)
    jd0 = julian_date(plant.epoch)
    acts = actuators_from_matrix(plant.M, plant.i_max)
    mag = Magnetometer(bias=plant.mag_bias, soft_iron=plant.mag_soft_iron)
    q0 = quat_from_rotvec(np.deg2rad(20) * np.array([0.6, 0.8, 0.0]))
    omega0 = np.deg2rad([0.1, -0.05, 0.08])
    last = slice(-int(orbit.period), None)
    P = orbit.period

    print(f"orbit period {P:.1f} s, {N_ORBITS} orbits at {DT} s, metrics over the last orbit")
    print("building the periodic-LQR schedules (acquisition + fine)...", flush=True)
    ctrl = build_controller(params, plant, acts, orbit, jd0)
    K = np.asarray(ctrl.fine.K, float)
    print(f"fine schedule K shape {K.shape}")

    out = {"setup": {"seed_main": 100, "n_orbits": N_ORBITS, "dt": DT,
                     "orbit_period_s": float(P), "params_path": PARAMS_PATH,
                     "K_shape": list(K.shape)}}

    # ---------------- T1a: the gain ratio itself --------------------------
    print("\n" + "=" * 72)
    print("T1  PREDICTION FROM THE GAIN RATIO")
    print("=" * 72)
    R = gain_ratios(K)                      # (256, 3), seconds
    r_med = np.median(R, axis=0)
    print("r_i = |K[:,3+i]| / |K[:,i]|  [s of attitude offset per rad/s of fake rate]")
    for i, ax in enumerate("xyz"):
        print(f"  axis {ax}: median {r_med[i]:8.1f} s   range [{R[:, i].min():.1f}, {R[:, i].max():.1f}] s")
    print(f"  over all axes and bins: median {np.median(R):.1f} s, "
          f"range [{R.min():.1f}, {R.max():.1f}] s")
    out["T1"] = {"r_median_per_axis_s": r_med.tolist(),
                 "r_min_per_axis_s": R.min(axis=0).tolist(),
                 "r_max_per_axis_s": R.max(axis=0).tolist(),
                 "r_median_all_s": float(np.median(R)),
                 "r_range_all_s": [float(R.min()), float(R.max())]}

    # ---------------- calibration + the baseline run ----------------------
    cals = {}
    for seed in (100, 101):
        print(f"\ncalibration pass, seed {seed}...", flush=True)
        cals[seed], med = calibration_pass(plant, mag, orbit, jd0, seed)
        print(f"  post-calibration direction error (median): {med:.3f} deg")

    def go(seed, feed, bias_rw=1.0e-5, sigma_u=1.0e-5, label=""):
        print(f"\nrun: seed {seed}, feed={feed}, bias_rw={bias_rw:g}, sigma_u={sigma_u:g} "
              f"{label}", flush=True)
        ctrl.mode = 3
        r = run_logged(body, acts, orbit, jd0, ctrl, q0, omega0, seed, plant.m_res,
                       mag, cals[seed], bias_rw=bias_rw, sigma_u=sigma_u, feed=feed)
        pm = float(np.mean(r["pointing_err_deg"][last]))
        km = float(np.mean(r["knowledge_err_deg"][last]))
        print(f"  pointing mean {pm:6.3f} deg  p95 {np.percentile(r['pointing_err_deg'][last], 95):6.3f}"
              f"   knowledge mean {km:6.3f} deg   sat_frac {r['sat_frac']:.3f}")
        return r

    base = go(100, "estimated", label="(baseline, expect ~7.49 deg)")
    base_pm = float(np.mean(base["pointing_err_deg"][last]))
    base_km = float(np.mean(base["knowledge_err_deg"][last]))

    # ---------------- T1b: predicted vs observed --------------------------
    t = base["t"]
    db = base["db"]
    idx = bin_index(t, P, K.shape[0])
    pred_vec_bin = R[idx] * db                       # per-bin r, elementwise
    pred_deg_bin = np.degrees(np.linalg.norm(pred_vec_bin, axis=1))
    pred_vec_med = r_med * db                        # median r over bins
    pred_deg_med = np.degrees(np.linalg.norm(pred_vec_med, axis=1))
    pred_vec_full = full_offset_prediction(K, t, db, P)
    pred_deg_full = np.degrees(np.linalg.norm(pred_vec_full, axis=1))

    obs = base["pointing_err_deg"]
    print("\nlast-orbit means [deg]:")
    print(f"  observed pointing                         {np.mean(obs[last]):7.3f}")
    print(f"  predicted |r(bin) (*) db(t)|              {np.mean(pred_deg_bin[last]):7.3f}"
          f"   ratio pred/obs = {np.mean(pred_deg_bin[last]) / np.mean(obs[last]):.3f}")
    print(f"  predicted |median(r) (*) db(t)|           {np.mean(pred_deg_med[last]):7.3f}"
          f"   ratio pred/obs = {np.mean(pred_deg_med[last]) / np.mean(obs[last]):.3f}")
    print(f"  predicted |K_theta^-1 K_omega db| (exact) {np.mean(pred_deg_full[last]):7.3f}"
          f"   ratio pred/obs = {np.mean(pred_deg_full[last]) / np.mean(obs[last]):.3f}")
    out["T1"].update({
        "observed_pointing_mean_deg": float(np.mean(obs[last])),
        "observed_knowledge_mean_deg": base_km,
        "pred_mean_deg_per_bin_r": float(np.mean(pred_deg_bin[last])),
        "pred_mean_deg_median_r": float(np.mean(pred_deg_med[last])),
        "pred_mean_deg_full_matrix": float(np.mean(pred_deg_full[last])),
        "ratio_pred_over_obs_per_bin_r": float(np.mean(pred_deg_bin[last]) / np.mean(obs[last])),
        "ratio_pred_over_obs_median_r": float(np.mean(pred_deg_med[last]) / np.mean(obs[last])),
        "ratio_pred_over_obs_full_matrix": float(np.mean(pred_deg_full[last]) / np.mean(obs[last])),
    })

    # ---------------- T2: interventions -----------------------------------
    print("\n" + "=" * 72)
    print("T2  DIRECT INTERVENTION ON THE PATH")
    print("=" * 72)
    true_rate = go(100, "true_rate", label="(estimated attitude, TRUE rate)")
    true_att = go(100, "true_att", label="(TRUE attitude, estimated rate)")
    perfect = go(100, "estimated", bias_rw=0.0, sigma_u=1e-9,
                 label="(perfect gyro, expect ~2.59 deg)")

    rows = [("baseline (est q, est rate)", base),
            ("true rate,  est attitude ", true_rate),
            ("true attitude, est rate  ", true_att),
            ("perfect gyro (bias_rw=0) ", perfect)]
    print("\n                              pointing mean  p95    knowledge mean  knowledge med")
    for name, r in rows:
        pm = np.mean(r["pointing_err_deg"][last])
        pp = np.percentile(r["pointing_err_deg"][last], 95)
        km = np.mean(r["knowledge_err_deg"][last])
        kd = np.median(r["knowledge_err_deg"][last])
        print(f"  {name}   {pm:7.3f}  {pp:7.3f}   {km:7.3f}        {kd:7.3f}")
    out["T2"] = {key: {"pointing_mean_deg": float(np.mean(r["pointing_err_deg"][last])),
                       "pointing_p95_deg": float(np.percentile(r["pointing_err_deg"][last], 95)),
                       "knowledge_mean_deg": float(np.mean(r["knowledge_err_deg"][last])),
                       "knowledge_median_deg": float(np.median(r["knowledge_err_deg"][last])),
                       "sat_frac": float(r["sat_frac"])}
                 for key, r in (("baseline", base), ("true_rate", true_rate),
                                ("true_attitude", true_att), ("perfect_gyro", perfect))}

    # T1 re-tested against the ISOLATED rate path: the true-attitude run is the
    # configuration the r-prediction is actually derived for (perfect attitude
    # feedback, fake rate -db still present).
    print("\nT1 prediction re-tested on the true-attitude run (isolated rate path):")
    ta = true_att
    i_ta = bin_index(ta["t"], P, K.shape[0])
    obs_ta = float(np.mean(ta["pointing_err_deg"][last]))
    p_bin = np.degrees(np.linalg.norm(R[i_ta] * ta["db"], axis=1))
    p_med = np.degrees(np.linalg.norm(r_med * ta["db"], axis=1))
    p_full = np.degrees(np.linalg.norm(full_offset_prediction(K, ta["t"], ta["db"], P), axis=1))
    print(f"  observed pointing (true attitude, est rate)  {obs_ta:7.3f} deg")
    for tag, p in (("r(bin) (*) db", p_bin), ("median(r) (*) db", p_med),
                   ("-K_theta^-1 K_omega db", p_full)):
        print(f"  predicted {tag:24s} {np.mean(p[last]):7.3f} deg   "
              f"ratio pred/obs = {np.mean(p[last]) / obs_ta:.3f}")
    out["T1"]["isolated_rate_path"] = {
        "observed_pointing_mean_deg": obs_ta,
        "pred_mean_deg_per_bin_r": float(np.mean(p_bin[last])),
        "pred_mean_deg_median_r": float(np.mean(p_med[last])),
        "pred_mean_deg_full_matrix": float(np.mean(p_full[last])),
        "ratio_pred_over_obs_per_bin_r": float(np.mean(p_bin[last]) / obs_ta),
        "ratio_pred_over_obs_median_r": float(np.mean(p_med[last]) / obs_ta),
        "ratio_pred_over_obs_full_matrix": float(np.mean(p_full[last]) / obs_ta)}

    # ---------------- T3: direction ---------------------------------------
    print("\n" + "=" * 72)
    print("T3  DIRECTION TEST  (cos between pointing-error rotvec and r (*) db)")
    print("=" * 72)
    out["T3"] = {}
    runs_t3 = {"100": base}
    runs_t3["101"] = go(101, "estimated", label="(baseline, seed 101)")
    # the isolated rate path: pointing error here is not contaminated by the
    # attitude-estimate error, so it is the cleanest direction test available.
    runs_t3["100_true_att"] = true_att
    for seed, r in runs_t3.items():
        d = r["db"][last]
        pv = r["pe_vec"][last]
        i2 = bin_index(r["t"][last], P, K.shape[0])
        for tag, pred in (("r(bin) (*) db", R[i2] * d),
                          ("median(r) (*) db", r_med * d),
                          ("-K_theta^-1 K_omega db", full_offset_prediction(K, r["t"], r["db"], P)[last])):
            nn = np.linalg.norm(pred, axis=1) * np.linalg.norm(pv, axis=1)
            ok = nn > 0
            cos = np.sum(pred[ok] * pv[ok], axis=1) / nn[ok]
            print(f"  run {seed:12s}  {tag:24s}  median cos {np.median(cos):+.3f}   "
                  f"median |cos| {np.median(np.abs(cos)):.3f}   "
                  f"frac |cos|>0.9 {np.mean(np.abs(cos) > 0.9):.3f}")
            out["T3"].setdefault(str(seed), {})[tag] = {
                "median_cos": float(np.median(cos)),
                "median_abs_cos": float(np.median(np.abs(cos))),
                "frac_abs_cos_gt_0.9": float(np.mean(np.abs(cos) > 0.9))}
        out["T3"][str(seed)]["pointing_mean_deg"] = float(np.mean(r["pointing_err_deg"][last]))
        out["T3"][str(seed)]["knowledge_mean_deg"] = float(np.mean(r["knowledge_err_deg"][last]))

    # ---------------- T4: bias-error size ---------------------------------
    print("\n" + "=" * 72)
    print("T4  BIAS-ERROR SIZE  |db| = |b_hat - b_true|  [deg/s]")
    print("=" * 72)
    out["T4"] = {}
    for name, r in (("default gyro (bias_rw 1e-5)", base),
                    ("perfect gyro (bias_rw 0)   ", perfect),
                    ("seed 101, default gyro     ", runs_t3["101"])):
        n_db = np.degrees(np.linalg.norm(r["db"][last], axis=1))
        per_axis = np.degrees(np.abs(r["db"][last]))
        print(f"  {name}  |db| median {np.median(n_db):.5f}  p95 {np.percentile(n_db, 95):.5f} deg/s")
        print(f"       per-axis median {np.median(per_axis, axis=0)} deg/s")
        out["T4"][name.strip()] = {
            "db_norm_median_deg_s": float(np.median(n_db)),
            "db_norm_p95_deg_s": float(np.percentile(n_db, 95)),
            "db_per_axis_median_deg_s": np.median(per_axis, axis=0).tolist(),
            "db_per_axis_p95_deg_s": np.percentile(per_axis, 95, axis=0).tolist(),
            "b_true_drift_deg_s": float(np.degrees(np.linalg.norm(
                r["b_true"][-1] - r["b_true"][0])))}

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
