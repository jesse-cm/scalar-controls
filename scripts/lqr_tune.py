#!/usr/bin/env python3
"""Experiment 4: LQR for the identified coils, with the field inside the design.

    python3 scripts/lqr_tune.py

[A] The existing time-invariant LQR + projection (lqr.py) cannot point, even
    sign-corrected: the along-B part of every torque request is discarded and
    the residual spacecraft dipole out-muscles the low-gain averaging regime.
[B] Periodic LQR near the target: Bryson weights, rho sweep, and how coarsely
    the gain schedule may be quantised (flight storage). 64 bins is too coarse
    for the stiff designs; 256 bins (18 KB float32) is safe.
[C] Acquisition from 120 deg / 1 deg/s: rate damping (omega x B) until
    0.3 deg/s, then the periodic schedule. This is the C-26 supervisor.
[D] [C] with the MEKF in the loop at 1 Hz, ideal sensors.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import IDENTITY_QUAT, quat_from_rotvec  # noqa: E402
from scalar.constants import DEG  # noqa: E402
from scalar.control.periodic_lqr import (PeriodicLQR, TwoModeController, bryson_weights,  # noqa: E402
                                         periodic_lqr_schedule, quantize_schedule)
from scalar.control.tune import (actuators_from_matrix, simulate_closed_loop_estimated,  # noqa: E402
                                 tune_lqr_weights)
from scalar.dynamics import RigidBody  # noqa: E402
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.sysid.magnetorquer_id import s3_coil_matrix  # noqa: E402
from scalar.timeutil import julian_date  # noqa: E402


def stats(r, P):
    last = r.pointing_err_deg[-int(P):]
    return (f"settle {r.settle_time()/3600:5.2f} h | last orbit mean {np.mean(last):6.2f} "
            f"p95 {np.percentile(last,95):6.2f} | max rate {r.rate_deg_s.max():5.2f} deg/s | sat {r.sat_frac:.2f}")


def main():
    body = RigidBody(); M, i_max = s3_coil_matrix(); acts = actuators_from_matrix(M, i_max)
    orbit = CircularOrbit(inclination=98.61 * DEG); jd0 = julian_date(dt.datetime(2026, 9, 11, 12))
    P = orbit.period
    m_res = np.array([2e-3, -1e-3, 0.5e-3])              # residual dipole [A m^2], ~3 % of a coil
    q_far, w_far = np.array([0.5, 0.5, 0.5, 0.5]), np.deg2rad([1.0, -0.6, 0.8])
    q_near, w_near = quat_from_rotvec(np.deg2rad(20) * np.array([0.6, 0.8, 0.0])), np.deg2rad([0.1, -0.05, 0.08])
    print("Actuator m_max [A m^2]:", np.round(acts.m_max_vector, 4), f" orbit {P/60:.1f} min, "
          f"residual dipole {np.linalg.norm(m_res)*1e3:.1f} mA m^2")

    print("\n[A] Time-invariant LQR + projection (lqr.py, sign-corrected), 4 random weight sets, 120 deg start:")
    for score, w, r in tune_lqr_weights(body, acts, orbit, jd0, q_far, w_far, n_candidates=4, n_orbits=3):
        print(f"   q={w['q_weight']:.2g} w={w['w_weight']:.2g} r={w['r_weight']:.2g}: {stats(r, P)}")

    print("\n[B] Periodic LQR near target (20 deg, 0.1 deg/s), Bryson 10 deg / 0.2 deg/s, truth feedback, 4 orbits:")
    schedules = {}
    for rho in (0.1, 0.3, 1.0):
        Q, R = bryson_weights(10 * DEG, 0.2 * DEG, acts.m_max_vector, rho)
        K, _ = periodic_lqr_schedule(body.inertia, orbit, jd0, IDENTITY_QUAT, 1.0, Q, R)
        schedules[rho] = K
        for nb in (None, 256, 64):
            Kq = K if nb is None else quantize_schedule(K, nb)
            r = simulate_closed_loop_estimated(body, acts, orbit, jd0, q0=q_near, omega0=w_near, n_orbits=4,
                                               use_truth=True, controller=PeriodicLQR(Kq, P, acts, IDENTITY_QUAT), m_res=m_res)
            print(f"   rho={rho:<4} bins={str(nb or len(K)):<5} ({Kq.size*4/1024:6.1f} KB): {stats(r, P)}")

    print("\n[C] Acquisition from 120 deg / 1 deg/s, 256-bin schedule, truth feedback, 6 orbits:")
    for rho in (0.1, 0.3, 1.0):
        K256 = quantize_schedule(schedules[rho], 256)
        for two in (False, True):
            lqr = PeriodicLQR(K256, P, acts, IDENTITY_QUAT)
            ctrl = TwoModeController(lqr) if two else lqr
            r = simulate_closed_loop_estimated(body, acts, orbit, jd0, q0=q_far, omega0=w_far, n_orbits=6,
                                               use_truth=True, controller=ctrl, m_res=m_res)
            print(f"   rho={rho:<4} {'damp->LQR' if two else 'LQR only ':<9}: {stats(r, P)}")

    print("\n[D] Two-mode, rho 0.3, 256 bins, MEKF in the loop at 1 Hz (ideal sensors), 6 orbits:")
    K256 = quantize_schedule(schedules[0.3], 256)
    r = simulate_closed_loop_estimated(body, acts, orbit, jd0, q0=q_far, omega0=w_far, n_orbits=6, seed=2,
                                       controller=TwoModeController(PeriodicLQR(K256, P, acts, IDENTITY_QUAT)), m_res=m_res)
    last = slice(-int(P), None)
    print(f"   {stats(r, P)} | knowledge median {np.median(r.knowledge_err_deg[last]):.2f} deg")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].semilogy(r.t / 3600, r.pointing_err_deg, label="pointing (truth vs target)")
    ax[0].semilogy(r.t / 3600, r.knowledge_err_deg, label="knowledge (MEKF vs truth)", alpha=.7)
    ax[0].set_ylabel("deg"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Rate damping -> periodic LQR (256-bin, rho 0.3), S3 coils on S6 channels, MEKF in loop, 1 Hz")
    ax[1].semilogy(r.t / 3600, r.rate_deg_s); ax[1].set_ylabel("|omega| deg/s"); ax[1].grid(alpha=.3)
    ax[2].plot(r.t / 3600, r.dipole, lw=.6); ax[2].set_ylabel("dipole [A m^2]"); ax[2].set_xlabel("h"); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("outputs/lqr_two_mode.png", dpi=120)
    print("plot -> outputs/lqr_two_mode.png")


if __name__ == "__main__":
    main()
