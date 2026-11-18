#!/usr/bin/env python3
"""Experiment 5: the whole chain — learned sun front end -> MEKF -> periodic LQR.

    python3 scripts/closed_loop_nn.py

Fine-pointing mode (periodic LQR, 256-bin schedule) closed on the MEKF at 1 Hz
with a flight-like sun sensor unit under Earth albedo. Compares the analytic
opposing-pair reconstruction with the MLP-B front end, for a stiff and a soft
controller. Pointing error is truth vs target; the controller never sees truth.

Runs the closed loop with its current defaults (coil dipole held over the step,
2 plant substeps, errors sampled at one instant). The numbers in
docs/learned-estimation-and-control.md came from the original loop; they are
reproduced by the baseline scenarios, which pin it
(``python3 scripts/matrix.py --gate``), not by this script.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import IDENTITY_QUAT, quat_from_rotvec  # noqa: E402
from scalar.constants import DEG  # noqa: E402
from scalar.control.periodic_lqr import (PeriodicLQR, bryson_weights,  # noqa: E402
                                         periodic_lqr_schedule, quantize_schedule)
from scalar.control.tune import actuators_from_matrix, simulate_closed_loop_estimated  # noqa: E402
from scalar.dynamics import RigidBody  # noqa: E402
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.learning.sun_mlp import flight_like_sensor, train_sun_mlp  # noqa: E402
from scalar.sysid.magnetorquer_id import s3_coil_matrix  # noqa: E402
from scalar.timeutil import julian_date  # noqa: E402


def main():
    body = RigidBody(); M, i_max = s3_coil_matrix(); acts = actuators_from_matrix(M, i_max)
    orbit = CircularOrbit(inclination=98.61 * DEG); jd0 = julian_date(dt.datetime(2026, 9, 11, 12))
    P = orbit.period
    q0 = quat_from_rotvec(np.deg2rad(20) * np.array([0.6, 0.8, 0.0])); w0 = np.deg2rad([0.1, -0.05, 0.08])
    m_res = np.array([2e-3, -1e-3, 0.5e-3])

    sensor = flight_like_sensor(np.random.default_rng(1))
    mlp_b, _ = train_sun_mlp(sensor, extra=True, epochs=15, seed=0)
    fronts = {
        "analytic pairs": None,
        "MLP-B front end": lambda c, E, nb, ds: mlp_b.reconstruct(c, E, sensor, np.concatenate([nb, [ds]])),
    }
    designs = {
        "stiff (10 deg, 0.2 deg/s, rho 0.3)": (10.0, 0.2, 0.3),
        "low rate weight (10 deg, 1.0 deg/s, rho 1.0)": (10.0, 1.0, 1.0),
    }
    print("Fine pointing from 20 deg, MEKF in the loop at 1 Hz, flight-like sun unit, residual dipole, 4 orbits.")
    print("Last-orbit pointing error (truth vs target) and knowledge error (MEKF vs truth), deg:\n")
    print(f"{'controller':<36} {'front end':<16} {'albedo':>6} {'point mean':>11} {'p95':>7} {'know med':>9} {'p95':>6} {'sat':>5}")
    for dname, (th, wm, rho) in designs.items():
        Q, R = bryson_weights(th * DEG, wm * DEG, acts.m_max_vector, rho)
        K, _ = periodic_lqr_schedule(body.inertia, orbit, jd0, IDENTITY_QUAT, 1.0, Q, R)
        K256 = quantize_schedule(K, 256)
        for fname, front in fronts.items():
            for albedo in (0.0, 0.3, 0.5):
                ctrl = PeriodicLQR(K256, P, acts, IDENTITY_QUAT)
                r = simulate_closed_loop_estimated(body, acts, orbit, jd0, q0=q0, omega0=w0, n_orbits=4,
                                                   seed=2, controller=ctrl, m_res=m_res,
                                                   albedo_coeff=albedo, sun_sensor=sensor, sun_front=front)
                last = slice(-int(P), None)
                pe, ke = r.pointing_err_deg[last], r.knowledge_err_deg[last]
                print(f"{dname:<36} {fname:<16} {albedo:>6.1f} {np.mean(pe):>11.2f} {np.percentile(pe,95):>7.2f} "
                      f"{np.median(ke):>9.2f} {np.percentile(ke,95):>6.2f} {r.sat_frac:>5.2f}")
        print()


if __name__ == "__main__":
    main()
