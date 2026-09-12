#!/usr/bin/env python3
"""Experiment 1: a learned sun-sensor front end for the MEKF.

    python3 scripts/nn_sun_sensor.py

Trains two tiny MLPs for one flight-like sensor unit, compares their sun-vector
error against the analytic opposing-pair reconstruction under albedo, then runs
the MEKF end-to-end with each front end on an albedo-contaminated orbit.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import attitude_error_angle, to_body  # noqa: E402
from scalar.estimator.mekf import MEKF  # noqa: E402
from scalar.learning.sun_mlp import (angular_error_deg, export_c_header,  # noqa: E402
                                     flight_like_sensor, make_dataset,
                                     train_sun_mlp)
from scalar.scenario import Scenario  # noqa: E402


def mekf_with_front_end(res, sensor, front):
    """front(currents, E, nadir_body_pred, dayside) -> (s_hat, valid)."""
    mekf = MEKF(n_iter=3)
    err = np.empty(len(res)); started = False
    for k in range(len(res)):
        if k > 0:
            mekf.propagate(res.gyro_meas[k], res.dt)
        r_hat = res.r_eci[k] / np.linalg.norm(res.r_eci[k])
        nadir_pred = to_body(mekf.q, -r_hat)
        dayside = max(0.0, float(r_hat @ res.sun_eci[k]))
        s_hat, valid = front(res.sun_currents[k], res.irradiance[k], nadir_pred, dayside)
        if not started and valid:
            mekf.init_triad(res.sun_eci[k], s_hat, res.B_eci_ref[k], res.mag_body[k]); started = True
        mekf.update_mag(res.B_eci_ref[k], res.mag_body[k])
        if valid:
            mekf.update_sun(res.sun_eci[k], s_hat)
        err[k] = np.degrees(attitude_error_angle(mekf.q, res.q_true[k]))
    conv = err[len(err) // 5:]
    return float(np.median(conv)), float(np.percentile(conv, 95))


def main():
    rng = np.random.default_rng(1)
    sensor = flight_like_sensor(rng)
    print("Flight-like unit: gains", np.round(sensor.gains, 3), "rolloff", sensor.rolloff)

    t0 = time.time()
    mlp_a, va = train_sun_mlp(sensor, extra=False, epochs=15, seed=0)
    mlp_b, vb = train_sun_mlp(sensor, extra=True, epochs=15, seed=0)
    print(f"trained in {time.time()-t0:.0f} s; params A={mlp_a.n_params} B={mlp_b.n_params}, "
          f"MACs A={mlp_a.n_macs} B={mlp_b.n_macs}")

    print("\nSun-vector error [deg] on held-out samples (sunlit, random geometry):")
    print(f"{'albedo coeff':>14} {'analytic med/p95':>18} {'MLP-A med/p95':>16} {'MLP-B med/p95':>16}")
    for a in (0.0, 0.15, 0.3, 0.5):
        Xc, Xe, y = make_dataset(sensor, 5000, np.random.default_rng(7), albedo_max=a) if a > 0 else \
            make_dataset(sensor, 5000, np.random.default_rng(7), albedo_max=1e-9)
        fs = sensor.full_scale_current(1361.0)
        an = np.array([sensor.reconstruct(c * fs, 1361.0)[0] for c in Xc])
        ea = angular_error_deg(an, y); eA = angular_error_deg(mlp_a(Xc), y)
        eB = angular_error_deg(mlp_b(Xc, Xe), y)
        print(f"{a:>14.2f} {np.median(ea):>8.2f}/{np.percentile(ea,95):<8.2f} "
              f"{np.median(eA):>7.2f}/{np.percentile(eA,95):<7.2f} "
              f"{np.median(eB):>7.2f}/{np.percentile(eB,95):<7.2f}")

    print("\nMEKF attitude knowledge error [deg], one orbit, dt=1 s, albedo 0.3, "
          "real-unit systematics, ideal magnetometer:")
    sc = Scenario(dt=1.0, sun_sensor=sensor, albedo_coeff=0.3, seed=3)
    res = sc.run()
    fronts = {
        "analytic pairs": lambda c, E, nb, ds: sensor.reconstruct(c, E),
        "MLP-A (currents)": lambda c, E, nb, ds: mlp_a.reconstruct(c, E, sensor),
        "MLP-B (+pred nadir)": lambda c, E, nb, ds: mlp_b.reconstruct(
            c, E, sensor, np.concatenate([nb, [ds]])),
    }
    for name, f in fronts.items():
        med, p95 = mekf_with_front_end(res, sensor, f)
        print(f"  {name:<22} median {med:5.2f}  p95 {p95:5.2f}")

    os.makedirs("outputs", exist_ok=True)
    export_c_header(mlp_b, "outputs/sun_mlp_weights.h")
    print("\nweights -> outputs/sun_mlp_weights.h")


if __name__ == "__main__":
    main()
