#!/usr/bin/env python3
"""Experiment 3: learn the magnetorquer configuration from torque-pulse bursts.

    python3 scripts/sysid_magnetorquer.py

Truth is the S3 coil set paired onto S6's three channels plus a residual
spacecraft dipole. The identifier sees only gyro, magnetometer and coil-current
samples at the burst rate and must recover the 3x3 dipole matrix. Sweeps burst
sample rate (the A9 decision), pulse count and pulse duration, comparing the
naive per-sample (differentiated) fit with the impulse-form fit.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.dynamics import RigidBody, simulate_free_tumble  # noqa: E402
from scalar.sysid.magnetorquer_id import (PulseSchedule, identify_dipole_matrix,  # noqa: E402
                                          identify_dipole_matrix_impulse,
                                          inertia_shape_from_tumble, s3_coil_matrix,
                                          simulate_burst_campaign)


def main():
    M_true, i_max = s3_coil_matrix()
    m_res = np.array([2e-3, -1e-3, 0.5e-3])      # plausible residual dipole [A m^2]
    body = RigidBody()
    np.set_printoptions(precision=4, suppress=True)
    print("Truth dipole/amp matrix M [A m^2 / A] (S3 coils on S6 channels):")
    print(M_true)
    print("channel dipole at full drive [A m^2]:", np.round(np.diag(M_true) * i_max, 4))
    print("residual dipole truth:", m_res)

    tl = simulate_free_tumble(body, np.array([0, 0, 0, 1.0]), np.deg2rad([3, -2, 4.0]), 0.1, 3000)
    rng = np.random.default_rng(0)
    g = tl.omega + rng.normal(0, 6.6e-5 / np.sqrt(0.1), tl.omega.shape)
    I_hat, cond = inertia_shape_from_tumble(g, np.gradient(g, 0.1, axis=0))
    print("\nInertia shape (unit trace) from 300 s free tumble at 10 Hz:")
    print(" truth", np.round(np.diag(body.inertia / np.trace(body.inertia)), 4),
          " ident", np.round(np.diag(I_hat), 4), f" (sigma_min/sigma_max {cond:.1e})")

    def rel(M):
        return np.linalg.norm(M - M_true) / np.linalg.norm(M_true)

    def col_err(M):
        return [np.linalg.norm(M[:, j] - M_true[:, j]) / np.linalg.norm(M_true[:, j]) for j in range(3)]

    print("\nRelative error of M, per-sample (differentiated) vs impulse-form fit.")
    print("Per-column errors show the weak Z coil (13x less dipole) separately.")
    print(f"{'rate':>6} {'pulses':>7} {'pulse':>7} {'per-sample':>11} {'impulse':>9}  {'impulse per column x/y/z':>26}")
    for dur in (0.32, 2.0):
        for hz in (1.0, 10.0):
            for n_p in (30, 90):
                sched = PulseSchedule(3, i_max, n_pulses=n_p, duration=dur)
                d = simulate_burst_campaign(M_true, m_res, sched, body=body,
                                            sample_hz=hz, seed=int(hz * 10 + n_p))
                Md, *_ = identify_dipole_matrix(body.inertia, d["omega"], d["omega_dot"],
                                                d["B_body"], d["currents"])
                Mi, mr, npul, c = identify_dipole_matrix_impulse(
                    body.inertia, d["t"], d["omega"], d["B_body"], d["currents"], window=1.0)
                ce = col_err(Mi)
                print(f"{hz:>5.0f}Hz {n_p:>7d} {dur:>6.2f}s {rel(Md):>11.3f} {rel(Mi):>9.3f}  "
                      f"{ce[0]:>7.3f} {ce[1]:>7.3f} {ce[2]:>7.3f}")

    sched = PulseSchedule(3, i_max, n_pulses=90, duration=2.0)
    d = simulate_burst_campaign(M_true, m_res, sched, body=body, sample_hz=10.0, seed=1)
    Mi, mr, npul, c = identify_dipole_matrix_impulse(body.inertia, d["t"], d["omega"],
                                                     d["B_body"], d["currents"])
    print(f"\nIdentified M, impulse form, 10 Hz, 90 x 2 s pulses ({sched.total_time/60:.0f} min):")
    print(Mi)
    print("identified residual dipole:", mr, " truth:", m_res)
    print("\nInertia-scale caveat: motion data fixes M/|I| only. A 10 % inertia "
          "error is a 10 % dipole error, no more.")


if __name__ == "__main__":
    main()
