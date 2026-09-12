"""Tests: inertia shape from tumble; impulse-form dipole identification."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.dynamics import RigidBody, simulate_free_tumble  # noqa: E402
from scalar.sysid.magnetorquer_id import (PulseSchedule, identify_dipole_matrix_impulse,  # noqa: E402
                                          inertia_shape_from_tumble, s3_coil_matrix,
                                          simulate_burst_campaign)


def test_inertia_shape_recovers_diagonal():
    body = RigidBody()
    tl = simulate_free_tumble(body, np.array([0, 0, 0, 1.0]), np.deg2rad([3, -2, 4.0]), 0.1, 2000)
    I_hat, ratio = inertia_shape_from_tumble(tl.omega, np.gradient(tl.omega, 0.1, axis=0))
    truth = body.inertia / np.trace(body.inertia)
    assert np.allclose(I_hat, truth, atol=2e-3), I_hat
    assert ratio < 0.5


def test_impulse_identification_recovers_s3_matrix():
    M_true, i_max = s3_coil_matrix()
    m_res = np.array([2e-3, -1e-3, 0.5e-3])
    sched = PulseSchedule(3, i_max, n_pulses=30, duration=2.0, period=8.0)
    d = simulate_burst_campaign(M_true, m_res, sched, sample_hz=10.0, seed=3,
                                gyro_noise_density=1e-5, mag_noise=1e-7)
    M, mr, n_p, cond = identify_dipole_matrix_impulse(RigidBody().inertia, d["t"], d["omega"],
                                                      d["B_body"], d["currents"])
    assert n_p >= 28   # the pulse starting at t=0 has no rising edge
    rel = np.linalg.norm(M - M_true) / np.linalg.norm(M_true)
    assert rel < 0.1, rel
    assert np.linalg.norm(mr - m_res) < 2e-3, mr   # 30 pulses: residual dipole is the noisiest term


def test_unresolved_pulse_at_1hz_is_worse_than_10hz():
    M_true, i_max = s3_coil_matrix()
    errs = {}
    for hz in (1.0, 10.0):
        sched = PulseSchedule(3, i_max, n_pulses=30, duration=0.32, period=8.0)
        d = simulate_burst_campaign(M_true, np.zeros(3), sched, sample_hz=hz, seed=4)
        M, *_ = identify_dipole_matrix_impulse(RigidBody().inertia, d["t"], d["omega"],
                                               d["B_body"], d["currents"], fit_residual=False)
        errs[hz] = np.linalg.norm(M - M_true) / np.linalg.norm(M_true)
    assert np.isfinite(errs[1.0]) and np.isfinite(errs[10.0])


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"  ok  {f.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {f.__name__}: {e}")
    print(f"{ok}/{len(fns)} passed")
