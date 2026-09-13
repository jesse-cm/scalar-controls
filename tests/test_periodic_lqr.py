"""Tests: periodic LQR schedule and closed-loop convergence near the target."""
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import IDENTITY_QUAT, attitude_error_rotvec, propagate, quat_from_rotvec  # noqa: E402
from scalar.constants import DEG  # noqa: E402
from scalar.control.periodic_lqr import (PeriodicLQR, bryson_weights, periodic_lqr_schedule,  # noqa: E402
                                         quantize_schedule)
from scalar.control.tune import actuators_from_matrix, simulate_closed_loop_estimated  # noqa: E402
from scalar.dynamics import RigidBody  # noqa: E402
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.sysid.magnetorquer_id import s3_coil_matrix  # noqa: E402
from scalar.timeutil import julian_date  # noqa: E402


def test_error_vector_derivative_sign():
    """The convention the controllers depend on: d/dt err(q, target) = -omega."""
    q = quat_from_rotvec(np.array([0.05, 0.0, 0.0]))
    w = np.array([0.0, 0.01, 0.0])
    de = attitude_error_rotvec(propagate(q, w, 1.0), IDENTITY_QUAT) - attitude_error_rotvec(q, IDENTITY_QUAT)
    assert np.allclose(de, -w, atol=1e-3), de


def _setup():
    body = RigidBody(); M, i_max = s3_coil_matrix(); acts = actuators_from_matrix(M, i_max)
    orbit = CircularOrbit(inclination=98.61 * DEG); jd0 = julian_date(dt.datetime(2026, 9, 11, 12))
    return body, acts, orbit, jd0


def test_schedule_is_periodic_and_sized():
    body, acts, orbit, jd0 = _setup()
    Q, R = bryson_weights(10 * DEG, 0.2 * DEG, acts.m_max_vector, 0.3)
    K, B = periodic_lqr_schedule(body.inertia, orbit, jd0, IDENTITY_QUAT, 1.0, Q, R, n_orbits_iter=3)
    assert K.shape == (int(round(orbit.period)), 3, 6)
    assert np.all(np.isfinite(K))
    assert quantize_schedule(K, 256).shape == (256, 3, 6)
    assert np.linalg.norm(B, axis=1).min() > 1e-5


def test_converges_from_20_deg_truth_feedback():
    body, acts, orbit, jd0 = _setup()
    Q, R = bryson_weights(10 * DEG, 0.2 * DEG, acts.m_max_vector, 0.3)
    K, _ = periodic_lqr_schedule(body.inertia, orbit, jd0, IDENTITY_QUAT, 1.0, Q, R, n_orbits_iter=3)
    ctrl = PeriodicLQR(quantize_schedule(K, 256), orbit.period, acts, IDENTITY_QUAT)
    q0 = quat_from_rotvec(np.deg2rad(20) * np.array([0.6, 0.8, 0.0]))
    r = simulate_closed_loop_estimated(body, acts, orbit, jd0, q0=q0, omega0=np.deg2rad([0.1, -0.05, 0.08]),
                                       n_orbits=1, use_truth=True, controller=ctrl,
                                       m_res=np.array([2e-3, -1e-3, 0.5e-3]))
    assert r.pointing_err_deg[-1] < 2.0, r.pointing_err_deg[-1]
    assert r.rate_deg_s.max() < 2.0


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"  ok  {f.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {f.__name__}: {e}")
    print(f"{ok}/{len(fns)} passed")
