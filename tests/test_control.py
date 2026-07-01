"""Tests for scalar.control.lqr — LQR magnetorquer detumble/stabilization."""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.actuators.magnetorquer import default_actuator_set  # noqa: E402
from scalar.attitude import IDENTITY_QUAT  # noqa: E402
from scalar.control.lqr import MagnetorquerLQR, lqr_gain  # noqa: E402
from scalar.dynamics import RigidBody, default_inertia, simulate_controlled  # noqa: E402
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.timeutil import julian_date  # noqa: E402

JD0 = julian_date(dt.datetime(2024, 3, 20, 12))


def _controller(**kw):
    body = RigidBody()
    return body, MagnetorquerLQR(body.inertia, default_actuator_set(),
                                 CircularOrbit(), JD0, **kw)


def test_lqr_gain_stabilizes_plant():
    """The LQR closed-loop A - B K must be Hurwitz (all Re(eig) < 0)."""
    I = default_inertia()
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    B = np.zeros((6, 3))
    B[3:, :] = np.linalg.inv(I)
    K = lqr_gain(I)
    assert np.all(np.linalg.eigvals(A - B @ K).real < 0)


def test_zero_torque_at_target():
    _, ctrl = _controller()
    assert np.allclose(ctrl.desired_torque(ctrl.q_target, ctrl.w_target), 0.0)


def test_realized_torque_is_perpendicular_to_B():
    _, ctrl = _controller()
    tau = ctrl(0.0, IDENTITY_QUAT, np.deg2rad([4.0, -3.0, 5.0]))
    B = ctrl.B_hist[-1]
    assert abs(np.dot(tau, B)) < 1e-20            # magnetic torque has no ‖B part
    assert np.linalg.norm(tau) > 0.0              # but is doing something


def test_closed_loop_detumbles():
    body, ctrl = _controller()
    w0 = np.deg2rad(6.0) * np.array([1.0, -0.6, 0.8])
    tl, taus = simulate_controlled(body, IDENTITY_QUAT, w0, dt=2.0,
                                   n_steps=int(2 * CircularOrbit().period / 2),
                                   torque_fn=ctrl)
    w_start = np.linalg.norm(tl.omega[0])
    w_end = np.linalg.norm(tl.omega[-1])
    assert w_end < 0.5 * w_start                  # rate substantially reduced
    # torque never exceeds what the actuators can produce
    assert np.max(np.linalg.norm(taus, axis=1)) < 0.2 * 6e-5 * 1.01


def test_saturation_bounds_dipole():
    _, ctrl = _controller()
    ctrl(0.0, IDENTITY_QUAT, np.deg2rad([30.0, 30.0, 30.0]))  # huge rate
    m = ctrl.m_hist[-1]
    assert abs(m[0]) <= 0.2 + 1e-12 and abs(m[1]) <= 0.2 + 1e-12
    assert abs(m[2]) <= 0.02 + 1e-12              # weak coil axis


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
