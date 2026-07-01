"""Tests for scalar.dynamics — torque-free rigid-body invariants."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import IDENTITY_QUAT, is_unit  # noqa: E402
from scalar.dynamics import (RigidBody, default_inertia,  # noqa: E402
                             simulate_free_tumble)

DEG = np.pi / 180.0


def test_default_inertia_distinct_principal_moments():
    I = default_inertia()
    d = np.diag(I)
    assert d[0] != d[1] != d[2]           # not a degenerate symmetric top
    assert np.all(d > 0)


def test_energy_conserved():
    body = RigidBody()
    w0 = np.array([5.0, -3.0, 2.0]) * DEG  # a few deg/s tumble
    tl = simulate_free_tumble(body, IDENTITY_QUAT, w0, dt=0.05, n_steps=4000)
    E0 = body.kinetic_energy(tl.omega[0])
    E = np.array([body.kinetic_energy(w) for w in tl.omega])
    assert np.all(np.abs(E - E0) / E0 < 1e-6)


def test_inertial_angular_momentum_conserved():
    body = RigidBody()
    w0 = np.array([6.0, 1.0, -4.0]) * DEG
    tl = simulate_free_tumble(body, IDENTITY_QUAT, w0, dt=0.05, n_steps=4000)
    H = np.array([body.angular_momentum_eci(q, w)
                  for q, w in zip(tl.q, tl.omega)])
    H0 = H[0]
    drift = np.linalg.norm(H - H0, axis=1) / np.linalg.norm(H0)
    assert np.max(drift) < 1e-4           # inertial H vector stays fixed


def test_quaternions_stay_unit():
    body = RigidBody()
    w0 = np.array([10.0, 5.0, -8.0]) * DEG
    tl = simulate_free_tumble(body, IDENTITY_QUAT, w0, dt=0.05, n_steps=2000)
    assert all(is_unit(q, tol=1e-9) for q in tl.q)


def test_no_spin_stays_put():
    body = RigidBody()
    tl = simulate_free_tumble(body, IDENTITY_QUAT, np.zeros(3), dt=0.1,
                              n_steps=100)
    assert np.allclose(tl.q[-1], IDENTITY_QUAT, atol=1e-12)
    assert np.allclose(tl.omega[-1], 0.0, atol=1e-12)


def test_axis_spin_keeps_rate_constant():
    """A pure spin about a principal axis has constant angular velocity."""
    body = RigidBody()
    w0 = np.array([0.0, 0.0, 7.0]) * DEG   # about principal z
    tl = simulate_free_tumble(body, IDENTITY_QUAT, w0, dt=0.05, n_steps=2000)
    assert np.allclose(tl.omega, w0, atol=1e-10)


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
