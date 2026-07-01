"""Tests for scalar.attitude — pins the frame/quaternion conventions.

Runnable two ways:
    python tests/test_attitude.py     # standalone, prints PASS/FAIL
    pytest tests/test_attitude.py     # if pytest is installed
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar import attitude as att  # noqa: E402


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_normalize_and_unit():
    q = att.normalize([1.0, 2.0, 3.0, 4.0])
    assert att.is_unit(q)


def test_to_body_to_eci_roundtrip():
    rng = _rng(1)
    for _ in range(50):
        q = att.rand_quat(rng)
        v = rng.standard_normal(3)
        back = att.to_eci(q, att.to_body(q, v))
        assert np.allclose(back, v, atol=1e-12)


def test_dcm_orthonormal_and_quat_roundtrip():
    rng = _rng(2)
    for _ in range(50):
        q = att.rand_quat(rng)
        C = att.dcm(q)
        assert np.allclose(C @ C.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(C), 1.0, atol=1e-12)
        q2 = att.quat_from_dcm(C)
        # q and -q are the same rotation; compare via error angle
        assert att.attitude_error_angle(q2, q) < 1e-9


def test_quat_mul_matches_dcm_composition():
    rng = _rng(3)
    for _ in range(50):
        a = att.rand_quat(rng)
        b = att.rand_quat(rng)
        assert np.allclose(att.dcm(att.quat_mul(a, b)), att.dcm(a) @ att.dcm(b),
                           atol=1e-12)


def test_small_angle_roundtrip():
    rng = _rng(4)
    for _ in range(50):
        dtheta = rng.standard_normal(3) * 1e-3  # small
        q = att.quat_from_small_angle(dtheta)
        assert np.allclose(att.small_angle_from_quat(q), dtheta, atol=1e-8)


def test_spin_sense_is_physical():
    """A body spinning +Omega about +z must carry a body-fixed +x axis toward
    +y when viewed in ECI (right-hand rule). Pins the sign of propagate."""
    Omega = 0.3  # rad/s about +z
    q = att.IDENTITY_QUAT.copy()
    dt = 1e-3
    T_quarter = (np.pi / 2) / Omega
    n = int(round(T_quarter / dt))
    for _ in range(n):
        q = att.propagate(q, np.array([0.0, 0.0, Omega]), dt)
    x_body_in_eci = att.to_eci(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(x_body_in_eci, [0.0, 1.0, 0.0], atol=1e-3), x_body_in_eci


def test_propagate_matches_closed_form_spin():
    """Full closed-form check against an analytic constant-rate rotation."""
    rng = _rng(5)
    axis = att.normalize(np.append(rng.standard_normal(3), 0.0))[:3]
    axis = axis / np.linalg.norm(axis)
    Omega = 0.7
    omega = Omega * axis
    q = att.IDENTITY_QUAT.copy()
    dt = 1e-3
    t_end = 2.0
    n = int(round(t_end / dt))
    for _ in range(n):
        q = att.propagate(q, omega, dt)
    # closed form: C_{B<-I}(t) = exp(-[omega]x t)  ->  quat_from_rotvec(-omega t)
    q_closed = att.quat_from_rotvec(-omega * (n * dt))
    assert att.attitude_error_angle(q, q_closed) < 1e-6


def test_propagate_preserves_unit_norm():
    rng = _rng(6)
    q = att.rand_quat(rng)
    for _ in range(1000):
        q = att.propagate(q, rng.standard_normal(3), 0.01)
        assert att.is_unit(q, tol=1e-9)


def test_error_angle_known_offset():
    q_true = att.IDENTITY_QUAT
    q_est = att.quat_from_rotvec([0.0, 0.0, np.deg2rad(12.0)])
    assert np.isclose(np.rad2deg(att.attitude_error_angle(q_est, q_true)), 12.0,
                      atol=1e-9)
    assert att.attitude_error_angle(q_true, q_true) == 0.0


# --- standalone runner ------------------------------------------------------
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
