"""Tests for scalar.environment.orbit and .magfield."""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.constants import DEG, MU_EARTH, R_EARTH_M  # noqa: E402
from scalar.environment.magfield import (B0_SURFACE_T, dipole_axis_eci,  # noqa: E402
                                         magnetic_field_eci)
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.timeutil import julian_date  # noqa: E402

ORB = CircularOrbit()  # defaults: 550 km, 51.6 deg
JD = julian_date(dt.datetime(2024, 3, 20, 12))


def test_radius_and_speed_constant():
    ts = np.linspace(0, ORB.period, 40)
    for t in ts:
        r, v = ORB.state(t)
        assert np.isclose(np.linalg.norm(r), ORB.radius, rtol=1e-12)
        assert np.isclose(np.linalg.norm(v), np.sqrt(MU_EARTH / ORB.radius),
                          rtol=1e-12)
        assert abs(np.dot(r, v)) < 1e-3 * ORB.radius  # velocity ⟂ radius


def test_period_reasonable():
    assert 5600.0 < ORB.period < 5900.0            # ~95.6 min for 550 km


def test_max_latitude_equals_inclination():
    ts = np.linspace(0, ORB.period, 2000)
    max_sin_lat = max(abs(ORB.position(t)[2]) / ORB.radius for t in ts)
    assert np.isclose(max_sin_lat, np.sin(51.6 * DEG), atol=1e-3)


def test_ascending_node_in_equatorial_plane():
    # arg_lat0 = 0 by default -> at t=0 the satellite is at the ascending node,
    # which lies in the equatorial plane (z = 0).
    assert abs(ORB.position(0.0)[2]) < 1e-6


def test_field_magnitude_at_leo():
    for t in np.linspace(0, ORB.period, 60):
        B = magnetic_field_eci(ORB.position(t), JD)
        assert 1.5e-5 < np.linalg.norm(B) < 5.5e-5    # tens of microtesla


def test_pole_to_equator_ratio_is_two():
    m_hat = dipole_axis_eci(JD)
    r = R_EARTH_M + 550e3
    # perpendicular direction to the dipole axis
    perp = np.cross(m_hat, [0, 0, 1.0])
    perp = perp / np.linalg.norm(perp)
    B_pole = magnetic_field_eci(r * m_hat, JD)
    B_eq = magnetic_field_eci(r * perp, JD)
    assert np.isclose(np.linalg.norm(B_pole) / np.linalg.norm(B_eq), 2.0,
                      rtol=1e-9)
    # equatorial magnitude equals B0 scaled by (Re/r)^3
    assert np.isclose(np.linalg.norm(B_eq),
                      B0_SURFACE_T * (R_EARTH_M / r) ** 3, rtol=1e-9)


def test_dipole_tilt_angle():
    m_hat = dipole_axis_eci(JD)
    # dipole moment points ~opposite the north pole; angle from -z is the tilt
    ang = np.rad2deg(np.arccos(np.clip(np.dot(m_hat, [0, 0, -1.0]), -1, 1)))
    assert abs(ang - 9.7) < 0.1


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
