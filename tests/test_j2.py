"""Tests for the J2 secular orbit perturbation (scalar.environment.orbit).

The default orbit must be unchanged (fixed elements); with ``j2=True`` the
ascending node must regress at the standard first-order rate, recoverable from
the propagated trajectory itself.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scalar.constants import DEG
from scalar.environment.orbit import CircularOrbit


def test_default_is_unperturbed():
    """Without j2, both secular rates are exactly zero."""
    o = CircularOrbit()
    assert o.raan_dot == 0.0
    assert o.arg_lat_dot_sec == 0.0


def test_default_positions_unchanged():
    """j2=False must reproduce the original fixed-element trajectory."""
    o = CircularOrbit()
    # closed form: at t=0, arg_lat0=0, raan=0 -> position on the +x axis
    r0 = o.position(0.0)
    assert np.allclose(r0, [o.radius, 0.0, 0.0], atol=1e-6)
    # after a full period the satellite returns to the same point
    assert np.allclose(o.position(o.period), r0, atol=1e-3)


def test_nodal_regression_rate():
    """RAAN rate at 550 km / 51.6 deg is ~-5 deg/day (analytic value -4.63)."""
    o = CircularOrbit(j2=True)
    deg_per_day = np.degrees(o.raan_dot) * 86400.0
    assert -5.0 < deg_per_day < -4.0, deg_per_day
    assert abs(deg_per_day - (-4.63)) < 0.1, deg_per_day


def test_regression_recovered_from_trajectory():
    """The node measured from (r x v) must drift at exactly raan_dot."""
    o = CircularOrbit(j2=True)

    def raan_from_state(t):
        r, v = o.state(t)
        h = np.cross(r, v)
        node = np.cross([0.0, 0.0, 1.0], h)   # line of nodes
        return np.arctan2(node[1], node[0])

    d = np.degrees(raan_from_state(86400.0) - raan_from_state(0.0))
    analytic = np.degrees(o.raan_dot) * 86400.0
    assert abs(d - analytic) < 1e-3, (d, analytic)


def test_inclination_preserved():
    """J2 secular theory drifts the node but not the inclination."""
    o = CircularOrbit(j2=True)

    def inc_from_state(t):
        r, v = o.state(t)
        h = np.cross(r, v)
        return np.degrees(np.arccos(h[2] / np.linalg.norm(h)))

    assert abs(inc_from_state(0.0) - inc_from_state(5 * 86400.0)) < 1e-6


def test_polar_orbit_no_regression():
    """A 90 deg orbit has cos(i)=0, so the node does not regress."""
    o = CircularOrbit(inclination=90.0 * DEG, j2=True)
    assert abs(o.raan_dot) < 1e-15


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all J2 tests passed")
