"""Integration tests for scalar.scenario — the assembled truth+sensor timeline."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import angle_between, is_unit, to_body  # noqa: E402
from scalar.scenario import Scenario  # noqa: E402


def _run(dt=1.0, duration=None):
    sc = Scenario(dt=dt, seed=7)
    if duration is not None:
        sc.duration = duration
    return sc.run()


def test_shapes_and_alignment():
    res = _run(dt=2.0, duration=1000.0)
    n = len(res)
    for arr, cols in [(res.q_true, 4), (res.omega_true, 3), (res.r_eci, 3),
                      (res.sun_eci, 3), (res.B_eci, 3), (res.gyro_meas, 3),
                      (res.mag_body, 3), (res.sun_body, 3)]:
        assert arr.shape == (n, cols)
    for arr in (res.jd, res.irradiance, res.lit, res.sun_valid):
        assert arr.shape == (n,)


def test_truth_quaternions_unit_and_orbit_radius():
    res = _run(dt=2.0, duration=2000.0)
    assert all(is_unit(q, tol=1e-8) for q in res.q_true)
    radii = np.linalg.norm(res.r_eci, axis=1)
    assert np.allclose(radii, radii[0], rtol=1e-9)   # circular


def test_eclipse_occurs_over_an_orbit():
    """Vernal-equinox epoch, RAAN 0 -> low beta angle -> ~1/3 eclipse."""
    res = _run(dt=1.0)   # full orbit
    frac = res.eclipse_fraction
    assert 0.15 < frac < 0.45
    assert res.lit.any() and (~res.lit).any()


def test_sun_valid_implies_lit():
    res = _run(dt=1.0)
    # a valid sun fix never occurs in eclipse (no noise false-positives)...
    assert np.all(res.sun_valid <= res.lit)
    # ...and every sunlit sample yields a valid fix
    assert res.sun_valid[res.lit].all()
    # invalid sun rows are NaN, valid rows are finite unit vectors
    assert np.all(np.isnan(res.sun_body[~res.sun_valid]))
    assert np.all(np.isfinite(res.sun_body[res.sun_valid]))


def test_measurements_track_truth():
    res = _run(dt=2.0, duration=1500.0)
    # magnetometer body reading ~ rotated true field (within a few sigma)
    for k in range(0, len(res), 50):
        B_true_body = to_body(res.q_true[k], res.B_eci[k])
        assert np.linalg.norm(res.mag_body[k] - B_true_body) < 3e-6  # >>3e-7 noise
    # gyro ~ omega + bias (within a few sigma of ARW at this dt)
    resid = res.gyro_meas - res.omega_true - res.gyro_bias_true
    assert np.all(np.linalg.norm(resid, axis=1) < 5e-3)


def test_sun_sensor_direction_reasonable_when_lit():
    res = _run(dt=2.0, duration=3000.0)
    lit_idx = np.where(res.sun_valid)[0]
    errs = []
    for k in lit_idx[::20]:
        s_true_body = to_body(res.q_true[k], res.sun_eci[k])
        errs.append(np.rad2deg(angle_between(res.sun_body[k], s_true_body)))
    assert np.median(errs) < 8.0   # coarse sun sensor, few-degree class


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
