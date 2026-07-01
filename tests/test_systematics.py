"""Tests for the systematic (unaveraged) error sources: albedo + field model
error, and their end-to-end effect on the MEKF."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import angle_between  # noqa: E402
from scalar.environment.magfield import (magnetic_field_eci,  # noqa: E402
                                         truth_field_eci)
from scalar.estimator.mekf import run_mekf  # noqa: E402
from scalar.scenario import Scenario  # noqa: E402
from scalar.sensors.sun_sensor import SunSensorArray  # noqa: E402

E = 1361.0


# --- albedo -----------------------------------------------------------------
def test_albedo_biases_sun_toward_nadir():
    arr = SunSensorArray(noise_frac=0.0)          # deterministic
    rng = np.random.default_rng(0)
    sun = np.array([1.0, 0.0, 0.0])
    nadir = np.array([0.0, 0.0, 1.0])
    s0, _ = arr.reconstruct(arr.measure(sun, E, True, rng), E)
    s1, _ = arr.reconstruct(
        arr.measure(sun, E, True, rng, albedo_dir_body=nadir,
                    albedo_irradiance=0.3 * E), E)
    assert np.allclose(s0, sun)                   # no albedo -> exact
    assert s1[2] > 0.05                           # pulled toward nadir (+z)
    assert np.rad2deg(angle_between(s1, sun)) > 2.0


def test_albedo_off_by_default_in_scenario():
    res = Scenario(dt=2.0, duration=1500.0, seed=1).run()
    # default albedo_coeff=0 -> sun error is only noise (~few deg), unbiased mean
    from scalar.attitude import to_body
    lit = res.sun_valid
    errs = [np.rad2deg(angle_between(res.sun_body[k],
            to_body(res.q_true[k], res.sun_eci[k]))) for k in np.where(lit)[0]]
    assert np.median(errs) < 5.0


# --- field model error ------------------------------------------------------
def test_field_error_rotates_by_amplitude():
    r = np.array([6.9e6, 1.0e6, 2.0e6])
    jd = 2460390.0
    B0 = magnetic_field_eci(r, jd)
    for deg in (3.0, 7.0, 10.0):
        Bt = truth_field_eci(r, jd, deg)
        assert abs(np.rad2deg(angle_between(B0, Bt)) - deg) < 0.5
        assert np.isclose(np.linalg.norm(B0), np.linalg.norm(Bt))  # magnitude kept


def test_field_error_zero_is_identity():
    r = np.array([6.9e6, 1.0e6, 2.0e6])
    jd = 2460390.0
    assert np.allclose(magnetic_field_eci(r, jd), truth_field_eci(r, jd, 0.0))


def test_scenario_truth_vs_reference_field():
    res_ideal = Scenario(dt=2.0, duration=1500.0, seed=1).run()
    assert np.allclose(res_ideal.B_eci, res_ideal.B_eci_ref)   # truth == model
    res_err = Scenario(dt=2.0, duration=1500.0, seed=1,
                       field_model_error_deg=7.0).run()
    assert not np.allclose(res_err.B_eci, res_err.B_eci_ref)
    # per-sample truth-vs-reference direction error ~7 deg
    angs = [np.rad2deg(angle_between(a, b))
            for a, b in zip(res_err.B_eci, res_err.B_eci_ref)]
    assert abs(np.median(angs) - 7.0) < 0.6


# --- end-to-end: systematics dominate and degrade the MEKF ------------------
def test_systematics_degrade_mekf_as_expected():
    ideal = run_mekf(Scenario(dt=1.0, seed=3).run()).summary()
    real = run_mekf(Scenario(dt=1.0, seed=3, albedo_coeff=0.3,
                             field_model_error_deg=5.0).run()).summary()
    # unaveraged error enters ~1:1 -> several-degree median, well above ideal
    assert real["median_deg"] > 5 * ideal["median_deg"]
    assert 2.0 < real["median_deg"] < 10.0


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
