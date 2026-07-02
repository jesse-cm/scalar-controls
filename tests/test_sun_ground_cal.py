"""Tests for on-ground sun-sensor response-curve calibration."""
import numpy as np

from scalar.sensors.sun_sensor import SunSensorArray
from scalar.calibration.sun_onground import (
    calibrate_sun_response, sweep_currents, SunResponseCal)


ANGLES = np.linspace(0, 75, 30) * np.pi / 180
E = 1361.0


def test_recovers_gains_and_rolloff_noisefree():
    gains = np.array([1.14, 0.92, 1.05, 0.97, 1.08, 0.90])
    ss = SunSensorArray(gains=gains, rolloff=0.22)
    cur = sweep_currents(ss, ANGLES, E)          # noise-free
    cal = calibrate_sun_response(cur, ANGLES, E, sensor=ss)
    assert np.allclose(cal.gains, gains, atol=1e-6)
    assert np.allclose(cal.rolloff, 0.22, atol=1e-6)


def test_recovers_with_bench_noise():
    gains = np.array([1.10, 0.95, 1.03, 0.98, 1.06, 0.93])
    ss = SunSensorArray(gains=gains, rolloff=0.18, noise_frac=0.005)
    rng = np.random.default_rng(0)
    cur = sweep_currents(ss, ANGLES, E, rng=rng)
    cal = calibrate_sun_response(cur, ANGLES, E, sensor=ss)
    assert np.allclose(cal.gains, gains, atol=0.02)
    assert np.allclose(cal.rolloff, 0.18, atol=0.03)


def test_ideal_sensor_gives_unit_gain_zero_rolloff():
    ss = SunSensorArray()      # ideal
    cur = sweep_currents(ss, ANGLES, E)
    cal = calibrate_sun_response(cur, ANGLES, E, sensor=ss)
    assert np.allclose(cal.gains, 1.0, atol=1e-6)
    assert np.allclose(cal.rolloff, 0.0, atol=1e-6)


def test_correct_currents_restores_balance():
    gains = np.array([1.20, 0.85, 1.10, 0.90, 1.05, 0.95])
    ss = SunSensorArray(gains=gains, rolloff=0.15)
    cur = sweep_currents(ss, ANGLES, E)
    cal = calibrate_sun_response(cur, ANGLES, E, sensor=ss)
    corrected = cal.correct_currents(cur)
    # after dividing by recovered gains, all diodes share one response curve
    fs = ss.full_scale_current(E)
    ideal = np.array([ss._angular_response(np.cos(a)) for a in ANGLES]) * fs
    for i in range(6):
        assert np.allclose(corrected[i], ideal, rtol=0, atol=1e-3 * fs)


def test_full_scale_estimated_when_no_sensor():
    gains = np.array([1.14, 0.92, 1.05, 0.97, 1.08, 0.90])
    ss = SunSensorArray(gains=gains, rolloff=0.20)
    cur = sweep_currents(ss, ANGLES, E)
    # No sensor / no full_scale: the bench irradiance scale is unknown, so only
    # *relative* gains are recoverable (the head-on estimate folds the largest
    # gain into full-scale). Relative gains and roll-off must still be exact.
    cal = calibrate_sun_response(cur, ANGLES, E)
    assert np.allclose(cal.gains / cal.gains.mean(), gains / gains.mean(),
                       atol=0.02)
    assert np.allclose(cal.rolloff, 0.20, atol=0.02)


def test_residual_small_for_model_data():
    ss = SunSensorArray(gains=np.full(6, 1.05), rolloff=0.1)
    cur = sweep_currents(ss, ANGLES, E)
    cal = calibrate_sun_response(cur, ANGLES, E, sensor=ss)
    assert np.all(cal.residual_rms < 1e-6)


def test_too_few_angles_raises():
    ss = SunSensorArray()
    ang = np.array([0.0, 0.3])
    cur = sweep_currents(ss, ang, E)
    try:
        calibrate_sun_response(cur, ang, E, sensor=ss)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for <3 angles")


def test_shape_mismatch_raises():
    ss = SunSensorArray()
    cur = sweep_currents(ss, ANGLES, E)
    try:
        calibrate_sun_response(cur, ANGLES[:-1], E, sensor=ss)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched angles")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print("all sun-ground-cal tests passed")
