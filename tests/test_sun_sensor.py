"""Tests for scalar.sensors.sun_sensor."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import angle_between, normalize  # noqa: E402
from scalar.sensors.sun_sensor import SunSensorArray  # noqa: E402

E = 1361.0  # irradiance [W/m^2]


def test_noiseless_reconstruction_along_axis():
    arr = SunSensorArray()
    s = np.array([1.0, 0.0, 0.0])
    c = arr.currents(s, E)
    # +x face lit, -x face dark
    assert c[0] > 0 and c[1] == 0.0
    s_hat, valid = arr.reconstruct(c, E)
    assert valid and np.allclose(s_hat, s, atol=1e-12)


def test_noiseless_reconstruction_generic_direction():
    arr = SunSensorArray()
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = normalize(rng.standard_normal(3))
        c = arr.currents(s, E)
        s_hat, valid = arr.reconstruct(c, E)
        assert valid
        assert np.rad2deg(angle_between(s_hat, s)) < 1e-9


def test_back_faces_are_dark():
    arr = SunSensorArray()
    s = normalize([1.0, 1.0, 1.0])
    c = arr.currents(s, E)
    # the three faces pointing away from the sun read exactly zero
    assert c[1] == 0.0 and c[3] == 0.0 and c[5] == 0.0
    assert c[0] > 0 and c[2] > 0 and c[4] > 0


def test_eclipse_gives_no_valid_measurement():
    arr = SunSensorArray()
    rng = np.random.default_rng(1)
    s = normalize([0.3, -0.7, 0.5])
    c = arr.measure(s, E, lit=False, rng=rng)
    # only dark floor + noise, no directional signal
    s_hat, valid = arr.reconstruct(c, E)
    assert not valid and s_hat is None


def test_full_scale_current_positive_and_scales():
    arr = SunSensorArray(responsivity=0.5, area=1e-5)
    assert arr.full_scale_current(E) > 0
    assert np.isclose(arr.full_scale_current(2 * E), 2 * arr.full_scale_current(E))


def test_noisy_accuracy_is_a_few_degrees():
    """Coarse sun sensor: with 2% per-face noise the reconstructed direction
    should be good to a few degrees (median), and roughly unbiased."""
    arr = SunSensorArray(noise_frac=0.02)
    rng = np.random.default_rng(2)
    s = normalize([1.0, 1.0, 1.0])
    errs = []
    recs = []
    for _ in range(2000):
        s_hat, valid = arr.measure_sun_vector(s, E, lit=True, rng=rng)
        assert valid
        errs.append(np.rad2deg(angle_between(s_hat, s)))
        recs.append(s_hat)
    errs = np.array(errs)
    assert np.median(errs) < 5.0            # typical coarse-sensor accuracy
    assert np.percentile(errs, 95) < 12.0
    mean_dir = normalize(np.mean(recs, axis=0))
    assert np.rad2deg(angle_between(mean_dir, s)) < 1.0   # ~unbiased


def test_noise_scales_error():
    """Lower per-face noise -> tighter reconstruction."""
    s = normalize([1.0, 0.5, 0.2])

    def median_err(nf):
        arr = SunSensorArray(noise_frac=nf)
        rng = np.random.default_rng(3)
        e = [np.rad2deg(angle_between(
            arr.measure_sun_vector(s, E, True, rng)[0], s)) for _ in range(1000)]
        return np.median(e)

    assert median_err(0.005) < median_err(0.04)


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
