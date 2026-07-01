"""Tests for scalar.sensors.gyro and .magnetometer."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import rand_quat, to_body  # noqa: E402
from scalar.sensors.gyro import Gyro  # noqa: E402
from scalar.sensors.magnetometer import Magnetometer  # noqa: E402


# --- gyro -------------------------------------------------------------------
def test_gyro_ideal_is_exact():
    g = Gyro(rate_noise_density=0.0, bias_rw=0.0, initial_bias=np.zeros(3))
    rng = np.random.default_rng(0)
    w = np.array([0.01, -0.02, 0.03])
    assert np.allclose(g.measure(w, 0.1, rng), w)


def test_gyro_bias_offsets_mean():
    bias = np.array([0.002, -0.001, 0.0005])
    g = Gyro(rate_noise_density=1e-4, bias_rw=0.0, initial_bias=bias)
    rng = np.random.default_rng(1)
    w = np.array([0.05, 0.0, -0.05])
    samples = np.array([g.measure(w, 0.1, rng) for _ in range(20000)])
    assert np.allclose(samples.mean(axis=0), w + bias, atol=2e-4)


def test_gyro_white_noise_scales_with_dt():
    g = Gyro(rate_noise_density=6.6e-5, bias_rw=0.0, initial_bias=np.zeros(3))
    rng = np.random.default_rng(2)
    w = np.zeros(3)
    for dt in (0.01, 0.1):
        s = np.array([g.measure(w, dt, rng) for _ in range(20000)])
        expected = g.rate_noise_density / np.sqrt(dt)
        assert np.isclose(s.std(), expected, rtol=0.05)


def test_gyro_bias_random_walk_grows():
    g = Gyro(rate_noise_density=0.0, bias_rw=1e-4, initial_bias=np.zeros(3))
    dt = 0.1
    n = 5000
    finals = []
    for _ in range(300):
        g.reset()
        rng = np.random.default_rng(1000 + _)
        for _ in range(n):
            g.measure(np.zeros(3), dt, rng)
        finals.append(g.bias.copy())
    finals = np.array(finals)
    expected_std = g.bias_rw * np.sqrt(n * dt)      # RW std after total time
    assert np.isclose(finals.std(), expected_std, rtol=0.1)


def test_gyro_reset_restores_initial_bias():
    b0 = np.array([0.003, 0.0, -0.002])
    g = Gyro(bias_rw=1e-3, initial_bias=b0)
    rng = np.random.default_rng(3)
    for _ in range(100):
        g.measure(np.zeros(3), 0.1, rng)
    assert not np.allclose(g.bias, b0)
    g.reset()
    assert np.allclose(g.bias, b0)


# --- magnetometer -----------------------------------------------------------
def test_mag_ideal_matches_rotation():
    m = Magnetometer(noise_std=0.0)
    rng = np.random.default_rng(4)
    q = rand_quat(rng)
    B_eci = np.array([2e-5, -1e-5, 3e-5])
    assert np.allclose(m.measure(B_eci, q, rng), to_body(q, B_eci))


def test_mag_noise_and_bias():
    bias = np.array([1e-7, -2e-7, 0.5e-7])
    m = Magnetometer(noise_std=3e-7, bias=bias)
    rng = np.random.default_rng(5)
    q = rand_quat(rng)
    B_eci = np.array([2e-5, -1e-5, 3e-5])
    s = np.array([m.measure(B_eci, q, rng) for _ in range(20000)])
    truth = to_body(q, B_eci)
    assert np.allclose(s.mean(axis=0), truth + bias, atol=1e-8)
    assert np.allclose(s.std(axis=0), 3e-7, rtol=0.05)


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
