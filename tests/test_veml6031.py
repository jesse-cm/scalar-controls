"""Tests: VEML6031X00 sun-sensor model (saturation at the flight setting)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.constants import SOLAR_CONSTANT_1AU  # noqa: E402
from scalar.sensors.sun_sensor import SunSensorArray  # noqa: E402
from scalar.sensors.veml6031 import COUNT_MAX, VEML6031Array  # noqa: E402

E = SOLAR_CONSTANT_1AU
NOISELESS = SunSensorArray(noise_frac=0.0)


def _counts(it_ms, s):
    v = VEML6031Array(integration_ms=it_ms, base=NOISELESS, dark_counts=0.0)
    return v.measure(s, E, True, np.random.default_rng(0))


def test_face_on_sun_clips_at_25ms():
    c = _counts(25, [1.0, 0.0, 0.0])
    assert c[0] == COUNT_MAX                   # +X face saturates
    assert c[1] == 0                           # -X dark


def test_face_on_sun_fits_at_6p25ms():
    c = _counts(6.25, [1.0, 0.0, 0.0])
    assert abs(c[0] - 38_287) <= 1             # 133,300 lx / 3.4816 lx/count
    assert c[0] < COUNT_MAX


def test_clip_cone_25ms():
    # 57,042 / 133,300 = cos(64.7 deg): inside that cone the face saturates
    for ang, clipped in ((60.0, True), (70.0, False)):
        a = np.radians(ang)
        c = _counts(25, [np.cos(a), np.sin(a), 0.0])
        assert (c[0] == COUNT_MAX) == clipped


def test_clipping_biases_direction_and_6p25_does_not():
    a = np.radians(15.0)
    s = np.array([np.cos(a), np.sin(a), 0.0])
    for it_ms, max_err in ((25, None), (6.25, 0.5)):
        v = VEML6031Array(integration_ms=it_ms, base=NOISELESS, dark_counts=0.0)
        s_hat, ok = v.reconstruct(v.measure(s, E, True, np.random.default_rng(0)), E)
        err = np.degrees(np.arccos(np.clip(s_hat @ s, -1, 1)))
        if max_err is None:
            assert err > 15.0                  # +X clipped at 0.43 of 0.97 -> ~16 deg
        else:
            assert err < max_err


def test_fill_mode_recovers_single_clipped_axis():
    a = np.radians(15.0)
    s = np.array([np.cos(a), np.sin(a), 0.0])
    v = VEML6031Array(integration_ms=25, base=NOISELESS, dark_counts=0.0, clip_mode="fill")
    s_hat, ok = v.reconstruct(v.measure(s, E, True, np.random.default_rng(0)), E)
    assert ok and np.degrees(np.arccos(np.clip(s_hat @ s, -1, 1))) < 0.5


def test_two_minus_z_sensors_averaged():
    v = VEML6031Array(integration_ms=6.25, base=NOISELESS, dark_counts=0.0)
    c = v.measure([0.0, 0.0, -1.0], E, True, np.random.default_rng(0))
    assert c[4] == c[5] and c[6] == 0
    s_hat, ok = v.reconstruct(c, E)
    assert ok and s_hat[2] < -0.999
