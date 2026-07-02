"""Tests for the higher-fidelity, calibratable sensor systematics.

Magnetometer: soft-iron (D) turns the field sphere into an ellipsoid; hard-iron
(b) offsets it. Sun sensor: cover-glass FOV roll-off, per-diode gain mismatch,
and physical normal misalignment each add a systematic reconstruction error, and
all default to the ideal sensor (baselines unchanged).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scalar.sensors.magnetometer import Magnetometer
from scalar.sensors.sun_sensor import CUBE_FACE_NORMALS, SunSensorArray

Q_ID = np.array([0.0, 0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# Magnetometer
# --------------------------------------------------------------------------- #
def test_mag_defaults_ideal():
    """Identity soft-iron + zero bias + zero noise -> exact rotation of B."""
    mag = Magnetometer(noise_std=0.0)
    B = np.array([2e-5, -1e-5, 3e-5])
    out = mag.measure(B, Q_ID, np.random.default_rng(0))
    assert np.allclose(out, B, atol=1e-12)


def test_mag_hard_iron_offset():
    """A hard-iron bias appears as a constant additive offset."""
    b = np.array([1e-6, -2e-6, 5e-7])
    mag = Magnetometer(noise_std=0.0, bias=b)
    B = np.array([2e-5, 0.0, 0.0])
    out = mag.measure(B, Q_ID, np.random.default_rng(0))
    assert np.allclose(out - b, B, atol=1e-12)


def test_mag_soft_iron_makes_ellipsoid():
    """A non-identity soft-iron turns a constant-|B| sphere into an ellipsoid."""
    D = np.diag([1.15, 0.9, 1.0])
    mag = Magnetometer(noise_std=0.0, soft_iron=D)
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(500, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    B0 = 3e-5
    mags = np.array([np.linalg.norm(mag.measure(B0 * d, Q_ID, rng)) for d in dirs])
    # true magnitude is constant; distorted magnitude must spread
    assert mags.max() / mags.min() > 1.2, mags.max() / mags.min()


# --------------------------------------------------------------------------- #
# Sun sensor
# --------------------------------------------------------------------------- #
def _incidence_dir(theta_deg, az_deg=37.0):
    th, az = np.radians(theta_deg), np.radians(az_deg)
    return np.array([np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)])


def _recon_error(ss, theta_deg):
    s = _incidence_dir(theta_deg)
    shat, valid = ss.measure_sun_vector(s, 1361.0, True, np.random.default_rng(2))
    assert valid
    return np.degrees(np.arccos(np.clip(np.dot(shat, s), -1.0, 1.0)))


def test_sun_defaults_ideal():
    """Ideal sensor (no noise/gains/rolloff/misalign) reconstructs exactly."""
    ss = SunSensorArray(noise_frac=0.0)
    # residual is the dark-current floor only (~1e-6 deg), far below any real error
    assert _recon_error(ss, 40.0) < 1e-4


def test_rolloff_below_cosine():
    """Roll-off response is strictly below the ideal cosine at high incidence."""
    ss = SunSensorArray(noise_frac=0.0, rolloff=0.4)
    cos60 = np.cos(np.radians(60.0))
    assert ss._angular_response(cos60) < cos60
    # and equals cosine at normal incidence
    assert abs(ss._angular_response(1.0) - 1.0) < 1e-12


def test_each_systematic_adds_error():
    """Gain mismatch, roll-off and misalignment each push recon error above ideal."""
    tn = CUBE_FACE_NORMALS.copy()
    tn[0] = tn[0] + np.array([0.0, 0.05, 0.03])
    tn[0] /= np.linalg.norm(tn[0])
    cases = {
        "gains": SunSensorArray(noise_frac=0.0,
                                gains=np.array([1.12, 0.9, 1.05, 0.95, 1.08, 0.92])),
        "rolloff": SunSensorArray(noise_frac=0.0, rolloff=0.4),
        "misalign": SunSensorArray(noise_frac=0.0, true_normals=tn),
    }
    for name, ss in cases.items():
        assert _recon_error(ss, 45.0) > 0.3, (name, _recon_error(ss, 45.0))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all sensor-systematics tests passed")
