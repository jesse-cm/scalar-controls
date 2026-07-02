"""Tests for the real IGRF field model (scalar.environment.igrf).

Validates that the IGRF truth field is physically sane at LEO, that the batch
and scalar evaluations agree, that the frame conventions round-trip, and that
IGRF departs from the tilted-dipole stand-in by the expected non-dipole amount.
"""
import datetime
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scalar.environment.igrf import (igrf_field_eci, igrf_field_eci_batch,
                                      jd_to_datetime)
from scalar.environment.magfield import magnetic_field_eci
from scalar.environment.orbit import CircularOrbit
from scalar.timeutil import julian_date

EPOCH = datetime.datetime(2024, 3, 20, 12, 0, 0)
JD0 = julian_date(EPOCH)


def _trajectory(n=200):
    orb = CircularOrbit()
    ts = np.linspace(0.0, orb.period, n)
    r = np.array([orb.position(t) for t in ts])
    jd = JD0 + ts / 86400.0
    return r, jd


def test_magnitude_at_leo():
    """|B| at 550 km must fall in the physical 15-55 uT band everywhere."""
    r, jd = _trajectory()
    B = igrf_field_eci_batch(r, jd)
    mag_nT = np.linalg.norm(B, axis=1) * 1e9
    assert mag_nT.min() > 15_000.0, mag_nT.min()
    assert mag_nT.max() < 55_000.0, mag_nT.max()


def test_batch_matches_scalar():
    """The vectorised path must agree with per-sample evaluation."""
    r, jd = _trajectory(30)
    Bb = igrf_field_eci_batch(r, jd)
    for i in range(0, len(r), 7):
        Bs = igrf_field_eci(r[i], jd[i])
        rel = np.linalg.norm(Bs - Bb[i]) / np.linalg.norm(Bs)
        assert rel < 1e-6, (i, rel)


def test_jd_datetime_roundtrip():
    """jd_to_datetime must invert julian_date to sub-second accuracy."""
    dt = jd_to_datetime(JD0)
    err = abs((dt - EPOCH).total_seconds())
    assert err < 1.0, err


def test_continuity_along_orbit():
    """Field must vary smoothly: no sample-to-sample jump > 30% of local |B|."""
    r, jd = _trajectory(400)
    B = igrf_field_eci_batch(r, jd)
    dB = np.linalg.norm(np.diff(B, axis=0), axis=1)
    local = np.linalg.norm(B[:-1], axis=1)
    assert np.all(dB < 0.30 * local), float((dB / local).max())


def test_departs_from_dipole():
    """IGRF's non-dipole terms give a several-degree direction error vs dipole."""
    r, jd = _trajectory()
    Bi = igrf_field_eci_batch(r, jd)
    Bd = np.array([magnetic_field_eci(r[i], jd[i]) for i in range(len(r))])
    cos = np.sum(Bi * Bd, axis=1) / (
        np.linalg.norm(Bi, axis=1) * np.linalg.norm(Bd, axis=1))
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    # median a few degrees, but real IGRF has local excursions (SAA) well above.
    assert 3.0 < np.median(ang) < 20.0, np.median(ang)
    assert ang.max() > 15.0, ang.max()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all IGRF tests passed")
