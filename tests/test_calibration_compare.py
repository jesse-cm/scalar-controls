"""Tests for the calibration-technique comparison harness.

Verifies the monotone benefit ordering that is the whole point of the study:
each successively richer calibration cannot make attitude knowledge worse, and
the star-tracker-referenced full-matrix recovery is the most accurate.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import calibration_compare as cc  # noqa: E402


def _medians(seed=0):
    out, diag = cc.run_one(seed=seed, n_orbits=1.0, dt=3.0)
    return {r: float(np.median(out[r][0])) for r in cc.REGIMES}, diag


def test_regimes_present():
    med, _ = _medians()
    assert set(med) == set(cc.REGIMES)
    assert all(np.isfinite(v) for v in med.values())


def test_calibration_helps_vs_raw():
    med, _ = _medians()
    # every calibrated regime beats the uncalibrated baseline
    for r in ("on-orbit mag", "both", "ST-referenced"):
        assert med[r] < med["none"], (r, med)


def test_monotone_richer_is_better():
    med, _ = _medians()
    # adding the sun on-ground cal to on-orbit mag does not hurt; ST-ref best
    assert med["both"] <= med["on-orbit mag"] + 0.3
    assert med["ST-referenced"] <= med["both"] + 0.3


def test_st_reference_recovers_full_matrix():
    _, diag = _medians()
    # full soft-iron matrix + hard-iron bias recovered tightly
    assert diag["mag_ref_D_err"] < 5e-3, diag["mag_ref_D_err"]
    assert diag["mag_ref_bias_err_nT"] < 100.0, diag["mag_ref_bias_err_nT"]


def test_ellipsoid_hits_noise_floor():
    _, diag = _medians()
    # scalar residual collapses to a few hundred nT (mag noise floor scale)
    assert diag["mag_ell_resid_nT"] < 800.0, diag["mag_ell_resid_nT"]


def test_sun_gain_recovery():
    _, diag = _medians()
    assert diag["sun_gain_err"] < 0.02, diag["sun_gain_err"]


if __name__ == "__main__":
    test_regimes_present()
    test_calibration_helps_vs_raw()
    test_monotone_richer_is_better()
    test_st_reference_recovers_full_matrix()
    test_ellipsoid_hits_noise_floor()
    test_sun_gain_recovery()
    print("all calibration_compare tests passed")
