"""Tests for the attitude-independent on-orbit magnetometer calibration."""
import numpy as np
from scipy.linalg import sqrtm

from scalar.calibration import (
    fit_magnetometer_ellipsoid, apply_mag_calibration, MagCalibration)


def _synthetic(N=3000, seed=0, Dtrue=None, btrue=None, sigma=3e-7):
    rng = np.random.default_rng(seed)
    if Dtrue is None:
        Dtrue = np.array([[1.10, 0.04, -0.03],
                          [0.04, 0.93, 0.02],
                          [-0.03, 0.02, 1.06]])
    if btrue is None:
        btrue = np.array([2.5e-6, -1.8e-6, 3.2e-6])
    dirs = rng.normal(size=(N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    F = rng.uniform(18e-6, 46e-6, N)
    Bbody = dirs * F[:, None]
    m = (Dtrue @ Bbody.T).T + btrue + rng.normal(0, sigma, size=(N, 3))
    return m, F, Dtrue, btrue


def test_recovers_bias():
    m, F, Dtrue, btrue = _synthetic()
    cal = fit_magnetometer_ellipsoid(m, F)
    # bias recovered to well under 0.1 uT
    assert np.linalg.norm(cal.v - btrue) < 1e-7


def test_symmetric_softiron_recovered_to_identity():
    # For symmetric D (no rotation), W D = I exactly (up to sign-free sqrtm).
    m, F, Dtrue, btrue = _synthetic()
    cal = fit_magnetometer_ellipsoid(m, F)
    WD = cal.W @ Dtrue
    assert np.allclose(WD, np.eye(3), atol=2e-2)


def test_general_softiron_recovered_up_to_rotation():
    # Non-symmetric D: W D should be orthogonal (a pure rotation), because the
    # magnitude constraint cannot see an overall rotation.
    Dtrue = np.array([[1.08, 0.06, 0.01],
                      [-0.05, 0.96, 0.03],
                      [0.02, -0.04, 1.03]])
    m, F, _, btrue = _synthetic(Dtrue=Dtrue, seed=5)
    cal = fit_magnetometer_ellipsoid(m, F)
    R = cal.W @ Dtrue
    assert np.allclose(R @ R.T, np.eye(3), atol=3e-2)


def test_scalar_residual_collapses_to_noise_floor():
    sigma = 3e-7
    m, F, _, _ = _synthetic(sigma=sigma)
    raw = np.abs(np.linalg.norm(m, axis=1) - F)
    cal = fit_magnetometer_ellipsoid(m, F)
    Bcal = apply_mag_calibration(m, cal)
    res = np.abs(np.linalg.norm(Bcal, axis=1) - F)
    # calibration removes the bulk of the scalar error ...
    assert res.mean() < 0.2 * raw.mean()
    # ... down to roughly the per-axis noise floor
    assert cal.residual_rms < 5 * sigma


def test_apply_matches_stored_matrices():
    m, F, _, _ = _synthetic(N=50)
    cal = fit_magnetometer_ellipsoid(m, F)
    one = apply_mag_calibration(m[0], cal)
    assert np.allclose(one, cal.W @ (m[0] - cal.v))
    many = apply_mag_calibration(m, cal)
    assert many.shape == m.shape
    assert np.allclose(many[0], one)


def test_constant_F_is_degenerate():
    # A *constant* field magnitude puts every sample on a single ellipsoid, so
    # the rhs = F^2 quadric fit is rank-deficient (scale-free) and must be
    # rejected: the attitude-independent method relies on F varying along the
    # orbit (~18-46 uT), which naturally breaks the degeneracy.
    rng = np.random.default_rng(1)
    N = 2000
    Dtrue = np.diag([1.12, 0.90, 1.05])
    dirs = rng.normal(size=(N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    F0 = 30e-6
    m = (Dtrue @ (dirs * F0).T).T + np.array([1e-6, -0.5e-6, 0.8e-6])
    try:
        fit_magnetometer_ellipsoid(m, F0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for constant-F (degenerate)")


def test_too_few_samples_raises():
    m = np.random.default_rng(0).normal(size=(9, 3))
    try:
        fit_magnetometer_ellipsoid(m, 30e-6)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for <10 samples")


def test_bad_shape_raises():
    try:
        fit_magnetometer_ellipsoid(np.zeros((10, 2)), 30e-6)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong shape")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print("all mag-calibration tests passed")
