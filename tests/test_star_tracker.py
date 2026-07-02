"""Tests for the star-tracker sensor and star-tracker-referenced calibration."""
import numpy as np

from scalar.attitude import (rand_quat, to_body, attitude_error_angle,
                             quat_mul, quat_from_small_angle)
from scalar.sensors.star_tracker import StarTracker
from scalar.calibration.startracker_ref import (
    calibrate_mag_with_reference, calibrate_sun_gains_with_reference,
    MagRefCalibration)

_ARCSEC = np.pi / 180.0 / 3600.0


def test_boresight_normalised_and_frame_orthonormal():
    st = StarTracker(boresight=[0, 0, 3])
    assert np.allclose(np.linalg.norm(st.boresight), 1.0)
    C = st._cam
    assert np.allclose(C.T @ C, np.eye(3), atol=1e-12)
    assert np.allclose(C[:, 2], st.boresight)


def test_noise_level_matches_spec():
    st = StarTracker(sigma_cross=10 * _ARCSEC, roll_factor=6.0)
    rng = np.random.default_rng(0)
    q = rand_quat(rng)
    errs = []
    for _ in range(4000):
        qm, ok = st.measure(q, rng)   # no exclusion vectors -> always valid
        assert ok
        errs.append(attitude_error_angle(qm, q))
    rms = np.sqrt(np.mean(np.square(errs)))
    expected = np.sqrt(2 * st.sigma_cross**2 + st.sigma_roll**2)
    assert abs(rms - expected) / expected < 0.1


def test_roll_axis_is_coarser():
    # error about the boresight should exceed cross-boresight error
    st = StarTracker(boresight=[0, 0, 1], sigma_cross=10 * _ARCSEC, roll_factor=6.0)
    rng = np.random.default_rng(1)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    dth = []
    for _ in range(3000):
        qm, _ = st.measure(q, rng)
        # small-angle vector = 2*vec part
        dq = quat_mul(qm, np.array([0, 0, 0, 1.0]))
        dth.append(2 * dq[:3])
    dth = np.array(dth)
    sx, sy, sz = dth.std(axis=0)
    assert sz > 3 * sx and sz > 3 * sy


def test_sun_exclusion_gates_off():
    st = StarTracker(boresight=[0, 0, 1], sun_exclusion=np.deg2rad(30))
    rng = np.random.default_rng(0)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    # Sun 10 deg off boresight -> inside keep-out -> no solution
    sun = np.array([np.sin(np.deg2rad(10)), 0, np.cos(np.deg2rad(10))])
    qm, ok = st.measure(q, rng, sun_body=sun)
    assert not ok and qm is None
    # Sun 60 deg off -> outside keep-out -> solution returned
    sun2 = np.array([np.sin(np.deg2rad(60)), 0, np.cos(np.deg2rad(60))])
    qm2, ok2 = st.measure(q, rng, sun_body=sun2)
    assert ok2 and qm2 is not None


def test_earth_exclusion_and_rate_gate():
    st = StarTracker(boresight=[0, 0, 1], earth_exclusion=np.deg2rad(25),
                     max_rate=np.deg2rad(1.5))
    rng = np.random.default_rng(0)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    nadir = np.array([np.sin(np.deg2rad(10)), 0, np.cos(np.deg2rad(10))])
    assert not st.measure(q, rng, nadir_body=nadir)[1]
    # high slew rate blurs stars
    assert not st.measure(q, rng, rate=np.deg2rad(3.0))[1]
    # benign geometry + slow rate -> valid
    assert st.measure(q, rng, rate=np.deg2rad(0.5))[1]


# --- star-tracker-referenced calibration --------------------------------

def _mag_ref_data(N=600, seed=0):
    rng = np.random.default_rng(seed)
    D = np.array([[1.10, 0.04, -0.03],
                  [0.05, 0.93, 0.02],
                  [-0.02, 0.03, 1.06]])
    b = np.array([2.5e-6, -1.8e-6, 3.2e-6])
    q = np.array([rand_quat(rng) for _ in range(N)])
    dirs = rng.normal(size=(N, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    F = rng.uniform(18e-6, 46e-6, N)
    B_eci = dirs * F[:, None]
    B_body = np.array([to_body(q[k], B_eci[k]) for k in range(N)])
    m = (D @ B_body.T).T + b + rng.normal(0, 3e-7, size=(N, 3))
    return m, q, B_eci, D, b


def test_mag_reference_recovers_full_matrix():
    m, q, B_eci, D, b = _mag_ref_data()
    cal = calibrate_mag_with_reference(m, q, B_eci)
    # full D recovered (including rotation, unlike the ellipsoid fit)
    assert np.max(np.abs(cal.D - D)) < 0.01
    assert np.linalg.norm(cal.b - b) < 1e-7


def test_mag_reference_correction_reduces_direction_error():
    m, q, B_eci, D, b = _mag_ref_data()
    cal = calibrate_mag_with_reference(m, q, B_eci)
    B_body = np.array([to_body(q[k], B_eci[k]) for k in range(len(m))])
    Bcal = cal.apply(m)

    def de(a, bb):
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        bb = bb / np.linalg.norm(bb, axis=1, keepdims=True)
        return np.degrees(np.arccos(np.clip(np.sum(a * bb, 1), -1, 1)))
    assert np.median(de(Bcal, B_body)) < 0.2 * np.median(de(m, B_body))


def test_mag_reference_too_few_samples():
    m, q, B_eci, _, _ = _mag_ref_data(N=3)
    try:
        calibrate_mag_with_reference(m, q, B_eci)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for <4 samples")


def test_sun_gains_with_reference():
    from scalar.sensors.sun_sensor import SunSensorArray
    rng = np.random.default_rng(2)
    gains = np.array([1.14, 0.92, 1.05, 0.97, 1.08, 0.90])
    ss = SunSensorArray(gains=gains)
    E = 1361.0
    N = 1500
    q = np.array([rand_quat(rng) for _ in range(N)])
    sd = rng.normal(size=(N, 3)); sd /= np.linalg.norm(sd, axis=1, keepdims=True)
    fs = ss.full_scale_current(E)
    currents = np.empty((N, 6))
    for k in range(N):
        currents[k] = ss.measure(to_body(q[k], sd[k]), E, True, rng)
    ghat = calibrate_sun_gains_with_reference(currents, q, sd, ss.normals, fs)
    # relative gains recovered (absolute scale tied to full_scale estimate)
    assert np.allclose(ghat / ghat.mean(), gains / gains.mean(), atol=0.05)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print("all star-tracker tests passed")
