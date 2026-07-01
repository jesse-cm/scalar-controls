"""Tests for the MEKF: TRIAD init, Jacobian sign, end-to-end convergence."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import (attitude_error_angle, normalize, quat_from_small_angle,  # noqa: E402
                             quat_mul, rand_quat, to_body)
from scalar.estimator.mekf import (MEKF, discrete_process_noise, run_mekf,  # noqa: E402
                                   skew)
from scalar.estimator.triad import triad  # noqa: E402
from scalar.scenario import Scenario  # noqa: E402


def test_triad_recovers_attitude():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rand_quat(rng)
        r1 = normalize(rng.standard_normal(3))
        r2 = normalize(rng.standard_normal(3))
        if abs(np.dot(r1, r2)) > 0.98:
            continue
        b1, b2 = to_body(q, r1), to_body(q, r2)
        q_hat = triad(r1, b1, r2, b2)
        assert np.rad2deg(attitude_error_angle(q_hat, q)) < 1e-9


def test_measurement_jacobian_sign():
    """d(C(q) r) under a small multiplicative error dtheta must equal
    -[b_pred]x dtheta — the exact H used in update_vector."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rand_quat(rng)
        r = normalize(rng.standard_normal(3))
        dtheta = 1e-5 * rng.standard_normal(3)
        b_pred = to_body(q, r)
        q_true = quat_mul(quat_from_small_angle(dtheta), q)
        b_true = to_body(q_true, r)
        linear = b_pred - skew(b_pred) @ dtheta
        assert np.allclose(b_true, linear, atol=1e-9)


def test_process_noise_psd_symmetric():
    Q = discrete_process_noise(6e-5, 1e-5, 0.5)
    assert np.allclose(Q, Q.T)
    assert np.all(np.linalg.eigvalsh(Q) >= -1e-20)


def test_mekf_converges_within_budget():
    """End-to-end: over one orbit (with an eclipse) the MEKF should hold
    attitude-knowledge error well inside the 10-15 deg budget."""
    res = Scenario(dt=1.0, seed=3).run()
    out = run_mekf(res)
    s = out.summary()
    # sanity: filter tracks, does not diverge
    assert s["median_deg"] < 10.0
    assert s["p95_deg"] < 15.0
    assert s["lit_median_deg"] < 6.0
    # eclipse degrades but must not blow up
    assert s["eclipse_median_deg"] < 15.0


def test_mekf_beats_uncorrected_propagation():
    """The updates must actually help vs dead-reckoning the biased gyro."""
    res = Scenario(dt=1.0, seed=5).run()
    filtered = run_mekf(res).summary()["median_deg"]

    # dead-reckon: propagate truth q0 on the raw (biased) gyro, no updates
    from scalar.attitude import propagate
    q = res.q_true[0].copy()
    errs = []
    for k in range(len(res)):
        if k > 0:
            q = propagate(q, res.gyro_meas[k], res.dt)
        errs.append(np.rad2deg(attitude_error_angle(q, res.q_true[k])))
    deadreckon = float(np.median(errs[len(errs) // 5:]))

    assert filtered < deadreckon


def test_mekf_estimates_gyro_bias():
    res = Scenario(dt=1.0, seed=7).run()
    out = run_mekf(res)
    # by the end, estimated bias should approach the (near-constant) true bias
    b_err0 = np.linalg.norm(out.b_est[0] - res.gyro_bias_true[0])
    b_errN = np.linalg.norm(out.b_est[-1] - res.gyro_bias_true[-1])
    assert b_errN < b_err0
    assert np.rad2deg(b_errN) < 0.1   # within 0.1 deg/s


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
