"""Tests for the USQUE unscented attitude filter."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import attitude_error_angle, quat_from_rotvec  # noqa: E402
from scalar.estimator.ukf import USQUE, dq_from_grp, grp_from_dq, run_ukf  # noqa: E402
from scalar.scenario import Scenario  # noqa: E402


def test_grp_roundtrip():
    rng = np.random.default_rng(0)
    for a, f in [(1.0, 4.0), (0.0, 1.0), (1.0, 2.0)]:
        for _ in range(50):
            rv = rng.standard_normal(3)
            rv = rv / np.linalg.norm(rv) * rng.uniform(0.0, 2.0)  # up to ~115 deg
            dq = quat_from_rotvec(rv)
            dq2 = dq_from_grp(grp_from_dq(dq, a, f), a, f)
            assert attitude_error_angle(dq2, dq) < 1e-9


def test_usque_converges_within_budget():
    res = Scenario(dt=1.0, seed=3).run()
    s = run_ukf(res).summary()
    assert s["median_deg"] < 3.0
    assert s["p95_deg"] < 10.0
    assert s["lit_median_deg"] < 3.0
    assert s["eclipse_median_deg"] < 10.0


def test_usque_matches_mekf_baseline():
    """Under truth==model both filters should be sub-degree and close."""
    from scalar.estimator.mekf import run_mekf
    res = Scenario(dt=1.0, seed=11).run()
    m = run_mekf(res).summary()["median_deg"]
    u = run_ukf(res).summary()["median_deg"]
    assert abs(m - u) < 1.0
    assert u < 2.0


def test_usque_estimates_bias():
    res = Scenario(dt=1.0, seed=7).run()
    out = run_ukf(res)
    b0 = np.linalg.norm(out.b_est[0] - res.gyro_bias_true[0])
    bN = np.linalg.norm(out.b_est[-1] - res.gyro_bias_true[-1])
    assert bN < b0
    assert np.rad2deg(bN) < 0.15


def test_usque_cold_start_converges():
    """No TRIAD seed (identity + large P) should still converge — the unscented
    transform handles the large initial error."""
    res = Scenario(dt=1.0, seed=5).run()
    s = run_ukf(res, USQUE(), init_triad=False).summary()
    assert s["median_deg"] < 5.0


def test_coarse_dt_divergence_is_the_first_order_transition():
    """At a coarse step with fast tumble the MEKF's *first-order* covariance
    transition diverges — but the exact-F MEKF and the USQUE both stay bounded.
    So the coarse-dt robustness is the exact transition, not the unscented
    machinery per se (a key exploration finding)."""
    from scalar.estimator.mekf import MEKF, run_mekf
    from scalar.montecarlo import randomized_scenario

    first_order_diverged = 0
    for s in range(8):
        sc = randomized_scenario(s, dt=3.0)
        rng = np.random.default_rng(2000 + s)
        ax = rng.standard_normal(3)
        ax /= np.linalg.norm(ax)
        sc.omega0 = np.deg2rad(8.0) * ax          # fast tumble
        res = sc.run()
        w = len(res) // 5
        if run_mekf(res, MEKF(exact_transition=False)).err_deg[w:].max() > 30.0:
            first_order_diverged += 1
        assert run_mekf(res, MEKF(exact_transition=True)).err_deg[w:].max() < 20.0
        assert run_ukf(res).err_deg[w:].max() < 20.0
    assert first_order_diverged >= 1              # first-order does blow up


def test_usque_beats_mekf_from_large_initial_error():
    """The genuine UKF win: from a 120-deg initial error (cold acquisition) the
    sigma-point filter converges far faster than the first-order MEKF, and
    collapsing the UKF sigma spread (alpha->0) reproduces the MEKF — proving the
    two are distinct, correct filters (not accidentally identical)."""
    from scalar.attitude import quat_from_rotvec, quat_mul
    from scalar.constants import DEG
    from scalar.estimator.mekf import MEKF

    res = Scenario(dt=1.0, seed=3).run()
    axis = np.array([1.0, 0.2, -0.3])
    axis /= np.linalg.norm(axis)
    q_bad = quat_mul(quat_from_rotvec(np.deg2rad(120) * axis), res.q_true[0])
    P0 = np.diag(np.concatenate([(90 * DEG) ** 2 * np.ones(3),
                                 (0.5 * DEG) ** 2 * np.ones(3)]))

    def converge(cls, **kw):
        est = cls(q=q_bad.copy(), P=P0.copy(), **kw)
        errs = []
        for k in range(40):
            if k > 0:
                est.propagate(res.gyro_meas[k], res.dt)
            est.update_mag(res.B_eci_ref[k], res.mag_body[k])
            if res.sun_valid[k]:
                est.update_sun(res.sun_eci[k], res.sun_body[k])
            errs.append(np.rad2deg(attitude_error_angle(est.q, res.q_true[k])))
        return np.array(errs)

    em = converge(MEKF)
    eu = converge(USQUE)
    ec = converge(USQUE, alpha=1e-3)              # sigma spread -> 0

    assert eu[15] < 3.0                           # UKF has converged
    assert em[15] > 2 * eu[15]                    # MEKF still far off
    # collapsed UKF tracks the MEKF, not the full UKF
    assert abs(ec[15] - em[15]) < abs(eu[15] - em[15])


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
