"""Tests: the harness returns exactly what the wrapped simulators return, and the ledger keeps its fixed columns.

Every equivalence test builds the reference the way the experiment script does
(direct calls into scalar.*) on a shortened run, and compares with ``==``.
"""
import datetime as dt
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import IDENTITY_QUAT, quat_from_rotvec  # noqa: E402
from scalar.constants import DEG  # noqa: E402
from scalar.control.periodic_lqr import (PeriodicLQR, TwoModeController, bryson_weights,  # noqa: E402
                                         periodic_lqr_schedule, quantize_schedule)
from scalar.control.tune import (ClosedLoopResult, actuators_from_matrix,  # noqa: E402
                                 simulate_closed_loop_estimated, tune_lqr_weights)
from scalar.dynamics import RigidBody  # noqa: E402
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.harness import (DOC_CHECKS, FIXED_METRICS, LEDGER_COLUMNS, SCRIPT_SEEDS, SEEDS, Metrics,  # noqa: E402
                            ScenarioSpec, append_rows, baseline_specs, ledger_row, read_ledger,
                            rl_tune_specs, run_case, run_cases, run_matrix)
from scalar.harness import runner  # noqa: E402
from scalar.harness.baseline import DocCheck, projected_weight_draws  # noqa: E402
from scalar.harness.gate import gate_seeds, gate_table, matches  # noqa: E402
from scalar.harness.ledger import format_seed_set, parse_seed_set  # noqa: E402
from scalar.harness.metrics import COIL_VOLTAGE, actuation_stats, sunlit_mask  # noqa: E402
from scalar.learning.sun_mlp import SunMLP, angular_error_deg, flight_like_sensor, make_dataset  # noqa: E402
from scalar.sysid.magnetorquer_id import s3_coil_matrix  # noqa: E402
from scalar.timeutil import julian_date  # noqa: E402

M_RES = np.array([2e-3, -1e-3, 0.5e-3])
Q_FAR, W_FAR = np.array([0.5, 0.5, 0.5, 0.5]), np.deg2rad([1.0, -0.6, 0.8])
Q_NEAR = quat_from_rotvec(np.deg2rad(20) * np.array([0.6, 0.8, 0.0]))
W_NEAR = np.deg2rad([0.1, -0.05, 0.08])


def _spec(pattern):
    hits = [s for s in baseline_specs() if pattern in s.key]
    assert len(hits) == 1, (pattern, [s.key for s in hits])
    return hits[0]


def _closed_loop_setup():
    body = RigidBody()
    M, i_max = s3_coil_matrix()
    acts = actuators_from_matrix(M, i_max)
    orbit = CircularOrbit(inclination=98.61 * DEG)
    jd0 = julian_date(dt.datetime(2026, 9, 11, 12))
    return body, acts, i_max, orbit, jd0


def _k256(body, acts, orbit, jd0, th=10.0, wm=0.2, rho=0.3):
    Q, R = bryson_weights(th * DEG, wm * DEG, acts.m_max_vector, rho)
    K, _ = periodic_lqr_schedule(body.inertia, orbit, jd0, IDENTITY_QUAT, 1.0, Q, R)
    return quantize_schedule(K, 256)


def _raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except ValueError:
        return True
    return False


def _fake_mlp(extra, seed=0):
    rng = np.random.default_rng(seed)
    n_in = 10 if extra else 6
    w = [(rng.standard_normal((8, n_in)).astype(np.float32), rng.standard_normal(8).astype(np.float32)),
         (rng.standard_normal((3, 8)).astype(np.float32), rng.standard_normal(3).astype(np.float32))]
    return SunMLP(w, extra)


# --------------------------------------------------------------------------- ledger


def test_ledger_columns_are_the_plans_fixed_set():
    assert LEDGER_COLUMNS[:6] == ("scenario", "estimator", "controller", "seed_set", "git_hash", "date")
    assert FIXED_METRICS == ("know_med_all", "know_p95_all", "know_med_lit", "know_p95_lit", "know_med_ecl",
                             "know_p95_ecl", "point_mean", "point_p95", "settle_h", "sat_frac", "duty_ch0",
                             "duty_ch1", "duty_ch2", "coil_power_w", "max_rate_deg_s", "diverged")
    metric_cols = LEDGER_COLUMNS[6:]
    assert metric_cols[-1] == "diverged__frac" and len(metric_cols) == 2 * 15 + 1
    assert all(c.endswith(("__med", "__p95")) for c in metric_cols[:-1])
    assert tuple(SEEDS) == tuple(range(40))


def test_ledger_row_aggregates_across_seeds_and_leaves_inapplicable_cells_empty():
    spec = _spec("e4D_acq120_mekf")
    ms = [Metrics(point_mean=float(i), settle_h=(np.inf if i >= 38 else 2.0), diverged=float(i >= 36))
          for i in range(40)]
    row = ledger_row(spec, range(40), ms, git_hash="abc1234", date="2026-01-01")
    assert row["seed_set"] == "0-39" and row["scenario"] == "e4D_acq120_mekf"
    assert row["point_mean__med"] == np.median(np.arange(40.0))
    assert row["point_mean__p95"] == np.percentile(np.arange(40.0), 95)
    assert row["settle_h__med"] == 2.0 and row["settle_h__p95"] == np.inf     # 2 of 40 never settled
    assert row["diverged__frac"] == 0.1
    assert np.isnan(row["know_med_all__med"])                                 # not measured -> empty
    path = os.path.join(tempfile.mkdtemp(), "ledger.csv")
    append_rows(path, [row])
    append_rows(path, [ledger_row(spec, [2], ms[2:3], git_hash="abc1234", date="2026-01-01")])
    with open(path) as f:
        lines = f.read().splitlines()
    assert lines[0].split(",") == list(LEDGER_COLUMNS) and len(lines) == 3
    back = read_ledger(path)
    assert back[0]["settle_h__p95"] == np.inf and np.isnan(back[0]["know_med_all__med"])
    assert back[1]["seed_set"] == "2" and back[1]["point_mean__med"] == back[1]["point_mean__p95"] == 2.0
    with open(path, "w") as f:
        f.write("scenario,other\n")
    assert _raises(append_rows, path, [row])                                  # never appends under a foreign header


def test_seed_set_format_roundtrip():
    for seeds in (list(range(40)), [2], [100, 101, 102], [0, 5, 9]):
        assert parse_seed_set(format_seed_set(seeds)) == seeds
    assert format_seed_set(range(40)) == "0-39" and format_seed_set([0, 5, 9]) == "0,5,9"


# --------------------------------------------------------------------------- spec


def test_validate_rejects_switches_the_wrapped_code_does_not_have():
    cl = _spec("e4D_acq120_mekf")
    assert cl.validate() is cl
    assert _raises(cl.replace(field_truth="igrf").validate)        # tune.py has a dipole truth only
    assert _raises(cl.replace(mekf_n_iter=1).validate)             # tune.py hard-codes n_iter=3
    assert _raises(cl.replace(controller="none").validate)
    est = _spec("e2_alongtrack_20km")
    assert _raises(est.replace(controller="two_mode").validate)
    assert _raises(est.replace(m_res=(1e-3, 0.0, 0.0)).validate)
    assert _raises(lambda: run_cases([cl, cl], [0], workers=1))    # one row per (scenario, estimator, controller)


def test_sensor_overrides_reach_the_wrapped_objects_and_defaults_stay_the_class_defaults():
    from scalar.sensors.gyro import Gyro
    from scalar.sensors.magnetometer import Magnetometer
    s = ScenarioSpec(name="x", controller="periodic_lqr")
    g, m = runner._gyro(s), runner._magnetometer(s)
    assert (g.rate_noise_density, g.bias_rw) == (Gyro().rate_noise_density, Gyro().bias_rw)
    assert np.array_equal(g.initial_bias, Gyro().initial_bias)
    assert m.noise_std == Magnetometer().noise_std and not m.bias.any() and np.array_equal(m.soft_iron, np.eye(3))
    s2 = s.replace(gyro_bias_rw=0.0, gyro_noise_density=1e-6, gyro_initial_bias_deg_s=(0.0, 0.0, 0.0),
                   mag_noise_std=1e-9, mag_bias_T=(1e-6, 0.0, 0.0), sun_noise_frac=0.5, mekf_sigma_sun=0.2)
    g, m = runner._gyro(s2), runner._magnetometer(s2)
    assert (g.bias_rw, g.rate_noise_density, m.noise_std) == (0.0, 1e-6, 1e-9)
    assert not g.bias.any() and m.bias[0] == 1e-6
    assert runner._sun_sensor(s2).noise_frac == 0.5 and runner._mekf(s2).sigma_sun == 0.2
    unit = runner._sun_sensor(s.replace(sun_unit="flight_like"))
    ref = flight_like_sensor(np.random.default_rng(1))
    assert np.array_equal(unit.gains, ref.gains) and np.array_equal(unit.true_normals, ref.true_normals)


# --------------------------------------------------------------------------- metrics


def test_actuation_metrics_follow_the_existing_definitions():
    _, acts, i_max, orbit, jd0 = _closed_loop_setup()
    mm = acts.m_max_vector
    dip = np.array([[mm[0], 0, 0], [-0.5 * mm[0], 0.25 * mm[1], 0], [0, 0, -mm[2]], [0, 0, 0]])
    r = ClosedLoopResult(np.arange(4.0), np.zeros(4), np.zeros(4), np.array([0.1, 0.4, 0.2, 0.3]), dip, 0.5)
    a = actuation_stats(r, acts, i_max)
    assert a["sat_frac"] == 0.5 and a["max_rate_deg_s"] == 0.4
    assert np.allclose([a["duty_ch0"], a["duty_ch1"], a["duty_ch2"]], [0.375, 0.0625, 0.25])
    assert np.isclose(a["coil_power_w"], np.sum(np.array([0.375, 0.0625, 0.25]) * i_max * COIL_VOLTAGE))
    assert actuation_stats(r, acts, i_max, slice(-2, None))["sat_frac"] == 0.5
    lit = sunlit_mask(orbit, jd0, np.arange(0.0, orbit.period, 10.0))
    assert 0.55 < lit.mean() < 0.72                                 # roughly a third of this orbit is eclipse


# --------------------------------------------------------------------------- equivalence with the scripts' calls


def test_closed_loop_two_mode_equals_the_lqr_tune_call():
    body, acts, _, orbit, jd0 = _closed_loop_setup()
    P = orbit.period
    ctrl = TwoModeController(PeriodicLQR(_k256(body, acts, orbit, jd0), P, acts, IDENTITY_QUAT))
    r = simulate_closed_loop_estimated(body, acts, orbit, jd0, q0=Q_FAR, omega0=W_FAR, n_orbits=0.05, seed=2,
                                       controller=ctrl, m_res=M_RES)                      # lqr_tune.py block [D]
    m = run_case(_spec("e4D_acq120_mekf").replace(n_orbits=0.05), 2)
    last = slice(-int(P), None)
    assert m.point_mean == np.mean(r.pointing_err_deg[last])
    assert m.point_p95 == np.percentile(r.pointing_err_deg[last], 95)
    assert m.know_med_all == np.median(r.knowledge_err_deg[last])
    assert m.sat_frac == r.sat_frac and m.max_rate_deg_s == r.rate_deg_s.max()
    assert m.settle_h == r.settle_time() / 3600.0 and m.diverged == 1.0
    assert m.wall_s > 0
    other = run_case(_spec("e4D_acq120_mekf").replace(n_orbits=0.05), 3)
    assert other.know_med_all != m.know_med_all                     # the seed reaches the sensor noise


def test_closed_loop_flight_unit_equals_the_closed_loop_nn_call_and_scores_the_last_orbit():
    body, acts, _, orbit, jd0 = _closed_loop_setup()
    P = orbit.period
    sensor = flight_like_sensor(np.random.default_rng(1))
    ctrl = PeriodicLQR(_k256(body, acts, orbit, jd0), P, acts, IDENTITY_QUAT)
    r = simulate_closed_loop_estimated(body, acts, orbit, jd0, q0=Q_NEAR, omega0=W_NEAR, n_orbits=1.03, seed=2,
                                       controller=ctrl, m_res=M_RES, albedo_coeff=0.3, sun_sensor=sensor,
                                       sun_front=None)                                    # closed_loop_nn.py
    spec = _spec("e5_fine20_alb0.3 | mekf+analytic | periodic_lqr th=10deg w=0.2dps").replace(n_orbits=1.03)
    m = run_case(spec, 2)
    last = slice(-int(P), None)
    assert len(r.t) > int(P)                                        # the slice really drops the first samples
    pe, ke = r.pointing_err_deg[last], r.knowledge_err_deg[last]
    assert (m.point_mean, m.point_p95) == (np.mean(pe), np.percentile(pe, 95))
    assert (m.know_med_all, m.know_p95_all) == (np.median(ke), np.percentile(ke, 95))
    lit = sunlit_mask(orbit, jd0, r.t[last])
    assert 0 < lit.sum() < lit.size
    assert m.know_med_lit == np.median(ke[lit]) and m.know_p95_ecl == np.percentile(ke[~lit], 95)
    assert m.sat_frac == r.sat_frac and m.diverged == float(pe.max() > 90.0)
    m_last = run_case(spec.replace(actuation_window="last_orbit"), 2)
    assert m_last.max_rate_deg_s == r.rate_deg_s[last].max() and m_last.point_mean == m.point_mean


def test_closed_loop_truth_feedback_ignores_the_seed_and_reports_no_knowledge():
    spec = _spec("e4B_near20_truth | truth | periodic_lqr th=10deg w=0.2dps rho=0.3 bins=256").replace(n_orbits=0.05)
    a, b = run_case(spec, 0), run_case(spec, 7)
    assert a.fixed()["point_mean"] == b.fixed()["point_mean"] and a.max_rate_deg_s == b.max_rate_deg_s
    assert np.isnan(a.know_med_all) and np.isnan(a.know_p95_ecl)


def test_closed_loop_mlp_front_end_is_wired_like_closed_loop_nn():
    body, acts, _, orbit, jd0 = _closed_loop_setup()
    P = orbit.period
    spec = _spec("e5_fine20_alb0.3 | mekf+mlp_b | periodic_lqr th=10deg w=0.2dps").replace(n_orbits=0.05)
    mlp = _fake_mlp(extra=True)
    saved = dict(runner._MLP_CACHE)
    runner._MLP_CACHE[runner.mlp_key(spec)] = mlp                   # stand-in weights: no torch training here
    try:
        m = run_case(spec, 2)
    finally:
        runner._MLP_CACHE.clear()
        runner._MLP_CACHE.update(saved)
    sensor = flight_like_sensor(np.random.default_rng(1))
    front = lambda c, E, nb, ds: mlp.reconstruct(c, E, sensor, np.concatenate([nb, [ds]]))   # noqa: E731
    r = simulate_closed_loop_estimated(body, acts, orbit, jd0, q0=Q_NEAR, omega0=W_NEAR, n_orbits=0.05, seed=2,
                                       controller=PeriodicLQR(_k256(body, acts, orbit, jd0), P, acts, IDENTITY_QUAT),
                                       m_res=M_RES, albedo_coeff=0.3, sun_sensor=sensor, sun_front=front)
    assert m.point_mean == np.mean(r.pointing_err_deg) and m.know_med_all == np.median(r.knowledge_err_deg)


def test_projected_lqr_specs_are_the_tuners_draws_and_runs():
    body, acts, _, orbit, jd0 = _closed_loop_setup()
    out = tune_lqr_weights(body, acts, orbit, jd0, Q_FAR, W_FAR, n_candidates=4, n_orbits=0.005)   # lqr_tune.py [A]
    draws = projected_weight_draws(4, 0)
    assert sorted(draws) == sorted((w["q_weight"], w["w_weight"], w["r_weight"]) for _, w, _ in out)
    specs = [s for s in baseline_specs(["4"]) if s.controller == "projected_lqr"]
    assert [(s.proj_q_weight, s.proj_w_weight, s.proj_r_weight) for s in specs] == draws
    _, w, r = out[0]
    spec = [s for s in specs if s.proj_q_weight == w["q_weight"]][0].replace(n_orbits=0.005)
    m = run_case(spec, 0)
    assert m.point_mean == np.mean(r.pointing_err_deg) and m.sat_frac == r.sat_frac


def test_three_mode_specs_equal_rl_tune_evaluate():
    from scalar.control.rl_tune import default_plant, evaluate
    params = np.array([10.0, 0.2, 0.3, 0.3, 1.5, 5e3, 12.0, 0.3, 0.5])
    ref = evaluate(params, default_plant(mag_calibrated=True), seed=1, n_orbits_acq=0.05, n_orbits_fine=0.05)
    acq, fine = rl_tune_specs(params, mag_calibrated=True, n_orbits_acq=0.05, n_orbits_fine=0.05)
    ma, mf = run_case(acq, 1), run_case(fine, 1)
    assert ma.extra["cal_dir_err_med"] == ref["cal_dir_err_med"]
    assert ma.settle_h == ref["acq_settle_h"] and ma.diverged == ref["acq_diverged"]
    assert ma.point_mean == ref["acq_point_mean"]
    assert (mf.point_mean, mf.point_p95) == (ref["fine_point_mean"], ref["fine_point_p95"])
    assert mf.max_rate_deg_s == ref["fine_max_rate"] and mf.sat_frac == ref["fine_sat_frac"]
    assert np.isclose(mf.coil_power_w, ref["fine_power_w"], rtol=1e-12, atol=0.0)
    assert mf.know_med_all == ref["fine_knowledge_med"] and mf.diverged == ref["fine_diverged"]


def test_estimation_equals_the_absolute_without_gps_call():
    from scalar.environment.magfield import magnetic_field_eci
    from scalar.estimator.mekf import MEKF, run_mekf
    from scalar.scenario import Scenario
    orbit = CircularOrbit(inclination=98.61 * DEG, j2=True)
    res = Scenario(orbit=orbit, dt=1.0, seed=5, duration=0.1 * orbit.period).run()
    shift = 20.0 * 1e3 / (orbit.radius * orbit.mean_motion)
    ref = np.array([magnetic_field_eci(orbit.position(t + shift), jd) for t, jd in zip(res.t, res.jd)])
    ang = np.degrees(np.arccos(np.clip(np.sum(ref * res.B_eci, 1) / np.linalg.norm(ref, axis=1)
                                       / np.linalg.norm(res.B_eci, axis=1), -1, 1)))
    out = run_mekf(res.__class__(**{**res.__dict__, "B_eci_ref": ref}), MEKF(n_iter=3)).summary()
    m = run_case(_spec("e2_alongtrack_20km").replace(n_orbits=0.1), 5)
    assert (m.know_med_all, m.know_p95_all) == (out["median_deg"], out["p95_deg"])
    assert m.extra["field_dir_err_med"] == np.median(ang) and m.extra["clock_equiv_s"] == shift
    assert np.isnan(m.point_mean) and np.isnan(m.coil_power_w)      # no controller: nothing invented


def test_mekf_front_loop_equals_run_mekf_for_the_analytic_front():
    from scalar.estimator.mekf import MEKF, run_mekf
    from scalar.scenario import Scenario
    sensor = flight_like_sensor(np.random.default_rng(1))
    res = Scenario(dt=1.0, sun_sensor=sensor, albedo_coeff=0.3, seed=3, duration=600.0).run()   # nn_sun_sensor.py
    err = runner._mekf_with_front(res, MEKF(n_iter=3), lambda c, E, nb, ds: sensor.reconstruct(c, E))
    assert np.array_equal(err, run_mekf(res, MEKF(n_iter=3)).err_deg)
    m = run_case(_spec("e1_mekf_alb0.3 | mekf+analytic").replace(n_orbits=600.0 / CircularOrbit().period), 3)
    assert len(err) == 601 and m.know_med_all == np.median(err[len(err) // 5:])


def test_sysid_equals_the_sysid_magnetorquer_call():
    from scalar.sysid.magnetorquer_id import (PulseSchedule, identify_dipole_matrix,
                                              identify_dipole_matrix_impulse, simulate_burst_campaign)
    M_true, i_max = s3_coil_matrix()
    body = RigidBody()
    sched = PulseSchedule(3, i_max, n_pulses=3, duration=2.0)
    d = simulate_burst_campaign(M_true, M_RES, sched, body=body, sample_hz=10.0, seed=4)
    Md, *_ = identify_dipole_matrix(body.inertia, d["omega"], d["omega_dot"], d["B_body"], d["currents"])
    Mi, mr, _, _ = identify_dipole_matrix_impulse(body.inertia, d["t"], d["omega"], d["B_body"], d["currents"],
                                                  window=1.0)
    rel = lambda M: np.linalg.norm(M - M_true) / np.linalg.norm(M_true)   # noqa: E731
    m = run_case(_spec("e3_pulse2s_10Hz_n90").replace(n_pulses=3), 4)
    assert m.extra["M_rel_err_per_sample"] == rel(Md) and m.extra["M_rel_err_impulse"] == rel(Mi)
    assert m.extra["M_col_err_z"] == np.linalg.norm(Mi[:, 2] - M_true[:, 2]) / np.linalg.norm(M_true[:, 2])
    assert m.extra["m_res_err_mAm2"] == np.linalg.norm(mr - M_RES) * 1e3
    assert all(np.isnan(v) for v in m.fixed().values())             # no fixed metric applies to sysid


def test_sun_vector_equals_the_nn_sun_sensor_call():
    sensor = flight_like_sensor(np.random.default_rng(1))
    Xc, Xe, y = make_dataset(sensor, 300, np.random.default_rng(7), albedo_max=0.3)
    fs = sensor.full_scale_current(1361.0)
    e = angular_error_deg(np.array([sensor.reconstruct(c * fs, 1361.0)[0] for c in Xc]), y)
    m = run_case(_spec("e1_sunvec_alb0.3 | sunvec:analytic").replace(dataset_n=300), 7)
    assert (m.extra["sun_err_med"], m.extra["sun_err_p95"]) == (np.median(e), np.percentile(e, 95))
    mlp = _fake_mlp(extra=True)
    spec = _spec("e1_sunvec_alb0.3 | sunvec:mlp_b").replace(dataset_n=300)
    saved = dict(runner._MLP_CACHE)
    runner._MLP_CACHE[runner.mlp_key(spec)] = mlp
    try:
        mb = run_case(spec, 7)
    finally:
        runner._MLP_CACHE.clear()
        runner._MLP_CACHE.update(saved)
    assert mb.extra["sun_err_med"] == np.median(angular_error_deg(mlp(Xc, Xe), y))


# --------------------------------------------------------------------------- matrix, baseline, gate


def test_run_matrix_rows_and_parallel_equals_serial():
    specs = [_spec("e2_alongtrack_20km").replace(n_orbits=0.03),
             _spec("e1_sunvec_alb0.3 | sunvec:analytic").replace(dataset_n=100)]
    seeds = [0, 1, 2]
    serial = run_cases(specs, seeds, workers=1)
    par = run_cases(specs, seeds, workers=2)
    assert set(serial) == set(par) == {(s.key, sd) for s in specs for sd in seeds}
    for k in serial:
        assert serial[k].extra == par[k].extra
        assert np.array_equal(list(serial[k].fixed().values()), list(par[k].fixed().values()), equal_nan=True)
    rows = run_matrix(specs, seeds, workers=1, git_hash="abc1234", date="2026-01-01")
    assert [r["scenario"] for r in rows] == [s.name for s in specs] and rows[0]["seed_set"] == "0-2"
    assert set(rows[0]) == set(LEDGER_COLUMNS)
    assert rows[0]["know_med_all__med"] == np.median([serial[(specs[0].key, sd)].know_med_all for sd in seeds])
    assert rows[0]["know_med_all__p95"] >= rows[0]["know_med_all__med"]
    assert np.isnan(rows[1]["know_med_all__med"])                   # sun-vector error has no ledger column


def test_baseline_mirrors_the_five_scripts():
    specs = baseline_specs()
    keys = [s.key for s in specs]
    assert len(keys) == len(set(keys)) == 62
    assert {s.experiment for s in specs} == {"1", "2", "3", "4", "5"}
    assert [len(baseline_specs([e])) for e in "12345"] == [15, 7, 8, 20, 12]
    assert all(c.key in keys for c in DOC_CHECKS) and set(SCRIPT_SEEDS) == set(keys)
    # The plan assumes seed 2 produced the doc. Only lqr_tune [D] and closed_loop_nn use it.
    by_exp = {e: {SCRIPT_SEEDS[s.key] for s in baseline_specs([e])} for e in "12345"}
    assert by_exp == {"1": {7, 3}, "2": {5}, "3": {40, 100, 130, 190}, "4": {0, 2}, "5": {2}}
    g = gate_seeds(specs)
    assert g[_spec("e3_pulse2s_10Hz_n90").key] == [1, 2, 190] and g[_spec("e4D_acq120_mekf").key] == [2]


def test_experiment_6_is_the_tuned_files_and_only_runs_when_asked_for():
    import contextlib
    import importlib.util
    import io
    import json
    from scalar.control.rl_tune import default_plant, evaluate, load_params
    from scalar.harness.baseline import TUNED_FILES, select_specs, tuned_controller_specs
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Not in the default selection: the baseline and the gate see the same 62 scenarios as before.
    assert [s.key for s in select_specs()] == [s.key for s in baseline_specs()]
    assert [s.key for s in select_specs("12345")] == [s.key for s in baseline_specs()]
    assert baseline_specs(["6"]) == [] and not any("s6_" in k for k in SCRIPT_SEEDS)
    six = select_specs(["6"])
    assert six == tuned_controller_specs() and select_specs(["5", "6"]) == baseline_specs(["5"]) + six
    assert len({s.key for s in six}) == len(six) == 2 * len(TUNED_FILES) == 10
    assert {s.experiment for s in six} == {"6"} and all(s.validate() is s for s in six)

    # Each file as it says it was produced: its parameters, calibration pass iff it stored a calibration result.
    for i, stem in enumerate(TUNED_FILES):
        with open(os.path.join(root, "results", stem + ".json")) as f:
            d = json.load(f)
        cal = "cal_dir_err_med" in d["train_metrics"]
        acq, fine = six[2 * i], six[2 * i + 1]
        tag = ("calmag_" if cal else "rawmag_") + stem
        assert (acq.name, fine.name) == ("s6_acq120_" + tag, "s6_fine20_" + tag)
        assert acq.mag_calibrated == fine.mag_calibrated == cal
        assert (acq.start_mode, fine.start_mode, acq.n_orbits, fine.n_orbits) == (1, 3, 4, 3)
        p = d["params"]
        got = (acq.theta_max_deg, acq.omega_max_deg_s, acq.rho, acq.enter_deg_s, acq.exit_deg_s, acq.k_damp)
        assert got == tuple(p[k] for k in ("theta_max", "omega_max", "rho", "enter", "exit", "k_damp"))
        assert acq.fine_weights == fine.fine_weights == (p.get("fine_theta_max", p["theta_max"]),
                                                         p.get("fine_omega_max", p["omega_max"]),
                                                         p.get("fine_rho", p["rho"]))
    by_file = {stem: six[2 * i] for i, stem in enumerate(TUNED_FILES)}
    assert [by_file[s].mag_calibrated for s in TUNED_FILES] == [False, False, True, True, True]
    assert by_file["lqr_tuned2"].controller_label == by_file["lqr_tuned_cal"].controller_label   # same parameters

    # Experiment 6L: the acquisition episode of two files at 6 orbits, under its own id and scenario names.
    from scalar.harness.baseline import LONG_ACQ_FILES, long_acquisition_specs
    long = select_specs(["6L"])
    assert long == long_acquisition_specs() and select_specs(["6", "6L"]) == six + long
    assert LONG_ACQ_FILES == ("lqr_tuned_3mode", "lqr_tuned_ms") and len(long) == len(LONG_ACQ_FILES)
    for stem, s in zip(LONG_ACQ_FILES, long):
        assert s.validate() is s and s.experiment == "6L" and s.name == "s6_acq120x6orb_calmag_" + stem
        assert s == by_file[stem].replace(experiment="6L", name=s.name, n_orbits=6)   # nothing else differs
    assert not {s.key for s in long} & {s.key for s in six}

    # A six-parameter file without the calibration pass equals rl_tune.evaluate (the nine-parameter,
    # calibrated case is test_three_mode_specs_equal_rl_tune_evaluate).
    params, _ = load_params(os.path.join(root, "results", "lqr_tuned.json"))
    ref = evaluate(params, default_plant(mag_calibrated=False), seed=1, n_orbits_acq=0.05, n_orbits_fine=0.05)
    ma, mf = (run_case(s.replace(n_orbits=0.05), 1) for s in six[:2])
    assert "cal_dir_err_med" not in ref and "cal_dir_err_med" not in ma.extra
    assert (ma.settle_h, ma.diverged, ma.point_mean) == (ref["acq_settle_h"], ref["acq_diverged"],
                                                         ref["acq_point_mean"])
    assert (mf.point_mean, mf.point_p95, mf.diverged) == (ref["fine_point_mean"], ref["fine_point_p95"],
                                                          ref["fine_diverged"])
    assert mf.max_rate_deg_s == ref["fine_max_rate"] and mf.know_med_all == ref["fine_knowledge_med"]

    # scripts/matrix.py: id 6 is selectable, the default is not changed by it, --gate refuses it.
    mod_spec = importlib.util.spec_from_file_location("matrix_script", os.path.join(root, "scripts", "matrix.py"))
    matrix = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(matrix)

    def listed(*argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert matrix.main([*argv, "--list"]) == 0
        return buf.getvalue().splitlines()

    assert listed()[-1] == "62 scenarios" and not any("s6_" in line for line in listed())
    assert listed("--experiments", "6")[-1] == "10 scenarios"
    assert listed("--experiments", "6L")[-1] == "2 scenarios"
    assert not any("x6orb" in line for line in listed("--experiments", "6"))
    assert listed("--experiments", "6", "--match", "s6_fine20_*_lqr_tuned_3mode |*")[-1] == "1 scenarios"
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            matrix.main(["--experiments", "6", "--gate", "--no-ledger"])
            raise AssertionError("--gate accepted experiment 6")
        except SystemExit as e:
            assert e.code == 2


def test_gate_compares_at_the_docs_printed_precision_and_never_decides_loose_claims():
    assert matches(DocCheck("k", 2, "point_mean", "3.0", 124), 3.0199) is True
    assert matches(DocCheck("k", 2, "point_mean", "3.0", 124), 3.06) is False
    assert matches(DocCheck("k", 2, "point_p95", "8", 124), 7.946) is True
    assert matches(DocCheck("k", 0, "M_rel_err_impulse", "0.048", 87), 0.0484) is True
    assert matches(DocCheck("k", 0, "settle_h", "3.1–4.8", 120, kind="range", lo=3.1, hi=4.8), 4.83) is True
    assert matches(DocCheck("k", 0, "point_mean", "90–110", 107, kind="range", lo=90, hi=110), 89.11) is False
    assert matches(DocCheck("k", 0, "settle_h", "3.1–4.8", 120, kind="range", lo=3.1, hi=4.8), np.inf) is False
    assert matches(DocCheck("k", 1, "m_res_err_mAm2", "~0.3", 92, kind="approx"), 0.31) is None
    spec = _spec("e2_alongtrack_0km")
    rows = gate_table([spec], {(spec.key, 5): Metrics(know_med_all=0.418, know_p95_all=0.9,
                                                      extra={"clock_equiv_s": 0.0, "field_dir_err_med": 0.0})})
    got = {r["metric"]: r["match"] for r in rows}
    assert got == {"clock_equiv_s": True, "field_dir_err_med": True, "know_med_all": True, "know_p95_all": False}
    assert all(r["match_seed2"] is None for r in rows)              # seed 2 not run -> no verdict


def test_baseline_reproduces_doc_line_57_at_the_scripts_seed():
    # docs/learned-estimation-and-control.md, section 2 table, 0 km row: MEKF median 0.42, p95 0.63 (seed 5).
    spec = _spec("e2_alongtrack_0km")
    m = run_case(spec, SCRIPT_SEEDS[spec.key])
    assert (f"{m.know_med_all:.2f}", f"{m.know_p95_all:.2f}") == ("0.42", "0.63")


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"  ok  {f.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} passed")
