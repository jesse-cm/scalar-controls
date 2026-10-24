"""run_case / run_matrix: drive the existing simulators from a ScenarioSpec.

Nothing here integrates dynamics or filters by itself except
``_mekf_with_front`` (the per-step MEKF loop with a pluggable sun front end,
which today lives only inside scripts/nn_sun_sensor.py and returns two numbers).
Everything else is a call into scalar.control.tune, scalar.control.rl_tune,
scalar.scenario, scalar.sysid and scalar.learning.

SEED. ``seed`` is passed straight to the wrapped simulator's own ``seed``
argument (sensor-noise generator). Initial conditions, the flight-like sun
unit (``sun_unit_seed``) and the MLP training (``mlp_train_seed``) are part of
the scenario, not of the seed.
"""
from __future__ import annotations

import datetime as _dt
import multiprocessing as mp
import sys
import time

import numpy as np

from ..attitude import IDENTITY_QUAT, attitude_error_angle, to_body
from ..constants import DEG
from ..control.periodic_lqr import (PeriodicLQR, TwoModeController, bryson_weights,
                                    periodic_lqr_schedule, quantize_schedule)
from ..control.rl_tune import Plant, build_controller, calibration_pass, clip_params
from ..control.tune import actuators_from_matrix, simulate_closed_loop_estimated
from ..dynamics import RigidBody
from ..environment.magfield import magnetic_field_eci
from ..environment.orbit import CircularOrbit
from ..estimator.mekf import MEKF, run_mekf
from ..scenario import Scenario
from ..sensors.gyro import Gyro
from ..sensors.magnetometer import Magnetometer
from ..sensors.sun_sensor import SunSensorArray
from ..sysid.magnetorquer_id import (PulseSchedule, identify_dipole_matrix,
                                     identify_dipole_matrix_impulse, s3_coil_matrix,
                                     simulate_burst_campaign)
from ..timeutil import julian_date
from .ledger import ledger_row
from .metrics import Metrics, actuation_stats, knowledge_stats, pointing_stats, sunlit_mask
from .spec import ScenarioSpec

__all__ = ["run_case", "run_cases", "run_matrix", "mlp_key", "train_shared", "stderr_progress"]

# ---------------------------------------------------------------------------
# builders: spec -> the objects the wrapped simulators take
# ---------------------------------------------------------------------------


def _orbit(spec):
    return CircularOrbit(altitude=spec.altitude_m, inclination=spec.inclination_deg * DEG,
                         raan=spec.raan_deg * DEG, arg_lat0=spec.arg_lat0_deg * DEG, j2=spec.j2)


def _epoch(spec):
    return _dt.datetime(*spec.epoch)


def _inertia(spec):
    return RigidBody().inertia if spec.inertia is None else np.array(spec.inertia, float)


def _gyro(spec):
    kw = {}
    if spec.gyro_noise_density is not None:
        kw["rate_noise_density"] = spec.gyro_noise_density
    if spec.gyro_bias_rw is not None:
        kw["bias_rw"] = spec.gyro_bias_rw
    if spec.gyro_initial_bias_deg_s is not None:
        kw["initial_bias"] = np.array(spec.gyro_initial_bias_deg_s, float) * DEG
    return Gyro(**kw)


def _magnetometer(spec):
    kw = {}
    if spec.mag_noise_std is not None:
        kw["noise_std"] = spec.mag_noise_std
    if spec.mag_bias_T is not None:
        kw["bias"] = np.array(spec.mag_bias_T, float)
    if spec.mag_soft_iron is not None:
        kw["soft_iron"] = np.array(spec.mag_soft_iron, float)
    return Magnetometer(**kw)


def _sun_sensor(spec):
    if spec.sun_unit == "flight_like":
        from ..learning.sun_mlp import flight_like_sensor
        s = flight_like_sensor(np.random.default_rng(spec.sun_unit_seed), gain_sigma=spec.sun_gain_sigma,
                               rolloff=spec.sun_rolloff, tilt_deg=spec.sun_tilt_deg)
    else:
        s = SunSensorArray()
    if spec.sun_noise_frac is not None:
        s.noise_frac = spec.sun_noise_frac
    return s


def _mekf(spec):
    kw = {"n_iter": spec.mekf_n_iter}
    for name, key in (("mekf_sigma_sun", "sigma_sun"), ("mekf_sigma_mag", "sigma_mag"),
                      ("mekf_sigma_v", "sigma_v"), ("mekf_sigma_u", "sigma_u")):
        if getattr(spec, name) is not None:
            kw[key] = getattr(spec, name)
    return MEKF(**kw)


def _plant(spec):
    """rl_tune.Plant for this spec, so build_controller / calibration_pass are reused as they are."""
    M, i_max = s3_coil_matrix()
    mag = _magnetometer(spec)
    return Plant(_inertia(spec), M, i_max, inclination_deg=spec.inclination_deg, epoch=_epoch(spec),
                 m_res=np.array(spec.m_res, float), mag_bias=mag.bias, mag_soft_iron=mag.soft_iron,
                 mag_calibrated=spec.mag_calibrated, scale_M=spec.coil_scale,
                 scale_inertia=spec.inertia_scale)


# ---------------------------------------------------------------------------
# the learned sun front end: trained once per (unit, variant), shared with workers
# ---------------------------------------------------------------------------
_MLP_CACHE = {}


def mlp_key(spec):
    """Cache key of the trained MLP a spec needs, or None for the analytic front end."""
    if spec.sun_front == "analytic":
        return None
    return (spec.sun_front, spec.sun_unit, spec.sun_unit_seed, spec.sun_gain_sigma, spec.sun_rolloff,
            spec.sun_tilt_deg, spec.sun_noise_frac, spec.mlp_train_seed, spec.mlp_epochs)


def _mlp(spec):
    key = mlp_key(spec)
    if key not in _MLP_CACHE:
        from ..learning.sun_mlp import train_sun_mlp
        _MLP_CACHE[key], _ = train_sun_mlp(_sun_sensor(spec), extra=(spec.sun_front == "mlp_b"),
                                           epochs=spec.mlp_epochs, seed=spec.mlp_train_seed)
    return _MLP_CACHE[key]


def train_shared(specs):
    """Train every MLP the specs need, once, in this process. Returns {key: SunMLP}."""
    for s in specs:
        if mlp_key(s) is not None:
            _mlp(s)
    return {mlp_key(s): _MLP_CACHE[mlp_key(s)] for s in specs if mlp_key(s) is not None}


def _sun_front(spec, sensor):
    """Front-end callable in tune.py's signature, or None for the analytic reconstruction."""
    if spec.sun_front == "analytic":
        return None
    mlp = _mlp(spec)
    if spec.sun_front == "mlp_a":
        return lambda c, E, nb, ds: mlp.reconstruct(c, E, sensor)
    return lambda c, E, nb, ds: mlp.reconstruct(c, E, sensor, np.concatenate([nb, [ds]]))


# ---------------------------------------------------------------------------
# kind = "closed_loop"
# ---------------------------------------------------------------------------


def _periodic(spec, plant, acts_design, acts_true, orbit, jd0):
    # The scripts design the schedule at a 1 s step from the nominal inertia and coils.
    Q, R = bryson_weights(spec.theta_max_deg * DEG, spec.omega_max_deg_s * DEG,
                          acts_design.m_max_vector, spec.rho)
    K, _ = periodic_lqr_schedule(plant.inertia, orbit, jd0, IDENTITY_QUAT, 1.0, Q, R)
    if spec.schedule_bins is not None:
        K = quantize_schedule(K, spec.schedule_bins)
    return PeriodicLQR(K, orbit.period, acts_true, IDENTITY_QUAT)


def _controller(spec, plant, acts_design, acts_true, orbit, jd0):
    """(controller or None, K_weights or None) for simulate_closed_loop_estimated."""
    if spec.controller == "projected_lqr":
        return None, {"q_weight": spec.proj_q_weight, "w_weight": spec.proj_w_weight,
                      "r_weight": spec.proj_r_weight}
    if spec.controller == "periodic_lqr":
        return _periodic(spec, plant, acts_design, acts_true, orbit, jd0), None
    if spec.controller == "two_mode":
        ctrl = TwoModeController(_periodic(spec, plant, acts_design, acts_true, orbit, jd0),
                                 k_damp=spec.k_damp, enter_deg_s=spec.enter_deg_s,
                                 exit_deg_s=spec.exit_deg_s)
        ctrl.mode = spec.start_mode
        return ctrl, None
    # three_mode: exactly what rl_tune.evaluate does
    params = clip_params([spec.theta_max_deg, spec.omega_max_deg_s, spec.rho, spec.enter_deg_s,
                          spec.exit_deg_s, spec.k_damp, *spec.fine_weights])
    ctrl = build_controller(params, plant, acts_design, orbit, jd0)
    ctrl.lqr.actuators = acts_true
    ctrl.fine.actuators = acts_true
    ctrl.mode = spec.start_mode
    return ctrl, None


def _run_closed_loop(spec, seed):
    plant = _plant(spec)
    body = RigidBody(plant.inertia * plant.scale_inertia)
    acts_design = actuators_from_matrix(plant.M, plant.i_max)
    acts_true = actuators_from_matrix(plant.M * plant.scale_M, plant.i_max)
    orbit = _orbit(spec)
    jd0 = julian_date(_epoch(spec))
    extra = {}

    mag = _magnetometer(spec)
    cal = None
    if spec.mag_calibrated:
        cal, extra["cal_dir_err_med"] = calibration_pass(plant, mag, orbit, jd0, seed)
    sensor = _sun_sensor(spec)
    ctrl, k_weights = _controller(spec, plant, acts_design, acts_true, orbit, jd0)

    r = simulate_closed_loop_estimated(
        body, acts_true, orbit, jd0, k_weights,
        q0=np.array(spec.q0, float), omega0=np.array(spec.omega0_deg_s, float) * DEG,
        dt=spec.dt, n_orbits=spec.n_orbits, seed=seed, albedo_coeff=spec.albedo_coeff,
        use_truth=(spec.estimator == "truth"), controller=ctrl, m_res=np.array(spec.m_res, float),
        sun_sensor=sensor, sun_front=_sun_front(spec, sensor), sigma_sun=spec.mekf_sigma_sun,
        magnetometer=mag, mag_calibration=cal, gyro=_gyro(spec),
        plant_hold=spec.plant_hold, plant_substeps=spec.plant_substeps)

    n_last = int(orbit.period / spec.dt)        # the scripts' "last orbit" slice
    last = slice(-n_last, None)
    m = Metrics(extra=extra)
    vals = pointing_stats(r, n_last, spec.settle_thresh_deg, spec.settle_hold, spec.diverge_deg)
    vals.update(actuation_stats(r, acts_true, plant.i_max,
                                last if spec.actuation_window == "last_orbit" else slice(None)))
    if spec.estimator == "mekf":                # knowledge of a filter that is not in the loop is not reported
        vals.update(knowledge_stats(r.knowledge_err_deg[last], sunlit_mask(orbit, jd0, r.t[last])))
    for k, v in vals.items():
        setattr(m, k, v)
    return m


# ---------------------------------------------------------------------------
# kind = "estimation"
# ---------------------------------------------------------------------------


def _mekf_with_front(res, mekf, front):
    """MEKF over a SimResult with a pluggable sun front end; returns the error series [deg].

    Same step order as estimator.mekf.run_mekf (propagate, TRIAD seed at the
    first valid sun, magnetometer update, sun update), with the sun vector taken
    from ``front(currents, E, nadir_body_pred, dayside)`` as tune.py does.
    """
    err = np.empty(len(res))
    started = False
    for k in range(len(res)):
        if k > 0:
            mekf.propagate(res.gyro_meas[k], res.dt)
        r_hat = res.r_eci[k] / np.linalg.norm(res.r_eci[k])
        nadir_pred = to_body(mekf.q, -r_hat)
        dayside = max(0.0, float(r_hat @ res.sun_eci[k]))
        s_hat, valid = front(res.sun_currents[k], res.irradiance[k], nadir_pred, dayside)
        if not started and valid:
            mekf.init_triad(res.sun_eci[k], s_hat, res.B_eci_ref[k], res.mag_body[k])
            started = True
        mekf.update_mag(res.B_eci_ref[k], res.mag_body[k])
        if valid:
            mekf.update_sun(res.sun_eci[k], s_hat)
        err[k] = np.degrees(attitude_error_angle(mekf.q, res.q_true[k]))
    return err


def _run_estimation(spec, seed):
    orbit = _orbit(spec)
    sensor = _sun_sensor(spec)
    sc = Scenario(epoch=_epoch(spec), orbit=orbit, body=RigidBody(_inertia(spec)),
                  q0=np.array(spec.q0, float), omega0=np.array(spec.omega0_deg_s, float) * DEG,
                  dt=spec.dt, duration=spec.n_orbits * orbit.period, gyro=_gyro(spec),
                  magnetometer=_magnetometer(spec), sun_sensor=sensor, seed=seed,
                  albedo_coeff=spec.albedo_coeff, field_model_error_deg=spec.field_model_error_deg,
                  field_truth=spec.field_truth)
    res = sc.run()
    extra = {"eclipse_fraction": res.eclipse_fraction}

    # Estimator's reference field evaluated at a position shifted along track (experiment 2).
    v = orbit.radius * orbit.mean_motion
    dt_shift = spec.ref_along_track_error_km * 1e3 / v
    if spec.ref_along_track_error_km != 0.0:
        ref = np.array([magnetic_field_eci(orbit.position(t + dt_shift), jd)
                        for t, jd in zip(res.t, res.jd)])
        res = res.__class__(**{**res.__dict__, "B_eci_ref": ref})
    cosang = (np.sum(res.B_eci_ref * res.B_eci, 1)
              / np.linalg.norm(res.B_eci_ref, axis=1) / np.linalg.norm(res.B_eci, axis=1))
    extra["field_dir_err_med"] = float(np.median(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
    extra["clock_equiv_s"] = float(dt_shift)

    front = _sun_front(spec, sensor)
    if front is None:
        err = run_mekf(res, _mekf(spec)).err_deg
    else:
        err = _mekf_with_front(res, _mekf(spec), front)
    w = len(err) // 5                          # MEKFResult.summary(): skip the convergence transient
    m = Metrics(extra=extra)
    for k, v_ in knowledge_stats(err[w:], res.lit[w:]).items():
        setattr(m, k, v_)
    return m


# ---------------------------------------------------------------------------
# kind = "sysid"
# ---------------------------------------------------------------------------


def _run_sysid(spec, seed):
    M_true, i_max = s3_coil_matrix()
    m_res = np.array(spec.m_res, float)
    body = RigidBody(_inertia(spec))
    sched = PulseSchedule(M_true.shape[1], i_max, duration=spec.pulse_duration_s,
                          period=spec.pulse_period_s, n_pulses=spec.n_pulses)
    kw = {}
    if spec.gyro_noise_density is not None:
        kw["gyro_noise_density"] = spec.gyro_noise_density
    if spec.mag_noise_std is not None:
        kw["mag_noise"] = spec.mag_noise_std
    d = simulate_burst_campaign(M_true, m_res, sched, body=body, orbit=_orbit(spec),
                                jd0=julian_date(_epoch(spec)), q0=np.array(spec.q0, float),
                                omega0=np.array(spec.omega0_deg_s, float) * DEG,
                                dt_sim=spec.sysid_dt_sim, sample_hz=spec.sample_hz, seed=seed, **kw)
    Md, *_ = identify_dipole_matrix(body.inertia, d["omega"], d["omega_dot"], d["B_body"], d["currents"])
    Mi, mr, n_used, cond = identify_dipole_matrix_impulse(
        body.inertia, d["t"], d["omega"], d["B_body"], d["currents"], window=spec.impulse_window_s)

    def rel(M):
        return float(np.linalg.norm(M - M_true) / np.linalg.norm(M_true))

    cols = [float(np.linalg.norm(Mi[:, j] - M_true[:, j]) / np.linalg.norm(M_true[:, j]))
            for j in range(M_true.shape[1])]
    extra = {"M_rel_err_per_sample": rel(Md), "M_rel_err_impulse": rel(Mi),
             "M_col_err_x": cols[0], "M_col_err_y": cols[1], "M_col_err_z": cols[2],
             "m_res_err_mAm2": float(np.linalg.norm(mr - m_res) * 1e3),
             "n_pulses_used": float(n_used), "campaign_min": sched.total_time / 60.0}
    return Metrics(extra=extra)


# ---------------------------------------------------------------------------
# kind = "sun_vector"
# ---------------------------------------------------------------------------


def _run_sun_vector(spec, seed):
    from ..learning.sun_mlp import angular_error_deg, make_dataset
    sensor = _sun_sensor(spec)
    # scripts/nn_sun_sensor.py asks for albedo_max=1e-9 where the table says 0.
    amax = spec.dataset_albedo_max if spec.dataset_albedo_max > 0 else 1e-9
    Xc, Xe, y = make_dataset(sensor, spec.dataset_n, np.random.default_rng(seed), albedo_max=amax)
    if spec.sun_front == "analytic":
        fs = sensor.full_scale_current(1361.0)
        pred = np.array([sensor.reconstruct(c * fs, 1361.0)[0] for c in Xc])
    elif spec.sun_front == "mlp_a":
        pred = _mlp(spec)(Xc)
    else:
        pred = _mlp(spec)(Xc, Xe)
    e = angular_error_deg(pred, y)
    return Metrics(extra={"sun_err_med": float(np.median(e)), "sun_err_p95": float(np.percentile(e, 95))})


_RUNNERS = {"closed_loop": _run_closed_loop, "estimation": _run_estimation,
            "sysid": _run_sysid, "sun_vector": _run_sun_vector}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def run_case(spec: ScenarioSpec, seed: int) -> Metrics:
    """Run one scenario at one seed and return its Metrics."""
    spec.validate()
    t0 = time.time()
    m = _RUNNERS[spec.kind](spec, int(seed))
    m.wall_s = time.time() - t0
    return m


def _init_worker(shared):
    _MLP_CACHE.update(shared)


def _work(task):
    spec, seed = task
    return spec.key, seed, run_case(spec, seed)


def run_cases(specs, seeds, workers=None, progress=None):
    """Run every (spec, seed) pair. Returns {(spec.key, seed): Metrics}.

    ``seeds`` is one iterable for all specs, or a dict {spec.key: iterable}.
    MLP front ends are trained once here and handed to the workers, so every
    seed of a row sees the same weights. ``workers=1`` runs in this process.
    """
    specs = list(specs)
    keys = [s.key for s in specs]
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"duplicate (scenario, estimator, controller): {dup}")
    for s in specs:
        s.validate()
    per = seeds if isinstance(seeds, dict) else {s.key: list(seeds) for s in specs}
    tasks = [(s, int(sd)) for s in specs for sd in per[s.key]]
    shared = train_shared(specs)
    out = {}
    workers = max(1, mp.cpu_count() - 2) if workers is None else int(workers)

    def note(i, key, seed, m):
        if progress:
            progress(f"[{i}/{len(tasks)}] {m.wall_s:6.1f} s  seed {seed:<4d} {key}")

    if workers <= 1 or len(tasks) <= 1:
        for i, task in enumerate(tasks, 1):
            key, seed, m = _work(task)
            out[(key, seed)] = m
            note(i, key, seed, m)
        return out
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, len(tasks)), initializer=_init_worker, initargs=(shared,)) as pool:
        for i, (key, seed, m) in enumerate(pool.imap_unordered(_work, tasks, chunksize=1), 1):
            out[(key, seed)] = m
            note(i, key, seed, m)
    return out


def run_matrix(specs, seeds, workers=None, progress=None, git_hash=None, date=None):
    """Run specs x seeds and return one ledger row (dict) per spec, in spec order.

    Each metric is reported as median and p95 across the seed set (rule 4).
    """
    specs = list(specs)
    seeds = [int(s) for s in seeds]
    cases = run_cases(specs, seeds, workers=workers, progress=progress)
    return [ledger_row(s, seeds, [cases[(s.key, sd)] for sd in seeds], git_hash=git_hash, date=date)
            for s in specs]


def stderr_progress(msg):
    print(msg, file=sys.stderr, flush=True)
