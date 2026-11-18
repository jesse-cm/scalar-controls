"""LQR tuning against an *identified* plant, closed on the *estimated* state.

Two things the existing ``MagnetorquerLQR`` demo leaves open:

1. Its actuator set (2 rods + PCB coil) matches no SCALAR source. Here the
   actuator set is built from a dipole matrix M — the output of
   ``scalar.sysid.magnetorquer_id`` — so the gain is for the coils actually
   measured.
2. It feeds back the *true* attitude. Flight has only the MEKF's estimate, and
   only the magnetometer's *measured* body field for the dipole projection.
   ``simulate_closed_loop_estimated`` closes the loop that way, at the 1 Hz the
   flight software can actually run, and reports both pointing error (truth vs
   target) and knowledge error (estimate vs truth).

The weight search is deliberately dumb (log-uniform random over Q/R), because
the objective is cheap enough that cleverness is not worth the code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..actuators.magnetorquer import ActuatorSet, Magnetorquer
from ..attitude import IDENTITY_QUAT, attitude_error_angle, to_body
from ..dynamics import RigidBody, check_plant_switches, plant_step
from ..environment.magfield import magnetic_field_eci
from ..environment.sun import eclipse_factor, solar_irradiance, sun_direction_eci
from ..estimator.mekf import MEKF
from ..sensors.gyro import Gyro
from ..sensors.magnetometer import Magnetometer
from ..sensors.sun_sensor import SunSensorArray
from .lqr import MagnetorquerLQR

__all__ = ["actuators_from_matrix", "ClosedLoopResult", "ProjectedLQRAdapter", "ERROR_TIMINGS",
           "LEGACY_LOOP", "simulate_closed_loop_estimated", "tune_lqr_weights", "tune_periodic_lqr"]

ERROR_TIMINGS = ("aligned", "legacy")
# The loop as it ran for docs/learned-estimation-and-control.md (experiments 4 and 5): torque held over
# the control step, one plant substep, errors recorded one step late. Only the baseline scenarios of
# scalar.harness.baseline pin it, so that scripts/matrix.py --gate still reproduces the notes.
LEGACY_LOOP = {"plant_hold": "torque", "plant_substeps": 1, "error_timing": "legacy"}


def actuators_from_matrix(M, i_max):
    """ActuatorSet whose torquer j points along column j of M with m_max=|M_j| i_max_j."""
    M = np.asarray(M, float)
    ts = []
    for j in range(M.shape[1]):
        col = M[:, j]
        n = np.linalg.norm(col)
        if n > 0:
            ts.append(Magnetorquer(col / n, n * float(i_max[j]), f"ch{j}"))
    return ActuatorSet(ts)


@dataclass
class ClosedLoopResult:
    t: np.ndarray
    pointing_err_deg: np.ndarray
    knowledge_err_deg: np.ndarray
    rate_deg_s: np.ndarray
    dipole: np.ndarray
    sat_frac: float

    def settle_time(self, thresh_deg=10.0, hold=0.9):
        """First time after which >= ``hold`` of the remaining samples are under thresh."""
        e = self.pointing_err_deg
        for k in range(len(e)):
            if np.mean(e[k:] < thresh_deg) >= hold and e[k] < thresh_deg:
                return float(self.t[k])
        return np.inf


class ProjectedLQRAdapter:
    """Wrap the time-invariant ``MagnetorquerLQR`` as a dipole controller."""

    def __init__(self, inertia, actuators, orbit, jd0, K_weights, q_target=None):
        self.ctrl = MagnetorquerLQR(inertia, actuators, orbit, jd0,
                                    q_target=q_target, **K_weights)
        self.actuators = actuators
        self.q_target = self.ctrl.q_target

    def dipole(self, t, q, omega, B_meas):
        # sign-corrected version of MagnetorquerLQR.desired_torque (see periodic_lqr.py)
        from ..attitude import attitude_error_rotvec
        x = np.concatenate([-attitude_error_rotvec(q, self.ctrl.q_target),
                            np.asarray(omega, float) - self.ctrl.w_target])
        tau_d = -self.ctrl.K @ x
        return self.actuators.command(np.cross(B_meas, tau_d) / (B_meas @ B_meas))


def simulate_closed_loop_estimated(body, actuators, orbit, jd0, K_weights=None,
                                   q0=None, omega0=None, dt=1.0, duration=None,
                                   n_orbits=3, seed=0, q_target=None,
                                   albedo_coeff=0.0, use_truth=False, controller=None,
                                   m_res=None, sun_sensor=None, sun_front=None,
                                   sigma_sun=None, magnetometer=None,
                                   mag_calibration=None, gyro=None,
                                   plant_hold="dipole", plant_substeps=2, error_timing="aligned"):
    """Truth -> sensors -> MEKF -> controller -> coils, at ``dt`` (1 s = flight ceiling).

    ``controller`` is any object with ``dipole(t, q, omega, B_meas) -> m`` and a
    ``q_target``; default is the projected time-invariant LQR with ``K_weights``.
    ``sun_sensor`` overrides the ideal array (e.g. a flight-like unit) and
    ``sun_front(currents, E, nadir_body_pred, dayside) -> (s_hat, valid)``
    replaces the analytic reconstruction (e.g. the learned MLP front end).

    The truth is integrated in ``plant_substeps`` substeps per ``dt``; sensors,
    estimator and controller still run once per ``dt``. ``plant_hold="torque"``
    freezes the body torque of the control instant over ``dt``;
    ``plant_hold="dipole"`` freezes the dipole (command plus ``m_res``) and lets
    the torque follow the true body field inside the step
    (``scalar.dynamics.plant_step``). The defaults (dipole held, 2 substeps) are
    what a coil does, converged in the substep count (tests/test_plant_step.py);
    ``plant_hold="torque", plant_substeps=1`` is the original plant
    (docs/crosscheck-basilisk.md, discrepancy D).

    ``error_timing="aligned"`` (default): sample k of the pointing error, the
    knowledge error and the rate is the truth at ``t[k]``, before the plant
    step, against the estimate after the measurements of ``t[k]`` and the
    target. ``"legacy"``: the truth after the plant step (at ``t[k] + dt``) is
    filed under ``t[k]``, as the loop did before; the knowledge error then
    carries an extra ``|omega| dt`` of time offset. The simulated trajectory is
    the same either way. ``LEGACY_LOOP`` holds all three original settings.
    """
    check_plant_switches(plant_hold, plant_substeps)
    if error_timing not in ERROR_TIMINGS:
        raise ValueError(f"error_timing={error_timing!r} not in {ERROR_TIMINGS}")
    aligned = error_timing == "aligned"
    rng = np.random.default_rng(seed)
    gyro = Gyro() if gyro is None else gyro
    mag = Magnetometer() if magnetometer is None else magnetometer
    sun = SunSensorArray() if sun_sensor is None else sun_sensor
    gyro.reset()
    mekf = MEKF(n_iter=3) if sigma_sun is None else MEKF(n_iter=3, sigma_sun=sigma_sun)
    ctrl = controller or ProjectedLQRAdapter(body.inertia, actuators, orbit, jd0,
                                             K_weights, q_target)
    duration = n_orbits * orbit.period if duration is None else duration
    n = int(duration / dt)
    q, w = np.asarray(q0, float), np.asarray(omega0, float)
    t_arr = np.arange(n) * dt
    pe = np.empty(n); ke = np.empty(n); rate = np.empty(n); mh = np.empty((n, 3))
    n_sat = 0
    started = False
    m_max = actuators.m_max_vector

    def field_eci(time):                    # truth field along the orbit, for the dipole hold
        return magnetic_field_eci(orbit.position(time), jd0 + time / 86400.0)

    for k in range(n):
        t = t_arr[k]
        jd = jd0 + t / 86400.0
        r = orbit.position(t)
        B_eci = magnetic_field_eci(r, jd)
        s_eci = sun_direction_eci(jd)
        E = solar_irradiance(jd)
        lit = eclipse_factor(r, s_eci) > 0
        r_hat = r / np.linalg.norm(r)
        # --- sensors ---
        g = gyro.measure(w, dt, rng)
        Bm = mag.measure(B_eci, q, rng)
        if mag_calibration is not None:
            Bm = mag_calibration.apply(Bm)
        dayside = max(0.0, float(r_hat @ s_eci))
        albedo_E = albedo_coeff * E * dayside
        currents = sun.measure(to_body(q, s_eci), E, lit, rng, to_body(q, -r_hat), albedo_E)
        if sun_front is None:
            s_hat, valid = sun.reconstruct(currents, E)
        else:
            s_hat, valid = sun_front(currents, E, to_body(mekf.q, -r_hat), dayside)
        # --- estimator ---
        if k > 0:
            mekf.propagate(g, dt)
        if not started and valid:
            mekf.init_triad(s_eci, s_hat, B_eci, Bm); started = True
        mekf.update_mag(B_eci, Bm)
        if valid:
            mekf.update_sun(s_eci, s_hat)
        # --- controller on the estimate, field from the magnetometer ---
        q_c, w_c, B_c = (q, w, to_body(q, B_eci)) if use_truth else (mekf.q, g - mekf.b, Bm)
        m = ctrl.dipole(t, q_c, w_c, B_c)
        n_sat += int(np.any(np.abs([m @ tq.axis for tq in actuators.torquers]) >= 0.999 * m_max))
        if aligned:                         # truth, estimate and target at the same instant t_k
            pe[k] = np.degrees(attitude_error_angle(q, ctrl.q_target))
            ke[k] = np.degrees(attitude_error_angle(mekf.q, q))
            rate[k] = np.degrees(np.linalg.norm(w))
        m_dist = np.zeros(3) if m_res is None else np.asarray(m_res, float)
        # --- plant (RK4 on omega, exp-map on q; what is held over dt is plant_hold) ---
        if plant_hold == "torque":
            tau = np.cross(m + m_dist, to_body(q, B_eci))   # truth torque incl. residual dipole
            q, w = plant_step(body, q, w, dt, tau, plant_substeps)
        else:
            q, w = plant_step(body, q, w, dt, substeps=plant_substeps, dipole=m + m_dist,
                              field_fn=field_eci, t=t)
        if not aligned:                     # legacy: the truth at t_k + dt filed under t_k
            pe[k] = np.degrees(attitude_error_angle(q, ctrl.q_target))
            ke[k] = np.degrees(attitude_error_angle(mekf.q, q))
            rate[k] = np.degrees(np.linalg.norm(w))
        mh[k] = m
    return ClosedLoopResult(t_arr, pe, ke, rate, mh, n_sat / n)


def tune_lqr_weights(body, actuators, orbit, jd0, q0, omega0, n_candidates=24,
                     seed=0, n_orbits=3, dt=1.0, use_truth=True, verbose=False, **loop):
    """Random log-uniform search over (q_weight, w_weight, r_weight).

    ``loop`` goes to ``simulate_closed_loop_estimated`` (e.g. ``**LEGACY_LOOP``).

    Objective = settle time to <10 deg (s) + 0.5 * mean pointing error over the
    last orbit (deg) * 100 + saturation-fraction penalty. Returns the sorted
    list of (score, weights, result).
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_candidates):
        wts = {"q_weight": 10 ** rng.uniform(-1, 2),
               "w_weight": 10 ** rng.uniform(1, 5),
               "r_weight": 10 ** rng.uniform(7, 12)}
        res = simulate_closed_loop_estimated(body, actuators, orbit, jd0, wts, q0,
                                             omega0, dt=dt, n_orbits=n_orbits,
                                             seed=seed, use_truth=use_truth, **loop)
        last = res.pointing_err_deg[-int(orbit.period / dt):]
        st = res.settle_time()
        score = (min(st, n_orbits * orbit.period) + 50.0 * float(np.mean(last))
                 + 2000.0 * res.sat_frac)
        out.append((score, wts, res))
        if verbose:
            print(f"  [{i:2d}] q={wts['q_weight']:.2g} w={wts['w_weight']:.2g} "
                  f"r={wts['r_weight']:.2g}  settle={st/3600:.2f} h  "
                  f"last-orbit mean={np.mean(last):.1f} deg  sat={res.sat_frac:.2f}  "
                  f"score={score:.0f}")
    out.sort(key=lambda x: x[0])
    return out


def tune_periodic_lqr(body, actuators, orbit, jd0, q0, omega0, q_target=None,
                      theta_max_deg=10.0, omega_max_deg_s=0.2, rhos=(0.01, 0.1, 1.0, 10.0),
                      dt=1.0, n_orbits=4, seed=0, n_bins=None, verbose=False, **loop):
    """Sweep Bryson's rho for the periodic LQR; returns sorted [(score, rho, ctrl, result)].

    ``loop`` goes to ``simulate_closed_loop_estimated`` (e.g. ``**LEGACY_LOOP``)."""
    from ..attitude import IDENTITY_QUAT
    from .periodic_lqr import (PeriodicLQR, bryson_weights, periodic_lqr_schedule,
                               quantize_schedule)
    q_target = IDENTITY_QUAT.copy() if q_target is None else np.asarray(q_target, float)
    out = []
    for rho in rhos:
        Q, R = bryson_weights(np.deg2rad(theta_max_deg), np.deg2rad(omega_max_deg_s),
                              actuators.m_max_vector, rho)
        K, _ = periodic_lqr_schedule(body.inertia, orbit, jd0, q_target, dt, Q, R)
        if n_bins:
            K = quantize_schedule(K, n_bins)
        ctrl = PeriodicLQR(K, orbit.period, actuators, q_target)
        res = simulate_closed_loop_estimated(body, actuators, orbit, jd0, q0=q0, omega0=omega0,
                                             dt=dt, n_orbits=n_orbits, seed=seed,
                                             use_truth=True, controller=ctrl, **loop)
        last = res.pointing_err_deg[-int(orbit.period / dt):]
        st = res.settle_time()
        score = min(st, n_orbits * orbit.period) + 50.0 * float(np.mean(last)) + 2000.0 * res.sat_frac
        out.append((score, rho, ctrl, res))
        if verbose:
            print(f"  rho={rho:<6g} settle={st/3600:5.2f} h  last-orbit mean={np.mean(last):6.2f} deg "
                  f"p95={np.percentile(last,95):6.2f}  sat={res.sat_frac:.2f}  score={score:.0f}")
    out.sort(key=lambda x: x[0])
    return out
