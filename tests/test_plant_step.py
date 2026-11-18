"""Tests: the plant step is independent of the control step (verification-loop plan, phase 1).

Two switches, ``plant_hold`` and ``plant_substeps``, added after the Basilisk
cross-check (docs/crosscheck-basilisk.md, discrepancies C and D). Rule 3 of the
plan: each switch ships with an assertion that turning it on moves a number.

    C  the truth quaternion is advanced with a second-order step
       -> free tumble at the README rate against an independent ODE solution:
          the attitude error falls 4x per doubling of plant_substeps.
    D  the plant froze the body torque for a whole control step
       -> the cross-check's detumble orbit: plant_hold="dipole" moves the truth
          attitude by more than 1 deg in one orbit, and that number is converged
          in the substep count.

The detumble case is set up as scripts/crosscheck_basilisk.py does it (control
configuration, rate-damping law fed the truth at 1 Hz) without importing it.
"""
import datetime as dt
import os
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import IDENTITY_QUAT, attitude_error_angle, dcm, propagate, to_body  # noqa: E402
from scalar.constants import DEG  # noqa: E402
from scalar.control.periodic_lqr import PeriodicLQR, TwoModeController  # noqa: E402
from scalar.control.tune import LEGACY_LOOP, actuators_from_matrix, simulate_closed_loop_estimated  # noqa: E402
from scalar.dynamics import RigidBody, plant_step, simulate_controlled, simulate_free_tumble  # noqa: E402
from scalar.environment.magfield import magnetic_field_eci  # noqa: E402
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.harness import ScenarioSpec, baseline_specs, run_case  # noqa: E402
from scalar.sysid.magnetorquer_id import s3_coil_matrix  # noqa: E402
from scalar.timeutil import julian_date  # noqa: E402

# docs/crosscheck-basilisk.md "control" row: 550 km, 98.61 deg, 2026-09-11 12:00, 120 deg / 1.4 deg/s
Q_CONTROL, W_CONTROL = np.array([0.5, 0.5, 0.5, 0.5]), np.deg2rad([1.0, -0.6, 0.8])
# "readme" row (scalar/scenario.py defaults): identity, 7.1 deg/s, 0.5 s step
W_README, DT_README = np.deg2rad([4.0, -3.0, 5.0]), 0.5
K_DAMP = 5e3                                  # TwoModeController.k_damp
CONVERGED_SUBSTEPS = 2                        # smallest N whose doubling moves the detumble attitude < 0.01 deg


def _max_angle_deg(qa, qb):
    """Largest angle [deg] between two attitude histories, sample by sample."""
    Ca, Cb = np.array([dcm(q) for q in qa]), np.array([dcm(q) for q in qb])
    return float(_dcm_angle_deg(Ca, Cb).max())


def _dcm_angle_deg(Ca, Cb):
    tr = np.einsum("nij,nij->n", Ca, Cb)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def _control_setup():
    orbit = CircularOrbit(inclination=98.61 * DEG)
    jd0 = julian_date(dt.datetime(2026, 9, 11, 12))
    acts = actuators_from_matrix(*s3_coil_matrix())

    def field(t):
        return magnetic_field_eci(orbit.position(t), jd0 + t / 86400.0)

    return RigidBody(), acts, orbit, jd0, field


def _damping_law(orbit, acts):
    """m = sat(k omega x B): TwoModeController held in mode 1 (an enter threshold of 0 deg/s is never met)."""
    return TwoModeController(PeriodicLQR(np.zeros((1, 3, 6)), orbit.period, acts), k_damp=K_DAMP, enter_deg_s=0.0)


def _detumble(n_steps, plant_hold="torque", plant_substeps=1):
    """Truth timeline of the damping law at 1 Hz on the truth rate and field, from simulate_controlled."""
    body, acts, orbit, _, field = _control_setup()
    ctrl = _damping_law(orbit, acts)

    def dipole(t, q, w):
        return ctrl.dipole(t, q, w, to_body(q, field(t)))

    if plant_hold == "torque":
        tl, _ = simulate_controlled(body, Q_CONTROL, W_CONTROL, 1.0, n_steps,
                                    lambda t, q, w: np.cross(dipole(t, q, w), to_body(q, field(t))),
                                    plant_substeps=plant_substeps)
    else:
        tl, _ = simulate_controlled(body, Q_CONTROL, W_CONTROL, 1.0, n_steps, plant_hold="dipole",
                                    plant_substeps=plant_substeps, dipole_fn=dipole, field_fn=field)
    return tl


# --------------------------------------------------------------------------- the integrators' defaults are the original plant


def _original_step(body, q, w, dt_, tau=None):
    """The integrator as it stood before the switches (dynamics.py and tune.py at 5d56674), verbatim."""
    k1 = body.omega_dot(w, tau)
    k2 = body.omega_dot(w + 0.5 * dt_ * k1, tau)
    k3 = body.omega_dot(w + 0.5 * dt_ * k2, tau)
    k4 = body.omega_dot(w + dt_ * k3, tau)
    w_next = w + (dt_ / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return propagate(q, 0.5 * (w + w_next), dt_), w_next


def test_defaults_reproduce_the_original_integrators_bit_for_bit():
    body = RigidBody()
    n = 400
    tl = simulate_free_tumble(body, IDENTITY_QUAT, W_README, DT_README, n)
    q, w = IDENTITY_QUAT.copy(), W_README.copy()
    for i in range(n):
        q, w = _original_step(body, q, w, DT_README)
        assert np.array_equal(tl.q[i + 1], q) and np.array_equal(tl.omega[i + 1], w)

    def torque(t, q_, w_):
        return 1e-7 * np.array([np.sin(0.01 * t), -0.5, np.cos(0.02 * t)]) - 1e-5 * w_

    tl, taus = simulate_controlled(body, Q_CONTROL, W_CONTROL, 1.0, n, torque)
    q, w = Q_CONTROL.copy(), W_CONTROL.copy()
    for i in range(n):
        tau = torque(i * 1.0, q, w)
        assert np.array_equal(taus[i], tau)
        q, w = _original_step(body, q, w, 1.0, tau)
        assert np.array_equal(tl.q[i + 1], q) and np.array_equal(tl.omega[i + 1], w)
    assert np.array_equal(taus[n], taus[n - 1])

    # The closed loop defaults to the corrected plant; the baseline scenarios pin the original one.
    spec = ScenarioSpec(name="x")
    assert (spec.plant_hold, spec.plant_substeps) == ("dipole", 2)
    closed = [s for s in baseline_specs() if s.kind == "closed_loop"]
    assert len(closed) == 32
    assert all((s.plant_hold, s.plant_substeps, s.error_timing) == ("torque", 1, "legacy") for s in closed)
    assert LEGACY_LOOP == {"plant_hold": "torque", "plant_substeps": 1, "error_timing": "legacy"}


def test_substeps_are_the_same_as_a_shorter_plant_step():
    body = RigidBody()
    fine = simulate_free_tumble(body, IDENTITY_QUAT, W_README, 0.5, 600)
    sub = simulate_free_tumble(body, IDENTITY_QUAT, W_README, 1.0, 300, plant_substeps=2)
    assert np.array_equal(sub.q, fine.q[::2]) and np.array_equal(sub.omega, fine.omega[::2])
    assert np.array_equal(sub.t, np.arange(301.0))


# --------------------------------------------------------------------------- C: second-order quaternion step


def test_free_tumble_attitude_error_falls_4x_per_halving_of_the_plant_step():
    """Discrepancy C at the README rate, one orbit, against an ODE solution that shares no code with the plant.

    Reference: Euler's equation and dC/dt = -[omega]x C (attitude.py, KINEMATICS)
    integrated together by DOP853 at rtol 1e-12. The cross-check measured
    1.002 / 0.251 / 0.063 / 0.016 deg against Basilisk at 1 / 0.5 / 0.25 / 0.125 s.
    """
    body = RigidBody()
    n = int(CircularOrbit(inclination=51.6 * DEG).period)

    def rhs(t, y):
        w, C = y[:3], y[3:].reshape(3, 3)
        return np.concatenate([body.omega_dot(w), -np.cross(w, C, axisb=0, axisc=0).ravel()])

    sol = solve_ivp(rhs, (0.0, n), np.concatenate([W_README, np.eye(3).ravel()]), method="DOP853",
                    rtol=1e-12, atol=1e-14, t_eval=np.arange(n + 1.0))
    assert sol.success
    C_ref = sol.y[3:].T.reshape(-1, 3, 3)

    err = {}
    for N in (1, 2, 4, 8):
        tl = simulate_free_tumble(body, IDENTITY_QUAT, W_README, 1.0, n, plant_substeps=N)
        e = _dcm_angle_deg(np.array([dcm(q) for q in tl.q]), C_ref)
        err[N] = e.max()
        assert e.argmax() == n                                      # grows through the orbit
        assert np.abs(tl.omega - sol.y[:3].T).max() < 1e-6 * DEG    # the rate is not the problem (RK4: 5.5e-7 deg/s at 1 s)
    assert err[1] == pytest.approx(1.002, abs=0.002)
    assert err[2] == pytest.approx(0.251, abs=0.001)                # the README default: 0.5 s step
    for N in (1, 2, 4):
        assert err[N] / err[2 * N] == pytest.approx(4.0, abs=0.01)

    # rule 3 for plant_substeps: at the README defaults (0.5 s) the switch moves the truth by 0.25 deg/orbit
    coarse = simulate_free_tumble(body, IDENTITY_QUAT, W_README, DT_README, 2 * n)
    fine = simulate_free_tumble(body, IDENTITY_QUAT, W_README, DT_README, 2 * n, plant_substeps=8)
    assert _max_angle_deg(coarse.q, fine.q) > 0.2


# --------------------------------------------------------------------------- D: what the coil holds


def test_dipole_hold_moves_the_detumble_attitude_by_more_than_1_deg_and_is_converged():
    """Discrepancy D on the cross-check's detumble orbit (rule 3 for plant_hold).

    The cross-check: torque held 1.81 deg from Basilisk (same field both sides),
    dipole held 0.0013 deg. Here the two plants are compared with each other.
    """
    n = int(_control_setup()[2].period)
    N = CONVERGED_SUBSTEPS
    default = _detumble(n)
    dip = {k: _detumble(n, "dipole", k) for k in (N // 2, N, 2 * N)}

    assert _max_angle_deg(default.q, dip[N].q) >= 1.0                # the effect (measured 1.86)
    assert _max_angle_deg(default.q, dip[N].q) == pytest.approx(1.86, abs=0.05)
    assert _max_angle_deg(dip[N].q, dip[2 * N].q) < 0.01             # converged (measured 0.0037)
    assert _max_angle_deg(dip[N // 2].q, dip[N].q) > 0.01            # and N is the smallest such count
    # both plants still detumble to the same place: the rate is not what moves
    rate = [np.degrees(np.linalg.norm(tl.omega[-1])) for tl in (default, dip[N])]
    assert rate[0] == pytest.approx(0.0062, abs=0.0002) and rate[1] == pytest.approx(0.0062, abs=0.0002)

    # it is the hold, not the step: the torque-hold plant with a finer step stays where it was
    torque_fine = _detumble(n, "torque", 2 * N)
    assert _max_angle_deg(default.q, torque_fine.q) < 0.05
    assert _max_angle_deg(torque_fine.q, dip[2 * N].q) >= 1.0


def test_dipole_hold_with_a_fixed_field_and_no_rotation_is_the_torque_hold():
    """With B_body constant over the step the two holds are the same torque: m x B."""
    body = RigidBody()
    B, m = np.array([2e-5, -1e-5, 3e-5]), np.array([0.01, 0.02, -0.005])
    q, w = plant_step(body, IDENTITY_QUAT, np.zeros(3), 1e-3, np.cross(m, B))
    qd, wd = plant_step(body, IDENTITY_QUAT, np.zeros(3), 1e-3, dipole=m, field_fn=lambda t: B)
    assert np.allclose(wd, w, rtol=1e-6, atol=0.0) and np.linalg.norm(w) > 0
    assert attitude_error_angle(qd, q) < 1e-12


# --------------------------------------------------------------------------- the closed-loop simulator and the harness


@pytest.mark.parametrize("error_timing", ["aligned", "legacy"])
@pytest.mark.parametrize("plant_hold,plant_substeps", [("torque", 1), ("torque", 3), ("dipole", 1), ("dipole", 2)])
def test_closed_loop_simulator_steps_the_same_plant(plant_hold, plant_substeps, error_timing):
    """tune.simulate_closed_loop_estimated on the truth is simulate_controlled, switch for switch.

    Sample k is the truth at t_k (aligned) or at t_k + dt (legacy record)."""
    body, acts, orbit, jd0, _ = _control_setup()
    n = 300
    r = simulate_closed_loop_estimated(body, acts, orbit, jd0, q0=Q_CONTROL, omega0=W_CONTROL, duration=n,
                                       use_truth=True, controller=_damping_law(orbit, acts),
                                       plant_hold=plant_hold, plant_substeps=plant_substeps,
                                       error_timing=error_timing)
    tl = _detumble(n, plant_hold, plant_substeps)
    k = slice(0, n) if error_timing == "aligned" else slice(1, n + 1)
    assert np.array_equal(r.rate_deg_s, [np.degrees(np.linalg.norm(w)) for w in tl.omega[k]])
    assert np.array_equal(r.pointing_err_deg, [np.degrees(attitude_error_angle(q, IDENTITY_QUAT)) for q in tl.q[k]])


def test_closed_loop_simulator_defaults_to_the_dipole_held_plant_in_two_substeps():
    body, acts, orbit, jd0, _ = _control_setup()
    n = 300
    r = simulate_closed_loop_estimated(body, acts, orbit, jd0, q0=Q_CONTROL, omega0=W_CONTROL, duration=n,
                                       use_truth=True, controller=_damping_law(orbit, acts))
    tl = _detumble(n, "dipole", CONVERGED_SUBSTEPS)
    assert np.array_equal(r.pointing_err_deg, [np.degrees(attitude_error_angle(q, IDENTITY_QUAT)) for q in tl.q[:n]])


def test_spec_switches_reach_the_simulator():
    (base,) = [s for s in baseline_specs(["4"]) if s.name == "e4D_acq120_mekf"]
    off = base.replace(n_orbits=0.05)
    on = off.replace(plant_hold="dipole", plant_substeps=2)
    m_off, m_on = run_case(off, 2), run_case(on, 2)
    assert m_on.max_rate_deg_s != m_off.max_rate_deg_s and m_on.point_mean != m_off.point_mean

    from scalar.harness import runner
    plant = runner._plant(on)
    orbit, jd0 = runner._orbit(on), julian_date(runner._epoch(on))
    acts = actuators_from_matrix(plant.M, plant.i_max)
    ctrl, _ = runner._controller(on, plant, acts, acts, orbit, jd0)
    r = simulate_closed_loop_estimated(RigidBody(plant.inertia), acts, orbit, jd0, q0=Q_CONTROL, omega0=W_CONTROL,
                                       n_orbits=0.05, seed=2, controller=ctrl, m_res=np.array(on.m_res),
                                       plant_hold="dipole", plant_substeps=2, error_timing="legacy")
    assert m_on.point_mean == np.mean(r.pointing_err_deg) and m_on.max_rate_deg_s == r.rate_deg_s.max()
    assert m_on.know_med_all == np.median(r.knowledge_err_deg)


def test_switch_values_are_checked():
    body, acts, orbit, jd0, field = _control_setup()
    for bad in (dict(plant_hold="coil"), dict(plant_substeps=0), dict(plant_substeps=2.0), dict(error_timing="late")):
        with pytest.raises(ValueError):
            simulate_closed_loop_estimated(body, acts, orbit, jd0, q0=Q_CONTROL, omega0=W_CONTROL, duration=2, **bad)
        with pytest.raises(ValueError):
            ScenarioSpec(name="x", controller="periodic_lqr", **bad).validate()
    with pytest.raises(ValueError):
        simulate_free_tumble(body, IDENTITY_QUAT, W_README, 1.0, 2, plant_substeps=0)
    with pytest.raises(ValueError):                                 # dipole hold needs the dipole and the field
        simulate_controlled(body, Q_CONTROL, W_CONTROL, 1.0, 2, lambda t, q, w: np.zeros(3), plant_hold="dipole")
    with pytest.raises(ValueError):
        simulate_controlled(body, Q_CONTROL, W_CONTROL, 1.0, 2, dipole_fn=lambda t, q, w: np.zeros(3), field_fn=field)
    ScenarioSpec(name="x", controller="periodic_lqr", plant_hold="dipole", plant_substeps=4).validate()
    # Scenario, the burst campaign and the static dataset integrate their own truth: no switch to honour there
    for kind, kw in (("estimation", dict(knowledge_window="after_first_fifth")), ("sysid", {}), ("sun_vector", {})):
        ScenarioSpec(name="x", kind=kind, **kw).validate()
        for on in (dict(plant_hold="torque"), dict(plant_substeps=1), dict(error_timing="legacy")):
            with pytest.raises(ValueError):
                ScenarioSpec(name="x", kind=kind, **kw, **on).validate()
