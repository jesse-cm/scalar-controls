"""Tests: the closed loop records pointing and knowledge error at one instant (defect B1).

The loop used to file the truth after the plant step (t_k + dt) under t_k, next to
the estimate at t_k, so the knowledge error carried |omega| dt of time offset.
``error_timing="aligned"`` (the default) compares truth, estimate and target at
t_k; ``"legacy"`` keeps the old record for the baseline scenarios that reproduce
docs/learned-estimation-and-control.md. Neither changes the simulated trajectory.

The estimator here is exact by construction: a stand-in MEKF whose attitude is
the truth the magnetometer was sampled at, at t_k. Its knowledge error must
then be zero while the body turns at 1.4 deg/s.
"""
import datetime as dt
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import IDENTITY_QUAT, to_body  # noqa: E402
from scalar.constants import DEG  # noqa: E402
from scalar.control import tune  # noqa: E402
from scalar.control.periodic_lqr import PeriodicLQR, TwoModeController  # noqa: E402
from scalar.control.tune import LEGACY_LOOP, actuators_from_matrix, simulate_closed_loop_estimated  # noqa: E402
from scalar.dynamics import RigidBody  # noqa: E402
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.harness import ScenarioSpec, baseline_specs, run_case  # noqa: E402
from scalar.sysid.magnetorquer_id import s3_coil_matrix  # noqa: E402
from scalar.timeutil import julian_date  # noqa: E402

Q0, W0 = np.array([0.5, 0.5, 0.5, 0.5]), np.deg2rad([1.0, -0.6, 0.8])     # 120 deg / 1.4 deg/s
N_STEPS = 120


class _TruthMagnetometer:
    """Noise-free magnetometer that remembers the attitude it was sampled at."""

    def __init__(self):
        self.q = None

    def measure(self, B_eci, q, rng, t=None):
        self.q = np.array(q, float)
        return to_body(q, B_eci)


def _exact_mekf(mag):
    class ExactMEKF:
        """MEKF stand-in: after the measurements of t_k its attitude is the truth at t_k."""

        def __init__(self, **kw):
            self.q, self.b = IDENTITY_QUAT.copy(), np.zeros(3)

        def propagate(self, g, dt_):
            pass

        def init_triad(self, *a):
            pass

        def update_mag(self, B_eci, Bm):
            self.q = mag.q.copy()

        def update_sun(self, *a):
            pass

    return ExactMEKF


def _run(monkeypatch, **kw):
    body = RigidBody()
    acts = actuators_from_matrix(*s3_coil_matrix())
    orbit = CircularOrbit(inclination=98.61 * DEG)
    jd0 = julian_date(dt.datetime(2026, 9, 11, 12))
    mag = _TruthMagnetometer()
    monkeypatch.setattr(tune, "MEKF", _exact_mekf(mag))
    ctrl = TwoModeController(PeriodicLQR(np.zeros((1, 3, 6)), orbit.period, acts), enter_deg_s=0.0)  # rate damping
    return simulate_closed_loop_estimated(body, acts, orbit, jd0, q0=Q0, omega0=W0, duration=N_STEPS,
                                          use_truth=True, controller=ctrl, magnetometer=mag, **kw)


def test_exact_estimator_has_zero_knowledge_error_while_turning(monkeypatch):
    r = _run(monkeypatch)
    assert r.rate_deg_s.min() > 1.0                                  # the body turns throughout
    assert np.max(r.knowledge_err_deg) < 1e-9                        # fails on the old record (1.4 deg)


def test_legacy_record_is_one_step_late_and_the_trajectory_is_the_same(monkeypatch):
    new = _run(monkeypatch, plant_hold="torque", plant_substeps=1)
    old = _run(monkeypatch, **LEGACY_LOOP)
    # the old record: the exact estimate is |omega| dt behind the truth it is compared with
    assert old.knowledge_err_deg[0] == pytest.approx(np.linalg.norm(W0) / DEG * 1.0, rel=0.01)
    assert np.all(old.knowledge_err_deg > 1.0)
    # same trajectory, sampled one step apart: sample k+1 aligned is sample k legacy
    assert np.array_equal(new.pointing_err_deg[1:], old.pointing_err_deg[:-1])
    assert np.array_equal(new.rate_deg_s[1:], old.rate_deg_s[:-1])
    assert np.array_equal(new.dipole, old.dipole) and new.sat_frac == old.sat_frac
    assert new.pointing_err_deg[0] == pytest.approx(120.0, abs=1e-9)  # aligned starts at the initial state


def test_error_timing_is_checked_and_reaches_the_harness():
    body = RigidBody()
    acts = actuators_from_matrix(*s3_coil_matrix())
    orbit = CircularOrbit(inclination=98.61 * DEG)
    with pytest.raises(ValueError):
        simulate_closed_loop_estimated(body, acts, orbit, 2461295.0, q0=Q0, omega0=W0, duration=2,
                                       error_timing="late")
    with pytest.raises(ValueError):
        ScenarioSpec(name="x", controller="periodic_lqr", error_timing="late").validate()
    ScenarioSpec(name="x", kind="estimation", knowledge_window="after_first_fifth").validate()
    with pytest.raises(ValueError):                                 # no closed loop, no record to time
        ScenarioSpec(name="x", kind="estimation", knowledge_window="after_first_fifth",
                     error_timing="legacy").validate()

    (base,) = [s for s in baseline_specs(["4"]) if s.name == "e4D_acq120_mekf"]
    legacy = base.replace(n_orbits=0.02)
    aligned = legacy.replace(error_timing="aligned")
    m_old, m_new = run_case(legacy, 2), run_case(aligned, 2)
    assert m_old.know_med_all != m_new.know_med_all and m_old.point_mean != m_new.point_mean
