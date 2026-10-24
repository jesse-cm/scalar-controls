"""Rigid-body attitude dynamics for a CubeSat: free tumble and applied torque.

Two integrators share one rigid body (Euler's equation):

  * ``simulate_free_tumble`` - no torque. This is the truth the estimator
    must track in the determination runs.
  * ``simulate_controlled`` - a body torque from a callback each step
    (zero-order hold), used by the magnetorquer control code in
    ``scalar.control``. The actuator model itself is in ``scalar.actuators``.

    Euler:       I omega_dot = tau - omega x (I omega)   (tau = 0 in free tumble)
    Kinematics:  q_{B<-I} advanced by the attitude exponential map.

In free tumble the angular velocity obeys a q-independent ODE, so we
RK4-integrate omega on its own and advance the quaternion with the validated
``attitude.propagate`` using the step-averaged rate. The controlled integrator
uses the same scheme with the torque held over the step. Correctness of the
free-tumble case is checked by the two torque-free invariants: rotational
kinetic energy and the inertial angular-momentum vector are both conserved.

PLANT STEP vs CONTROL STEP. ``plant_step`` advances the truth over one control
step and is the one place the three simulators (here and in
``scalar.control.tune``) integrate. Two switches, both defaulting to what the
code did before they existed (bit for bit):

  * ``substeps`` (``plant_substeps``): the control step is integrated in N equal
    substeps. The quaternion update is second order in the substep, so its
    error falls 4x per doubling of N.
  * hold (``plant_hold``): ``"torque"`` freezes the body torque computed at the
    control instant for the whole control step. ``"dipole"`` freezes the coil
    dipole instead and re-evaluates  tau = m x B_body  inside the step, at
    every RK4 stage, with the attitude and the orbit position of that stage -
    what a real coil does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attitude import propagate, to_body, to_eci

__all__ = ["RigidBody", "default_inertia", "simulate_free_tumble",
           "simulate_controlled", "TruthTimeline", "plant_step", "PLANT_HOLDS"]

PLANT_HOLDS = ("torque", "dipole")


def default_inertia(mass=1.3, side=0.1, asymmetry=(0.9, 1.0, 1.15)):
    """Diagonal inertia [kg m^2] of a ~1U CubeSat.

    Base is a uniform cube ((1/6) m a^2 per axis); the ``asymmetry`` factors
    make the three principal moments distinct (a real bus is never a perfect
    uniform cube), which gives non-trivial tumbling instead of a degenerate
    symmetric top.
    """
    base = mass * side * side / 6.0
    return np.diag(base * np.asarray(asymmetry, dtype=float))


@dataclass
class RigidBody:
    """Rigid body with a (generally full) inertia tensor; torque is optional."""

    inertia: np.ndarray = None  # 3x3 [kg m^2]

    def __post_init__(self):
        if self.inertia is None:
            self.inertia = default_inertia()
        self.inertia = np.asarray(self.inertia, dtype=float)
        self.inv_inertia = np.linalg.inv(self.inertia)

    def omega_dot(self, omega, torque=None):
        """Angular acceleration [rad/s^2]: I^-1 (tau - omega x (I omega)).

        ``torque`` is the external body torque [N·m] (None -> torque-free).
        """
        rhs = -np.cross(omega, self.inertia @ omega)
        if torque is not None:
            rhs = rhs + np.asarray(torque, dtype=float)
        return self.inv_inertia @ rhs

    def kinetic_energy(self, omega):
        """Rotational kinetic energy [J] = 1/2 omega^T I omega (conserved)."""
        return 0.5 * float(omega @ self.inertia @ omega)

    def angular_momentum_eci(self, q, omega):
        """Angular momentum [kg m^2/s] in ECI = C^T (I omega) (conserved vector)."""
        return to_eci(q, self.inertia @ omega)


@dataclass
class TruthTimeline:
    """Truth attitude/rate history from a free-tumble or controlled run."""

    t: np.ndarray            # (N,)   time from epoch [s]
    q: np.ndarray            # (N, 4) attitude q_{B<-I}, scalar-last
    omega: np.ndarray        # (N, 3) body rate [rad/s]

    def __len__(self):
        return len(self.t)


def check_plant_switches(plant_hold, plant_substeps):
    """Raise ValueError unless the two plant switches hold legal values."""
    if plant_hold not in PLANT_HOLDS:
        raise ValueError(f"plant_hold={plant_hold!r} not in {PLANT_HOLDS}")
    if not isinstance(plant_substeps, (int, np.integer)) or plant_substeps < 1:
        raise ValueError(f"plant_substeps={plant_substeps!r} must be an integer >= 1")


def plant_step(body, q, w, dt, torque=None, substeps=1, dipole=None, field_fn=None, t=0.0):
    """Advance the truth ``(q, w)`` over one control step ``dt``; returns ``(q, w)``.

    The step is integrated in ``substeps`` equal substeps: RK4 on omega, then
    the exponential-map quaternion update with the substep-averaged rate.

    Torque hold (``dipole is None``): the body torque ``torque`` [N·m] is frozen
    in the body frame for the whole control step (None -> torque-free).

    Dipole hold (``dipole`` given): the body dipole ``dipole`` [A·m^2] is frozen
    and the torque is ``dipole x B_body`` at every RK4 stage, with
    ``B_body = C(q_stage) field_fn(t_stage)``; ``field_fn(time) -> B_eci`` [T]
    and ``t`` is the time at the start of the control step. The stage attitudes
    come from the same averaged-rate update, so the scheme stays second order.
    """
    h = dt / substeps
    if dipole is None:
        for _ in range(substeps):
            k1 = body.omega_dot(w, torque)
            k2 = body.omega_dot(w + 0.5 * h * k1, torque)
            k3 = body.omega_dot(w + 0.5 * h * k2, torque)
            k4 = body.omega_dot(w + h * k3, torque)
            w_next = w + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            q = propagate(q, 0.5 * (w + w_next), h)
            w = w_next
        return q, w
    m = np.asarray(dipole, dtype=float)
    B_end = field_fn(t)
    for j in range(substeps):
        B_start, B_mid, B_end = B_end, field_fn(t + (j + 0.5) * h), field_fn(t + (j + 1) * h)
        k1 = body.omega_dot(w, np.cross(m, to_body(q, B_start)))
        w2 = w + 0.5 * h * k1
        k2 = body.omega_dot(w2, np.cross(m, to_body(propagate(q, 0.5 * (w + w2), 0.5 * h), B_mid)))
        w3 = w + 0.5 * h * k2
        k3 = body.omega_dot(w3, np.cross(m, to_body(propagate(q, 0.5 * (w + w3), 0.5 * h), B_mid)))
        w4 = w + h * k3
        k4 = body.omega_dot(w4, np.cross(m, to_body(propagate(q, 0.5 * (w + w4), h), B_end)))
        w_next = w + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q = propagate(q, 0.5 * (w + w_next), h)
        w = w_next
    return q, w


def simulate_free_tumble(body, q0, omega0, dt, n_steps, plant_substeps=1):
    """Integrate torque-free dynamics, returning a :class:`TruthTimeline`.

    RK4 on omega + exponential-map quaternion update with the step-averaged
    rate (exact for constant omega, ~2nd order otherwise). The timeline is
    sampled every ``dt``; ``plant_substeps`` > 1 integrates each ``dt`` in that
    many substeps.
    """
    check_plant_switches("torque", plant_substeps)
    q = np.asarray(q0, dtype=float)
    w = np.asarray(omega0, dtype=float)
    qs = np.empty((n_steps + 1, 4))
    ws = np.empty((n_steps + 1, 3))
    qs[0], ws[0] = q, w
    for i in range(n_steps):
        q, w = plant_step(body, q, w, dt, substeps=plant_substeps)
        qs[i + 1], ws[i + 1] = q, w
    t = np.arange(n_steps + 1) * dt
    return TruthTimeline(t=t, q=qs, omega=ws)


def simulate_controlled(body, q0, omega0, dt, n_steps, torque_fn=None, plant_hold="torque",
                        plant_substeps=1, dipole_fn=None, field_fn=None):
    """Integrate closed-loop dynamics under a control torque.

    ``plant_hold="torque"`` (default): ``torque_fn(t, q, omega)`` returns the
    body torque [N·m] applied over the step (zero-order hold, the
    discrete-control convention).

    ``plant_hold="dipole"``: ``dipole_fn(t, q, omega)`` returns the body dipole
    [A·m^2] commanded at the control instant and held over the step, and
    ``field_fn(t)`` the ECI field [T]; the torque ``m x B_body`` is re-evaluated
    inside the step (see :func:`plant_step`).

    Either way the callback runs once per ``dt`` and the truth is integrated in
    ``plant_substeps`` substeps per ``dt``. Returns (:class:`TruthTimeline`,
    torques) where ``torques`` is (n_steps+1, 3), the torque at each control
    instant, with the last row repeated for shape alignment.
    """
    check_plant_switches(plant_hold, plant_substeps)
    if plant_hold == "torque" and (torque_fn is None or dipole_fn is not None or field_fn is not None):
        raise ValueError('plant_hold="torque" takes torque_fn only')
    if plant_hold == "dipole" and (torque_fn is not None or dipole_fn is None or field_fn is None):
        raise ValueError('plant_hold="dipole" takes dipole_fn and field_fn, not torque_fn')
    q = np.asarray(q0, dtype=float)
    w = np.asarray(omega0, dtype=float)
    qs = np.empty((n_steps + 1, 4))
    ws = np.empty((n_steps + 1, 3))
    taus = np.zeros((n_steps + 1, 3))
    qs[0], ws[0] = q, w
    for i in range(n_steps):
        if plant_hold == "torque":
            tau = np.asarray(torque_fn(i * dt, q, w), dtype=float)
            taus[i] = tau
            q, w = plant_step(body, q, w, dt, tau, plant_substeps)
        else:
            m = np.asarray(dipole_fn(i * dt, q, w), dtype=float)
            taus[i] = np.cross(m, to_body(q, field_fn(i * dt)))
            q, w = plant_step(body, q, w, dt, substeps=plant_substeps, dipole=m, field_fn=field_fn, t=i * dt)
        qs[i + 1], ws[i + 1] = q, w
    taus[n_steps] = taus[n_steps - 1] if n_steps > 0 else 0.0
    t = np.arange(n_steps + 1) * dt
    return TruthTimeline(t=t, q=qs, omega=ws), taus
