"""Rigid-body attitude dynamics for a freely tumbling CubeSat.

Determination-only sim: there is **no control torque and no actuator**. The
spacecraft tumbles under torque-free rigid-body dynamics (Euler's equation),
which is the truth the estimator must track.

    Euler:       I omega_dot = -omega x (I omega)        (tau = 0)
    Kinematics:  q_{B<-I} advanced by the attitude exponential map.

Angular velocity obeys a q-independent ODE (torque-free), so we RK4-integrate
omega on its own and advance the quaternion with the validated
``attitude.propagate`` using the step-averaged rate. Correctness is checked by
the two torque-free invariants: rotational kinetic energy and the inertial
angular-momentum vector are both conserved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attitude import propagate, to_eci

__all__ = ["RigidBody", "default_inertia", "simulate_free_tumble", "TruthTimeline"]


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
    """Torque-free rigid body with a (generally full) inertia tensor."""

    inertia: np.ndarray = None  # 3x3 [kg m^2]

    def __post_init__(self):
        if self.inertia is None:
            self.inertia = default_inertia()
        self.inertia = np.asarray(self.inertia, dtype=float)
        self.inv_inertia = np.linalg.inv(self.inertia)

    def omega_dot(self, omega):
        """Angular acceleration [rad/s^2] from Euler's equation (tau = 0)."""
        H = self.inertia @ omega
        return self.inv_inertia @ (-np.cross(omega, H))

    def kinetic_energy(self, omega):
        """Rotational kinetic energy [J] = 1/2 omega^T I omega (conserved)."""
        return 0.5 * float(omega @ self.inertia @ omega)

    def angular_momentum_eci(self, q, omega):
        """Angular momentum [kg m^2/s] in ECI = C^T (I omega) (conserved vector)."""
        return to_eci(q, self.inertia @ omega)


@dataclass
class TruthTimeline:
    """Truth attitude/rate history from a free-tumble run."""

    t: np.ndarray            # (N,)   time from epoch [s]
    q: np.ndarray            # (N, 4) attitude q_{B<-I}, scalar-last
    omega: np.ndarray        # (N, 3) body rate [rad/s]

    def __len__(self):
        return len(self.t)


def simulate_free_tumble(body, q0, omega0, dt, n_steps):
    """Integrate torque-free dynamics, returning a :class:`TruthTimeline`.

    RK4 on omega + exponential-map quaternion update with the step-averaged
    rate (exact for constant omega, ~2nd order otherwise).
    """
    q = np.asarray(q0, dtype=float)
    w = np.asarray(omega0, dtype=float)
    qs = np.empty((n_steps + 1, 4))
    ws = np.empty((n_steps + 1, 3))
    qs[0], ws[0] = q, w
    for i in range(n_steps):
        k1 = body.omega_dot(w)
        k2 = body.omega_dot(w + 0.5 * dt * k1)
        k3 = body.omega_dot(w + 0.5 * dt * k2)
        k4 = body.omega_dot(w + dt * k3)
        w_next = w + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q = propagate(q, 0.5 * (w + w_next), dt)
        w = w_next
        qs[i + 1], ws[i + 1] = q, w
    t = np.arange(n_steps + 1) * dt
    return TruthTimeline(t=t, q=qs, omega=ws)
