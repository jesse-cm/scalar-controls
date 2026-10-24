"""LQR magnetorquer attitude controller.

Underactuated magnetic control: the torque is ``tau = m x B``, which can only lie
in the plane perpendicular to B, so at any instant we have 2-DOF authority. The
system is nonetheless controllable *over an orbit* because B rotates through the
body frame. The controller is the textbook split:

  1. **LQR feedback** on the linearized attitude+rate plant gives a *desired*
     3-axis torque  tau_d = -K [dtheta; domega].
  2. **Torque -> dipole projection**  m = (B x tau_d) / |B|²  realizes the part of
     tau_d perpendicular to B (the achievable part); the along-B part is dropped.
  3. **Saturation** through the actuator set (2 rods + 1 PCB coil).

Linear plant (small error about the target, gyroscopic term neglected near the
target rate):
    dtheta_dot = domega,   I domega_dot = tau
    A = [[0, I3],[0, 0]],  B = [[0],[I^-1]]
solved via the continuous-time algebraic Riccati equation.

State feedback here uses the *true* attitude/rate (full-state LQR). Closing the
loop on the MEKF estimate instead (the separation principle) is the next step.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from ..attitude import IDENTITY_QUAT, attitude_error_rotvec, to_body
from ..environment.magfield import magnetic_field_eci

__all__ = ["lqr_gain", "MagnetorquerLQR"]


def lqr_gain(inertia, q_weight=1.0, w_weight=1.0e3, r_weight=1.0e10):
    """LQR gain K (3x6) for the linearized attitude+rate plant.

    ``q_weight``/``w_weight`` weight attitude-error and rate-error cost; the
    large ``r_weight`` reflects that magnetic torque is tiny (m~0.2 A·m², B~3e-5 T
    -> tau~1e-6 N·m), so the input is cheap to penalize and the loop is soft.
    """
    inertia = np.asarray(inertia, dtype=float)
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    B = np.zeros((6, 3))
    B[3:, :] = np.linalg.inv(inertia)
    Q = np.diag(np.concatenate([q_weight * np.ones(3), w_weight * np.ones(3)]))
    R = r_weight * np.eye(3)
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P


class MagnetorquerLQR:
    """Callable controller: ``(t, q, omega) -> body torque`` for simulate_controlled.

    Logs the commanded dipole, realized torque, and body field each call.
    """

    def __init__(self, inertia, actuators, orbit, jd0, q_target=None,
                 w_target=None, q_weight=1.0, w_weight=1.0e3, r_weight=1.0e10):
        self.K = lqr_gain(inertia, q_weight, w_weight, r_weight)
        self.actuators = actuators
        self.orbit = orbit
        self.jd0 = jd0
        self.q_target = (IDENTITY_QUAT.copy() if q_target is None
                         else np.asarray(q_target, float))
        self.w_target = (np.zeros(3) if w_target is None
                         else np.asarray(w_target, float))
        self.m_hist, self.tau_hist, self.B_hist = [], [], []

    def desired_torque(self, q, omega):
        """Full 3-axis LQR torque request (before the magnetic projection)."""
        # d/dt attitude_error_rotvec(q, q_target) = -omega with q_{B<-I}; the
        # plant in lqr_gain assumes dtheta_dot = +domega, so the error enters negated.
        dtheta = -attitude_error_rotvec(q, self.q_target)
        domega = np.asarray(omega, float) - self.w_target
        return -self.K @ np.concatenate([dtheta, domega])

    def field_body(self, t, q):
        r = self.orbit.position(t)
        return to_body(q, magnetic_field_eci(r, self.jd0 + t / 86400.0))

    def __call__(self, t, q, omega):
        B_body = self.field_body(t, q)
        tau_d = self.desired_torque(q, omega)
        m_desired = np.cross(B_body, tau_d) / (B_body @ B_body)   # ⊥B projection
        tau, m = self.actuators.torque(m_desired, B_body)
        self.m_hist.append(m)
        self.tau_hist.append(tau)
        self.B_hist.append(B_body)
        return tau
