"""Fine-pointing LQI: periodic LQR plus an integral state on attitude error.

WHY
    A slowly walking gyro-bias estimation error db enters the rate feedback as a
    fake rate and the proportional law balances it with an attitude offset
    (~ K_omega/K_theta x db, ~300 s x db). An integral on attitude error removes
    that static balance: at equilibrium the integrator, not the attitude,
    carries the K_omega db term.

MODEL (nine states, design convention dtheta_dot = +omega)
    x = [dtheta; domega; z],   z_{k+1} = z_k + dt dtheta_k
    A = [[I, dt I, 0], [0, I, 0], [dt I, 0, I]]
    G_k = [[dt^2/2 I^-1 (-[B_k]x)], [dt I^-1 (-[B_k]x)], [0]]
    Q = blkdiag(Q6, I3 / z_max^2),  z_max = theta_max_fine * T_i
Same backward periodic Riccati iteration as ``periodic_lqr_schedule``, same
256-bin quantisation.

SIGN
    d/dt attitude_error_rotvec(q, q_target) = -omega, so dtheta = -aer (as in
    PeriodicLQR.dipole) and the integrator accumulates that negated error.

ANTI-WINDUP / RESET
    * integrate only while |dtheta| < ``gate_rad`` and the unclipped command is
      inside every channel's limit (conditional integration);
    * clamp each |z_i| <= ``z_clamp``;
    * reset z to zero on every entry into fine mode: the integrator is only
      called while the supervisor is in mode 3, so a missed step (gap in t, or
      t going backwards = new episode) means the mode was left; ``reset()`` is
      also public.

Nothing in periodic_lqr.py is changed; ``LQIThreeModeController`` is a
ThreeModeController whose ``fine`` slot holds a ``PeriodicLQI``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..attitude import IDENTITY_QUAT, attitude_error_rotvec, to_body
from ..environment.magfield import magnetic_field_eci
from .periodic_lqr import PeriodicLQR, ThreeModeController, bryson_weights, quantize_schedule

__all__ = ["periodic_lqi_schedule", "lqi_weights", "PeriodicLQI", "LQIThreeModeController"]


def lqi_weights(theta_max_rad, omega_max_rad_s, m_max, rho, T_i_s):
    """Bryson Q6/R from periodic_lqr plus an integral weight 1/(theta_max*T_i)^2."""
    Q6, R = bryson_weights(theta_max_rad, omega_max_rad_s, m_max, rho)
    z_max = theta_max_rad * T_i_s
    Q = np.zeros((9, 9))
    Q[:6, :6] = Q6
    Q[6:, 6:] = np.eye(3) / z_max ** 2
    return Q, R


def periodic_lqi_schedule(inertia, orbit, jd0, q_target, dt, Q, R, n_orbits_iter=4, B_body=None):
    """Return (K (N,3,9), B_body (N,3)) over one orbit at step ``dt``."""
    Iinv = np.linalg.inv(np.asarray(inertia, float))
    N = int(round(orbit.period / dt))
    if B_body is None:
        ts = np.arange(N) * dt
        B_body = np.array([to_body(q_target, magnetic_field_eci(orbit.position(t), jd0 + t / 86400.0))
                           for t in ts])
    I3 = np.eye(3)
    A = np.eye(9)
    A[:3, 3:6] = dt * I3
    A[6:, :3] = dt * I3
    G = np.zeros((N, 9, 3))
    for k, B in enumerate(B_body):
        Bx = np.array([[0, -B[2], B[1]], [B[2], 0, -B[0]], [-B[1], B[0], 0]])
        Gt = Iinv @ (-Bx)
        G[k, :3] = 0.5 * dt * dt * Gt
        G[k, 3:6] = dt * Gt
    P = Q.copy()
    K = np.zeros((N, 3, 9))
    for it in range(n_orbits_iter):
        for k in range(N - 1, -1, -1):
            Gk = G[k]
            Kk = np.linalg.solve(R + Gk.T @ P @ Gk, Gk.T @ P @ A)
            P = Q + A.T @ P @ (A - Gk @ Kk)
            if it == n_orbits_iter - 1:
                K[k] = Kk
    return K, B_body


@dataclass
class PeriodicLQI(PeriodicLQR):
    """m = -K(phase) [dtheta; domega; z]; z integrates dtheta with anti-windup."""

    dt: float = 1.0
    gate_rad: float = np.deg2rad(10.0)
    z_clamp: float = np.inf
    z: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _last_t: float = None
    n_resets: int = 0
    n_frozen: int = 0

    def reset(self):
        self.z = np.zeros(3)
        self._last_t = None
        self.n_resets += 1

    def dipole(self, t, q, omega, B_meas=None):
        if self._last_t is None or t <= self._last_t or t - self._last_t > 1.5 * self.dt:
            self.reset()                                  # fine mode (re-)entered
        self._last_t = t
        dth = -attitude_error_rotvec(q, self.q_target)    # sign: d/dt aer = -omega
        x = np.concatenate([dth, np.asarray(omega, float) - self.w_target, self.z])
        m_des = -self.gain(t) @ x
        m = self.actuators.command(m_des)
        saturated = any(abs(float(m_des @ tq.axis)) > tq.m_max for tq in self.actuators.torquers)
        if np.linalg.norm(dth) < self.gate_rad and not saturated:
            self.z = np.clip(self.z + self.dt * dth, -self.z_clamp, self.z_clamp)
        else:
            self.n_frozen += 1
        return m


@dataclass
class LQIThreeModeController(ThreeModeController):
    """ThreeModeController with a PeriodicLQI in the fine slot; resets z when mode 3 is left."""

    def dipole(self, t, q, omega, B_meas):
        m = super().dipole(t, q, omega, B_meas)
        if self.mode != 3 and isinstance(self.fine, PeriodicLQI) and self.fine._last_t is not None:
            self.fine.reset()
        return m


def build_lqi_fine(plant_inertia, acts, orbit, jd0, fth_deg, fwm_deg_s, frho, T_i_s,
                   gate_deg=10.0, z_clamp_factor=1.0, n_bins=256, dt=1.0, B_body=None):
    """Fine-mode PeriodicLQI from the tuned fine Bryson weights plus T_i.

    ``z_clamp`` = z_clamp_factor * z_max (the Bryson integral scale), per axis.
    """
    th = np.deg2rad(fth_deg)
    Q, R = lqi_weights(th, np.deg2rad(fwm_deg_s), acts.m_max_vector, frho, T_i_s)
    K, B_body = periodic_lqi_schedule(plant_inertia, orbit, jd0, IDENTITY_QUAT, dt, Q, R,
                                      n_orbits_iter=5, B_body=B_body)
    return PeriodicLQI(quantize_schedule(K, n_bins), orbit.period, acts, IDENTITY_QUAT,
                       dt=dt, gate_rad=np.deg2rad(gate_deg), z_clamp=z_clamp_factor * th * T_i_s)
