"""Periodic (time-varying) LQR for magnetorquer-only attitude control.

WHY THE TIME-INVARIANT LQR + PROJECTION FAILS FOR POINTING
----------------------------------------------------------
``lqr.py`` designs a 3-axis torque law as if the actuator were a reaction
wheel, then projects the request onto the plane perpendicular to B. The part
of the request along B is thrown away every step, the gain never accounts for
it, and for large-angle pointing the closed loop simply never converges (the
tuning sweep in ``scripts/lqr_tune.py`` found 100 % saturation and >100 deg
error for every weight set).

The standard remedy (Psiaki 2001; Wisniewski & Blanke 1999) is to put the
field *inside* the design. Near the target attitude the body-frame field
B_body(t) ~ C(q_target) B_eci(t) is a known periodic function of orbit phase,
so the linearised plant with the DIPOLE as the input,

    x = [dtheta; domega],   x_{k+1} = A x_k + G_k m_k
    A   = [[I3, dt I3], [0, I3]]
    G_k = [[dt^2/2 I^-1 (-[B_k]x)], [dt I^-1 (-[B_k]x)]]

is linear periodic. Iterating the discrete Riccati recursion backward over a
few orbits converges to the periodic solution, giving a gain schedule
K_k (3x6) indexed by orbit phase. The controller is then just  m = -K_k x,
followed by per-channel saturation. The along-B direction is handled by the
Riccati solution itself: it knows it cannot act there now and will be able
to a quarter orbit later.

WHAT FLIGHT HAS TO STORE
------------------------
One orbit at 1 Hz is ~5700 gains x 18 floats = 410 KB — fine on the RP2350,
not on the STM32L031 (32 KB flash). ``quantize_schedule`` bins the schedule
into ``n_bins`` orbit phases (64 bins = 4.6 KB float32, 1.2 KB int16), which
is what a payload-MCU implementation would carry. The gains are only valid
for the target attitude and orbit they were designed for; a new target means
a new schedule, uplinked or computed on the RP2350.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..attitude import IDENTITY_QUAT, attitude_error_rotvec, to_body
from ..environment.magfield import magnetic_field_eci

__all__ = ["periodic_lqr_schedule", "quantize_schedule", "PeriodicLQR",
           "bryson_weights", "TwoModeController", "ThreeModeController"]


def bryson_weights(theta_max_rad, omega_max_rad_s, m_max, rho=1.0):
    """Bryson's rule: Q = diag(1/x_max^2), R = rho * diag(1/m_max^2)."""
    Q = np.diag(np.concatenate([np.full(3, 1 / theta_max_rad ** 2),
                                np.full(3, 1 / omega_max_rad_s ** 2)]))
    R = rho * np.diag(1 / np.asarray(m_max, float) ** 2)
    return Q, R


def periodic_lqr_schedule(inertia, orbit, jd0, q_target, dt, Q, R, n_orbits_iter=4):
    """Return (K (N,3,6), B_body (N,3)) over one orbit at step ``dt``."""
    inertia = np.asarray(inertia, float)
    Iinv = np.linalg.inv(inertia)
    N = int(round(orbit.period / dt))
    ts = np.arange(N) * dt
    B_body = np.array([to_body(q_target, magnetic_field_eci(orbit.position(t), jd0 + t / 86400.0))
                       for t in ts])
    A = np.eye(6); A[:3, 3:] = dt * np.eye(3)
    G = np.zeros((N, 6, 3))
    for k, B in enumerate(B_body):
        Bx = np.array([[0, -B[2], B[1]], [B[2], 0, -B[0]], [-B[1], B[0], 0]])
        Gt = Iinv @ (-Bx)
        G[k, :3] = 0.5 * dt * dt * Gt
        G[k, 3:] = dt * Gt
    P = Q.copy()
    K = np.zeros((N, 3, 6))
    for it in range(n_orbits_iter):
        for k in range(N - 1, -1, -1):
            Gk = G[k]
            Kk = np.linalg.solve(R + Gk.T @ P @ Gk, Gk.T @ P @ A)
            P = Q + A.T @ P @ (A - Gk @ Kk)
            if it == n_orbits_iter - 1:
                K[k] = Kk
    return K, B_body


def quantize_schedule(K, n_bins):
    """Average the per-step schedule into ``n_bins`` orbit-phase bins."""
    N = len(K)
    edges = np.linspace(0, N, n_bins + 1).astype(int)
    return np.array([K[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])


@dataclass
class PeriodicLQR:
    """m = -K(phase) [dtheta; domega], then per-channel saturation via ``actuators``."""

    K: np.ndarray            # (N,3,6) or (n_bins,3,6)
    period: float
    actuators: object
    q_target: np.ndarray = None
    w_target: np.ndarray = None

    def __post_init__(self):
        self.q_target = IDENTITY_QUAT.copy() if self.q_target is None else np.asarray(self.q_target, float)
        self.w_target = np.zeros(3) if self.w_target is None else np.asarray(self.w_target, float)

    def gain(self, t):
        return self.K[int((t % self.period) / self.period * len(self.K)) % len(self.K)]

    def dipole(self, t, q, omega, B_meas=None):
        # SIGN: with q = q_{B<-I} and propagate()'s left-update by rotvec(-w dt),
        # d/dt attitude_error_rotvec(q, q_target) = -omega. The Riccati design
        # assumes dtheta_dot = +domega, so the error vector enters negated.
        # (lqr.py's desired_torque has the same convention mismatch.)
        x = np.concatenate([-attitude_error_rotvec(q, self.q_target),
                            np.asarray(omega, float) - self.w_target])
        return self.actuators.command(-self.gain(t) @ x)


@dataclass
class TwoModeController:
    """Rate damping until |omega| < ``enter``; periodic LQR after; back out above ``exit``.

    Mode 1 is the gyro-based damping law  m = k (omega x B)  (torque
    -k|B|^2 omega_perp), which is what DetumbleManager's B-dot approximates
    without a gyro. Mode 2 is the pointing schedule. The linear design is only
    trusted near zero rate, so this is the same supervisor structure C-26 asks
    about: DetumbleManager as mode 1 on the RP2350, the pointing law as mode 2.
    """

    lqr: PeriodicLQR
    k_damp: float = 5e3           # dipole per (rad/s * T); saturates anyway
    enter_deg_s: float = 0.3
    exit_deg_s: float = 1.5
    mode: int = 1

    @property
    def q_target(self):
        return self.lqr.q_target

    def dipole(self, t, q, omega, B_meas):
        rate = np.degrees(np.linalg.norm(omega))
        if self.mode == 1 and rate < self.enter_deg_s:
            self.mode = 2
        elif self.mode == 2 and rate > self.exit_deg_s:
            self.mode = 1
        if self.mode == 1:
            return self.lqr.actuators.command(self.k_damp * np.cross(omega, B_meas))
        return self.lqr.dipole(t, q, omega, B_meas)


@dataclass
class ThreeModeController(TwoModeController):
    """TwoModeController plus a fine-pointing schedule with its own weights.

    Mode 3 engages when the *estimated* attitude error is below
    ``fine_enter_deg`` and hands back to the acquisition schedule above
    ``fine_exit_deg``. The point: the acquisition weights need a strong rate
    term to survive large angles, but that same rate term turns gyro-bias
    estimation error into a pointing offset (~300 s x bias error). Fine mode
    can afford a much softer rate weight.
    """

    fine: PeriodicLQR = None
    fine_enter_deg: float = 15.0
    fine_exit_deg: float = 30.0

    def dipole(self, t, q, omega, B_meas):
        rate = np.degrees(np.linalg.norm(omega))
        err = np.degrees(np.linalg.norm(attitude_error_rotvec(q, self.q_target)))
        if self.mode == 1 and rate < self.enter_deg_s:
            self.mode = 2
        elif self.mode in (2, 3) and rate > self.exit_deg_s:
            self.mode = 1
        elif self.mode == 2 and err < self.fine_enter_deg:
            self.mode = 3
        elif self.mode == 3 and err > self.fine_exit_deg:
            self.mode = 2
        if self.mode == 1:
            return self.lqr.actuators.command(self.k_damp * np.cross(omega, B_meas))
        if self.mode == 3 and self.fine is not None:
            return self.fine.dipole(t, q, omega, B_meas)
        return self.lqr.dipole(t, q, omega, B_meas)
