"""Unscented attitude filter — USQUE (Crassidis & Markley, 2003).

An Unscented Kalman Filter for the same 6-state attitude-determination problem
as the MEKF, but replacing the analytic Jacobians with sigma-point (unscented)
propagation of the mean and covariance. This captures second-order effects the
first-order MEKF linearizes away, which matters for large initial errors, fast
tumble, and strong systematic biases.

PARAMETERIZATION
----------------
Attitude error is carried as **Generalized Rodrigues Parameters (GRP)** with
tuning constants ``(a, f)``: dp = f * dq_vec / (a + dq_scalar). The default
a=1, f=4 makes dp equal the rotation vector for small angles (so the UKF and
MEKF share the same error metric); a=0, f=1 gives the classical Gibbs/Rodrigues
vector. These, plus the unscented scaling ``(alpha, beta, kappa)``, are the
"creative combination" knobs the exploration sweeps.

The reference quaternion q_{B<-I} is carried outside the state; the error state
[dp (3), gyro-bias (3)] has mean recentered to dp=0 after each propagate/update
(the multiplicative trick). Measurements are unit vectors (mag, sun), fused
sequentially.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..attitude import (attitude_error_angle, propagate, quat_conj, quat_mul,
                        to_body)
from ..constants import DEG
from .mekf import MEKFResult, discrete_process_noise
from .triad import triad

__all__ = ["USQUE", "run_ukf"]


def _chol(P):
    """Lower-triangular matrix square root, with jitter if not quite PD."""
    P = 0.5 * (P + P.T)
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(P + 1e-9 * np.eye(len(P)))


def grp_from_dq(dq, a, f):
    """Error quaternion (scalar-last) -> GRP vector."""
    dq = dq / np.linalg.norm(dq)
    if dq[3] < 0:
        dq = -dq                      # keep scalar >= 0 for a unique GRP
    return f * dq[:3] / (a + dq[3])


def dq_from_grp(dp, a, f):
    """GRP vector -> error quaternion (scalar-last)."""
    dp2 = float(dp @ dp)
    dq4 = (-a * dp2 + f * np.sqrt(f * f + (1.0 - a * a) * dp2)) / (f * f + dp2)
    dqv = ((a + dq4) / f) * dp
    return np.array([dqv[0], dqv[1], dqv[2], dq4])


@dataclass
class USQUE:
    """Unscented quaternion estimator (attitude + gyro bias)."""

    sigma_v: float = 3.8e-3 * DEG        # gyro ARW [rad/sqrt(s)]
    sigma_u: float = 1.0e-5              # gyro bias RW [rad/s/sqrt(s)]
    sigma_sun: float = 0.06              # sun-vector meas noise [rad]
    sigma_mag: float = 0.03             # mag-direction meas noise [rad]
    a: float = 1.0                       # GRP parameter
    f: float = 4.0                       # GRP scale (4 => dp ~ rotation vector)
    alpha: float = 1.0                   # unscented spread
    beta: float = 2.0                    # unscented prior-knowledge term
    kappa: float = 0.0                   # unscented secondary scaling
    q: np.ndarray = None
    b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray = None

    def __post_init__(self):
        n = 6
        self.n = n
        self.lam = self.alpha ** 2 * (n + self.kappa) - n
        self.gamma = np.sqrt(n + self.lam)
        self.Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (n + self.lam)
        self.Wc[0] = self.lam / (n + self.lam) + (1.0 - self.alpha ** 2 + self.beta)
        if self.q is None:
            self.q = np.array([0.0, 0.0, 0.0, 1.0])
        if self.P is None:
            self.P = np.diag(np.concatenate([
                (30 * DEG) ** 2 * np.ones(3), (0.5 * DEG) ** 2 * np.ones(3)]))
        self.q = np.asarray(self.q, float)
        self.b = np.asarray(self.b, float)
        self.P = np.asarray(self.P, float)

    def init_triad(self, sun_eci, sun_body, B_eci, B_body):
        self.q = triad(sun_eci, sun_body, B_eci / np.linalg.norm(B_eci),
                       B_body / np.linalg.norm(B_body))

    # --- sigma points around the current (dp=0, bias=b) mean ----------------
    def _sigma(self):
        n = self.n
        S = self.gamma * _chol(self.P)
        mean = np.concatenate([np.zeros(3), self.b])
        chi = np.zeros((2 * n + 1, 6))
        chi[0] = mean
        for i in range(n):
            chi[1 + i] = mean + S[:, i]
            chi[1 + n + i] = mean - S[:, i]
        return chi

    def _quat_of(self, chi_row):
        return quat_mul(dq_from_grp(chi_row[:3], self.a, self.f), self.q)

    # --- propagation --------------------------------------------------------
    def propagate(self, gyro_meas, dt):
        n = self.n
        gyro_meas = np.asarray(gyro_meas, float)
        chi = self._sigma()
        q_prop = np.array([propagate(self._quat_of(chi[i]),
                                     gyro_meas - chi[i, 3:], dt)
                           for i in range(2 * n + 1)])
        q0 = q_prop[0]
        chi_p = np.zeros((2 * n + 1, 6))
        for i in range(2 * n + 1):
            dq = quat_mul(q_prop[i], quat_conj(q0))
            chi_p[i, :3] = grp_from_dq(dq, self.a, self.f)
            chi_p[i, 3:] = chi[i, 3:]          # bias constant (RW enters via Q)
        xbar = self.Wm @ chi_p
        Q = discrete_process_noise(self.sigma_v, self.sigma_u, dt)
        P = Q.copy()
        for i in range(2 * n + 1):
            d = chi_p[i] - xbar
            P += self.Wc[i] * np.outer(d, d)
        # fold the mean attitude error into the reference; recenter dp -> 0
        self.q = quat_mul(dq_from_grp(xbar[:3], self.a, self.f), q0)
        self.b = xbar[3:]
        self.P = 0.5 * (P + P.T)

    # --- unit-vector measurement update (sequential) ------------------------
    def update_vector(self, r_eci, z_meas, sigma):
        n = self.n
        r_hat = np.asarray(r_eci, float) / np.linalg.norm(r_eci)
        z = np.asarray(z_meas, float)
        z = z / np.linalg.norm(z)
        chi = self._sigma()
        mean = np.concatenate([np.zeros(3), self.b])
        Z = np.zeros((2 * n + 1, 3))
        for i in range(2 * n + 1):
            zp = to_body(self._quat_of(chi[i]), r_hat)
            Z[i] = zp / np.linalg.norm(zp)
        zbar = self.Wm @ Z
        zbar /= np.linalg.norm(zbar)
        Pzz = (sigma ** 2) * np.eye(3)
        Pxz = np.zeros((6, 3))
        for i in range(2 * n + 1):
            dz = Z[i] - zbar
            dx = chi[i] - mean
            Pzz += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        K = Pxz @ np.linalg.inv(Pzz)
        dx = K @ (z - zbar)
        self.q = quat_mul(dq_from_grp(dx[:3], self.a, self.f), self.q)
        self.b = self.b + dx[3:]
        P = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (P + P.T)

    def update_mag(self, B_eci, B_body):
        self.update_vector(B_eci, B_body, self.sigma_mag)

    def update_sun(self, sun_eci, sun_body):
        self.update_vector(sun_eci, sun_body, self.sigma_sun)


def run_ukf(res, ukf=None, init_triad=True):
    """Run the USQUE over a SimResult -> MEKFResult (same interface as run_mekf)."""
    if ukf is None:
        ukf = USQUE()

    n = len(res)
    q_est = np.empty((n, 4))
    b_est = np.empty((n, 3))
    err = np.empty(n)
    P_theta = np.empty(n)

    started = False
    for k in range(n):
        if k > 0:
            ukf.propagate(res.gyro_meas[k], res.dt)
        if init_triad and not started and res.sun_valid[k]:
            ukf.init_triad(res.sun_eci[k], res.sun_body[k],
                           res.B_eci_ref[k], res.mag_body[k])
            started = True
        ukf.update_mag(res.B_eci_ref[k], res.mag_body[k])
        if res.sun_valid[k]:
            ukf.update_sun(res.sun_eci[k], res.sun_body[k])

        q_est[k] = ukf.q
        b_est[k] = ukf.b
        err[k] = np.rad2deg(attitude_error_angle(ukf.q, res.q_true[k]))
        P_theta[k] = np.rad2deg(np.sqrt(np.trace(ukf.P[:3, :3])))

    return MEKFResult(t=res.t, q_est=q_est, b_est=b_est, err_deg=err,
                      lit=res.lit, P_theta_deg=P_theta)
