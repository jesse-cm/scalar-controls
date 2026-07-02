"""Baseline Multiplicative Extended Kalman Filter (MEKF).

The estimator the SCALAR slide's "reference state + error state" box describes.
It fuses gyro rate (propagation) with magnetometer and sun-vector directions
(updates) to reconstruct 3D attitude, and estimates the gyro bias.

STATE
-----
A reference quaternion ``q`` (= q_{B<-I}) is carried outside the linear state.
The 6-D error state is  x = [dtheta (3), db (3)]  — a small body-frame attitude
error and the gyro-bias error — with covariance ``P`` (6x6). The multiplicative
convention is  q_true = dq(dtheta) (x) q,  with dq(dtheta) ~ [dtheta/2, 1].

PROPAGATION (gyro-driven)
-------------------------
    omega_hat = omega_gyro - b
    q      <- propagate(q, omega_hat, dt)              (exp-map, exact for const omega)
    P      <- F P F^T + Q
with F = [[expm(-[omega_hat]x dt), -I dt], [0, I]] and Q the standard
Crassidis-Markley discrete gyro process noise from the ARW (sigma_v) and
bias random walk (sigma_u).

UPDATE (unit-vector measurements)
---------------------------------
For a reference direction r_eci with body measurement b_meas (both unit),
    b_pred = C(q) r_eci,   y = b_meas - b_pred,   H = [-[b_pred]x , 0].
The sign follows from C(q_true) = (I + [dtheta]x) C(q) for this convention, so
d(C r) = -[C r]x dtheta. Standard Kalman gain, Joseph-form covariance update,
then the correction is injected multiplicatively into q and additively into b
and the error state resets to zero.

ECLIPSE
-------
When the sun vector is unavailable (eclipse), only the magnetometer updates.
A single vector cannot observe rotation about itself, so attitude about the
B-axis is momentarily unobservable and the filter coasts on the gyro — the
regime where the 10-15 deg knowledge budget is stressed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..attitude import (attitude_error_angle, propagate, quat_conj,
                        quat_from_small_angle, quat_mul, to_body)
from ..constants import DEG
from .triad import triad

__all__ = ["MEKF", "MEKFResult", "run_mekf"]


def skew(v):
    """Cross-product matrix [v]x such that [v]x u = v x u."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def discrete_process_noise(sigma_v, sigma_u, dt):
    """Crassidis-Markley 6x6 discrete gyro process noise.

    sigma_v: angle random walk / rate noise density [rad/sqrt(s)].
    sigma_u: gyro bias random walk [rad/s/sqrt(s)].
    """
    I3 = np.eye(3)
    q_tt = (sigma_v ** 2 * dt + sigma_u ** 2 * dt ** 3 / 3.0) * I3
    q_tb = -(sigma_u ** 2 * dt ** 2 / 2.0) * I3
    q_bb = (sigma_u ** 2 * dt) * I3
    Q = np.zeros((6, 6))
    Q[:3, :3], Q[:3, 3:], Q[3:, :3], Q[3:, 3:] = q_tt, q_tb, q_tb, q_bb
    return Q


@dataclass
class MEKF:
    """Multiplicative EKF for attitude + gyro bias."""

    sigma_v: float = 3.8e-3 * DEG              # gyro ARW [rad/sqrt(s)]
    sigma_u: float = 1.0e-5                    # gyro bias RW [rad/s/sqrt(s)]
    sigma_sun: float = 0.06                    # sun-vector meas noise [rad]
    sigma_mag: float = 0.03                    # mag-direction meas noise [rad]
    q: np.ndarray = None                       # reference attitude q_{B<-I}
    b: np.ndarray = field(default_factory=lambda: np.zeros(3))   # bias estimate
    P: np.ndarray = None                       # 6x6 covariance

    def __post_init__(self):
        if self.q is None:
            self.q = np.array([0.0, 0.0, 0.0, 1.0])
        if self.P is None:
            self.P = np.diag(np.concatenate([
                (30 * DEG) ** 2 * np.ones(3),      # attitude uncertainty
                (0.5 * DEG) ** 2 * np.ones(3)]))   # bias uncertainty [rad/s]
        self.q = np.asarray(self.q, float)
        self.b = np.asarray(self.b, float)
        self.P = np.asarray(self.P, float)

    # --- initialization -----------------------------------------------------
    def init_triad(self, sun_eci, sun_body, B_eci, B_body):
        """Seed the reference attitude from a two-vector TRIAD solution."""
        self.q = triad(sun_eci, sun_body, B_eci / np.linalg.norm(B_eci),
                       B_body / np.linalg.norm(B_body))

    # --- propagation --------------------------------------------------------
    def propagate(self, gyro_meas, dt):
        omega_hat = np.asarray(gyro_meas, float) - self.b
        self.q = propagate(self.q, omega_hat, dt)
        F = np.eye(6)
        F[:3, :3] = np.eye(3) - skew(omega_hat) * dt   # ~ expm(-[omega]x dt)
        F[:3, 3:] = np.eye(3) * dt                     # dtheta/dbias coupling
        Q = discrete_process_noise(self.sigma_v, self.sigma_u, dt)
        self.P = F @ self.P @ F.T + Q

    # --- measurement update -------------------------------------------------
    def update_vector(self, r_eci, b_meas, sigma):
        """Fuse one unit-vector measurement (body ``b_meas`` of ECI ``r_eci``)."""
        r_hat = np.asarray(r_eci, float) / np.linalg.norm(r_eci)
        b_pred = to_body(self.q, r_hat)
        b_meas = np.asarray(b_meas, float)
        b_meas = b_meas / np.linalg.norm(b_meas)

        H = np.zeros((3, 6))
        H[:, :3] = -skew(b_pred)
        R = (sigma ** 2) * np.eye(3)
        y = b_meas - b_pred
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        dtheta, db = dx[:3], dx[3:]
        self.q = quat_mul(quat_from_small_angle(dtheta), self.q)
        self.b = self.b + db
        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T   # Joseph form

    def update_mag(self, B_eci, B_body):
        self.update_vector(B_eci, B_body, self.sigma_mag)

    def update_sun(self, sun_eci, sun_body):
        self.update_vector(sun_eci, sun_body, self.sigma_sun)

    def update_attitude(self, q_meas, sigma):
        """Fuse a full absolute-attitude measurement (a star tracker).

        The measurement is a quaternion ``q_meas ~ q_{B<-I}``; the innovation is
        the small-angle rotation carrying the predicted attitude to the
        measured one, ``dtheta = 2 * vec(q_meas * q_pred^{-1})``, observed
        directly through the error state (``H = [I  0]``).

        ``sigma`` is the 1-sigma per-axis attitude error [rad] (scalar, or a
        3-vector for an anisotropic tracker).
        """
        q_meas = np.asarray(q_meas, float)
        # error quaternion measured-from-predicted, in the body/error frame
        dq = quat_mul(q_meas, quat_conj(self.q))
        dq = dq * np.sign(dq[3] if dq[3] != 0 else 1.0)   # shortest rotation
        y = 2.0 * dq[:3]                                   # small-angle vector

        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        sig = np.atleast_1d(np.asarray(sigma, float))
        if sig.size == 1:
            R = (sig[0] ** 2) * np.eye(3)
        else:
            R = np.diag(sig ** 2)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        dtheta, db = dx[:3], dx[3:]
        self.q = quat_mul(quat_from_small_angle(dtheta), self.q)
        self.b = self.b + db
        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T   # Joseph form


@dataclass
class MEKFResult:
    """MEKF run history + error metrics."""

    t: np.ndarray
    q_est: np.ndarray        # (N, 4)
    b_est: np.ndarray        # (N, 3)
    err_deg: np.ndarray      # (N,)  attitude knowledge error
    lit: np.ndarray          # (N,)  bool
    P_theta_deg: np.ndarray  # (N,)  1-sigma attitude uncertainty (sqrt trace)

    def summary(self):
        """Dict of headline error stats (deg), overall / lit / eclipse."""
        e = self.err_deg
        conv = e[len(e) // 5:]   # ignore the initial convergence transient
        out = {
            "median_deg": float(np.median(conv)),
            "p95_deg": float(np.percentile(conv, 95)),
            "final_deg": float(e[-1]),
            "max_deg": float(np.max(conv)),
        }
        if self.lit.any():
            out["lit_median_deg"] = float(np.median(e[self.lit]))
        if (~self.lit).any():
            out["eclipse_median_deg"] = float(np.median(e[~self.lit]))
        return out


def run_mekf(res, mekf=None, init_triad=True):
    """Run the MEKF over a :class:`~scalar.scenario.SimResult`.

    Propagates on the gyro every step, updates on the magnetometer every step,
    and updates on the sun vector whenever it is valid (sunlit). Returns a
    :class:`MEKFResult` with the attitude knowledge error vs the truth.
    """
    if mekf is None:
        mekf = MEKF()

    n = len(res)
    q_est = np.empty((n, 4))
    b_est = np.empty((n, 3))
    err = np.empty(n)
    P_theta = np.empty(n)

    started = False
    for k in range(n):
        if k > 0:
            mekf.propagate(res.gyro_meas[k], res.dt)

        # one-time TRIAD seed at the first sunlit sample (needs both vectors).
        # The estimator only knows its *reference* field (dipole), not the truth.
        if init_triad and not started and res.sun_valid[k]:
            mekf.init_triad(res.sun_eci[k], res.sun_body[k],
                            res.B_eci_ref[k], res.mag_body[k])
            started = True

        mekf.update_mag(res.B_eci_ref[k], res.mag_body[k])
        if res.sun_valid[k]:
            mekf.update_sun(res.sun_eci[k], res.sun_body[k])

        q_est[k] = mekf.q
        b_est[k] = mekf.b
        err[k] = np.rad2deg(attitude_error_angle(mekf.q, res.q_true[k]))
        P_theta[k] = np.rad2deg(np.sqrt(np.trace(mekf.P[:3, :3])))

    return MEKFResult(t=res.t, q_est=q_est, b_est=b_est, err_deg=err,
                      lit=res.lit, P_theta_deg=P_theta)
