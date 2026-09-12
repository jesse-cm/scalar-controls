"""Identify the magnetorquer dipole matrix (and residual dipole) from flight bursts.

THE PROBLEM
-----------
Three sources describe three different actuator suites for SCALAR (S3's five
PROVES face coils, S6's three-channel H-bridge board, S4's rods-plus-coil), and
none has been measured. Any LQR gain computed from a paper dipole is a gain for
a spacecraft that may not exist. The fix is to *measure* the actuator on orbit
from the data the A9 burst record is already specified to carry: coil current,
body field and body rate at >= 10 Hz around each torque pulse.

MODEL
-----
    I w_dot + w x (I w) = (M u + m_res) x B          (body frame)

  * u      channel currents [A]            (n_ch,)     — measured on the STM32
  * M      dipole per amp [A m^2 / A]      (3, n_ch)   — UNKNOWN, the thing we want
  * m_res  residual spacecraft dipole      (3,)        — UNKNOWN nuisance, real
  * B      body field [T]                             — magnetometer
  * w      body rate [rad/s]                          — gyro (bias cancels in w_dot)
  * I      inertia [kg m^2]                           — from CAD / ground measurement

Because  (M u) x B = -[B]x (u^T kron I3) vec(M),  the equation is LINEAR in
[vec(M); m_res] once I is known, and plain least squares solves it. One
unavoidable fact: motion data alone determines only M / |I| — scaling I and M
together leaves every trajectory unchanged — so the inertia scale must come from
the ground (a mass-properties measurement or CAD). The inertia *shape* can be
checked in flight from any torque-free segment (:func:`inertia_shape_from_tumble`).

EXCITATION
----------
Pulse one channel at a time, alternating polarity, so the columns of M are
separately excited. The gyro must be sampled fast enough to resolve the pulse:
at the software's current 1 Hz ceiling a 320 ms pulse leaves at most one sample
and the fit is meaningless — that is the quantitative case for A9's 10 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..attitude import to_body
from ..dynamics import RigidBody, simulate_controlled
from ..environment.magfield import magnetic_field_eci

__all__ = ["skew", "inertia_shape_from_tumble", "identify_dipole_matrix",
           "identify_dipole_matrix_impulse", "PulseSchedule",
           "simulate_burst_campaign", "s3_coil_matrix"]


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _G(w):
    """3x6 operator with G(w) @ [Ixx,Iyy,Izz,Ixy,Ixz,Iyz] = I w."""
    wx, wy, wz = w
    return np.array([[wx, 0, 0, wy, wz, 0],
                     [0, wy, 0, wx, 0, wz],
                     [0, 0, wz, 0, wx, wy]], float)


def _vec6_to_I(v):
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = v
    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


def inertia_shape_from_tumble(omega, omega_dot):
    """Inertia tensor up to scale (unit trace) from a torque-free tumble.

    Rows  G(w_dot) + [w]x G(w)  annihilate vec6(I); the smallest right singular
    vector is the shape. Needs a tumble that is not about a principal axis.
    Returns (I_unit_trace, condition_ratio) where a small ratio (sigma_min /
    sigma_max) means the data was informative.
    """
    A = np.vstack([_G(wd) + skew(w) @ _G(w) for w, wd in zip(omega, omega_dot)])
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    v = Vt[-1]
    I = _vec6_to_I(v)
    if np.trace(I) < 0:
        I = -I
    return I / np.trace(I), float(s[-1] / s[0])


def identify_dipole_matrix(inertia, omega, omega_dot, B_body, currents,
                           fit_residual=True):
    """Least-squares M (3 x n_ch) and m_res (3,) from burst samples.

    All arrays are per-sample rows: omega/omega_dot/B_body (N,3), currents (N,n_ch).
    Returns (M, m_res, rms_residual_torque, condition_number).
    """
    inertia = np.asarray(inertia, float)
    n_ch = currents.shape[1]
    rows, rhs = [], []
    for w, wd, B, u in zip(omega, omega_dot, B_body, currents):
        b = inertia @ wd + np.cross(w, inertia @ w)
        A_M = -skew(B) @ np.kron(u[None, :], np.eye(3))       # (3, 3*n_ch), vec col-major
        A = np.hstack([A_M, -skew(B)]) if fit_residual else A_M
        rows.append(A)
        rhs.append(b)
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    M = x[:3 * n_ch].reshape(3, n_ch, order="F")
    m_res = x[3 * n_ch:] if fit_residual else np.zeros(3)
    resid = A @ x - b
    return M, m_res, float(np.sqrt(np.mean(resid ** 2))), float(np.linalg.cond(A))


def identify_dipole_matrix_impulse(inertia, t, omega, B_body, currents,
                                   window=1.0, fit_residual=True):
    """Impulse-form least squares: one 3-row equation per pulse, no differentiation.

    Differentiating a noisy gyro amplifies its noise by ~1/dt, so the
    per-sample form gets *worse* as the burst rate rises. Integrating Euler's
    equation across each pulse instead,

        I (w_post - w_pre) + sum_k (w_k x I w_k) dt = -sum_k [B_k]x (u_k^T kron I3) dt vec(M)
                                                       -sum_k [B_k]x dt m_res

    uses the change in rate between a ``window``-second average before the
    pulse and one after it. Gyro noise then enters once per pulse, averaged
    over the window, instead of once per sample amplified. Pulses are located
    from the current record (|u| > 0). Returns (M, m_res, n_pulses, cond).
    """
    inertia = np.asarray(inertia, float)
    t = np.asarray(t, float); omega = np.asarray(omega, float)
    B_body = np.asarray(B_body, float); currents = np.asarray(currents, float)
    dt = float(np.median(np.diff(t)))
    on = np.any(np.abs(currents) > 0, axis=1)
    edges = np.flatnonzero(np.diff(on.astype(int)))
    starts = edges[on[edges + 1]] + 1
    ends = edges[~on[edges + 1]] + 1        # first index after the pulse
    n_w = max(1, int(round(window / dt)))
    n_ch = currents.shape[1]
    rows, rhs = [], []
    for s0 in starts:
        e0 = ends[ends > s0]
        if len(e0) == 0:
            continue
        e0 = e0[0]
        pre = slice(max(0, s0 - n_w), s0)
        post = slice(e0, min(len(t), e0 + n_w))
        if post.stop - post.start < 1 or pre.stop - pre.start < 1:
            continue
        w_pre, w_post = omega[pre].mean(axis=0), omega[post].mean(axis=0)
        mid = slice((pre.start + pre.stop) // 2, (post.start + post.stop) // 2)
        gyro_term = np.sum([np.cross(w, inertia @ w) for w in omega[mid]], axis=0) * dt
        lhs = inertia @ (w_post - w_pre) + gyro_term
        A_M = -np.sum([skew(B) @ np.kron(u[None, :], np.eye(3))
                       for B, u in zip(B_body[mid], currents[mid])], axis=0) * dt
        A_r = -np.sum([skew(B) for B in B_body[mid]], axis=0) * dt
        rows.append(np.hstack([A_M, A_r]) if fit_residual else A_M)
        rhs.append(lhs)
    A = np.vstack(rows); b = np.concatenate(rhs)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    M = x[:3 * n_ch].reshape(3, n_ch, order="F")
    m_res = x[3 * n_ch:] if fit_residual else np.zeros(3)
    return M, m_res, len(rows), float(np.linalg.cond(A))


def s3_coil_matrix():
    """Dipole-per-amp matrix for the S3 (PROVES DetumbleManager) coil set,
    paired onto S6's three H-bridge channels: X+ & X- in series, Y+ & Y-, Z-.

    m/I = N * A per coil (air core). Rectangular 53x45 mm, 96 turns; circular
    57.55 mm, 153 turns. Currents at 3.3 V: 0.254 A (13 ohm), 0.0219 A (150.7 ohm).
    Returns (M (3x3) [A m^2 / A], i_max (3,) [A]).
    """
    A_rect = 0.053 * 0.045
    A_circ = np.pi * (0.05755 / 2) ** 2
    m_x = 2 * 96 * A_rect          # two coils in series, same axis
    m_z = 153 * A_circ
    M = np.diag([m_x, m_x, m_z])
    i_max = np.array([3.3 / (2 * 13.0), 3.3 / (2 * 13.0), 3.3 / 150.7])
    return M, i_max


@dataclass
class PulseSchedule:
    """One channel at a time, alternating polarity, every ``period`` seconds."""

    n_ch: int
    i_max: np.ndarray
    duration: float = 0.32      # S3 TORQUE_DURATION
    period: float = 10.0
    n_pulses: int = 30

    def current(self, t):
        k = int(t // self.period)
        if k >= self.n_pulses or (t - k * self.period) > self.duration:
            return np.zeros(self.n_ch)
        u = np.zeros(self.n_ch)
        ch = k % self.n_ch
        u[ch] = self.i_max[ch] * (1.0 if (k // self.n_ch) % 2 == 0 else -1.0)
        return u

    @property
    def total_time(self):
        return self.period * self.n_pulses


def simulate_burst_campaign(M_true, m_res_true, schedule, body=None, orbit=None,
                            jd0=2460390.0, q0=None, omega0=None, dt_sim=0.01,
                            sample_hz=10.0, gyro_noise_density=6.6e-5,
                            mag_noise=3e-7, seed=0):
    """Truth run of a pulse campaign, sampled like the A9 burst record.

    Returns dict with sampled t, omega (gyro, noisy), omega_dot (central
    difference of the gyro), B_body (noisy), currents, and the truth timeline.
    """
    from ..environment.orbit import CircularOrbit
    from ..attitude import IDENTITY_QUAT
    rng = np.random.default_rng(seed)
    body = body or RigidBody()
    orbit = orbit or CircularOrbit()
    q0 = IDENTITY_QUAT if q0 is None else q0
    omega0 = np.deg2rad([1.0, -0.7, 0.5]) if omega0 is None else omega0
    n = int(schedule.total_time / dt_sim)
    log = {"u": [], "B": []}

    def torque_fn(t, q, w):
        B = to_body(q, magnetic_field_eci(orbit.position(t), jd0 + t / 86400.0))
        u = schedule.current(t)
        log["u"].append(u)
        log["B"].append(B)
        return np.cross(M_true @ u + m_res_true, B)

    tl, _ = simulate_controlled(body, q0, omega0, dt_sim, n, torque_fn)
    u_all = np.array(log["u"] + [log["u"][-1]])
    B_all = np.array(log["B"] + [log["B"][-1]])

    step = max(1, int(round(1.0 / (sample_hz * dt_sim))))
    idx = np.arange(0, len(tl.t), step)
    dt_s = tl.t[step] - tl.t[0]
    sig_w = gyro_noise_density / np.sqrt(dt_s)
    gyro = tl.omega[idx] + rng.normal(0, sig_w, (len(idx), 3)) + np.deg2rad(0.2)
    B_meas = B_all[idx] + rng.normal(0, mag_noise, (len(idx), 3))
    # average the current over each sample interval (what an ADC burst sees)
    u_s = np.array([u_all[i:i + step].mean(axis=0) for i in idx])
    wd = np.gradient(gyro, dt_s, axis=0)
    return {"t": tl.t[idx], "omega": gyro, "omega_dot": wd, "B_body": B_meas,
            "currents": u_s, "truth": tl, "dt": dt_s}
