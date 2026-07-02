"""Star-tracker-referenced in-flight calibration.

When the star tracker (:class:`scalar.sensors.star_tracker.StarTracker`) has a
valid solution, its attitude is accurate to arc-seconds — good enough to treat
as truth for the purpose of calibrating the coarse sensors. Rotating the
*modelled* environment vectors (the IGRF field, the Sun direction) into the
body frame with the tracker attitude gives a known reference direction to
compare each coarse-sensor reading against, with no ground equipment and no
attitude-independent trick.

Because the attitude is known, this recovers the **full** magnetometer
distortion matrix ``D`` (including the rigid rotation that the
attitude-*independent* ellipsoid fit cannot see), by a plain linear regression

    m_i = D (C(q_i) B_eci_i) + b + noise,

stacked over all star-tracker-valid samples and solved for the 12 unknowns
``[D | b]`` at once. The sun sensor's per-diode gains are recovered the same
way, from the ratio of each diode's current to the ideal cosine response at the
star-tracker-known incidence angle.

This is the third calibration regime in the comparison: unlike the on-ground
fit (controlled reference, pre-flight only) and the on-orbit ellipsoid fit
(no reference frame, recovers up to rotation), the star-tracker reference is
available in flight *and* fixes the full transformation — at the cost of
needing the tracker to be unobscured.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..attitude import to_body

__all__ = [
    "MagRefCalibration",
    "calibrate_mag_with_reference",
    "calibrate_sun_gains_with_reference",
]


@dataclass
class MagRefCalibration:
    """Full magnetometer calibration from a known attitude reference.

    ``B_cal = Dinv (m - b)`` recovers the true body-frame field (up to noise),
    including the rotation part that the attitude-independent fit leaves free.
    """

    D: np.ndarray       # recovered distortion matrix (3, 3)
    b: np.ndarray       # recovered hard-iron bias (3,)
    Dinv: np.ndarray    # inverse, for applying the correction
    residual_rms: float
    n_samples: int

    def apply(self, m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, float)
        if m.ndim == 1:
            return self.Dinv @ (m - self.b)
        return (self.Dinv @ (m - self.b).T).T


def calibrate_mag_with_reference(m, q_ref, B_eci):
    """Fit the full magnetometer model against a star-tracker attitude.

    Parameters
    ----------
    m : (N, 3) ndarray
        Raw magnetometer measurements [T].
    q_ref : (N, 4) ndarray
        Star-tracker attitudes ``q_{B<-I}`` at those samples (valid solutions
        only — gate on the tracker before calling).
    B_eci : (N, 3) ndarray
        Modelled field in ECI at those samples [T] (IGRF).

    Returns
    -------
    MagRefCalibration
    """
    m = np.asarray(m, float)
    q_ref = np.asarray(q_ref, float)
    B_eci = np.asarray(B_eci, float)
    N = m.shape[0]
    if not (m.shape == (N, 3) and q_ref.shape == (N, 4) and B_eci.shape == (N, 3)):
        raise ValueError("shapes must be m (N,3), q_ref (N,4), B_eci (N,3)")
    if N < 4:
        raise ValueError(f"need >=4 reference samples to fit [D|b]; got {N}")

    # true body-frame field from the reference attitude
    B_body = np.array([to_body(q_ref[k], B_eci[k]) for k in range(N)])
    # m = D B_body + b  ->  regress [B_body | 1] onto m
    A = np.column_stack([B_body, np.ones(N)])
    coef, *_ = np.linalg.lstsq(A, m, rcond=None)   # (4, 3)
    D = coef[:3].T
    b = coef[3]
    if abs(np.linalg.det(D)) < 1e-12:
        raise ValueError("recovered distortion matrix is singular")
    Dinv = np.linalg.inv(D)
    resid = (A @ coef) - m
    return MagRefCalibration(
        D=D, b=b, Dinv=Dinv,
        residual_rms=float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))),
        n_samples=int(N),
    )


def calibrate_sun_gains_with_reference(currents, q_ref, sun_eci, normals,
                                       full_scale, min_cos=0.2):
    """Recover per-diode sun-sensor gains from a known attitude reference.

    For each diode, at samples where the Sun is well within its FOV the ideal
    current is ``full_scale * cos(incidence)`` with the incidence angle known
    from the star-tracker attitude; the gain is the least-squares slope of the
    measured current against that ideal.

    Parameters
    ----------
    currents : (N, M) ndarray
        Per-sample diode currents [A].
    q_ref : (N, 4) ndarray
        Star-tracker attitudes.
    sun_eci : (N, 3) ndarray
        Modelled Sun direction in ECI (unit).
    normals : (M, 3) ndarray
        Nominal diode face normals (body).
    full_scale : float
        Head-on full-scale current [A].
    min_cos : float
        Only use samples with ``cos(incidence) > min_cos`` for a diode (a diode
        near its horizon carries almost no signal and only adds noise).

    Returns
    -------
    gains : (M,) ndarray
        Recovered per-diode gains (nominal sensor -> ones).
    """
    currents = np.asarray(currents, float)
    q_ref = np.asarray(q_ref, float)
    sun_eci = np.asarray(sun_eci, float)
    normals = np.asarray(normals, float)
    N, M = currents.shape
    sun_body = np.array([to_body(q_ref[k], sun_eci[k]) for k in range(N)])
    gains = np.ones(M)
    for i in range(M):
        cos_inc = sun_body @ normals[i]
        use = cos_inc > min_cos
        if use.sum() < 3:
            continue
        ideal = full_scale * cos_inc[use]
        meas = currents[use, i]
        # slope through origin: gain = <ideal, meas> / <ideal, ideal>
        gains[i] = float(ideal @ meas / (ideal @ ideal))
    return gains
