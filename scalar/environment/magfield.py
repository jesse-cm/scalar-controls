"""Geomagnetic field: a tilted centered-dipole model in ECI.

Honest scope note: the full model is IGRF, but this sim has no IGRF dependency
(numpy/scipy/matplotlib only). A tilted, Earth-rotating centered dipole is the
standard ADCS stand-in — it captures the two things the estimator actually
cares about: the field magnitude at LEO (~2e-5..5e-5 T) and, crucially, the
direction sweeping through the orbit (which sets when the mag vector alone
leaves attitude unobservable about B). It omits the non-dipole terms IGRF adds
(~5-10 deg local direction error). The interface is drop-in replaceable with a
real IGRF later.

Model:
    B(r) = B0 (Re/r)^3 [3 (m_hat . r_hat) r_hat - m_hat]
with B0 the mean equatorial surface field and m_hat the dipole moment axis.
The geomagnetic north pole sits ~9.7 deg from the rotation axis near 72.6 W;
the dipole moment points the opposite way (a bar magnet with its south pole in
the northern hemisphere), so the field points geographic-north at the equator
and into the ground at the north pole.
"""
from __future__ import annotations

import numpy as np

from ..constants import DEG, R_EARTH_M
from ..timeutil import gmst_rad

__all__ = ["B0_SURFACE_T", "dipole_axis_eci", "magnetic_field_eci"]

# Mean equatorial field at Earth's surface [T] (~31,200 nT, standard dipole B0).
B0_SURFACE_T = 3.12e-5

# Geomagnetic north pole location (approx, epoch ~2020) used for the tilt.
_POLE_COLAT = 9.7 * DEG          # tilt from the rotation (+z) axis
_POLE_EAST_LON = -72.6 * DEG     # east longitude of the pole in ECEF


def _pole_axis_ecef():
    """Unit vector toward the geomagnetic north pole, in ECEF."""
    return np.array([np.sin(_POLE_COLAT) * np.cos(_POLE_EAST_LON),
                     np.sin(_POLE_COLAT) * np.sin(_POLE_EAST_LON),
                     np.cos(_POLE_COLAT)])


def dipole_axis_eci(jd):
    """Unit dipole *moment* axis ``m_hat`` in ECI at Julian Date ``jd``.

    The moment points opposite the geomagnetic-north pole vector, and the whole
    thing rotates with Earth (ECEF->ECI via Rz(GMST)).
    """
    theta = gmst_rad(jd)
    c, s = np.cos(theta), np.sin(theta)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    m_hat_ecef = -_pole_axis_ecef()
    return rz @ m_hat_ecef


def magnetic_field_eci(r_eci, jd):
    """Geomagnetic flux density vector [T] in ECI at position ``r_eci`` [m]."""
    r_eci = np.asarray(r_eci, dtype=float)
    r = np.linalg.norm(r_eci)
    r_hat = r_eci / r
    m_hat = dipole_axis_eci(jd)
    strength = B0_SURFACE_T * (R_EARTH_M / r) ** 3
    return strength * (3.0 * np.dot(m_hat, r_hat) * r_hat - m_hat)
