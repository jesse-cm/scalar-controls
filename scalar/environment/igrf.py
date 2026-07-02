"""Real IGRF geomagnetic field in ECI, via the pure-Python ``ppigrf`` package.

This replaces the tilted-dipole *stand-in* (``magfield.py``) with the full
International Geomagnetic Reference Field: the same spherical-harmonic model
(degree/order 13) used operationally, including the non-dipole terms the dipole
omits. Those terms are exactly the ~5-10 deg local direction error the README's
``field_model_error_deg`` used to inject synthetically; with IGRF as the *truth*
field and the dipole kept as the estimator's onboard *reference*, that error
becomes a physical dipole-vs-IGRF discrepancy instead of a hand-tuned rotation.

Coordinate handling matches the rest of the sim's convention (``ECI = Rz(GMST)
ECEF``, see ``timeutil.gmst_rad``):

    ECI position  --Rz(GMST)^T-->  ECEF  -->  geocentric (r, colat, elon)
    ppigrf.igrf_gc  -->  (Br, Btheta, Bphi)  [nT, radial / south / east]
    spherical->cartesian ECEF  --Rz(GMST)-->  ECI field  [T]

``ppigrf`` evaluates the IGRF coefficients at a calendar ``date`` (secular
variation is an annual-timescale effect). Over a single orbit that date is
effectively constant, so :func:`igrf_field_eci_batch` evaluates the whole
trajectory's positions in one vectorised ``igrf_gc`` call at a representative
date -- ~300x faster than looping and physically indistinguishable from
per-sample dates over the minutes-to-hours a run spans.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from ..constants import JD_J2000
from ..timeutil import gmst_rad

__all__ = ["igrf_field_eci", "igrf_field_eci_batch", "jd_to_datetime"]

# Imported lazily so the rest of the sim (and the dipole model) never depends on
# ppigrf being installed -- only the IGRF truth path needs it.
_igrf_gc = None


def _get_igrf_gc():
    global _igrf_gc
    if _igrf_gc is None:
        try:
            from ppigrf import igrf_gc
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "The real IGRF field model needs the 'ppigrf' package "
                "(pip install ppigrf). The tilted-dipole model in "
                "scalar.environment.magfield needs no extra dependency."
            ) from exc
        _igrf_gc = igrf_gc
    return _igrf_gc


def jd_to_datetime(jd):
    """Calendar UTC ``datetime`` from a Julian Date (inverse of ``julian_date``)."""
    days = float(jd) - JD_J2000
    return _dt.datetime(2000, 1, 1, 12, 0, 0) + _dt.timedelta(days=days)


def _rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _eci_to_geocentric(r_eci, gmst):
    """(r_km, colat_deg, elon_deg) of an ECI position at Greenwich sidereal ``gmst``."""
    r_ecef = _rz(gmst).T @ np.asarray(r_eci, float)
    r = np.linalg.norm(r_ecef)
    colat = np.degrees(np.arccos(np.clip(r_ecef[2] / r, -1.0, 1.0)))
    elon = np.degrees(np.arctan2(r_ecef[1], r_ecef[0]))
    return r / 1e3, colat, elon


def _spherical_field_to_ecef(Br, Btheta, Bphi, colat_deg, elon_deg):
    """Geocentric spherical field components -> Cartesian ECEF vector [same units].

    ppigrf returns Br (radial/up), Btheta (toward increasing colatitude = south),
    Bphi (east). The local unit vectors in ECEF Cartesian are the standard
    spherical basis at (colatitude theta, east longitude phi).
    """
    th = np.radians(colat_deg)
    ph = np.radians(elon_deg)
    st, ct = np.sin(th), np.cos(th)
    sp, cp = np.sin(ph), np.cos(ph)
    r_hat = np.array([st * cp, st * sp, ct])
    th_hat = np.array([ct * cp, ct * sp, -st])
    ph_hat = np.array([-sp, cp, 0.0 * sp])
    return Br * r_hat + Btheta * th_hat + Bphi * ph_hat


def igrf_field_eci(r_eci, jd):
    """Real IGRF flux density [T] in ECI at position ``r_eci`` [m], Julian Date ``jd``.

    Drop-in replacement for :func:`scalar.environment.magfield.magnetic_field_eci`.
    """
    igrf_gc = _get_igrf_gc()
    gmst = gmst_rad(jd)
    r_km, colat, elon = _eci_to_geocentric(r_eci, gmst)
    Br, Bt, Bp = igrf_gc(r_km, colat, elon, jd_to_datetime(jd))
    Br, Bt, Bp = float(np.ravel(Br)[0]), float(np.ravel(Bt)[0]), float(np.ravel(Bp)[0])
    B_ecef_nT = _spherical_field_to_ecef(Br, Bt, Bp, colat, elon)
    B_eci_nT = _rz(gmst) @ B_ecef_nT
    return B_eci_nT * 1e-9   # nT -> T


def igrf_field_eci_batch(r_eci_array, jd_array):
    """Vectorised IGRF field [T] in ECI over a trajectory.

    ``r_eci_array`` is (N,3) [m], ``jd_array`` is (N,) Julian Dates. Returns
    (N,3) [T]. Uses a single representative date for the IGRF coefficients (the
    midpoint of the run) since secular variation is negligible over one run,
    while per-sample GMST is applied exactly for the frame rotation.
    """
    r = np.asarray(r_eci_array, float)
    jd = np.asarray(jd_array, float)
    n = len(r)
    igrf_gc = _get_igrf_gc()
    date = jd_to_datetime(jd[n // 2])

    gmst = gmst_rad(jd)                                   # (N,)
    cg, sg = np.cos(gmst), np.sin(gmst)
    # ECI -> ECEF for every sample: r_ecef = Rz(gmst)^T r_eci
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    xe = cg * x + sg * y
    ye = -sg * x + cg * y
    ze = z
    rr = np.sqrt(xe**2 + ye**2 + ze**2)
    colat = np.degrees(np.arccos(np.clip(ze / rr, -1.0, 1.0)))
    elon = np.degrees(np.arctan2(ye, xe))

    Br, Bt, Bp = igrf_gc(rr / 1e3, colat, elon, date)     # each (1, N)
    Br, Bt, Bp = np.ravel(Br), np.ravel(Bt), np.ravel(Bp)

    th = np.radians(colat)
    ph = np.radians(elon)
    st, ct = np.sin(th), np.cos(th)
    sp, cp = np.sin(ph), np.cos(ph)
    # spherical -> ECEF cartesian, per sample
    Bx = Br * st * cp + Bt * ct * cp - Bp * sp
    By = Br * st * sp + Bt * ct * sp + Bp * cp
    Bz = Br * ct - Bt * st
    # ECEF -> ECI: B_eci = Rz(gmst) B_ecef
    Bx_i = cg * Bx - sg * By
    By_i = sg * Bx + cg * By
    Bz_i = Bz
    return np.column_stack([Bx_i, By_i, Bz_i]) * 1e-9      # nT -> T
