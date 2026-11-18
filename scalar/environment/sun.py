"""Sun model: ECI direction, irradiance, and eclipse.

Two distinct quantities (do not conflate — this was an explicit design point):
  * **Sun direction** ``s_hat`` : a dimensionless unit vector in ECI. This is
    the reference the estimator ultimately compares its predicted body-frame
    sun vector against. Rays are treated as parallel (Sun at infinity).
  * **Irradiance** : a scalar [W/m^2], ~1361 at 1 AU, scaled by 1/r^2 for
    Earth's orbital eccentricity. Sets the photodiode signal level; for a
    *coarse* sun sensor it largely cancels when the direction is reconstructed.

Accuracy: the direction uses the Astronomical Almanac low-precision series
(~0.01 deg), reported in the mean equator/equinox of date and used directly as
ECI — far tighter than a coarse photodiode's several-degree error, so it is an
effectively exact reference here. "ECI" in this sim therefore means that
of-date frame, not J2000 (scalar/attitude.py, FRAMES).
"""
from __future__ import annotations

import numpy as np

from ..constants import AU_M, JD_J2000, R_EARTH_M, SOLAR_CONSTANT_1AU

__all__ = [
    "sun_direction_eci",
    "sun_distance_au",
    "solar_irradiance",
    "sun_state_eci",
    "eclipse_factor",
    "is_eclipsed",
]


def _sun_ecliptic(jd):
    """Return (lambda_sun_rad, obliquity_rad, distance_AU, mean_anomaly_rad)."""
    n = jd - JD_J2000
    L = np.deg2rad((280.460 + 0.9856474 * n) % 360.0)     # mean longitude
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)     # mean anomaly
    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)
    eps = np.deg2rad(23.439 - 4.0e-7 * n)                  # obliquity of ecliptic
    r_au = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
    return lam, eps, r_au, g


def sun_direction_eci(jd):
    """Unit vector from Earth to Sun in ECI at Julian Date ``jd``."""
    lam, eps, _, _ = _sun_ecliptic(jd)
    s = np.array([np.cos(lam),
                  np.cos(eps) * np.sin(lam),
                  np.sin(eps) * np.sin(lam)])
    return s / np.linalg.norm(s)


def sun_distance_au(jd):
    """Earth-Sun distance [AU] at ``jd`` (drives the 1/r^2 irradiance term)."""
    _, _, r_au, _ = _sun_ecliptic(jd)
    return float(r_au)


def solar_irradiance(jd):
    """Solar irradiance [W/m^2] at Earth's distance at ``jd``."""
    r_au = sun_distance_au(jd)
    return SOLAR_CONSTANT_1AU / (r_au * r_au)


def sun_state_eci(jd):
    """Convenience: (s_hat_eci, irradiance_W_m2) at ``jd``."""
    return sun_direction_eci(jd), solar_irradiance(jd)


def eclipse_factor(r_sat_eci, sun_dir_eci, r_body=R_EARTH_M):
    """Fraction of sunlight reaching the spacecraft (1 = full sun, 0 = umbra).

    Cylindrical-shadow model: Earth casts an infinite anti-solar cylinder of
    radius ``r_body``. The spacecraft is shadowed only when it is on the
    anti-solar side AND its distance from the Sun-Earth line is < r_body.
    This is the standard ADCS approximation (ignores penumbra and the ~0.5 deg
    solar disk), adequate for a sun-dropout study.
    """
    r = np.asarray(r_sat_eci, dtype=float)
    s = np.asarray(sun_dir_eci, dtype=float)
    s = s / np.linalg.norm(s)
    proj = float(np.dot(r, s))          # component along the Sun direction
    if proj >= 0.0:
        return 1.0                       # sunward hemisphere -> always lit
    perp = np.linalg.norm(r - proj * s)  # distance from the shadow axis
    return 1.0 if perp > r_body else 0.0


def is_eclipsed(r_sat_eci, sun_dir_eci, r_body=R_EARTH_M):
    """True when the spacecraft is in Earth's umbra (no direct sun)."""
    return eclipse_factor(r_sat_eci, sun_dir_eci, r_body) == 0.0
