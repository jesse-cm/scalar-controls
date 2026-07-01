"""Time conversions shared across the sim.

Astronomy needs Julian Date (JD) and Greenwich Mean Sidereal Time (GMST); the
propagator carries plain seconds-from-epoch. We keep UT1 == UTC (sub-second
difference is far below this sim's accuracy).
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from .constants import JD_J2000, JULIAN_CENTURY_DAYS

__all__ = ["julian_date", "gmst_rad"]


def julian_date(when):
    """Julian Date (UTC) from a timezone-aware or naive UTC ``datetime``.

    Uses the standard civil-calendar algorithm; valid for all Gregorian dates.
    """
    if when.tzinfo is not None:
        when = when.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    y, m = when.year, when.month
    day_frac = (when.day
                + (when.hour + (when.minute + (when.second
                   + when.microsecond / 1e6) / 60.0) / 60.0) / 24.0)
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return (int(365.25 * (y + 4716)) + int(30.6001 * (m + 1))
            + day_frac + b - 1524.5)


def gmst_rad(jd):
    """Greenwich Mean Sidereal Time [rad] for Julian Date ``jd`` (IAU 1982).

    This is the rotation angle of ECEF's +x (Greenwich meridian) about ECI +z,
    i.e. the ECI->ECEF rotation is Rz(gmst).
    """
    t = (jd - JD_J2000) / JULIAN_CENTURY_DAYS
    gmst_sec = (67310.54841
                + (876600.0 * 3600.0 + 8640184.812866) * t
                + 0.093104 * t ** 2
                - 6.2e-6 * t ** 3)
    gmst_deg = (gmst_sec / 240.0) % 360.0   # 1 second of time = 1/240 degree
    return np.deg2rad(gmst_deg)
