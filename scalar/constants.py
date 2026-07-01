"""Physical constants and unit conventions for the SCALAR attitude sim.

UNITS: strict SI everywhere unless a name says otherwise.
    length   metres          [m]
    time     seconds         [s]
    angle    radians         [rad]   (except where a name ends in _deg)
    mag flux tesla           [T]
    irradiance               [W/m^2]
Time is carried as Julian Date (JD, UTC) for astronomy and as seconds
from an epoch for propagation.
"""
from __future__ import annotations

import numpy as np

# --- Astronomy / time -------------------------------------------------------
JD_J2000 = 2451545.0            # Julian Date of the J2000.0 epoch (2000-01-01 12:00 TT)
JULIAN_CENTURY_DAYS = 36525.0
SECONDS_PER_DAY = 86400.0

# --- Sun --------------------------------------------------------------------
# Total solar irradiance at 1 AU (the "solar constant"). ~1361 W/m^2 is the
# modern accepted mean (SORCE/TSIS); it varies ~+/-0.05% with the solar cycle
# and ~+/-3.4% over the year from Earth's orbital eccentricity (handled via 1/r^2).
SOLAR_CONSTANT_1AU = 1361.0     # [W/m^2]
AU_M = 1.495978707e11           # astronomical unit [m]
SUN_LUMINOSITY = 3.828e26       # total radiant power of the Sun [W] (for reference)

# --- Earth ------------------------------------------------------------------
R_EARTH_M = 6378.137e3          # equatorial radius [m]
MU_EARTH = 3.986004418e14       # gravitational parameter [m^3/s^2]
OMEGA_EARTH = 7.292115e-5       # sidereal rotation rate [rad/s]
J2_EARTH = 1.08262668e-3        # second zonal harmonic

# --- Handy conversions ------------------------------------------------------
DEG = np.pi / 180.0             # multiply degrees by this to get radians
RAD = 180.0 / np.pi             # multiply radians by this to get degrees
