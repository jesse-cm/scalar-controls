"""Circular LEO orbit propagation in ECI.

Design point: 550 km altitude, 51.6 deg inclination (an ISS-like orbit chosen
because its ~35% eclipse fraction stresses the sun-dropout regime where the
determination budget is won or lost).

The default is a Keplerian *circular* orbit with fixed elements. Setting
``j2=True`` adds the first-order secular effect of Earth's oblateness (J2):
the ascending node regresses (``raan`` drifts) and the in-plane argument of
latitude drifts (the combined secular perigee + mean-anomaly rate). At
550 km / 51.6° the node regresses ~-4.6 deg/day. Over the few-orbit runs the
determination sim uses this is a small effect, but it matters for multi-day
propagation and for the sun-geometry / eclipse phasing it slowly shifts.

The propagator supplies (position, velocity) in ECI (the mean equator and
equinox of date, scalar/attitude.py; the elements are referred to that frame,
not to J2000): position feeds the
eclipse test and the geomagnetic field, velocity is available for any
velocity-frame diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import DEG, J2_EARTH, MU_EARTH, R_EARTH_M

__all__ = ["CircularOrbit"]


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@dataclass
class CircularOrbit:
    """A fixed-element circular orbit.

    Attributes are SI/radians. ``raan`` is the right ascension of the ascending
    node, ``arg_lat0`` the argument of latitude at t=0 (angle from the ascending
    node, which for a circular orbit fully places the satellite).
    """

    altitude: float = 550e3
    inclination: float = 51.6 * DEG
    raan: float = 0.0
    arg_lat0: float = 0.0
    j2: bool = False              # enable first-order J2 secular drift

    @property
    def radius(self):
        """Orbital radius [m] (constant for a circular orbit)."""
        return R_EARTH_M + self.altitude

    @property
    def mean_motion(self):
        """Mean motion n = sqrt(mu / a^3) [rad/s]."""
        return np.sqrt(MU_EARTH / self.radius ** 3)

    @property
    def period(self):
        """Orbital period [s]."""
        return 2.0 * np.pi / self.mean_motion

    @property
    def raan_dot(self):
        """Secular nodal regression rate dOmega/dt [rad/s] (0 unless j2)."""
        if not self.j2:
            return 0.0
        return (-1.5 * self.mean_motion * J2_EARTH
                * (R_EARTH_M / self.radius) ** 2 * np.cos(self.inclination))

    @property
    def arg_lat_dot_sec(self):
        """Secular argument-of-latitude drift [rad/s] beyond mean motion (0 unless j2).

        For a circular orbit this is the sum of the secular perigee rate and the
        secular mean-anomaly rate (both first-order J2, e=0).
        """
        if not self.j2:
            return 0.0
        fac = (1.5 * self.mean_motion * J2_EARTH
               * (R_EARTH_M / self.radius) ** 2)
        s2 = np.sin(self.inclination) ** 2
        argp_dot = fac * (2.0 - 2.5 * s2)          # perigee precession
        m_dot = fac * (1.0 - 1.5 * s2)             # mean-anomaly secular rate
        return argp_dot + m_dot

    def raan_at(self, t):
        """RAAN [rad] at time ``t`` seconds from epoch (drifts under J2)."""
        return self.raan + self.raan_dot * t

    def _plane(self, t=0.0):
        """Rotation from the orbital plane (x toward ascending node) to ECI."""
        return _rz(self.raan_at(t)) @ _rx(self.inclination)

    def state(self, t):
        """Return (r_eci, v_eci) [m], [m/s] at time ``t`` seconds from epoch."""
        u = self.arg_lat0 + (self.mean_motion + self.arg_lat_dot_sec) * t
        a = self.radius
        r_plane = a * np.array([np.cos(u), np.sin(u), 0.0])
        v_mag = a * self.mean_motion
        v_plane = v_mag * np.array([-np.sin(u), np.cos(u), 0.0])
        Q = self._plane(t)
        return Q @ r_plane, Q @ v_plane

    def position(self, t):
        """ECI position [m] at time ``t`` seconds from epoch."""
        return self.state(t)[0]
