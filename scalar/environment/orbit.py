"""Circular LEO orbit propagation in ECI.

Design point: 550 km altitude, 51.6 deg inclination (an ISS-like orbit chosen
because its ~35% eclipse fraction stresses the sun-dropout regime where the
determination budget is won or lost).

Kept deliberately simple — a Keplerian *circular* orbit with fixed elements
(no J2 nodal precession, which is ~a few deg/day and negligible over the
few-orbit runs this sim uses). The propagator supplies (position, velocity) in
ECI: position feeds the eclipse test and the geomagnetic field, velocity is
available for any velocity-frame diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import DEG, MU_EARTH, R_EARTH_M

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

    def _plane(self):
        """Rotation from the orbital plane (x toward ascending node) to ECI."""
        return _rz(self.raan) @ _rx(self.inclination)

    def state(self, t):
        """Return (r_eci, v_eci) [m], [m/s] at time ``t`` seconds from epoch."""
        u = self.arg_lat0 + self.mean_motion * t          # argument of latitude
        a = self.radius
        r_plane = a * np.array([np.cos(u), np.sin(u), 0.0])
        v_mag = a * self.mean_motion
        v_plane = v_mag * np.array([-np.sin(u), np.cos(u), 0.0])
        Q = self._plane()
        return Q @ r_plane, Q @ v_plane

    def position(self, t):
        """ECI position [m] at time ``t`` seconds from epoch."""
        return self.state(t)[0]
