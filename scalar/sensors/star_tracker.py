"""Star tracker ("camera") attitude sensor for a 1U CubeSat.

A star tracker images a starfield through a narrow-FOV camera, matches it
against an onboard catalog, and returns a full **absolute attitude** — a
quaternion ``q_{B<-I}`` — with arc-second-class accuracy. It is the highest
quality attitude sensor on the vehicle, but it is intermittent: it produces no
solution when a bright body (the Sun, or the sunlit Earth limb) is inside its
keep-out cone, or during high slew rates that blur the star images.

Measurement model
-----------------
The tracker returns a noisy version of the true attitude,

    q_meas = dq(delta_theta) * q_true,   delta_theta ~ N(0, Sigma),

with the small-angle error ``delta_theta`` expressed in the **camera** frame.
Its covariance is anisotropic: cross-boresight errors (pitch/yaw of the
boresight on the sky) are set by the centroiding accuracy and are small, while
the error *about* the boresight (roll) is coarser by a fixed factor, because
roll is constrained only by the tangential spread of the matched stars. This
2-3x roll penalty is the characteristic star-tracker error signature.

Availability gate
-----------------
A solution is returned only when

  * the Sun is more than ``sun_exclusion`` from the boresight, and
  * the Earth nadir is more than ``earth_exclusion`` from the boresight, and
  * the body rate is below ``max_rate`` (star images not smeared).

When gated off, :meth:`measure` returns ``(None, False)``.

Role in SCALAR
--------------
Two uses, both exercised by the calibration comparison:

  * as a direct, high-weight attitude update in the MEKF (``sigma`` of
    arc-seconds instead of the degree-level vector sensors), and
  * as an **in-flight reference**: when the tracker has a valid solution its
    attitude is accurate enough to be treated as truth, so the modelled field
    / Sun direction can be rotated into the body frame and compared with the
    magnetometer / sun-sensor reading to recover their systematics without any
    ground equipment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..attitude import quat_mul, quat_from_small_angle, to_body

__all__ = ["StarTracker"]

_ARCSEC = np.pi / 180.0 / 3600.0


@dataclass
class StarTracker:
    """Narrow-FOV star tracker returning an absolute attitude quaternion.

    Parameters
    ----------
    boresight : (3,) array
        Camera boresight in body coordinates (unit). Default body ``+z``.
    sigma_cross : float
        1-sigma cross-boresight (pitch/yaw) error [rad]. Default 10 arcsec.
    roll_factor : float
        Ratio of about-boresight (roll) error to cross-boresight error.
        Default 6 (a representative small-tracker roll penalty).
    sun_exclusion : float
        Sun keep-out half-angle from the boresight [rad]. Default 30 deg.
    earth_exclusion : float
        Earth-nadir keep-out half-angle from the boresight [rad]. Default 25 deg.
    max_rate : float
        Body-rate ceiling above which star images smear and no solution is
        returned [rad/s]. Default 1.5 deg/s.
    """

    boresight: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    sigma_cross: float = 10 * _ARCSEC
    roll_factor: float = 6.0
    sun_exclusion: float = np.deg2rad(30.0)
    earth_exclusion: float = np.deg2rad(25.0)
    max_rate: float = np.deg2rad(1.5)

    def __post_init__(self):
        b = np.asarray(self.boresight, float)
        self.boresight = b / np.linalg.norm(b)
        # An orthonormal camera frame whose z-axis is the boresight; the two
        # cross axes carry sigma_cross, the boresight axis carries the roll.
        z = self.boresight
        seed = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = seed - z * (seed @ z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        self._cam = np.column_stack([x, y, z])   # body <- camera basis

    @property
    def sigma_roll(self) -> float:
        """About-boresight 1-sigma error [rad]."""
        return self.roll_factor * self.sigma_cross

    def available(self, q, sun_body=None, nadir_body=None, rate=None) -> bool:
        """Whether a star solution is returned in this geometry.

        Parameters
        ----------
        q : (4,) array
            True attitude ``q_{B<-I}`` (used only if body-frame directions are
            not supplied directly).
        sun_body, nadir_body : (3,) array, optional
            Sun and Earth-nadir unit vectors already in body coordinates.
        rate : float, optional
            Body angular-rate magnitude [rad/s].
        """
        bs = self.boresight
        if sun_body is not None:
            if np.dot(bs, sun_body / np.linalg.norm(sun_body)) > np.cos(self.sun_exclusion):
                return False
        if nadir_body is not None:
            if np.dot(bs, nadir_body / np.linalg.norm(nadir_body)) > np.cos(self.earth_exclusion):
                return False
        if rate is not None and rate > self.max_rate:
            return False
        return True

    def measure(self, q_true, rng, sun_body=None, nadir_body=None, rate=None):
        """Return ``(q_meas, valid)``.

        ``q_meas`` is the noisy absolute attitude when the geometry gate passes,
        otherwise ``(None, False)``.
        """
        if not self.available(q_true, sun_body, nadir_body, rate):
            return None, False
        # error in the camera frame: (cross, cross, roll), mapped to body.
        e_cam = np.array([rng.normal(0.0, self.sigma_cross),
                          rng.normal(0.0, self.sigma_cross),
                          rng.normal(0.0, self.sigma_roll)])
        dtheta_body = self._cam @ e_cam
        q_meas = quat_mul(quat_from_small_angle(dtheta_body), np.asarray(q_true, float))
        return q_meas, True

    def attitude_sigma(self) -> float:
        """Scalar 1-sigma attitude error [rad], RMS over the three axes.

        Convenient isotropic ``sigma`` for a quaternion MEKF update.
        """
        return float(np.sqrt(2 * self.sigma_cross ** 2 + self.sigma_roll ** 2) / np.sqrt(3))
