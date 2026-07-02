"""Magnetometer model — ST LIS2MDL (3-axis magnetometer on FC_V5d).

Full measurement model (the standard magnetometer error model, e.g.
Crassidis/Markley, Alonso-Shuster TWOSTEP):

    B_meas_body = D (C_{B<-I}(q) B_eci) + b + noise,   noise ~ N(0, sigma^2 I)

The magnetometer reads the local geomagnetic field rotated into the body frame,
then distorted by the sensor itself:

  * ``soft_iron`` (D, 3x3): combined scale-factor, non-orthogonality and
    soft-iron distortion — a linear map that turns the field sphere the tip of
    B traces into an *ellipsoid*. Defaults to identity.
  * ``bias`` (b, 3-vector): hard-iron offset, constant in the body frame.
    Defaults to zero.

The estimator compares the measured direction against the modelled field to
constrain attitude; the on-orbit calibration module recovers (D, b) from the
attitude-independent locus of |B_meas| vs the known |B_model|.

Default noise ~3 mgauss RMS/axis (3e-7 T) is a representative published LIS2MDL
figure. The field magnitude at 550 km is tens of microtesla, so 3e-7 T is a
sub-percent, few-degree direction perturbation — the sun vector is the tighter
reference when the spacecraft is lit, and this is the *only* vector in eclipse.
Uncalibrated soft/hard-iron, by contrast, is a *systematic* several-degree error
that does not average away, which is why on-orbit calibration matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..attitude import to_body

__all__ = ["Magnetometer"]

# LIS2MDL RMS noise ~3 mgauss = 3e-7 T per axis.
_LIS2MDL_NOISE_T = 3.0e-7


@dataclass
class Magnetometer:
    """LIS2MDL-class 3-axis magnetometer with white noise, soft- and hard-iron.

    Attributes (SI):
        noise_std   white noise [T] per axis
        bias        hard-iron offset b [T], body frame (constant)
        soft_iron   3x3 distortion D (scale + non-orthogonality + soft-iron);
                    identity for an ideal sensor
    """

    noise_std: float = _LIS2MDL_NOISE_T          # [T] per axis
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [T], body
    soft_iron: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3

    def measure(self, B_eci, q, rng):
        """Return the noisy, distorted field [T] in body coordinates for ``q``."""
        B_body = to_body(q, np.asarray(B_eci, dtype=float))
        distorted = np.asarray(self.soft_iron, dtype=float) @ B_body
        return distorted + np.asarray(self.bias, dtype=float) + rng.normal(
            0.0, self.noise_std, size=3)
