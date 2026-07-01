"""Magnetometer model — ST LIS2MDL (3-axis magnetometer on FC_V5d).

Measurement model:
    B_meas_body = C_{B<-I}(q) B_eci + bias + noise,   noise ~ N(0, sigma^2 I)
The magnetometer reads the local geomagnetic field rotated into the body frame;
the estimator compares it against the modelled field direction to constrain
attitude. ``bias`` lumps hard-iron offset (constant in the body frame).

Default noise ~3 mgauss RMS/axis (3e-7 T) is a representative published LIS2MDL
figure. Note the field magnitude at 550 km is tens of microtesla, so 3e-7 T is
a sub-percent, few-degree direction perturbation — the sun vector is the tighter
reference when the spacecraft is lit, and this is the *only* vector in eclipse.
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
    """LIS2MDL-class 3-axis magnetometer with white noise and a hard-iron bias."""

    noise_std: float = _LIS2MDL_NOISE_T          # [T] per axis
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [T], body

    def measure(self, B_eci, q, rng):
        """Return the noisy field [T] in body coordinates for attitude ``q``."""
        B_body = to_body(q, np.asarray(B_eci, dtype=float))
        return B_body + np.asarray(self.bias, dtype=float) + rng.normal(
            0.0, self.noise_std, size=3)
