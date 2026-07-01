"""Rate gyro model — ST LSM6DSO (the gyro half of the 6-axis IMU on FC_V5d).

The accelerometer half is useless for attitude in orbit (free-fall -> it reads
only non-gravitational acceleration, ~noise), so only the gyro rate is modelled.

Measurement model (per axis):
    omega_meas = omega_true + bias + white,     white ~ N(0, sigma_w)
with the white term the angle-random-walk (ARW) noise and ``bias`` a slowly
drifting term (rate random walk). The continuous rate-noise density converts to
a per-sample standard deviation as sigma_w = rate_noise_density / sqrt(dt).

Default numbers are representative published LSM6DSO figures (rate noise density
~3.8 mdps/sqrt(Hz)); they are easy to override and are what the MEKF's process
noise must be matched to. The gyro carries state (its drifting bias), so call
:meth:`reset` at the start of each run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import DEG

__all__ = ["Gyro"]

# LSM6DSO gyro rate-noise density ~3.8 mdps/sqrt(Hz).
_LSM6DSO_NOISE_DENSITY = 3.8e-3 * DEG   # [rad/s / sqrt(Hz)]


@dataclass
class Gyro:
    """LSM6DSO-class rate gyro with ARW noise and a random-walk bias."""

    rate_noise_density: float = _LSM6DSO_NOISE_DENSITY   # [rad/s/sqrt(Hz)]
    bias_rw: float = 1.0e-5                               # [rad/s/sqrt(s)]
    initial_bias: np.ndarray = field(
        default_factory=lambda: np.array([0.2, -0.15, 0.1]) * DEG)  # [rad/s]
    bias: np.ndarray = None

    def __post_init__(self):
        self.reset()

    def reset(self, bias=None):
        """Reset the drifting bias to ``bias`` (or the configured initial bias)."""
        self.bias = (np.asarray(self.initial_bias, dtype=float).copy()
                     if bias is None else np.asarray(bias, dtype=float).copy())

    def measure(self, omega_true, dt, rng):
        """Return a noisy rate [rad/s] and advance the bias random walk.

        Measures with the *current* bias, then random-walks the bias forward one
        step so the next sample sees the updated value.
        """
        sigma_w = self.rate_noise_density / np.sqrt(dt)
        meas = (np.asarray(omega_true, dtype=float) + self.bias
                + rng.normal(0.0, sigma_w, size=3))
        self.bias = self.bias + rng.normal(0.0, self.bias_rw * np.sqrt(dt), size=3)
        return meas
