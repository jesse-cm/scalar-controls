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

Realism terms (verification-loop plan, row 1h). All default to off, and with
them off ``measure`` performs exactly the arithmetic above:

    omega_s    = (I + diag(s)) C_mis omega_true          scale factor, misalignment
    omega_meas = Q_lsb( omega_s + bias + k_T (T(t) - T_ref) + white )

  * ``scale_factor`` (s, a number or one per axis, fraction): each sensor axis
    reads (1 + s) times the rate along it. The plan gives 1 %.
  * ``misalignment`` (rotation vector, rad): the sensor triad is not the body
    triad. ``C_mis`` is the sensor-from-body DCM of this rotation vector, in the
    convention of ``attitude.py`` (``v_sensor = C_mis v_body``). The plan gives
    0.5 deg; about which axis it does not say.
  * ``temp_coeff`` (k_T, 3-vector, rad/s/K) with ``temperature`` (a callable
    ``t [s] -> board temperature [deg C]``) and ``temp_ref`` (the temperature
    at which ``bias`` holds): the zero-rate output drifts with temperature.
    The profile is the same kind of callable the magnetometer takes
    (``sensors.magnetometer.sinusoid_temperature`` / ``tabulated_temperature``),
    so one board temperature can drive both sensors. Neither the coefficient
    nor the profile is in any file of this repo: both are ASSUMED wherever a
    value is given, and are replaced by the plan's phase-5 bench item "gyro and
    magnetometer logs across a temperature ramp with the TMP112s".
  * ``quantisation`` (LSB, rad/s): the output is rounded to this step. The plan
    says "at the FS in use"; the full-scale range the flight code sets, and the
    step that goes with it, are not in this repo, so any value is ASSUMED.
    Clipping at full scale is not modelled.

``bias`` stays the random-walk state alone; the temperature drift is added on
top and is returned by :meth:`temperature_offset`. The drift needs a time and
``measure`` is not given one by the simulators, so the gyro keeps its own
clock: it advances by ``dt`` per sample and ``reset()`` rewinds it. A caller
that knows the time can pass ``t=`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..attitude import dcm, quat_from_rotvec
from ..constants import DEG

__all__ = ["Gyro"]

# LSM6DSO gyro rate-noise density ~3.8 mdps/sqrt(Hz).
_LSM6DSO_NOISE_DENSITY = 3.8e-3 * DEG   # [rad/s / sqrt(Hz)]


@dataclass
class Gyro:
    """LSM6DSO-class rate gyro with ARW noise and a random-walk bias.

    Realism terms (plan row 1h), all off by default (SI):
        scale_factor  fractional scale-factor error, a number or one per axis
        misalignment  rotation vector [rad] of the sensor-from-body DCM (v_sensor = C v_body)
        quantisation  output LSB [rad/s]; None = continuous output
        temp_coeff    zero-rate drift k_T [rad/s/K] per axis; None = no drift. Needs ``temperature``.
        temperature   callable t [s] -> board temperature [deg C]; None = no drift
        temp_ref      temperature [deg C] at which ``bias`` holds
    """

    rate_noise_density: float = _LSM6DSO_NOISE_DENSITY   # [rad/s/sqrt(Hz)]
    bias_rw: float = 1.0e-5                               # [rad/s/sqrt(s)]
    initial_bias: np.ndarray = field(
        default_factory=lambda: np.array([0.2, -0.15, 0.1]) * DEG)  # [rad/s]
    bias: np.ndarray = None
    scale_factor: Optional[np.ndarray] = None            # fraction, scalar or per axis
    misalignment: Optional[np.ndarray] = None            # rotation vector [rad], sensor from body
    quantisation: Optional[float] = None                 # [rad/s] LSB
    temp_coeff: Optional[np.ndarray] = None              # [rad/s/K]
    temperature: Optional[Callable[[float], float]] = None
    temp_ref: float = 0.0                                # [deg C]
    _t: float = field(default=0.0, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.reset()

    def reset(self, bias=None):
        """Reset the drifting bias to ``bias`` (or the configured initial bias); rewind the clock."""
        self.bias = (np.asarray(self.initial_bias, dtype=float).copy()
                     if bias is None else np.asarray(bias, dtype=float).copy())
        self._t = 0.0

    def sensing_matrix(self):
        """3x3 map from the true body rate to what the sensor axes read, before bias and noise.

        ``(I + diag(scale_factor)) @ C_mis``; the identity when both terms are off.
        """
        C = np.eye(3) if self.misalignment is None else dcm(
            quat_from_rotvec(np.asarray(self.misalignment, dtype=float)))
        if self.scale_factor is None:
            return C
        return np.diag(1.0 + np.broadcast_to(np.asarray(self.scale_factor, dtype=float), (3,))) @ C

    def temperature_offset(self, t):
        """Zero-rate drift k_T (T(t) - T_ref) [rad/s] at time ``t``; zeros when the drift is off."""
        if self.temp_coeff is None or self.temperature is None:
            return np.zeros(3)
        return np.asarray(self.temp_coeff, dtype=float) * (float(self.temperature(t)) - self.temp_ref)

    def measure(self, omega_true, dt, rng, t=None):
        """Return a noisy rate [rad/s] and advance the bias random walk.

        Measures with the *current* bias, then random-walks the bias forward one
        step so the next sample sees the updated value. ``t`` [s] is used only
        by the temperature drift; left out, the gyro's own clock supplies it
        (the sum of ``dt`` over the calls since ``reset``).
        """
        sigma_w = self.rate_noise_density / np.sqrt(dt)
        omega = np.asarray(omega_true, dtype=float)
        if self.scale_factor is not None or self.misalignment is not None:
            omega = self.sensing_matrix() @ omega
        meas = (omega + self.bias
                + rng.normal(0.0, sigma_w, size=3))
        if self.temp_coeff is not None and self.temperature is not None:
            meas = meas + self.temperature_offset(self._t if t is None else t)
        if self.quantisation:
            meas = np.round(meas / self.quantisation) * self.quantisation
        self._t += dt
        self.bias = self.bias + rng.normal(0.0, self.bias_rw * np.sqrt(dt), size=3)
        return meas
