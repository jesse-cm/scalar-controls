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

Realism terms (verification-loop plan, row 1c). All default to off, and with
them off ``measure`` performs exactly the arithmetic above:

    b(t)   = b + k_T (T(t) - T_ref)                 offset drift with temperature
    B_meas = Q_lsb( D B_body + b(t) + noise )       output rounded to the LSB grid

  * ``temp_coeff`` (k_T, 3-vector, T/K) with ``temperature`` (a callable
    ``t [s] -> board temperature [deg C]``, i.e. the profile the TMP112 next to
    the sensor would log) and ``temp_ref`` (the temperature at which ``bias``
    holds). Neither the coefficient nor the profile is in any file of this
    repo: both are ASSUMED wherever a value is given, and are replaced by the
    plan's phase-5 bench item "gyro and magnetometer logs across a temperature
    ramp with the TMP112s" (coefficient) and by TMP112 flight telemetry or a
    thermal model (profile).
  * ``quantisation`` (LSB, T): the output register step. 0.15 uT is the value
    the plan gives for row 1c (``LIS2MDL_LSB_T``).

``measure`` has no time argument in the simulators that call it, so the sensor
keeps its own sample clock: sample ``k`` is taken at ``k * sample_dt`` and
``reset()`` rewinds it. A caller that knows the time can pass ``t=`` instead.

Default noise ~3 mgauss RMS/axis (3e-7 T) is a representative published LIS2MDL
figure. The field magnitude at 550 km is tens of microtesla, so 3e-7 T is a
sub-percent, few-degree direction perturbation — the sun vector is the tighter
reference when the spacecraft is lit, and this is the *only* vector in eclipse.
Uncalibrated soft/hard-iron, by contrast, is a *systematic* several-degree error
that does not average away, which is why on-orbit calibration matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..attitude import to_body

__all__ = ["Magnetometer", "LIS2MDL_LSB_T", "sinusoid_temperature", "tabulated_temperature"]

# LIS2MDL RMS noise ~3 mgauss = 3e-7 T per axis.
_LIS2MDL_NOISE_T = 3.0e-7

# Output step [T]: "0.15 uT quantisation", docs/verification-loop-plan.md row 1c.
LIS2MDL_LSB_T = 0.15e-6


def sinusoid_temperature(mean_C, amplitude_K, period_s, phase_rad=0.0):
    """Board temperature ``t [s] -> deg C``: one sinusoid per ``period_s`` about ``mean_C``.

    A stand-in for a TMP112 log over an orbit (warm in sun, cold in eclipse);
    ``amplitude_K`` is half the peak-to-peak swing. Every argument is the
    caller's assumption until a TMP112 profile exists.
    """
    w = 2.0 * np.pi / float(period_s)
    return lambda t: float(mean_C) + float(amplitude_K) * np.sin(w * t + phase_rad)


def tabulated_temperature(t_s, temp_C, period_s=None):
    """Board temperature ``t [s] -> deg C`` interpolated from a log (e.g. TMP112 samples).

    With ``period_s`` the table repeats with that period (one logged orbit
    replayed); without it the end values are held.
    """
    t_s, temp_C = np.asarray(t_s, float), np.asarray(temp_C, float)
    if period_s is None:
        return lambda t: float(np.interp(t, t_s, temp_C))
    return lambda t: float(np.interp(t, t_s, temp_C, period=float(period_s)))


@dataclass
class Magnetometer:
    """LIS2MDL-class 3-axis magnetometer with white noise, soft- and hard-iron.

    Attributes (SI):
        noise_std   white noise [T] per axis
        bias        hard-iron offset b [T], body frame (constant)
        soft_iron   3x3 distortion D (scale + non-orthogonality + soft-iron);
                    identity for an ideal sensor
        quantisation  output LSB [T]; None = continuous output
        temp_coeff  offset temperature coefficient k_T [T/K], body frame;
                    None = no drift. Needs ``temperature``.
        temperature callable t [s] -> board temperature [deg C]; None = no drift
        temp_ref    temperature [deg C] at which ``bias`` holds
        sample_dt   spacing [s] of successive ``measure`` calls (the sample clock)
    """

    noise_std: float = _LIS2MDL_NOISE_T          # [T] per axis
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [T], body
    soft_iron: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3
    quantisation: Optional[float] = None         # [T] LSB
    temp_coeff: Optional[np.ndarray] = None      # [T/K], body
    temperature: Optional[Callable[[float], float]] = None
    temp_ref: float = 0.0                        # [deg C]
    sample_dt: float = 1.0                       # [s]
    _k: int = field(default=0, init=False, repr=False, compare=False)

    def reset(self, sample_dt=None):
        """Rewind the sample clock to t = 0 (and optionally change its step)."""
        self._k = 0
        if sample_dt is not None:
            self.sample_dt = float(sample_dt)

    def temperature_offset(self, t):
        """Offset drift k_T (T(t) - T_ref) [T] at time ``t``; zeros when the drift is off."""
        if self.temp_coeff is None or self.temperature is None:
            return np.zeros(3)
        return np.asarray(self.temp_coeff, dtype=float) * (float(self.temperature(t)) - self.temp_ref)

    def measure(self, B_eci, q, rng, t=None):
        """Return the noisy, distorted field [T] in body coordinates for ``q``.

        ``t`` [s] is used only by the temperature drift; left out, the sample
        clock supplies it (``k * sample_dt`` for the k-th call since ``reset``).
        """
        B_body = to_body(q, np.asarray(B_eci, dtype=float))
        distorted = np.asarray(self.soft_iron, dtype=float) @ B_body
        if self.temp_coeff is not None and self.temperature is not None:
            distorted = distorted + self.temperature_offset(self._k * self.sample_dt if t is None else t)
        self._k += 1
        out = distorted + np.asarray(self.bias, dtype=float) + rng.normal(
            0.0, self.noise_std, size=3)
        if self.quantisation:
            out = np.round(out / self.quantisation) * self.quantisation
        return out
