"""Coarse photodiode sun sensor for a 1U CubeSat.

Six body-mounted photodiodes, one per cube face, each with a Lambertian
(cosine) response: a diode's short-circuit photocurrent is proportional to the
irradiance times the cosine of the angle between its outward normal and the Sun
(the projected-area law), and zero when the Sun is behind the face.

    I_face = R * A * E * max(0, n_face . s_body)      [A]

with R the responsivity [A/W], A the active area [m^2], E the irradiance
[W/m^2], and s_body the unit Sun vector in body coordinates. In eclipse every
diode drops to its dark floor.

Reconstruction (opposing-face pairs): because I(+axis) - I(-axis) = g * s_axis
for any geometry (one of the pair is always dark), the Sun direction is just

    s_hat ~ normalize([ I+x - I-x , I+y - I-y , I+z - I-z ]).

The absolute scale (R, A, E) cancels in the normalization, so the reported
direction depends only on the *relative* diode signals — which is why a coarse
sun sensor's accuracy is set by its signal-to-noise, not its calibration.

Modelled error source: per-diode current noise (shot/read noise + gain
mismatch) as a fraction of full-scale. NOT modelled (documented systematics a
real coarse sensor also fights): Earth-albedo contamination, cover-glass
refraction/FOV roll-off, and diode misalignment. The EKF's measurement noise
should be sized to cover those in a real deployment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CUBE_FACE_NORMALS", "SunSensorArray"]

# Outward face normals of a 1U cube, in body axes, as opposing ± pairs.
CUBE_FACE_NORMALS = np.array([
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
])


@dataclass
class SunSensorArray:
    """Six-face coarse sun sensor.

    Attributes (SI):
        responsivity     diode responsivity [A/W]
        area             diode active area [m^2]
        noise_frac       1-sigma current noise as a fraction of full-scale
        dark_frac        dark/floor current as a fraction of full-scale
        min_signal_frac  reconstruction is 'valid' only if the recovered signal
                         vector exceeds this fraction of full-scale
    """

    responsivity: float = 0.5
    area: float = 1.0e-5
    noise_frac: float = 0.02
    dark_frac: float = 1.0e-3
    # Validity gate: the reconstructed signal vector is ~full-scale when lit and
    # only ~noise (few % of full-scale) in eclipse, so 0.2 cleanly separates the
    # two and keeps eclipse noise from ever faking a valid sun fix.
    min_signal_frac: float = 0.2
    normals: np.ndarray = None  # defaults to CUBE_FACE_NORMALS

    def __post_init__(self):
        if self.normals is None:
            self.normals = CUBE_FACE_NORMALS.copy()

    def full_scale_current(self, irradiance):
        """Photocurrent [A] of a diode facing the Sun head-on at ``irradiance``."""
        return self.responsivity * self.area * irradiance

    def currents(self, sun_dir_body, irradiance, lit=True):
        """Noiseless per-face photocurrents [A] (zeros in eclipse)."""
        if not lit:
            return np.zeros(len(self.normals))
        cos = self.normals @ np.asarray(sun_dir_body, dtype=float)
        return self.full_scale_current(irradiance) * np.clip(cos, 0.0, None)

    def measure(self, sun_dir_body, irradiance, lit, rng,
                albedo_dir_body=None, albedo_irradiance=0.0):
        """Noisy per-face photocurrents [A] including dark floor and albedo.

        If ``albedo_dir_body`` (body-frame nadir) and ``albedo_irradiance`` are
        given, Earth-reflected light adds a cosine photocurrent from that
        direction when lit — a *systematic* term that pulls the reconstructed
        sun vector toward Earth (unlike zero-mean noise). Photocurrent is
        non-negative, so the result is clamped at 0.
        """
        fs = self.full_scale_current(irradiance)
        ideal = self.currents(sun_dir_body, irradiance, lit)
        if lit and albedo_dir_body is not None and albedo_irradiance > 0.0:
            cos_a = self.normals @ np.asarray(albedo_dir_body, dtype=float)
            ideal = ideal + (self.responsivity * self.area * albedo_irradiance
                             * np.clip(cos_a, 0.0, None))
        noise = rng.normal(0.0, self.noise_frac * fs, size=ideal.shape)
        return np.clip(ideal + self.dark_frac * fs + noise, 0.0, None)

    def reconstruct(self, currents, irradiance):
        """Recover (s_hat_body, valid) from per-face currents.

        Uses opposing-pair differences, so it assumes ``normals`` is the default
        cube configuration (±x, ±y, ±z in order). ``valid`` is False when the
        lit signal is too weak to trust (e.g. eclipse).
        """
        c = np.asarray(currents, dtype=float)
        v = np.array([c[0] - c[1], c[2] - c[3], c[4] - c[5]])
        mag = np.linalg.norm(v)
        if mag <= self.min_signal_frac * self.full_scale_current(irradiance):
            return None, False
        return v / mag, True

    def measure_sun_vector(self, sun_dir_body, irradiance, lit, rng,
                           albedo_dir_body=None, albedo_irradiance=0.0):
        """Convenience: measure + reconstruct -> (s_hat_body or None, valid)."""
        c = self.measure(sun_dir_body, irradiance, lit, rng,
                         albedo_dir_body, albedo_irradiance)
        return self.reconstruct(c, irradiance)
