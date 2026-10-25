"""Vishay VEML6031X00 ambient-light sensors used as SCALAR's coarse sun sensor.

Seven bare, outward-facing sensors, one per face board (PROVES v5 / S3
devicetree): faces 0-3 on the four sides, faces 5 and 6 both on -Z, face 7 on
+Z. Which of faces 0-3 is +X/-X/+Y/-Y is unknown; the cube's side normals are
used in the order +X, -X, +Y, -Y.

Per sensor (datasheet VEML6031X00 rev 1.2):
    lux    = 133,300 lx * (E / E_1AU) * max(0, n . s)   direct sun
           + albedo term from the sim's albedo model, same lux scale
    counts = clip(round(gain_i * lux / resolution + noise + dark), 0, 65535)

``resolution`` [lx/count] at gain x0.5, photodiode 1/4 comes from Table 10:
max illuminance / 65535 (25 ms -> 0.8704, 6.25 ms -> 3.4816). The angular
response is the cosine (Fig. 7). Dark offset 4 counts typ. (p.2).

Noise and per-sensor gain are taken from a ``SunSensorArray`` ("base") so a
VEML array built from config B's ideal cube carries B's noise (2 % of
face-on full scale, now in lux) and gains (unit). Datasheet output tolerance
is +-10 %; pass ``gains`` to model it.

Reconstruction reuses ``SunSensorArray.reconstruct`` unchanged: counts are
converted back to lux, divided by the face-on lux to give relative signals,
the two -Z sensors are averaged, and the six resulting cube-face signals are
scaled to the base array's current units.

``clip_mode``: None (feed clipped counts as they are), "drop" (a sample with
any clipped sensor gives no sun vector), "fill" (a clipped face's axis
component is recovered from the unit-norm constraint with the irradiance
known: one clipped axis is exact up to noise; with several, the remaining
norm is split equally).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import SOLAR_CONSTANT_1AU
from .sun_sensor import SunSensorArray

__all__ = ["VEML6031_NORMALS", "VEML6031_FACES", "MAX_LUX_G05_PD4", "resolution_lx_per_count",
           "VEML6031Array", "LUX_1AU", "COUNT_MAX"]

LUX_1AU = 133_300.0          # solar illuminance outside the atmosphere at 1 AU [lx]
COUNT_MAX = 65535            # 16-bit output
DARK_COUNTS = 4.0            # ALS dark offset, typ. (datasheet p.2)

# Table 10 (PD 1/4), gain x0.5 column: integration time [ms] -> max illuminance [lx]
MAX_LUX_G05_PD4 = {400: 3565, 200: 7130, 100: 14260, 50: 28521,
                   25: 57042, 12.5: 114083, 6.25: 228167}

VEML6031_FACES = (0, 1, 2, 3, 5, 6, 7)
VEML6031_NORMALS = np.array([
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],   # faces 0, 1 (assumed +X, -X)
    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],   # faces 2, 3 (assumed +Y, -Y)
    [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],  # faces 5, 6 (-Z)
    [0.0, 0.0, 1.0],                      # face 7 (+Z)
])
# Index of the base cube's face (+x,-x,+y,-y,+z,-z) whose noise draw each sensor
# reuses, so a VEML run consumes the shared rng exactly like the cube does.
# The second -Z sensor (index 5) draws from the private rng instead.
_CUBE_DRAW = (0, 1, 2, 3, 5, None, 4)


def resolution_lx_per_count(integration_ms):
    return MAX_LUX_G05_PD4[integration_ms] / COUNT_MAX


@dataclass
class VEML6031Array:
    integration_ms: float = 25
    base: SunSensorArray = field(default_factory=SunSensorArray)
    gains: np.ndarray = None           # per-sensor gain (7); default: base gains mapped
    dark_counts: float = DARK_COUNTS
    clip_mode: str | None = None       # None | "drop" | "fill"
    aux_rng: np.random.Generator = None
    normals: np.ndarray = field(default_factory=lambda: VEML6031_NORMALS.copy())

    def __post_init__(self):
        self.resolution = resolution_lx_per_count(self.integration_ms)
        if self.gains is None:
            g = np.asarray(self.base.gains, dtype=float)
            self.gains = np.array([g[0], g[1], g[2], g[3], g[5], g[5], g[4]])
        if self.aux_rng is None:
            self.aux_rng = np.random.default_rng(0)
        self.last_clipped = np.zeros(len(self.normals), dtype=bool)

    @staticmethod
    def face_on_lux(irradiance):
        return LUX_1AU * irradiance / SOLAR_CONSTANT_1AU

    def illuminance(self, sun_dir_body, irradiance, lit=True,
                    albedo_dir_body=None, albedo_irradiance=0.0):
        """Noiseless per-sensor illuminance [lx], gains not applied."""
        lux = np.zeros(len(self.normals))
        if lit:
            cos = self.normals @ np.asarray(sun_dir_body, dtype=float)
            lux += self.face_on_lux(irradiance) * np.clip(cos, 0.0, None)
            if albedo_dir_body is not None and albedo_irradiance > 0.0:
                cos_a = self.normals @ np.asarray(albedo_dir_body, dtype=float)
                lux += self.face_on_lux(albedo_irradiance) * np.clip(cos_a, 0.0, None)
        return lux

    def counts(self, lux, noise_lux=None):
        """lux (+ noise, lux) -> 16-bit counts with gain, dark offset, clip."""
        x = self.gains * np.asarray(lux, dtype=float)
        if noise_lux is not None:
            x = x + noise_lux
        c = np.clip(np.round(x / self.resolution + self.dark_counts), 0, COUNT_MAX)
        return c

    def measure(self, sun_dir_body, irradiance, lit, rng,
                albedo_dir_body=None, albedo_irradiance=0.0):
        """Noisy 16-bit counts (7,). Same call signature as SunSensorArray.measure."""
        lux = self.illuminance(sun_dir_body, irradiance, lit, albedo_dir_body, albedo_irradiance)
        z6 = rng.normal(0.0, 1.0, size=6)          # same draws as the cube's measure()
        z = np.array([z6[i] if i is not None else self.aux_rng.normal() for i in _CUBE_DRAW])
        sigma_lux = self.base.noise_frac * LUX_1AU   # B: 2 % of face-on full scale
        c = self.counts(lux, sigma_lux * z)
        self.last_clipped = c >= COUNT_MAX
        return c

    def cube_signals(self, counts, irradiance):
        """Counts (7,) -> relative signals on the 6-face cube (+x,-x,+y,-y,+z,-z)."""
        rel = np.asarray(counts, dtype=float) * self.resolution / self.face_on_lux(irradiance)
        return np.array([rel[0], rel[1], rel[2], rel[3], rel[6], 0.5 * (rel[4] + rel[5])])

    def reconstruct(self, counts, irradiance):
        counts = np.asarray(counts, dtype=float)
        clipped = counts >= COUNT_MAX
        if self.clip_mode == "drop" and clipped.any():
            return None, False
        sig = self.cube_signals(counts, irradiance)
        if self.clip_mode == "fill" and clipped.any():
            # cube faces clipped: +x,-x,+y,-y,+z,-z
            cf = np.array([clipped[0], clipped[1], clipped[2], clipped[3], clipped[6],
                           clipped[4] or clipped[5]])
            v = np.array([sig[0] - sig[1], sig[2] - sig[3], sig[4] - sig[5]])
            ax_clip = cf[0::2] | cf[1::2]
            sign = np.where(cf[0::2], 1.0, -1.0)
            known = float(np.sum(v[~ax_clip] ** 2))
            k = int(ax_clip.sum())
            clip_rel = COUNT_MAX * self.resolution / self.face_on_lux(irradiance)
            mag = max(clip_rel, np.sqrt(max(0.0, 1.0 - known) / k))
            v[ax_clip] = sign[ax_clip] * mag
            n = np.linalg.norm(v)
            return v / n, True
        return self.base.reconstruct(sig * self.base.full_scale_current(irradiance), irradiance)
