"""Assemble the truth timeline + full sensor suite into one dataset.

A :class:`Scenario` wires together everything built so far:

    tumble dynamics (truth q, omega)  +  orbit (r_eci)  +  sun (s_eci, E, lit)
    +  geomagnetic field (B_eci)      ->  gyro / magnetometer / sun-sensor reads

and :meth:`Scenario.run` returns a :class:`SimResult` of aligned arrays. The
estimator (Task 7) consumes this: at each step it gets the gyro rate, the
body-frame magnetometer and sun-sensor vectors, and the *reference* inertial
directions (sun ephemeris + field model) it needs to form innovations — plus
the eclipse flag that tells it when the sun vector has dropped out.

The reference directions (``sun_eci``, ``B_eci``) are the same models the truth
uses: in a determination sim the estimator is assumed to know the sun ephemeris
and field model exactly; only the *sensors* are noisy.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np

from .attitude import IDENTITY_QUAT, to_body
from .constants import DEG
from .dynamics import RigidBody, simulate_free_tumble
from .environment.magfield import magnetic_field_eci, truth_field_eci
from .environment.orbit import CircularOrbit
from .environment.sun import eclipse_factor, solar_irradiance, sun_direction_eci
from .sensors.gyro import Gyro
from .sensors.magnetometer import Magnetometer
from .sensors.sun_sensor import SunSensorArray
from .timeutil import julian_date

__all__ = ["Scenario", "SimResult"]


@dataclass
class SimResult:
    """Aligned truth + measurement arrays over N samples."""

    t: np.ndarray            # (N,)   seconds from epoch
    jd: np.ndarray           # (N,)   Julian Date
    dt: float
    # --- truth ---
    q_true: np.ndarray       # (N, 4)
    omega_true: np.ndarray   # (N, 3) [rad/s]
    gyro_bias_true: np.ndarray  # (N, 3) [rad/s]
    # --- environment / references (known to the estimator) ---
    r_eci: np.ndarray        # (N, 3) [m]
    sun_eci: np.ndarray      # (N, 3) unit
    irradiance: np.ndarray   # (N,)   [W/m^2]
    lit: np.ndarray          # (N,)   bool (True = sunlit)
    B_eci: np.ndarray        # (N, 3) [T]  TRUTH field (what the magnetometer sees)
    B_eci_ref: np.ndarray    # (N, 3) [T]  estimator's dipole reference field
    # --- measurements ---
    gyro_meas: np.ndarray    # (N, 3) [rad/s]
    mag_body: np.ndarray     # (N, 3) [T]
    sun_body: np.ndarray     # (N, 3) unit, NaN where invalid
    sun_valid: np.ndarray    # (N,)   bool

    def __len__(self):
        return len(self.t)

    @property
    def eclipse_fraction(self):
        return float(np.mean(~self.lit))


@dataclass
class Scenario:
    """Configuration for one determination run."""

    epoch: _dt.datetime = field(
        default_factory=lambda: _dt.datetime(2024, 3, 20, 12, 0, 0))
    orbit: CircularOrbit = field(default_factory=CircularOrbit)
    body: RigidBody = field(default_factory=RigidBody)
    q0: np.ndarray = field(default_factory=lambda: IDENTITY_QUAT.copy())
    omega0: np.ndarray = field(
        default_factory=lambda: np.array([4.0, -3.0, 5.0]) * DEG)  # deg/s tumble
    dt: float = 0.5
    duration: float = None        # seconds; default = one orbital period
    gyro: Gyro = field(default_factory=Gyro)
    magnetometer: Magnetometer = field(default_factory=Magnetometer)
    sun_sensor: SunSensorArray = field(default_factory=SunSensorArray)
    seed: int = 0
    # --- systematic (unaveraged) error sources; 0 => truth == model ---
    albedo_coeff: float = 0.0            # Earth-albedo fraction on the sun sensor
    field_model_error_deg: float = 0.0   # IGRF-vs-dipole field direction error

    def run(self):
        """Generate the truth + sensor timeline as a :class:`SimResult`."""
        duration = self.orbit.period if self.duration is None else self.duration
        n_steps = int(round(duration / self.dt))
        jd0 = julian_date(self.epoch)
        rng = np.random.default_rng(self.seed)
        self.gyro.reset()

        truth = simulate_free_tumble(self.body, self.q0, self.omega0,
                                     self.dt, n_steps)
        n = len(truth)

        jd = jd0 + truth.t / 86400.0
        r_eci = np.empty((n, 3))
        sun_eci = np.empty((n, 3))
        irradiance = np.empty(n)
        lit = np.empty(n, dtype=bool)
        B_eci = np.empty((n, 3))
        B_eci_ref = np.empty((n, 3))
        gyro_meas = np.empty((n, 3))
        mag_body = np.empty((n, 3))
        sun_body = np.full((n, 3), np.nan)
        sun_valid = np.zeros(n, dtype=bool)
        gyro_bias_true = np.empty((n, 3))

        for k in range(n):
            q, w = truth.q[k], truth.omega[k]
            r = self.orbit.position(truth.t[k])
            s_eci = sun_direction_eci(jd[k])
            E = solar_irradiance(jd[k])
            is_lit = eclipse_factor(r, s_eci) > 0.0
            B_ref = magnetic_field_eci(r, jd[k])                      # estimator model
            B_true = truth_field_eci(r, jd[k], self.field_model_error_deg)  # measured

            r_eci[k], sun_eci[k], irradiance[k], lit[k] = r, s_eci, E, is_lit
            B_eci[k], B_eci_ref[k] = B_true, B_ref

            # Earth-albedo geometry: reflected light comes from nadir, strongest
            # where the sub-satellite point is sunlit (over the dayside).
            r_hat = r / np.linalg.norm(r)
            nadir_body = to_body(q, -r_hat)
            dayside = max(0.0, float(np.dot(r_hat, s_eci)))
            albedo_E = self.albedo_coeff * E * dayside

            gyro_bias_true[k] = self.gyro.bias
            gyro_meas[k] = self.gyro.measure(w, self.dt, rng)
            mag_body[k] = self.magnetometer.measure(B_true, q, rng)
            s_hat, valid = self.sun_sensor.measure_sun_vector(
                to_body(q, s_eci), E, is_lit, rng,
                albedo_dir_body=nadir_body, albedo_irradiance=albedo_E)
            sun_valid[k] = valid
            if valid:
                sun_body[k] = s_hat

        return SimResult(
            t=truth.t, jd=jd, dt=self.dt,
            q_true=truth.q, omega_true=truth.omega, gyro_bias_true=gyro_bias_true,
            r_eci=r_eci, sun_eci=sun_eci, irradiance=irradiance, lit=lit,
            B_eci=B_eci, B_eci_ref=B_eci_ref, gyro_meas=gyro_meas,
            mag_body=mag_body, sun_body=sun_body, sun_valid=sun_valid)
