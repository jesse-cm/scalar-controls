"""On-ground (pre-flight) sun-sensor response-curve calibration.

A coarse photodiode sun sensor's accuracy is set by the *relative* diode
signals (the absolute scale cancels in the reconstruction), so the systematics
that matter are the ones that distort the diodes relative to each other:

* per-diode **gain** (responsivity x active-area part-to-part spread), and
* the cover-glass **FOV roll-off** (the angular response falls below the ideal
  cosine toward grazing incidence).

Both are characterised before launch on an optical bench: the flat-sat is
mounted on a rate/goniometer table under a collimated solar-simulator beam of
known irradiance, and each diode is swept through a set of known incidence
angles while its short-circuit current is logged.  For one diode the model is

    I(theta) / I_fs = g * cos(theta) * (1 - rho * (1 - cos(theta))),

with ``g`` the gain, ``rho`` the roll-off and ``I_fs`` the full-scale
(head-on) current.  This is *nonlinear* in ``(g, rho)`` but becomes **linear**
under the reparameterisation ``a = g``, ``b = g*rho``::

    I/I_fs = a * cos - b * (cos - cos^2),

so a single linear least-squares solve per diode recovers ``a`` and ``b``, and
then ``g = a``, ``rho = b / a``.  The recovered curve is inverted to correct
in-flight currents before reconstruction.

This complements the on-orbit magnetometer calibration
(:mod:`scalar.calibration.mag_onorbit`): here the reference (the beam angle and
irradiance) is *controlled* on the ground, whereas the on-orbit method exploits
a known scalar (the field magnitude) with no attitude control at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SunResponseCal",
    "calibrate_sun_response",
    "sweep_currents",
]


@dataclass
class SunResponseCal:
    """Recovered per-diode response-curve calibration.

    Attributes
    ----------
    gains : (M,) ndarray
        Per-diode gain (relative responsivity x area). Nominal sensor -> ones.
    rolloff : (M,) ndarray
        Per-diode cover-glass roll-off coefficient. Nominal sensor -> zeros.
    residual_rms : (M,) ndarray
        RMS of the response-curve fit residual per diode (fraction of
        full-scale).
    n_angles : int
        Number of incidence angles used in the sweep.
    """

    gains: np.ndarray
    rolloff: np.ndarray
    residual_rms: np.ndarray
    n_angles: int

    def correct_currents(self, currents: np.ndarray) -> np.ndarray:
        """Divide out the recovered per-diode gains.

        The gain is the dominant relative distortion the reconstruction sees;
        dividing each diode current by its gain restores the balance the
        opposing-pair reconstruction assumes. (Roll-off is an angular shape
        term recovered for characterisation / modelling; the gain correction is
        what the reconstruction consumes.)
        """
        currents = np.asarray(currents, dtype=float)
        # gains index the diode axis: (M,) for a single reading, or the leading
        # axis of an (M, K) sweep.
        if currents.ndim == 1:
            return currents / self.gains
        return currents / self.gains[:, None]


def sweep_currents(sensor, angles_rad, irradiance, rng=None, noise_frac=None):
    """Simulate a bench sweep: per-diode current vs incidence angle.

    Each diode is illuminated head-on-plane and rotated through ``angles_rad``
    (incidence measured from the diode normal). Returns an ``(M, K)`` array of
    currents for ``M`` diodes and ``K`` angles, as an optical bench would log.

    Parameters
    ----------
    sensor : SunSensorArray
        The (systematic-bearing) sensor being characterised.
    angles_rad : (K,) array
        Incidence angles to sweep, radians from normal.
    irradiance : float
        Beam irradiance [W/m^2] (known, controlled).
    rng : np.random.Generator, optional
        If given, adds bench measurement noise.
    noise_frac : float, optional
        1-sigma bench noise as a fraction of full-scale. Defaults to the
        sensor's ``noise_frac`` when ``rng`` is given.
    """
    angles_rad = np.asarray(angles_rad, dtype=float)
    fs = sensor.full_scale_current(irradiance)
    cos = np.cos(angles_rad)
    # Build (M, K): gain_i * response(cos_k), using the sensor's own systematics
    M = len(sensor.gains)
    cur = np.empty((M, angles_rad.size))
    for k, c in enumerate(cos):
        cur[:, k] = fs * sensor.gains * sensor._angular_response(c)
    if rng is not None:
        nf = sensor.noise_frac if noise_frac is None else noise_frac
        cur = cur + rng.normal(0.0, nf * fs, size=cur.shape)
    return cur


def calibrate_sun_response(currents, angles_rad, irradiance, sensor=None,
                           full_scale=None):
    """Fit per-diode gain and roll-off from a bench sweep.

    Parameters
    ----------
    currents : (M, K) ndarray
        Per-diode currents at each swept angle (from :func:`sweep_currents` or
        real bench data).
    angles_rad : (K,) array
        The incidence angles swept, radians.
    irradiance : float
        Beam irradiance [W/m^2] used in the sweep.
    sensor : SunSensorArray, optional
        If given, its ``full_scale_current(irradiance)`` sets the current scale.
    full_scale : float, optional
        Explicit full-scale current [A]; overrides ``sensor``. If neither is
        given, the full-scale is estimated from the head-on (max-cos) column.

    Returns
    -------
    SunResponseCal
        Recovered per-diode gains, roll-off and fit residuals.
    """
    currents = np.asarray(currents, dtype=float)
    if currents.ndim != 2:
        raise ValueError(f"currents must be (M, K); got {currents.shape}")
    M, K = currents.shape
    angles_rad = np.asarray(angles_rad, dtype=float)
    if angles_rad.shape != (K,):
        raise ValueError(
            f"angles_rad must be ({K},) to match currents; got {angles_rad.shape}")
    if K < 3:
        raise ValueError(f"need at least 3 angles to fit the curve; got {K}")

    cos = np.cos(angles_rad)
    if full_scale is not None:
        fs = float(full_scale)
    elif sensor is not None:
        fs = sensor.full_scale_current(irradiance)
    else:
        # estimate: at the smallest incidence angle a nominal diode reads ~fs
        fs = currents[:, np.argmax(cos)].max() / max(cos.max(), 1e-9)

    # linear model per diode: I/fs = a*cos - b*(cos - cos^2)
    A = np.column_stack([cos, -(cos - cos ** 2)])
    gains = np.empty(M)
    rolloff = np.empty(M)
    resid = np.empty(M)
    for i in range(M):
        y = currents[i] / fs
        p, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = p
        gains[i] = a
        rolloff[i] = b / a if abs(a) > 1e-12 else 0.0
        resid[i] = float(np.sqrt(np.mean((A @ p - y) ** 2)))
    return SunResponseCal(gains=gains, rolloff=rolloff, residual_rms=resid,
                          n_angles=K)
