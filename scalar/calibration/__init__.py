"""Sensor calibration algorithms for SCALAR.

Two families:

* **On-ground** (:mod:`scalar.calibration.sun_onground`): pre-flight
  characterisation against a known reference (a collimated source on a rate
  table), fitting each sensor's response curve directly.
* **On-orbit** (:mod:`scalar.calibration.mag_onorbit`): in-flight,
  attitude-independent self-calibration that exploits a known scalar
  reference (the geomagnetic field magnitude from IGRF) to recover the
  sensor distortion without ever knowing the attitude.
"""
from .mag_onorbit import (
    MagCalibration,
    fit_magnetometer_ellipsoid,
    apply_mag_calibration,
)
from .sun_onground import (
    SunResponseCal,
    calibrate_sun_response,
    sweep_currents,
)
from .startracker_ref import (
    MagRefCalibration,
    calibrate_mag_with_reference,
    calibrate_sun_gains_with_reference,
)

__all__ = [
    "MagCalibration",
    "fit_magnetometer_ellipsoid",
    "apply_mag_calibration",
    "SunResponseCal",
    "calibrate_sun_response",
    "sweep_currents",
    "MagRefCalibration",
    "calibrate_mag_with_reference",
    "calibrate_sun_gains_with_reference",
]
