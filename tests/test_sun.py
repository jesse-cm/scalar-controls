"""Tests for scalar.environment.sun — validated against known astronomy."""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.environment import sun  # noqa: E402
from scalar.timeutil import julian_date  # noqa: E402


def _jd(y, m, d, hour=12):
    return julian_date(dt.datetime(y, m, d, hour, 0, 0))


def test_direction_is_unit():
    for jd in (_jd(2024, 1, 1), _jd(2024, 6, 21), _jd(2025, 9, 23)):
        assert np.isclose(np.linalg.norm(sun.sun_direction_eci(jd)), 1.0)


def test_declination_at_solstices_and_equinoxes():
    """Sun declination = asin(z_eci). Real values: ~+23.44 (Jun), ~-23.44 (Dec),
    ~0 at the equinoxes. Tolerances allow for the ~day-level date approximation."""
    def dec_deg(jd):
        return np.rad2deg(np.arcsin(sun.sun_direction_eci(jd)[2]))

    assert abs(dec_deg(_jd(2024, 6, 20)) - 23.44) < 0.5     # summer solstice
    assert abs(dec_deg(_jd(2024, 12, 21)) + 23.44) < 0.5    # winter solstice
    assert abs(dec_deg(_jd(2024, 3, 20))) < 1.0             # vernal equinox
    assert abs(dec_deg(_jd(2024, 9, 22))) < 1.0             # autumnal equinox


def test_irradiance_level_and_eccentricity():
    jan = sun.solar_irradiance(_jd(2024, 1, 3))   # near perihelion
    jul = sun.solar_irradiance(_jd(2024, 7, 4))   # near aphelion
    assert 1300.0 < jul < jan < 1430.0            # perihelion brighter
    assert (jan - jul) / jul > 0.05               # ~6.9% peak-to-peak


def test_eclipse_cylindrical_geometry():
    s = np.array([1.0, 0.0, 0.0])                 # sun toward +x
    R = 6378.137e3
    on_axis_shadow = np.array([-7000e3, 0.0, 0.0])
    sunward = np.array([7000e3, 0.0, 0.0])
    side_lit = np.array([-7000e3, R + 1e6, 0.0])
    side_dark = np.array([-7000e3, 3000e3, 0.0])

    assert sun.is_eclipsed(on_axis_shadow, s)
    assert not sun.is_eclipsed(sunward, s)
    assert not sun.is_eclipsed(side_lit, s)
    assert sun.is_eclipsed(side_dark, s)
    assert sun.eclipse_factor(sunward, s) == 1.0
    assert sun.eclipse_factor(on_axis_shadow, s) == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
