"""Tests: parameter transforms, criteria and reward logic of the CEM tuner (no simulation)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.control.rl_tune import (BOUNDS, THRESHOLDS, _from_z, _to_z, clip_params,  # noqa: E402
                                    criteria, reward)


def test_param_transform_roundtrip():
    p = np.array([10.0, 0.2, 0.3, 0.3, 1.5, 5e3, 10.0, 0.2, 0.3])
    assert np.allclose(_from_z(_to_z(p)), p)
    assert np.allclose(_from_z(np.zeros(9)), BOUNDS[:, 0]) and np.allclose(_from_z(np.ones(9)), BOUNDS[:, 1])
    assert np.all(clip_params([1e9] * 9) == BOUNDS[:, 1])


def _passing():
    m = {k: 0.5 * v for k, v in THRESHOLDS.items()}
    m["acq_diverged"] = 0.0; m["fine_diverged"] = 0.0
    return m


def test_criteria_all_pass_and_one_fail():
    m = _passing()
    assert all(r[3] for r in criteria(m))
    m["fine_point_mean"] = THRESHOLDS["fine_point_mean"] + 0.01
    rows = criteria(m)
    assert sum(not r[3] for r in rows) == 1 and [r for r in rows if not r[3]][0][0] == "fine_point_mean"


def test_reward_orders_pass_over_fail_over_divergence():
    good, bad, div = _passing(), _passing(), _passing()
    bad["fine_point_mean"] = 4 * THRESHOLDS["fine_point_mean"]
    div["fine_diverged"] = 1.0
    assert reward(good) > reward(bad) > reward(div)





def test_calibration_pass_recovers_direction():
    import datetime as dt
    from scalar.constants import DEG
    from scalar.control.rl_tune import calibration_pass, default_plant
    from scalar.environment.orbit import CircularOrbit
    from scalar.sensors.magnetometer import Magnetometer
    from scalar.timeutil import julian_date
    plant = default_plant(mag_calibrated=True)
    mag = Magnetometer(bias=plant.mag_bias, soft_iron=plant.mag_soft_iron)
    orbit = CircularOrbit(inclination=98.61 * DEG)
    cal, err_med = calibration_pass(plant, mag, orbit, julian_date(dt.datetime(2026, 9, 11, 12)), seed=0)
    assert np.linalg.norm(cal.v - plant.mag_bias) < 0.3e-6, cal.v        # bias to 0.3 uT
    assert err_med < 1.5, err_med                                          # direction to 1.5 deg


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"  ok  {f.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {f.__name__}: {e}")
    print(f"{ok}/{len(fns)} passed")
