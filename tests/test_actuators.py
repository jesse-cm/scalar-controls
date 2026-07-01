"""Tests for scalar.actuators.magnetorquer."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.actuators.magnetorquer import (ActuatorSet, Magnetorquer,  # noqa: E402
                                           default_actuator_set)


def test_dipole_law_air_core_vs_rod():
    coil = Magnetorquer.air_core([1, 0, 0], n_turns=100, area=6e-3, current_max=0.05)
    rod = Magnetorquer.rod([1, 0, 0], n_turns=100, area=6e-3, current_max=0.05,
                           core_gain=15.0)
    assert np.isclose(coil.m_max, 100 * 6e-3 * 0.05)
    assert np.isclose(rod.m_max, 15.0 * coil.m_max)     # core gain multiplies


def test_axis_is_normalized():
    t = Magnetorquer([0, 0, 3.0], 0.2)
    assert np.allclose(t.axis, [0, 0, 1.0])


def test_torque_is_m_cross_B_and_perp_to_B():
    acts = default_actuator_set()
    B = np.array([1e-5, -2e-5, 3e-5])
    tau, m = acts.torque([0.1, -0.1, 0.01], B)
    assert np.allclose(tau, np.cross(m, B))
    assert abs(np.dot(tau, B)) < 1e-18                  # no torque along B


def test_saturation_per_axis():
    acts = default_actuator_set(m_rod=0.2, m_coil=0.02)
    m = acts.command([10.0, -10.0, 10.0])               # way over-commanded
    assert np.allclose(m, [0.2, -0.2, 0.02])            # clipped to m_max each
    m2 = acts.command([0.1, 0.05, 0.005])               # within limits
    assert np.allclose(m2, [0.1, 0.05, 0.005])


def test_dipole_along_B_gives_no_torque():
    """Underactuation: commanding a dipole parallel to B produces zero torque."""
    acts = ActuatorSet([Magnetorquer([1, 0, 0], 1.0), Magnetorquer([0, 1, 0], 1.0),
                        Magnetorquer([0, 0, 1], 1.0)])
    B = np.array([0.0, 0.0, 4e-5])
    tau, m = acts.torque([0.0, 0.0, 0.5], B)            # dipole along B
    assert np.allclose(tau, 0.0)


def test_default_set_config():
    acts = default_actuator_set()
    names = [t.name for t in acts.torquers]
    assert names == ["rod-x", "rod-y", "pcb-coil-z"]
    assert np.allclose(acts.m_max_vector, [0.2, 0.2, 0.02])


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
