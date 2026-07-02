"""Tests for the Monte-Carlo comparison harness."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.montecarlo import (randomized_scenario, rank,  # noqa: E402
                               run_monte_carlo)


def test_randomized_scenarios_differ():
    a = randomized_scenario(0)
    b = randomized_scenario(1)
    assert not np.allclose(a.omega0, b.omega0)
    assert not np.allclose(a.q0, b.q0)


def test_monte_carlo_structure():
    stats = run_monte_carlo(n_runs=3, dt=3.0, duration=1800.0)
    assert set(stats) == {"MEKF", "IMEKF", "USQUE (UKF)", "MEKF cold-start",
                          "TRIAD-only"}
    for s in stats.values():
        assert s.run_median.shape == (3,)
        assert np.all(np.isfinite(s.run_median))


def test_mekf_beats_triad_only_in_eclipse():
    """The filter's whole point: coasting on the gyro through eclipse, where
    the gyro-free TRIAD-only estimator drifts badly holding its last fix."""
    stats = run_monte_carlo(n_runs=4, dt=2.0)   # full orbit -> real eclipse
    mekf_ecl = np.nanmedian(stats["MEKF"].run_eclipse_median)
    triad_ecl = np.nanmedian(stats["TRIAD-only"].run_eclipse_median)
    assert mekf_ecl < triad_ecl
    # and the MEKF should comfortably meet budget on most runs
    assert stats["MEKF"].budget_met_fraction() >= 0.75


def test_rank_orders_by_median():
    stats = run_monte_carlo(n_runs=3, dt=3.0, duration=1800.0)
    ordered = rank(stats)
    meds = [s.overall_median() for s in ordered]
    assert meds == sorted(meds)


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
