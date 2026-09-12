#!/usr/bin/env python3
"""Experiment 2: absolute attitude with no GPS — how good must position/time be?

    python3 scripts/absolute_without_gps.py

SCALAR has no GNSS. Absolute (inertial) attitude from mag + sun + gyro needs the
reference field B_eci(r, t), i.e. a position. Onboard that comes from an
uplinked TLE + the RV-3028 RTC + a propagator. This sweeps an along-track
position error (equivalently a clock error at 7.6 km/s) in the *estimator's*
reference field and reports the resulting MEKF knowledge error, so the team
can budget TLE age and clock drift instead of guessing.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.constants import DEG  # noqa: E402
from scalar.environment.magfield import magnetic_field_eci  # noqa: E402
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.estimator.mekf import MEKF, run_mekf  # noqa: E402
from scalar.scenario import Scenario  # noqa: E402


def main():
    orbit = CircularOrbit(inclination=98.61 * DEG, j2=True)   # S7's orbit, not S4's
    sc = Scenario(orbit=orbit, dt=1.0, seed=5)
    res = sc.run()
    v = orbit.radius * orbit.mean_motion
    print(f"Orbit 550 km / 98.61 deg, period {orbit.period/60:.1f} min, "
          f"eclipse {100*res.eclipse_fraction:.0f} %, v = {v/1e3:.2f} km/s")
    print(f"\n{'along-track err':>16} {'clock equiv':>12} {'field dir err':>14} "
          f"{'MEKF median':>12} {'p95':>7} {'eclipse med':>12}")
    for km in (0.0, 5.0, 20.0, 50.0, 100.0, 300.0, 1000.0):
        dt_shift = km * 1e3 / v
        ref = np.array([magnetic_field_eci(orbit.position(t + dt_shift), jd)
                        for t, jd in zip(res.t, res.jd)])
        ang = np.degrees(np.arccos(np.clip(np.sum(ref * res.B_eci, 1)
                        / np.linalg.norm(ref, axis=1) / np.linalg.norm(res.B_eci, axis=1), -1, 1)))
        res2 = res.__class__(**{**res.__dict__, "B_eci_ref": ref})
        out = run_mekf(res2, MEKF(n_iter=3)).summary()
        print(f"{km:>13.0f} km {dt_shift:>10.1f} s {np.median(ang):>12.2f}° "
              f"{out['median_deg']:>10.2f}° {out['p95_deg']:>6.2f}° "
              f"{out.get('eclipse_median_deg', float('nan')):>10.2f}°")
    print("\nSun-direction error from clock error is ~1 deg/day, negligible next to the above.")


if __name__ == "__main__":
    main()
