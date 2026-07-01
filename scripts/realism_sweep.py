#!/usr/bin/env python3
"""Expected MEKF accuracy vs systematic (unmodeled) error level.

    python3 scripts/realism_sweep.py [n_runs]

The as-shipped sim uses a perfect environment model (truth == estimator model),
so the only error is zero-mean sensor noise the filter averages away -> ~0.3 deg.
That is a *validation* baseline, not an in-orbit prediction. This sweep adds the
two dominant real-world systematics for this sensor suite — Earth albedo on the
coarse sun sensors and IGRF-vs-dipole field error — which are unaveraged and so
enter attitude error ~1:1, and reports the realistic expected accuracy.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.montecarlo import REALISM_LEVELS, run_realism_sweep  # noqa: E402


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(f"MEKF accuracy vs systematic-error level ({n_runs} randomized orbits each)\n")
    stats = run_realism_sweep(n_runs=n_runs, dt=1.0)

    hdr = f"{'level':<22}{'albedo':>7}{'field':>7}{'median':>9}{'p95':>9}{'eclipse':>9}{'budget<15':>11}"
    print(hdr)
    print("-" * len(hdr))
    for (name, alb, fld) in REALISM_LEVELS:
        s = stats[name]
        print(f"{name:<22}{alb:>7.2f}{fld:>6.0f}°{s.overall_median():>8.2f}°"
              f"{s.overall_p95():>8.2f}°{float(np.nanmedian(s.run_eclipse_median)):>8.2f}°"
              f"{s.budget_met_fraction()*100:>10.0f}%")

    print("\nThe ideal row is the validation baseline (filter + observability are "
          "sound).\nThe moderate/worst rows are the defensible in-orbit expectation.")


if __name__ == "__main__":
    main()
