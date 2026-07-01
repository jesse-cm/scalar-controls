#!/usr/bin/env python3
"""Monte-Carlo estimator comparison for the SCALAR determination sim.

    python3 scripts/monte_carlo.py [n_runs]

Runs the default variants (MEKF, MEKF cold-start, TRIAD-only) over many
randomized scenarios, prints a ranked table vs the 10-15 deg budget, and (if
matplotlib is present) saves a box plot of per-run error to outputs/.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.montecarlo import rank, run_monte_carlo  # noqa: E402


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(f"Running {n_runs} randomized scenarios (one orbit each)...\n")
    stats = run_monte_carlo(n_runs=n_runs, dt=1.0)

    hdr = f"{'variant':<18}{'median':>9}{'p95':>9}{'eclipse':>9}{'worst':>9}{'budget<15':>11}"
    print(hdr)
    print("-" * len(hdr))
    for s in rank(stats):
        print(f"{s.name:<18}{s.overall_median():>8.2f}°{s.overall_p95():>8.2f}°"
              f"{float(np.nanmedian(s.run_eclipse_median)):>8.2f}°{s.worst():>8.2f}°"
              f"{s.budget_met_fraction()*100:>10.0f}%")
    best = rank(stats)[0]
    print(f"\nMost accurate: {best.name} "
          f"(median {best.overall_median():.2f}°, "
          f"{best.budget_met_fraction()*100:.0f}% of runs within budget)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib unavailable; skipped plot)")
        return

    os.makedirs("outputs", exist_ok=True)
    names = [s.name for s in rank(stats)]
    data = [rank(stats)[i].run_median for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=names, showfliers=True)
    ax.axhline(15, ls="--", color="C3", lw=0.8, label="15° budget")
    ax.set_ylabel("per-run median attitude error [deg]")
    ax.set_title(f"Estimator comparison, {n_runs} randomized orbits")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = "outputs/monte_carlo.png"
    fig.savefig(path, dpi=120)
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    main()
