#!/usr/bin/env python3
"""Run one determination scenario through the MEKF and report accuracy.

    python3 scripts/run_mekf.py

Prints headline attitude-knowledge-error stats and (if matplotlib is present)
saves a plot of error vs time with the 1-sigma envelope and eclipse shading to
outputs/ (gitignored).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.estimator.mekf import run_mekf  # noqa: E402
from scalar.scenario import Scenario  # noqa: E402


def main():
    sc = Scenario(dt=0.5, seed=0)          # one orbit, ~35% eclipse
    res = sc.run()
    out = run_mekf(res)
    s = out.summary()

    print(f"Scenario: 550 km / 51.6 deg, {res.t[-1]/60:.1f} min "
          f"({res.eclipse_fraction*100:.0f}% eclipse), dt={res.dt}s")
    print(f"Initial tumble: {np.rad2deg(sc.omega0)} deg/s")
    print("-" * 52)
    print("Attitude knowledge error (post-convergence):")
    print(f"  median : {s['median_deg']:6.2f} deg")
    print(f"  95th   : {s['p95_deg']:6.2f} deg")
    print(f"  max    : {s['max_deg']:6.2f} deg")
    print(f"  sunlit : {s['lit_median_deg']:6.2f} deg (median)")
    print(f"  eclipse: {s['eclipse_median_deg']:6.2f} deg (median)")
    print(f"  budget : 10-15 deg  ->  {'MET' if s['p95_deg'] < 15 else 'MISSED'}")
    b_err = np.rad2deg(np.linalg.norm(out.b_est[-1] - res.gyro_bias_true[-1]))
    print(f"  gyro-bias error (final): {b_err:.3f} deg/s")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib unavailable; skipped plot)")
        return

    os.makedirs("outputs", exist_ok=True)
    t_min = res.t / 60.0
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # eclipse shading
    in_ecl = ~res.lit
    ax.fill_between(t_min, 0, 1, where=in_ecl, transform=ax.get_xaxis_transform(),
                    color="0.85", label="eclipse")
    ax.plot(t_min, out.err_deg, lw=1.0, color="C0", label="attitude error")
    ax.plot(t_min, out.P_theta_deg, lw=0.8, color="C1", alpha=0.7,
            label=r"1$\sigma$ estimate")
    ax.axhline(15, ls="--", color="C3", lw=0.8, label="15 deg budget")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("attitude knowledge error [deg]")
    ax.set_title("MEKF attitude determination (550 km / 51.6 deg)")
    ax.set_ylim(0, max(16, out.err_deg.max() * 1.1))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = "outputs/mekf_error.png"
    fig.savefig(path, dpi=120)
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    main()
