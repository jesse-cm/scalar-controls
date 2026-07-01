#!/usr/bin/env python3
"""Closed-loop LQR magnetorquer detumble demo.

    python3 scripts/detumble_lqr.py

Spins the 1U bus up to a tumble and runs the LQR controller (2 rods + 1 PCB
coil) closed-loop over several orbits, printing the rate decay and (if
matplotlib is present) saving a plot of |omega| and dipole usage to outputs/.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.actuators.magnetorquer import default_actuator_set  # noqa: E402
from scalar.attitude import IDENTITY_QUAT  # noqa: E402
from scalar.control.lqr import MagnetorquerLQR  # noqa: E402
from scalar.dynamics import RigidBody, simulate_controlled  # noqa: E402
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.timeutil import julian_date  # noqa: E402


def main():
    body = RigidBody()
    acts = default_actuator_set()
    orbit = CircularOrbit()
    jd0 = julian_date(dt.datetime(2024, 3, 20, 12))
    ctrl = MagnetorquerLQR(body.inertia, acts, orbit, jd0)

    w0 = np.deg2rad(8.0) * np.array([1.0, -0.6, 0.8])
    n_orbits = 4
    dt_s = 2.0
    n = int(n_orbits * orbit.period / dt_s)
    tl, taus = simulate_controlled(body, IDENTITY_QUAT, w0, dt_s, n, ctrl)

    w_deg = np.rad2deg(np.linalg.norm(tl.omega, axis=1))
    print(f"Actuators: 2 rods (0.2 A·m²) on X/Y + PCB coil (0.02 A·m²) on Z")
    print(f"Detumble over {n_orbits} orbits ({tl.t[-1]/3600:.1f} h), dt={dt_s}s")
    print("-" * 44)
    for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        i = int(frac * (len(w_deg) - 1))
        print(f"  t={tl.t[i]/3600:5.2f} h   |omega| = {w_deg[i]:6.3f} deg/s")
    print(f"  rate reduced {w_deg[0]/w_deg[-1]:.1f}x")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib unavailable; skipped plot)")
        return

    os.makedirs("outputs", exist_ok=True)
    m = np.array(ctrl.m_hist)
    th = tl.t[:len(m)] / 3600.0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    a1.plot(tl.t / 3600.0, w_deg, color="C0")
    a1.set_ylabel("|omega| [deg/s]")
    a1.set_title("LQR magnetorquer detumble (2 rods + PCB coil)")
    a1.grid(alpha=0.3)
    for j, name in enumerate(["rod-x", "rod-y", "coil-z"]):
        a2.plot(th, m[:, j], lw=0.7, label=name)
    a2.set_ylabel("dipole [A·m²]")
    a2.set_xlabel("time [h]")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    path = "outputs/detumble_lqr.png"
    fig.savefig(path, dpi=120)
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    main()
