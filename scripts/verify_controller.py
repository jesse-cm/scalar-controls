#!/usr/bin/env python3
"""Verify a tuned controller on HELD-OUT seeds and robustness variants. Exit 1 on any FAIL.

    python3 scripts/verify_controller.py [results/lqr_tuned.json]

Criteria and thresholds: scalar/control/rl_tune.THRESHOLDS. The tuner trained
on seeds 0-1; this runs seeds 100-102 on the nominal plant, plus the same plant
with 10 % coil/inertia error, on one seed each. Every row is value <= threshold.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.control.rl_tune import PARAM_NAMES, criteria, default_plant, evaluate, load_params  # noqa: E402

HELD_OUT_SEEDS = (100, 101, 102)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/lqr_tuned.json"
    params, meta = load_params(path)
    print("controller:", {k: round(float(v), 4) for k, v in zip(PARAM_NAMES, params)})
    variants = [("nominal", default_plant(mag_calibrated=True), HELD_OUT_SEEDS),
                ("coils -10 %, inertia +10 %", default_plant(mag_calibrated=True, scale_M=0.9, scale_inertia=1.1), HELD_OUT_SEEDS[:1]),
                ("coils +10 %, inertia -10 %", default_plant(mag_calibrated=True, scale_M=1.1, scale_inertia=0.9), HELD_OUT_SEEDS[:1])]
    all_ok = True
    for label, plant, seeds in variants:
        print(f"\n== {label}, seeds {seeds}")
        for s in seeds:
            m = evaluate(params, plant, seed=s)
            rows = criteria(m)
            ok = all(r[3] for r in rows)
            all_ok &= ok
            print(f"  seed {s}: {'PASS' if ok else 'FAIL'}   " + "  ".join(
                f"{n}={v:.2f}{'' if p else '!'}" for n, v, t, p in rows))
    print("\nVERDICT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
