#!/usr/bin/env python3
"""Tune LQR weights + mode thresholds by cross-entropy method against the criteria.

    python3 scripts/rl_tune_lqr.py [pop] [generations] [init.json] [out.json]

Trains on seeds 0,1 only. Writes results/lqr_tuned.json. Verification is a
separate command on held-out seeds: scripts/verify_controller.py.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.control.rl_tune import (PARAM_NAMES, cem_tune, criteria, default_plant,  # noqa: E402
                                    evaluate, load_params, reward, save_params)


def main():
    pop = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    plant = default_plant(mag_calibrated=True)
    hand = np.array([10.0, 0.2, 0.3, 0.3, 1.5, 5e3, 10.0, 0.2, 0.3])   # hand-picked design; fine = acquisition weights
    if len(sys.argv) > 3:
        hand, _ = load_params(sys.argv[3])                    # continue from a previous result
    out = sys.argv[4] if len(sys.argv) > 4 else "results/lqr_tuned.json"
    t0 = time.time()
    m = evaluate(hand, plant, seed=0)
    print(f"hand-picked design, seed 0 ({time.time()-t0:.0f} s): reward {reward(m):+.2f}")
    for name, v, thr, ok in criteria(m):
        print(f"   {'PASS' if ok else 'FAIL'}  {name:<16} {v:8.3f}  <= {thr}")
    print(f"\nCEM: pop {pop}, {gens} generations, train seeds (0, 1) ...")
    best, best_metrics, hist = cem_tune(plant, pop=pop, generations=gens, init=hand, sigma0=0.12, rng_seed=1)
    os.makedirs("results", exist_ok=True)
    save_params(out, best, best_metrics, {"history": hist, "train_seeds": [0, 1]})
    print("\nbest params:", {k: round(float(v), 4) for k, v in zip(PARAM_NAMES, best)})
    for name, v, thr, ok in criteria(best_metrics):
        print(f"   {'PASS' if ok else 'FAIL'}  {name:<16} {v:8.3f}  <= {thr}")
    print(f"\nsaved {out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
