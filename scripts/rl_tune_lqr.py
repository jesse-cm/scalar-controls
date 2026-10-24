#!/usr/bin/env python3
"""Tune LQR weights + mode thresholds by cross-entropy method against the criteria.

    python3 scripts/rl_tune_lqr.py [pop] [generations] [init.json] [out.json] [--seeds 0-7] [--workers 4]

Trains on seeds 0,1 unless --seeds says otherwise; the seeds used are written
to the output file as ``train_seeds``. Writes results/lqr_tuned.json.
Verification is a separate command on held-out seeds: scripts/verify_controller.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.control.rl_tune import (PARAM_NAMES, cem_tune, criteria, default_plant,  # noqa: E402
                                    evaluate, load_params, reward, save_params)
from scalar.harness.ledger import parse_seed_set  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pop", nargs="?", type=int, default=12, help="CEM population (default 12)")
    ap.add_argument("generations", nargs="?", type=int, default=5, help="CEM generations (default 5)")
    ap.add_argument("init", nargs="?", default=None, help="continue from this result file (default: the hand-picked design)")
    ap.add_argument("out", nargs="?", default="results/lqr_tuned.json", help="output file")
    ap.add_argument("--seeds", default="0,1", help="training seeds, e.g. 0-7 or 0,1,5 (default 0,1)")
    ap.add_argument("--workers", type=int, default=None, help="processes (default: cpu count - 1)")
    a = ap.parse_args(argv)
    pop, gens, out = a.pop, a.generations, a.out
    train_seeds = tuple(parse_seed_set(a.seeds))
    if not train_seeds or len(set(train_seeds)) != len(train_seeds):
        ap.error("--seeds must name at least one seed, each once")
    plant = default_plant(mag_calibrated=True)
    hand = np.array([10.0, 0.2, 0.3, 0.3, 1.5, 5e3, 10.0, 0.2, 0.3])   # hand-picked design; fine = acquisition weights
    if a.init is not None:
        hand, _ = load_params(a.init)                         # continue from a previous result
    t0 = time.time()
    m = evaluate(hand, plant, seed=0)
    print(f"hand-picked design, seed 0 ({time.time()-t0:.0f} s): reward {reward(m):+.2f}")
    for name, v, thr, ok in criteria(m):
        print(f"   {'PASS' if ok else 'FAIL'}  {name:<16} {v:8.3f}  <= {thr}")
    print(f"\nCEM: pop {pop}, {gens} generations, train seeds {train_seeds} ...")
    best, best_metrics, hist = cem_tune(plant, train_seeds=train_seeds, pop=pop, generations=gens, init=hand,
                                        sigma0=0.12, workers=a.workers, rng_seed=1)
    os.makedirs("results", exist_ok=True)
    save_params(out, best, best_metrics, {"history": hist, "train_seeds": list(train_seeds)})
    print("\nbest params:", {k: round(float(v), 4) for k, v in zip(PARAM_NAMES, best)})
    for name, v, thr, ok in criteria(best_metrics):
        print(f"   {'PASS' if ok else 'FAIL'}  {name:<16} {v:8.3f}  <= {thr}")
    print(f"\nsaved {out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
