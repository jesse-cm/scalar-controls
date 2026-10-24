#!/usr/bin/env python3
"""Run the scenario matrix and append rows to the ledger (verification-loop plan, phase 0).

    python3 scripts/matrix.py --list                       # the baseline scenarios, no run
    python3 scripts/matrix.py --gate                       # doc numbers vs harness, single-seed rows
    python3 scripts/matrix.py --seeds 0-39                 # baseline, experiments 1-5, 40 seeds
    python3 scripts/matrix.py --experiments 5 --seeds 0-39 --workers 8
    python3 scripts/matrix.py --match 'e4D*' --seeds 0-7 --no-ledger
    python3 scripts/matrix.py --experiments 6 --seeds 0-39 # the tuned controllers of results/lqr_tuned*.json

Experiment 6 (each results/lqr_tuned*.json as the acquisition and the
fine-pointing episode of rl_tune.evaluate) runs only when asked for by id; it
has no number in the --gate table and --gate refuses it.

Rows go to results/ledger.csv (appended; the file is created with the fixed
header). One row per (scenario, estimator, controller, seed-set, git hash,
date); each metric is the median and the 95th percentile across the seed set.

--gate runs every selected scenario at the plan's seed 2 and at the seed its
script hard-codes, prints the comparison against the numbers in
docs/learned-estimation-and-control.md, and appends one single-seed row per
(scenario, seed). It never changes a setting to make a number match.

--per-seed PATH writes every (scenario, seed) result, including the
experiment-specific numbers that have no ledger column (sun-vector error,
sysid matrix error, field-direction error), as CSV.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.harness import FIXED_METRICS, append_rows, ledger_row, run_cases  # noqa: E402
from scalar.harness.baseline import select_specs  # noqa: E402
from scalar.harness.gate import format_gate, gate_seeds, gate_table  # noqa: E402
from scalar.harness.ledger import current_git_hash, parse_seed_set  # noqa: E402
from scalar.harness.runner import stderr_progress  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_per_seed(path, specs, cases):
    by_key = {s.key: s for s in specs}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "scenario", "estimator", "controller", "seed", *FIXED_METRICS, "wall_s", "extra"])
        for (key, seed), m in sorted(cases.items(), key=lambda kv: (list(by_key).index(kv[0][0]), kv[0][1])):
            s = by_key[key]
            w.writerow([s.experiment, s.name, s.estimator_label, s.controller_label, seed,
                        *[repr(float(getattr(m, k))) for k in FIXED_METRICS], f"{m.wall_s:.2f}",
                        json.dumps(m.extra, sort_keys=True)])


def main(argv=None, select=None):
    """``select(experiments) -> specs`` replaces select_specs: the route for the row scripts
    (mag_realism.py, gyro_realism.py, plant_step_rows.py), which pass their own scenario list
    here instead of patching a module name that can change under them (tests/test_row_scripts.py)."""
    select = select_specs if select is None else select
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", default="1,2,3,4,5", help="comma list of experiment ids (default the baseline, 1-5; 6 = tuned controllers)")
    ap.add_argument("--match", default=None, help="glob on 'scenario | estimator | controller'")
    ap.add_argument("--seeds", default=None, help="seed set, e.g. 0-39 or 2 or 100,101,102")
    ap.add_argument("--gate", action="store_true", help="compare against the doc at seed 2 and the scripts' seeds")
    ap.add_argument("--list", action="store_true", help="list the selected scenarios and exit")
    ap.add_argument("--workers", type=int, default=None, help="processes (default: cpu count - 2)")
    ap.add_argument("--ledger", default=os.path.join(ROOT, "results", "ledger.csv"))
    ap.add_argument("--no-ledger", action="store_true", help="run and print, write nothing to the ledger")
    ap.add_argument("--per-seed", default=None, metavar="PATH", help="also write per-seed results as CSV")
    a = ap.parse_args(argv)

    experiments = [e.strip() for e in a.experiments.split(",")]
    if a.gate and "6" in experiments:
        ap.error("--gate compares experiments 1-5 against the doc; experiment 6 has no gate numbers")
    specs = select(experiments)
    if a.match:
        specs = [s for s in specs if fnmatch.fnmatch(s.key, a.match)]
    if not specs:
        ap.error("no scenario selected")
    if a.list:
        for s in specs:
            print(f"{s.experiment}  {s.kind:<11} {s.key}")
        print(f"{len(specs)} scenarios")
        return 0
    if a.gate == (a.seeds is not None):
        ap.error("give exactly one of --gate or --seeds")

    seeds = gate_seeds(specs) if a.gate else {s.key: parse_seed_set(a.seeds) for s in specs}
    n_cases = sum(len(v) for v in seeds.values())
    git_hash, t0 = current_git_hash(), time.time()
    print(f"{len(specs)} scenarios, {n_cases} cases, git {git_hash}", file=sys.stderr)
    cases = run_cases(specs, seeds, workers=a.workers, progress=stderr_progress)
    wall = time.time() - t0

    rows = []
    for s in specs:
        groups = [[sd] for sd in seeds[s.key]] if a.gate else [seeds[s.key]]
        for g in groups:
            rows.append(ledger_row(s, g, [cases[(s.key, sd)] for sd in g], git_hash=git_hash))
    if a.gate:
        print(format_gate(gate_table(specs, cases)))
    if a.per_seed:
        write_per_seed(a.per_seed, specs, cases)
        print(f"per-seed results -> {a.per_seed}", file=sys.stderr)
    if not a.no_ledger:
        append_rows(a.ledger, rows)
        print(f"{len(rows)} rows -> {a.ledger}", file=sys.stderr)
    cpu = sum(m.wall_s for m in cases.values())
    print(f"wall {wall:.0f} s, summed case time {cpu:.0f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
