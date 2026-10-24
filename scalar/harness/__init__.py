"""Verification-loop harness (docs/verification-loop-plan.md, phase 0).

    from scalar.harness import ScenarioSpec, run_case, run_matrix, baseline_specs, SEEDS

    rows = run_matrix(baseline_specs(["5"]), SEEDS)      # ledger rows, one per spec
    append_rows("results/ledger.csv", rows)

The harness wraps the existing simulators and changes none of them. See
``spec.py`` for the switches, ``metrics.py`` for the fixed metric set,
``ledger.py`` for the ledger columns, ``baseline.py`` for experiments 1-5.
"""
from __future__ import annotations

from .baseline import DOC_CHECKS, PLAN_GATE_SEED, SCRIPT_SEEDS, SEEDS, baseline_specs, rl_tune_specs
from .ledger import LEDGER_COLUMNS, append_rows, ledger_row, read_ledger
from .metrics import FIXED_METRICS, Metrics
from .runner import run_case, run_cases, run_matrix
from .spec import ScenarioSpec

__all__ = ["ScenarioSpec", "Metrics", "FIXED_METRICS", "run_case", "run_cases", "run_matrix",
           "LEDGER_COLUMNS", "ledger_row", "append_rows", "read_ledger",
           "baseline_specs", "rl_tune_specs", "SEEDS", "PLAN_GATE_SEED", "SCRIPT_SEEDS", "DOC_CHECKS"]
