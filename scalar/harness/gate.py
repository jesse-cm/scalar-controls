"""Phase-0 gate: harness value vs the number the doc states, as a table. A measurement, not a target.

A doc number is compared at the precision the doc prints it with (the doc
rarely prints two decimals). The harness value is shown to two decimals more
than that, so a two-decimal comparison against the scripts' own output can be
read off the same table.
"""
from __future__ import annotations

import numpy as np

from .baseline import DOC_CHECKS, PLAN_GATE_SEED, SCRIPT_SEEDS
from .metrics import FIXED_METRICS

__all__ = ["gate_seeds", "metric_value", "matches", "gate_table", "format_gate"]


def gate_seeds(specs, checks=DOC_CHECKS):
    """{spec.key: sorted seeds} = the plan's seed 2, the script's seed, and every doc-check seed."""
    out = {s.key: {PLAN_GATE_SEED} | ({SCRIPT_SEEDS[s.key]} if s.key in SCRIPT_SEEDS else set())
           for s in specs}
    for c in checks:
        if c.key in out:
            out[c.key].add(c.seed)
    return {k: sorted(v) for k, v in out.items()}


def metric_value(m, name):
    return float(getattr(m, name)) if name in FIXED_METRICS else float(m.extra[name])


def _decimals(text):
    """Decimals the doc prints: '0.048' -> 3, '8' -> 0, '3.1–4.8' -> 1, '~0.3' -> 1, 'to 20' -> 0."""
    return max(len(p.split(".")[1]) if "." in p else 0 for p in text.split("–"))


def matches(check, value):
    """True / False at the doc's printed precision; None for loosely worded ('approx') claims."""
    if check.kind == "approx" or np.isnan(value):
        return None
    if check.kind == "range":
        v = float(f"{value:.{_decimals(check.doc)}f}")
        return bool(check.lo <= v <= check.hi)
    nd = _decimals(check.doc)
    return f"{value:.{nd}f}" == f"{float(check.doc):.{nd}f}"


def gate_table(specs, cases, checks=DOC_CHECKS):
    """Rows (dicts) for every doc check whose spec is in ``specs`` and whose runs are in ``cases``."""
    by_key = {s.key: s for s in specs}
    rows = []
    for c in checks:
        if c.key not in by_key:
            continue
        s = by_key[c.key]
        at_doc = cases.get((c.key, c.seed))
        at_plan = cases.get((c.key, PLAN_GATE_SEED))
        v_doc = metric_value(at_doc, c.metric) if at_doc is not None else np.nan
        v_plan = metric_value(at_plan, c.metric) if at_plan is not None else np.nan
        nd = _decimals(c.doc) + 2
        rows.append({"experiment": s.experiment, "scenario": s.key, "metric": c.metric,
                     "doc": c.doc, "line": c.line, "seed": c.seed,
                     "harness": v_doc, "match": matches(c, v_doc),
                     "harness_seed2": v_plan, "match_seed2": matches(c, v_plan), "nd": nd})
    return rows


def _yn(x):
    return {True: "yes", False: "NO", None: "n/a"}[x]


def format_gate(rows):
    """Markdown table of gate rows."""
    out = ["| exp | scenario / estimator / controller | metric | doc (line) | script seed | harness | match "
           f"| harness @ seed {PLAN_GATE_SEED} | match |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        nd = r["nd"]
        out.append(f"| {r['experiment']} | {r['scenario']} | {r['metric']} | {r['doc']} (l.{r['line']}) "
                   f"| {r['seed']} | {r['harness']:.{nd}f} | {_yn(r['match'])} "
                   f"| {r['harness_seed2']:.{nd}f} | {_yn(r['match_seed2'])} |")
    n = sum(r["match"] is True for r in rows)
    no = sum(r["match"] is False for r in rows)
    out.append(f"\n{len(rows)} doc numbers: {n} match at the script's seed, {no} do not, "
               f"{len(rows) - n - no} loosely worded (no yes/no). At seed {PLAN_GATE_SEED}: "
               f"{sum(r['match_seed2'] is True for r in rows)} match, "
               f"{sum(r['match_seed2'] is False for r in rows)} do not.")
    return "\n".join(out)
