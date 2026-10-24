#!/usr/bin/env python3
"""Results scoreboard: what the ledger says, against the pass/fail limits. Runs no simulation.

    python3 scripts/report.py            # print the scoreboard
    python3 scripts/report.py --write    # the same, and regenerate docs/RESULTS.md

Reads results/ledger.csv, results/crosscheck_basilisk.json and hardware.json, nothing else.
Every number printed is computed from those three files at run time. The limits
are scalar.control.rl_tune.THRESHOLDS, plus the knowledge limit read from rule
2 of docs/verification-loop-plan.md. The same text is printed at the end of
every pytest run (conftest.py at the repository root).

Where a ledger row goes (``build`` below; ``tests/test_report.py`` holds it to this):

    A1, A2   scenario ``s6_acq*`` / ``s6_fine*``: the tuned controllers (experiment 6)
    B        scenario ``<base>+<tag>``: a fidelity switch, next to the ``<base>`` row with it off
    C1, C2   the other closed-loop scenarios of experiments 4 and 5
    E        scenarios with a knowledge error and no controller
    F        any other closed-loop scenario (a name this file has not seen)
    G        scenario ``hw_acq*`` / ``hw_fine*``: the as-built rows (experiment hw)
    not shown   scenarios whose rows have no number in any ledger column
    D        is not from the ledger: the cross-check JSON
    H        is not from the ledger: every entry of hardware.json

A new ``+tag`` or a new scenario name needs no change here; ``_FAMILY_WORDS``,
``_SCENARIO_WORDS`` and the other word tables only supply nicer labels.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scalar import hardware  # noqa: E402
from scalar.control.rl_tune import THRESHOLDS  # noqa: E402
from scalar.harness.baseline import baseline_specs  # noqa: E402
from scalar.harness.ledger import METRIC_COLUMNS, parse_seed_set, read_ledger  # noqa: E402
from scalar.harness.metrics import pointing_stats  # noqa: E402

LEDGER = os.path.join(ROOT, "results", "ledger.csv")
CROSSCHECK = os.path.join(ROOT, "results", "crosscheck_basilisk.json")
HARDWARE = hardware.HARDWARE_JSON
PLAN = os.path.join(ROOT, "docs", "verification-loop-plan.md")
RESULTS_MD = os.path.join(ROOT, "docs", "RESULTS.md")
COMMAND = "python3 scripts/report.py"
WIDTH = 100

# Which ledger metric each THRESHOLDS entry is checked on, and in which of the two episodes of a
# tuned controller ("acq" = the s6_acq* scenario, "fine" = the s6_fine* scenario).
LIMIT_SOURCE = {
    "acq_settle_h":    ("acq", "settle_h"),
    "acq_diverged":    ("acq", "diverged"),
    "fine_point_mean": ("fine", "point_mean"),
    "fine_point_p95":  ("fine", "point_p95"),
    "fine_max_rate":   ("fine", "max_rate_deg_s"),
    "fine_sat_frac":   ("fine", "sat_frac"),
    "fine_power_w":    ("fine", "coil_power_w"),
    "fine_diverged":   ("fine", "diverged"),
}
KNOWLEDGE_LIMIT = "know_p95"        # the one limit that is in the plan and not in THRESHOLDS

# ---- plain words for ledger labels; anything not listed is shown as it is in the ledger ----
_SCENARIO_WORDS = [    # (pattern on the scenario name, words; {n} = the n-th number in the name)
    (r"^s6_acq(\d+)_[a-z]+mag_(.+)$", "tuned file {1}, acquisition, start {0} deg off target"),
    (r"^s6_fine(\d+)_[a-z]+mag_(.+)$", "tuned file {1}, fine pointing, start {0} deg off target"),
    (r"^e1_sunvec_alb([\d.]+)$", "sun direction from the sun sensors alone, albedo up to {0}"),
    (r"^e1_mekf_alb([\d.]+)$", "albedo {0}"),
    (r"^e2_alongtrack_([\d.]+)km$", "position off by {0} km"),
    (r"^e3_pulse([\d.]+)s_([\d.]+)Hz_n(\d+)$", "coil identification, {2} pulses of {0} s, sampled at {1} Hz"),
    (r"^e4A_acq(\d+)_truth$", "start {0} deg off target, one fixed gain"),
    (r"^e4B_near(\d+)_truth$", "start {0} deg off target"),
    (r"^e4C_acq(\d+)_truth$", "start {0} deg off target"),
    (r"^e4D_acq(\d+)_mekf$", "start {0} deg off target"),
    (r"^e5_fine(\d+)_alb([\d.]+)$", "start {0} deg off target, albedo {1}"),
]
_ESTIMATOR_WORDS = {"truth": "true attitude", "mekf+analytic": "MEKF, analytic sun",
                    "mekf+mlp_a": "MEKF, learned sun A", "mekf+mlp_b": "MEKF, learned sun B"}
_CONTROLLER_WORDS = {"none": "none", "projected_lqr": "fixed-gain LQR", "periodic_lqr": "periodic LQR",
                     "two_mode": "two-mode", "three_mode": "three-mode"}
_MAG_WORDS = {"rawmag": "raw", "calmag": "calibrated"}
_FAMILY_WORDS = {"mag1c": "magnetometer errors"}
_SETTING_SUFFIX = (("_uncal", "on"), ("_cal", "on, calibrated"))     # <family>_uncal / <family>_cal
_VARIANT_WORDS = {"all": "all", "not_assumed": "datasheet+measured"}  # baseline.AS_BUILT_VARIANTS
_DEVICE_WORDS = {"gyro": "gyro (LSM6DSO)", "magnetometer": "magnetometer (LIS2MDL)",
                 "sun_sensor": "sun sensor (VEML6031 array)", "board_temperature": "board temperature",
                 "coils": "magnetic coils", "body": "spacecraft body"}


# ------------------------------------------------------------------ the report as data
@dataclass
class Check:
    """One value held against one limit. ``stat`` is 'med' (typical seed), 'p95' (bad seed) or 'frac'."""
    limit_name: str
    stat: str
    value: float
    limit: float
    passed: bool


@dataclass
class Row:
    cells: list                     # one string per column, without the trailing trace column
    scenario: str                   # raw ledger scenario (or the JSON run key): the last column
    group: str = ""                 # sub-heading the row sits under
    checks: list = field(default_factory=list)
    reference: bool = False         # repeated here for comparison; its home is another table
    source: dict = None             # the ledger row


@dataclass
class Table:
    key: str
    title: str
    note: str                       # what the columns mean and what counts as a pass
    columns: list                   # header per cell; "\n" marks a line break in the terminal
    rows: list = field(default_factory=list)
    align: str = ""                 # 'l' / 'r' per column; default: first left, the rest right
    group_name: str = ""            # header of the group column in the markdown rendering
    trace_name: str = "ledger scenario"


@dataclass
class Report:
    header: list = field(default_factory=list)          # plain lines
    tables: list = field(default_factory=list)
    not_shown: list = field(default_factory=list)       # (scenario, words, n_rows)
    accounting: dict = field(default_factory=dict)      # how the ledger rows split up
    home: dict = field(default_factory=dict)            # scenario -> table key or "not shown"
    problems: list = field(default_factory=list)        # things the report could not read

    def table(self, key):
        return next(t for t in self.tables if t.key == key)


# ------------------------------------------------------------------ small helpers
def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def passes(value, limit):
    """rl_tune.criteria's rule: every limit is 'value <= limit'. A missing value does not pass."""
    return (not _nan(value)) and bool(value <= limit)


def _num(v, never="inf"):
    if _nan(v):
        return "-"
    if math.isinf(v):
        return never
    a = abs(v)
    return f"{v:.1f}" if a >= 100 else f"{v:.2f}" if a >= 1 else f"{v:.3f}" if a >= 0.01 or a == 0 else f"{v:.2g}"


def _deg(v):
    if _nan(v):
        return "-"
    return f"{v:.1f}" if abs(v) >= 100 else f"{v:.2f}"


def _g(v):
    return f"{v:g}"


def _verdict(text, check):
    return f"{text} {'PASS' if check.passed else 'FAIL'}"


def plan_knowledge_limit(path=PLAN):
    """The 'knowledge p95 <= N deg' limit of rule 2 of the plan, read from the plan; None if not found."""
    try:
        with open(path, encoding="utf-8") as f:
            m = re.search(r"knowledge p95\s*(?:≤|<=)\s*([0-9.]+)\s*°", f.read())
    except OSError:
        return None
    return float(m.group(1)) if m else None


def _definitions():
    """The angles in the definitions of 'lost pointing' and 'settled', from scalar/harness/metrics.py."""
    p = inspect.signature(pointing_stats).parameters
    return {k: p[k].default for k in ("diverge_deg", "settle_thresh_deg", "settle_hold")}


def scenario_words(name):
    base = name.split("+", 1)[0]
    for pattern, words in _SCENARIO_WORDS:
        m = re.search(pattern, base)
        if m:
            return words.format(*m.groups())
    return base


def _n_seeds(row):
    try:
        return len(parse_seed_set(row["seed_set"]))
    except ValueError:
        return 0


def _has(row, metric):
    return any(not _nan(row[c]) for c in METRIC_COLUMNS if c.startswith(metric))


def _is_closed_loop(row):
    return _has(row, "point_") or _has(row, "diverged")


def _is_empty(row):
    return all(_nan(row[c]) for c in METRIC_COLUMNS)


def _lost_cell(row, limit_name):
    """'11/40 FAIL': seeds whose pointing error ended above the divergence angle, of the seeds run."""
    frac, n = row["diverged__frac"], _n_seeds(row)
    if _nan(frac):
        return "-", None
    lost = frac * n
    text = f"{round(lost)}/{n}" if n and abs(lost - round(lost)) < 0.01 else f"{100 * frac:.0f}%"
    chk = Check(limit_name, "frac", frac, THRESHOLDS[limit_name], passes(frac, THRESHOLDS[limit_name]))
    return _verdict(text, chk), chk


def _lost_limit_name(scenario):
    return "acq_diverged" if "acq" in scenario.split("+", 1)[0] else "fine_diverged"


def _limit_cell(row, limit_name, stat, fmt=_num, never="inf"):
    metric = LIMIT_SOURCE[limit_name][1]
    v = row[f"{metric}__{stat}"]
    if _nan(v):
        return "-", None
    chk = Check(limit_name, stat, v, THRESHOLDS[limit_name], passes(v, THRESHOLDS[limit_name]))
    return _verdict(never if math.isinf(v) else fmt(v), chk), chk


def _know_cell(row, stat, know_limit):
    v = row[f"know_p95_all__{stat}"]
    if know_limit is None or _nan(v):
        return _deg(v), None
    chk = Check(KNOWLEDGE_LIMIT, stat, v, know_limit, passes(v, know_limit))
    return _verdict(_deg(v), chk), chk


def _checks(*checks):
    return [c for c in checks if c is not None]


def _controller_cells(rows):
    """Short controller label per row: the kind in words, then only the settings that differ between
    the rows of that kind (the full label is in the ledger)."""
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["controller"].split()[0], []).append(r["controller"].split()[1:])
    out = []
    for r in rows:
        kind, *tokens = r["controller"].split()
        others = by_kind[kind]
        common = set(tokens).intersection(*map(set, others)) if len(others) > 1 else set(tokens)
        out.append(" ".join([_CONTROLLER_WORDS.get(kind, kind)] + [t for t in tokens if t not in common]))
    return out


def _seeds_words(row):
    n = _n_seeds(row)
    return f"seeds {row['seed_set']} ({n} seed{'s' if n != 1 else ''})"


# ------------------------------------------------------------------ ledger rows -> tables
def _select(rows):
    """(rows to show, why each other row is left out: {id(row): 'superseded' | 'single_seed'}).

    Left out: an earlier run of the same (scenario, estimator, controller, seed set); a single-seed
    row when the same (scenario, estimator, controller) also has a many-seed row.
    """
    latest = {}
    for r in rows:
        latest[(r["scenario"], r["estimator"], r["controller"], r["seed_set"])] = r
    kept = list(latest.values())
    kept_ids = {id(r) for r in kept}
    left_out = {id(r): "superseded" for r in rows if id(r) not in kept_ids}
    many = {(r["scenario"], r["estimator"], r["controller"]) for r in kept if _n_seeds(r) > 1}
    shown = []
    for r in kept:
        if _n_seeds(r) > 1 or (r["scenario"], r["estimator"], r["controller"]) not in many:
            shown.append(r)
        else:
            left_out[id(r)] = "single_seed"
    return shown, left_out


def _tuned_tables(rows, defs):
    a1 = Table("A1", "Tuned controllers: acquisition (start far from the target, tumbling)",
               f"One row per tuned file (its name in results/, without .json) and seed set. Lost "
               f"pointing = seeds whose pointing error went above {_g(defs['diverge_deg'])} deg in the last orbit, of the seeds "
               f"run; PASS = {_g(THRESHOLDS['acq_diverged'])} lost. Settle time = hours until the "
               f"pointing error is under {_g(defs['settle_thresh_deg'])} deg and stays there; 'never' = "
               f"it did not; PASS = {_g(THRESHOLDS['acq_settle_h'])} h or less.",
               ["tuned file", "magnetometer", "lost\npointing", "settle h,\ntypical\nseed", "settle h,\nbad seed"],
               align="llrrr", group_name="seeds")
    a2 = Table("A2", "Tuned controllers: fine pointing (start near the target)",
               f"Same files and seeds; the second of the two runs made for each file. Mean error = pointing error in degrees, averaged "
               f"over the last orbit; PASS = {_g(THRESHOLDS['fine_point_mean'])} deg or less. Lost "
               f"pointing as in A1; PASS = {_g(THRESHOLDS['fine_diverged'])} lost.",
               ["tuned file", "magnetometer", "lost\npointing", "mean error,\ntypical\nseed",
                "mean error,\nbad seed"], align="llrrr", group_name="seeds")
    a2b = Table("A2b", "Tuned controllers: fine pointing, the other limits, on the bad seed",
                f"Same ledger rows as A2; a PASS on the bad seed means the typical seed passes too. "
                f"p95 error = the pointing error that 95 % of the last orbit stays under, PASS = "
                f"{_g(THRESHOLDS['fine_point_p95'])} deg or less. Top rate = fastest rotation in the "
                f"last orbit, PASS = {_g(THRESHOLDS['fine_max_rate'])} deg/s or less. Coil at limit = "
                f"share of the last orbit with any coil at full output, PASS = "
                f"{_g(100 * THRESHOLDS['fine_sat_frac'])} % or less. Coil power = average, PASS = "
                f"{_g(THRESHOLDS['fine_power_w'])} W or less.",
                ["tuned file", "p95 error,\ndeg", "top rate,\ndeg/s", "coil at\nlimit, %", "coil\npower, W"],
                align="lrrrr", group_name="seeds")
    left = []
    parsed = []
    for r in rows:
        m = re.match(r"^s6_(acq|fine)\d+_([a-z]+mag)_(.+)$", r["scenario"])
        if m:
            parsed.append((r, *m.groups()))
        else:
            left.append(r)
    seed_order = list(dict.fromkeys(r["seed_set"] for r, *_ in parsed))
    parsed.sort(key=lambda p: seed_order.index(p[0]["seed_set"]))            # stable: file order kept
    for r, episode, mag, stem in parsed:
        group, magw = _seeds_words(r), _MAG_WORDS.get(mag, mag)
        if episode == "acq":
            lost, c0 = _lost_cell(r, "acq_diverged")
            typ, c1 = _limit_cell(r, "acq_settle_h", "med", never="never")
            bad, c2 = _limit_cell(r, "acq_settle_h", "p95", never="never")
            a1.rows.append(Row([stem, magw, lost, typ, bad], r["scenario"], group, _checks(c0, c1, c2), source=r))
        else:
            lost, c0 = _lost_cell(r, "fine_diverged")
            typ, c1 = _limit_cell(r, "fine_point_mean", "med", _deg)
            bad, c2 = _limit_cell(r, "fine_point_mean", "p95", _deg)
            a2.rows.append(Row([stem, magw, lost, typ, bad], r["scenario"], group, _checks(c0, c1, c2), source=r))
            p95, c3 = _limit_cell(r, "fine_point_p95", "p95", _deg)
            rate, c4 = _limit_cell(r, "fine_max_rate", "p95")
            sat, c5 = _limit_cell(r, "fine_sat_frac", "p95", lambda v: _num(100 * v))
            pwr, c6 = _limit_cell(r, "fine_power_w", "p95")
            a2b.rows.append(Row([stem, p95, rate, sat, pwr], r["scenario"], group, _checks(c3, c4, c5, c6), source=r))
    return [t for t in (a1, a2, a2b) if t.rows], left


def _switch_parts(scenario):
    base, tag = scenario.split("+", 1)
    for suffix, setting in _SETTING_SUFFIX:
        if tag.endswith(suffix) and len(tag) > len(suffix):
            return base, tag[:-len(suffix)], setting
    return base, tag, "on"


def _switch_table(tagged, shown, know_limit):
    t = Table("B", "Fidelity switches: the same run with one model error switched off and on",
              "'off' is the plain run (shown again from its own table). Lost pointing as in A1. "
              "Pointing error = mean over the last orbit, in degrees, on the typical and the bad seed. "
              "Knowledge error = angle between estimated and true attitude, in degrees: its median "
              "on the typical seed, and its 95th percentile on the bad seed"
              + (f", PASS = {_g(know_limit)} deg or less." if know_limit is not None
                 else " (no limit found in the plan)."),
              ["switch", "lost\npointing", "pointing\nerror,\ntypical", "pointing\nerror,\nbad seed",
               "knowledge\nmedian,\ntypical", "knowledge\np95,\nbad seed"], align="lrrrrr",
              group_name="switch, run")
    if any(_switch_parts(r["scenario"])[2] != "on" for r in tagged):
        t.note += " 'on, calibrated' = switch on, with the calibration step run before the run."
    groups = {}
    for r in tagged:
        base, family, setting = _switch_parts(r["scenario"])
        groups.setdefault((family, base, r["estimator"], r["controller"], r["seed_set"]), []).append((setting, r))
    for (family, base, est, ctrl, seeds), members in groups.items():
        same = [r for r in shown if r["scenario"] == base and r["seed_set"] == seeds]
        exact = [r for r in same if r["estimator"] == est and r["controller"] == ctrl]
        off = exact[-1] if exact else same[0] if len(same) == 1 else None
        kind = ctrl.split()[0]
        run = scenario_words(base) + (f", {_CONTROLLER_WORDS.get(kind, kind)}" if kind != "none"
                                      else ", estimator only")
        n = _n_seeds(members[0][1])
        group = f"{_FAMILY_WORDS.get(family, family)}: {run}; seeds {seeds} ({n})"
        if off is None:
            t.rows.append(Row(["off: no such row", "-", "-", "-", "-", "-"], base, group, reference=True))
        for setting, r in ([("off", off)] if off is not None else []) + members:
            lost, c0 = _lost_cell(r, _lost_limit_name(base))
            know, c1 = _know_cell(r, "p95", know_limit)
            t.rows.append(Row([setting, lost, _deg(r["point_mean__med"]), _deg(r["point_mean__p95"]),
                               _deg(r["know_med_all__med"]), know], r["scenario"], group,
                              _checks(c0, c1), reference=r is off, source=r))
    return t


def _closed_loop_table(key, title, note, rows, with_estimator):
    cols = (["attitude from"] if with_estimator else []) + [
        "controller", "lost\npointing", "pointing\nerror,\ntypical", "pointing\nerror,\nbad seed"]
    t = Table(key, title, note, cols, align=("l" if with_estimator else "") + "lrrr", group_name="run")
    by_scenario = {}
    for r in rows:
        by_scenario.setdefault(r["scenario"], []).append(r)
    for scenario, members in by_scenario.items():
        for r, ctrl in zip(members, _controller_cells(members)):
            lost, c0 = _lost_cell(r, _lost_limit_name(scenario))
            cells = ([_ESTIMATOR_WORDS.get(r["estimator"], r["estimator"])] if with_estimator else []) + [
                ctrl, lost, _deg(r["point_mean__med"]), _deg(r["point_mean__p95"])]
            t.rows.append(Row(cells, scenario, f"{scenario_words(scenario)}; {_seeds_words(r)}",
                              _checks(c0), source=r))
    return t


def _closed_note(defs):
    acq, fine = THRESHOLDS["acq_diverged"], THRESHOLDS["fine_diverged"]
    allowed = _g(acq) if acq == fine else f"{_g(acq)} (start far from the target) or {_g(fine)} (start near it)"
    return (f"Lost pointing = seeds whose pointing error went above {_g(defs['diverge_deg'])} deg in the "
            f"last orbit, of the seeds run; PASS = {allowed} lost. Pointing error = mean over the last "
            f"orbit, in degrees, on the typical and the bad seed (no limit applied here). Controller = "
            f"its kind, then the settings that differ between the rows of one run.")


def _open_loop_table(rows, know_limit):
    t = Table("E", "Estimator alone, no controller: knowledge error",
              "Knowledge error = angle between estimated and true attitude, in degrees. Median = the "
              "middle value over one run, on the typical seed. p95 = the value 95 % of a run stays "
              "under, on the typical and the bad seed"
              + (f"; PASS = {_g(know_limit)} deg or less." if know_limit is not None
                 else " (no limit found in the plan).")
              + " 'Position off by N km' = the spacecraft's own idea of where it is along its orbit is "
              "wrong by that much.",
              ["what was run", "estimator", "median,\ntypical\nseed", "p95,\ntypical\nseed", "p95,\nbad seed"],
              align="llrrr")
    for r in rows:
        typ, c1 = _know_cell(r, "med", know_limit)
        bad, c2 = _know_cell(r, "p95", know_limit)
        t.rows.append(Row([scenario_words(r["scenario"]), _ESTIMATOR_WORDS.get(r["estimator"], r["estimator"]),
                           _deg(r["know_med_all__med"]), typ, bad], r["scenario"],
                          "" if _n_seeds(r) > 1 else _seeds_words(r), _checks(c1, c2), source=r))
    return t


def _as_built_table(rows, defs):
    """G: the hw_* rows, (rows not of the form hw_<acq|fine><n>_<variant> are returned for table F)."""
    t = Table("G", "As built: the tuned controller with the values of hardware.json",
              "The three-mode controller of results/lqr_tuned_3mode.json (calibrated magnetometer), with "
              "the entries of hardware.json applied, the VEML6031 array as the sun sensor, and the plant "
              "holding the coil dipole over each step with 2 substeps. Values applied: 'all' = every entry "
              "with a value; 'datasheet+measured' = only entries whose source is not assumed, the rest as "
              "in experiment 6. Each row is the file as it stood at the row's git hash: after editing "
              "hardware.json, run `python3 scripts/matrix.py --experiments hw --seeds 0-39`. Lost pointing "
              f"as in A1; settle time as in A1, PASS = {_g(THRESHOLDS['acq_settle_h'])} h or less "
              f"(acquisition); mean error as in A2, PASS = {_g(THRESHOLDS['fine_point_mean'])} deg or less "
              "(fine pointing). Inputs: table H.",
              ["values\napplied", "episode", "lost\npointing", "mean\nerror,\ntypical", "mean\nerror,\nbad seed",
               "settle h,\ntypical", "settle h,\nbad seed"], align="llrrrrr", group_name="seeds")
    left = []
    for r in rows:
        m = re.match(r"^hw_(acq|fine)\d+_(.+)$", r["scenario"])
        if not m:
            left.append(r)
            continue
        episode, variant = m.groups()
        lost, c0 = _lost_cell(r, "acq_diverged" if episode == "acq" else "fine_diverged")
        if episode == "acq":
            mean_t, mean_b, c1, c2 = _deg(r["point_mean__med"]), _deg(r["point_mean__p95"]), None, None
            set_t, c3 = _limit_cell(r, "acq_settle_h", "med", never="never")
            set_b, c4 = _limit_cell(r, "acq_settle_h", "p95", never="never")
        else:
            mean_t, c1 = _limit_cell(r, "fine_point_mean", "med", _deg)
            mean_b, c2 = _limit_cell(r, "fine_point_mean", "p95", _deg)
            set_t, set_b, c3, c4 = _num(r["settle_h__med"], "never"), _num(r["settle_h__p95"], "never"), None, None
        t.rows.append(Row([_VARIANT_WORDS.get(variant, variant), "acquisition" if episode == "acq" else "fine pointing",
                           lost, mean_t, mean_b, set_t, set_b], r["scenario"], _seeds_words(r),
                          _checks(c0, c1, c2, c3, c4), source=r))
    return t, left


def _hw_value(v):
    if v is None:
        return "null (not applied)"
    if isinstance(v, str):
        return v
    if isinstance(v, tuple):
        return "[" + ",".join(_hw_value(x) for x in v) + "]"
    return f"{v:.4g}"


def _hardware_table(path):
    hw = hardware.load(path)
    counts = Counter("null" if e.value is None else e.source for e in hw.entries.values())
    t = Table("H", "Hardware inputs: hardware.json as it stands",
              "Every entry of hardware.json. Source: assumed = nobody measured it or read it off a datasheet; "
              "datasheet; measured = a bench result. Null = unknown, not applied: the code's own value "
              "stands. Read by the as-built rows (table G) and by scripts/gyro_realism.py, mag_realism.py "
              "and sun_realism.py (in table B: the gyro1h and mag1c rows, and the cube_tilt layout); the other tables never "
              "read it. Where each number comes from and which bench test replaces it: the entry's note "
              f"in the file. Now: {len(hw.entries)} entries, " + ", ".join(
                  f"{counts[k]} {k}" for k in (*hardware.SOURCES, "null")) + ".",
              ["value", "units", "source"], align="lll", group_name="device", trace_name="hardware.json entry")
    for e in hw.entries.values():
        t.rows.append(Row([_hw_value(e.value), e.units, e.source], e.key, _DEVICE_WORDS.get(e.device, e.device)))
    return t


def _gate_rows(node, path=()):
    """(run key, rows) for every dict under ``node`` that holds 'gate_rows_as_is'."""
    if isinstance(node, dict):
        if isinstance(node.get("gate_rows_as_is"), list):
            yield "/".join(path), node["gate_rows_as_is"]
        else:
            for k, v in node.items():
                yield from _gate_rows(v, path + (str(k),))


def _crosscheck_table(path):
    with open(path, encoding="utf-8") as f:
        runs = json.load(f).get("runs", {})
    t = Table("D", "Independent cross-check: this simulator against Basilisk, same set-up, as is",
              "One row per compared quantity and run. Largest difference = the worst sample of the run; "
              "PASS = within the limit, as recorded in results/crosscheck_basilisk.json. Causes of "
              "each FAIL: docs/crosscheck-basilisk.md.",
              ["compared quantity", "largest\ndifference", "limit", "result"], align="lrrl",
              group_name="run", trace_name="run in the JSON")
    for key, gate in _gate_rows(runs):
        parts = key.split("/")
        group = f"{parts[-1].replace('_', ' ')} ({', '.join(parts[:-1])} set-up)" if len(parts) > 1 else key
        for g in gate:
            chk = Check("crosscheck", "max", g["max"], g["threshold"], bool(g["passed"]))
            t.rows.append(Row([g["quantity"], f"{g['max']:.3g} {g['unit']}", f"{g['threshold']:.3g} {g['unit']}",
                               "PASS" if g["passed"] else "FAIL"], key, group, [chk]))
    return t


def build(ledger_path=LEDGER, crosscheck_path=CROSSCHECK, plan_path=PLAN, hardware_path=HARDWARE):
    """The whole report as data. Never raises for a missing file; says so in ``problems``."""
    rep = Report()
    know_limit = plan_knowledge_limit(plan_path)
    defs = _definitions()
    rel = lambda p: os.path.relpath(p, ROOT)  # noqa: E731
    try:
        rows = read_ledger(ledger_path)
    except (OSError, ValueError, KeyError) as e:
        rows = None
        rep.problems.append(f"{rel(ledger_path)} could not be read ({type(e).__name__}: {e}); no ledger tables.")

    rep.header.append(f"Generated by `{COMMAND}` (add --write to regenerate docs/RESULTS.md). No simulation is run.")
    if rows is not None:
        runs = Counter((r["git_hash"], r["date"]) for r in rows)
        rep.header.append(f"Source: {rel(ledger_path)}, {len(rows)} rows: "
                          + "; ".join(f"{n} from git {h}, {d}" for (h, d), n in runs.items()) + ".")
    rep.header.append(
        "Limits: THRESHOLDS in scalar/control/rl_tune.py"
        + (f", and knowledge p95 of {_g(know_limit)} deg or less from rule 2 of docs/verification-loop-plan.md."
           if know_limit is not None else "; the knowledge limit was not found in docs/verification-loop-plan.md.")
        + " The plan calls the pointing and coil-power limits placeholders.")
    rep.header.append("Typical seed = the median across the seeds of a row; bad seed = the 95th percentile "
                      "across them. A seed is one repeat of a run with different random noise.")
    rep.header.append("What these rows are taken to mean: docs/verification-loop-plan.md, section Status.")

    if rows is not None:
        shown, left_out = _select(rows)
        live = [r for r in shown if not _is_empty(r)]
        tagged = [r for r in live if "+" in r["scenario"]]
        plain = [r for r in live if "+" not in r["scenario"]]
        tuned_tables, unparsed = _tuned_tables([r for r in plain if r["scenario"].startswith("s6_")], defs)
        as_built, unparsed_hw = _as_built_table([r for r in plain if r["scenario"].startswith("hw_")], defs)
        plain = [r for r in plain if not r["scenario"].startswith(("s6_", "hw_"))] + unparsed + unparsed_hw
        experiment = {s.name: s.experiment for s in baseline_specs()}
        closed = [r for r in plain if _is_closed_loop(r)]
        open_loop = [r for r in plain if not _is_closed_loop(r)]
        september = [r for r in closed if experiment.get(r["scenario"]) in ("4", "5")]
        other = [r for r in closed if experiment.get(r["scenario"]) not in ("4", "5")]
        with_est = {r["scenario"] for r in september if r["estimator"] != "truth"}
        c1 = [r for r in september if r["scenario"] in with_est]
        c2 = [r for r in september if r["scenario"] not in with_est]
        note = _closed_note(defs)

        rep.tables += tuned_tables
        if tagged:
            rep.tables.append(_switch_table(tagged, live, know_limit))
        if c1:
            rep.tables.append(_closed_loop_table(
                "C1", "September hand-picked designs, estimator in the loop (experiments 4 and 5)",
                note + " Attitude from = what the controller was fed: the MEKF estimate, with the sun "
                "direction worked out by formula (analytic) or by the trained network (learned).", c1, True))
        if c2:
            same = all(r["point_mean__med"] == r["point_mean__p95"] for r in c2)
            rep.tables.append(_closed_loop_table(
                "C2", "September hand-picked designs, true attitude fed to the controller (experiment 4)",
                note + " No estimator in these runs."
                + (" The seed changes nothing here: typical and bad seed are equal in every row." if same else ""),
                c2, False))
    _add_crosscheck(rep, crosscheck_path)
    if rows is not None:
        if open_loop:
            rep.tables.append(_open_loop_table(open_loop, know_limit))
        if other:
            rep.tables.append(_closed_loop_table("F", "Other closed-loop rows", note, other, True))
        if as_built.rows:
            rep.tables.append(as_built)
    _add_hardware(rep, hardware_path)
    if rows is not None:
        for t in rep.tables:
            for row in t.rows:
                if row.source is not None and not row.reference:
                    rep.home.setdefault(row.scenario, t.key if t.key != "A2b" else "A2")
        in_tables = {id(row.source) for t in rep.tables for row in t.rows if row.source is not None}
        count = Counter()
        hidden = Counter()
        for r in rows:
            if r["scenario"] not in rep.home:
                hidden[r["scenario"]] += 1
                count["not_shown"] += 1
            elif id(r) in in_tables:
                count["in_tables"] += 1
            else:
                count[left_out.get(id(r), "no_number")] += 1
        for scenario, n in hidden.items():
            rep.home[scenario] = "not shown"
            rep.not_shown.append((scenario, scenario_words(scenario), n))
        rep.accounting = {"rows": len(rows), **{k: count[k] for k in
                                               ("in_tables", "not_shown", "single_seed", "superseded", "no_number")}}
    return rep


def _add_hardware(rep, path):
    try:
        rep.tables.append(_hardware_table(path))
    except (OSError, ValueError) as e:              # HardwareFileError is a ValueError
        rep.problems.append(f"{os.path.relpath(path, ROOT)} could not be read ({type(e).__name__}: {e}); "
                            f"no hardware table.")


def _add_crosscheck(rep, path):
    try:
        rep.tables.append(_crosscheck_table(path))
    except (OSError, ValueError, KeyError, TypeError) as e:
        rep.problems.append(f"{os.path.relpath(path, ROOT)} could not be read ({type(e).__name__}: {e}); "
                            f"no cross-check table.")


# ------------------------------------------------------------------ rendering
def _align(table, n):
    a = table.align or ("l" + "r" * (n - 1))
    return a + "l"                                   # the trace column


def _wrap(text, width=WIDTH, indent=""):
    return textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent,
                         break_on_hyphens=False)


def _not_shown_lines(rep):
    a = rep.accounting
    head = []
    if rep.not_shown:
        head.append("No number in any ledger column: the result of these runs is not one of the ledger's "
                    "fixed metrics. `python3 scripts/matrix.py --per-seed PATH` re-runs them and writes "
                    "it.")
    tail = ""
    if a:
        tail = (f"Of the {a['rows']} ledger rows: {a['in_tables']} are in the tables above; "
                f"{a['not_shown']} belong to the scenarios listed here; {a['single_seed']} are "
                f"single-seed rows of a run that also has a many-seed row; {a['superseded']} were "
                f"replaced by a later run of the same scenario and seeds"
                + (f"; {a['no_number']} more have no number" if a["no_number"] else "") + ".")
    return head, tail


def render_terminal(rep, width=WIDTH):
    """The scoreboard as plain text, no line longer than ``width``, no colour."""
    out = ["SCALAR results scoreboard", "=" * 25]
    for line in rep.header:
        out += _wrap(line.replace("`", ""), width)
    for p in rep.problems:
        out += _wrap("NOTE: " + p, width)
    for t in rep.tables:
        out += ["", f"{t.key}. {t.title}"]
        out += _wrap(t.note, width, "  ")
        heads = [c.split("\n") for c in t.columns] + [[t.trace_name]]
        depth = max(len(h) for h in heads)
        heads = [[""] * (depth - len(h)) + h for h in heads]
        cells = [row.cells + [row.scenario] for row in t.rows]
        widths = [max([len(x) for x in h] + [len(r[i]) for r in cells]) for i, h in enumerate(heads)]
        align = _align(t, len(t.columns))
        split = sum(widths) + 2 * (len(widths) - 1) > width      # trace column on its own line
        use = len(widths) - 1 if split else len(widths)

        def line(values):
            parts = [(v.ljust(widths[i]) if align[i] == "l" else v.rjust(widths[i])) for i, v in enumerate(values[:use])]
            return "  ".join(parts).rstrip()

        out.append("")
        for d in range(depth):
            out.append(line([h[d] for h in heads]))
        out.append("-" * min(width, sum(widths[:use]) + 2 * (use - 1)))
        group = None
        for row, c in zip(t.rows, cells):
            if row.group and row.group != group:
                group = row.group
                out += _wrap(f"[{group}]", width)
            out.append(line(c))
            if split:
                out += _wrap(f"{t.trace_name}: {row.scenario}", width, "    ")
    head, tail = _not_shown_lines(rep)
    if head or tail:
        out += ["", "Not shown"]
        out += [ln for h in head for ln in _wrap(h + " In brackets: how many ledger rows each has.", width, "  ")]
        if rep.not_shown:
            out += _wrap(", ".join(f"{s} ({n})" for s, _, n in rep.not_shown), width, "    ")
        if tail:
            out += _wrap(tail, width, "  ")
    return "\n".join(out) + "\n"


def _md(text):
    return text.replace("|", "\\|")


def _md_links(text):
    """docs/x.md -> a link relative to docs/ (RESULTS.md lives there); the Status pointer gets its anchor."""
    text = re.sub(r"docs/([\w.-]+\.md)(, section Status)?",
                  lambda m: f"[{m.group(0)}]({m.group(1)}{'#status' if m.group(2) else ''})", text)
    return re.sub(r"(?<![\w/(\[])((?:results|scalar)/[\w./-]+\.(?:json|py|csv))", r"[\1](../\1)", text)


def render_markdown(rep):
    """docs/RESULTS.md. Same tables as the terminal text."""
    out = ["# Results scoreboard", "",
           f"**Generated file. Do not edit it by hand:** run `{COMMAND} --write`.", ""]
    for line in rep.header:
        out.append(f"- {_md_links(line)}")
    for p in rep.problems:
        out.append(f"- NOTE: {p}")
    for t in rep.tables:
        out += ["", f"## {t.key}. {t.title}", "", _md_links(t.note), ""]
        grouped = any(row.group for row in t.rows)
        cols = ([t.group_name or "group"] if grouped else []) + [c.replace("\n", " ") for c in t.columns] + [t.trace_name]
        align = ("l" if grouped else "") + _align(t, len(t.columns))
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join("---" if a == "l" else "---:" for a in align) + "|")
        for row in t.rows:
            cells = ([row.group] if grouped else []) + row.cells + [f"`{row.scenario}`"]
            out.append("| " + " | ".join(_md(c) for c in cells) + " |")
    head, tail = _not_shown_lines(rep)
    if head or tail:
        out += ["", "## Not shown", ""]
        out += head
        if rep.not_shown:
            out += ["", "| ledger scenario | what it is | ledger rows |", "|---|---|---:|"]
            out += [f"| `{s}` | {_md(w)} | {n} |" for s, w, n in rep.not_shown]
        if tail:
            out += ["", tail]
    return "\n".join(out) + "\n"


def terminal_text():
    return render_terminal(build())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print the results scoreboard from results/ledger.csv; runs no simulation.")
    ap.add_argument("--write", action="store_true", help="also regenerate docs/RESULTS.md")
    args = ap.parse_args(argv)
    rep = build()
    sys.stdout.write(render_terminal(rep))
    if args.write:
        with open(RESULTS_MD, "w", encoding="utf-8") as f:
            f.write(render_markdown(rep))
        print(f"\nwrote {os.path.relpath(RESULTS_MD, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
