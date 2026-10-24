"""results/ledger.csv: one row per (scenario, estimator, controller, seed-set, git hash, date).

Columns are fixed (rule 1 of docs/verification-loop-plan.md): the six identity
columns, then for every metric of the fixed set its median and its 95th
percentile ACROSS the seed set (rule 4), named ``<metric>__med`` and
``<metric>__p95``. The divergence flag is reported as the fraction of seeds
that diverged, ``diverged__frac``. A single-seed row has med == p95 == the value.

Empty cell = the metric does not apply to that scenario. ``inf`` in a settle
column = at least that share of the seeds never settled.
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
import subprocess

import numpy as np

from .metrics import FIXED_METRICS

__all__ = ["IDENTITY_COLUMNS", "METRIC_COLUMNS", "LEDGER_COLUMNS", "format_seed_set",
           "parse_seed_set", "current_git_hash", "aggregate", "ledger_row", "append_rows",
           "read_ledger"]

IDENTITY_COLUMNS = ("scenario", "estimator", "controller", "seed_set", "git_hash", "date")
_PER_SEED = tuple(m for m in FIXED_METRICS if m != "diverged")
METRIC_COLUMNS = tuple(f"{m}__{s}" for m in _PER_SEED for s in ("med", "p95")) + ("diverged__frac",)
LEDGER_COLUMNS = IDENTITY_COLUMNS + METRIC_COLUMNS

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BIG = 1e300


def format_seed_set(seeds):
    """[0..39] -> '0-39'; [2] -> '2'; anything else -> '100,102,105'."""
    s = [int(x) for x in seeds]
    if not s:
        raise ValueError("empty seed set")
    if len(s) == 1:
        return str(s[0])
    if s == list(range(s[0], s[0] + len(s))):
        return f"{s[0]}-{s[-1]}"
    return ",".join(map(str, s))


def parse_seed_set(text):
    """Inverse of format_seed_set; also accepts mixed forms such as '0-3,7,100-102'."""
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def current_git_hash(repo=_REPO):
    """Short HEAD hash, with '+dirty' when the work tree differs from it. Read-only git calls."""
    try:
        h = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "--no-optional-locks", "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001  (no git, not a repo)
        return "unknown"


def _across_seeds(values):
    """(median, p95) across seeds; NaN seeds are skipped, +inf is kept and propagates."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan
    v = np.where(np.isposinf(v), _BIG, v)       # np.percentile interpolates inf - inf into NaN
    out = []
    for x in (np.median(v), np.percentile(v, 95)):
        out.append(np.inf if x > 1e200 else float(x))
    return out[0], out[1]


def aggregate(metrics_list):
    """{ledger metric column: value} from the per-seed Metrics of one row."""
    out = {}
    for m in _PER_SEED:
        out[f"{m}__med"], out[f"{m}__p95"] = _across_seeds([getattr(x, m) for x in metrics_list])
    d = np.asarray([x.diverged for x in metrics_list], float)
    d = d[~np.isnan(d)]
    out["diverged__frac"] = float(np.mean(d)) if d.size else np.nan
    return out


def ledger_row(spec, seeds, metrics_list, git_hash=None, date=None):
    """One ledger row (dict keyed by LEDGER_COLUMNS) for ``spec`` over ``seeds``."""
    seeds = list(seeds)
    if len(seeds) != len(metrics_list):
        raise ValueError("one Metrics per seed")
    row = {"scenario": spec.name, "estimator": spec.estimator_label,
           "controller": spec.controller_label, "seed_set": format_seed_set(seeds),
           "git_hash": current_git_hash() if git_hash is None else git_hash,
           "date": _dt.date.today().isoformat() if date is None else date}
    row.update(aggregate(metrics_list))
    return row


def _cell(v):
    if isinstance(v, str):
        return v
    if v is None or np.isnan(v):
        return ""
    if np.isinf(v):
        return "inf"
    return f"{v:.6g}"


def append_rows(path, rows):
    """Append rows to the ledger, creating it with the fixed header if it does not exist."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        if tuple(header) != LEDGER_COLUMNS:
            raise ValueError(f"{path}: header does not match the fixed ledger columns")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LEDGER_COLUMNS)
        for r in rows:
            w.writerow([_cell(r[c]) for c in LEDGER_COLUMNS])


def read_ledger(path):
    """List of row dicts; metric cells as float (NaN for empty)."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for c in METRIC_COLUMNS:
                r[c] = float(r[c]) if r[c] != "" else np.nan
            out.append(r)
    return out
