"""Tests: the results scoreboard (scripts/report.py) is built from the ledger and nothing else.

    - it builds from the committed results/ledger.csv and results/crosscheck_basilisk.json
    - every ledger scenario has exactly one home: one table, or the "not shown" list
    - every PASS / FAIL agrees with scalar.control.rl_tune.THRESHOLDS (and, for knowledge, with
      the limit in rule 2 of docs/verification-loop-plan.md), on the value in the ledger
    - a scenario name the report has never seen still shows up
    - docs/RESULTS.md is what the command writes today
    - the pytest hook in conftest.py prints it and cannot fail a run

No result number is typed in here: everything is compared with the ledger file read by ``csv``.
"""
import csv
import importlib.util
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scalar.control.rl_tune import THRESHOLDS, criteria  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                     # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


report = _load("scalar_report_under_test", os.path.join(ROOT, "scripts", "report.py"))


@pytest.fixture(scope="module")
def rep():
    return report.build()


@pytest.fixture(scope="module")
def ledger():
    with open(report.LEDGER, newline="") as f:
        return list(csv.DictReader(f))


def _cell(row, column):
    return float(row[column]) if row[column] != "" else float("nan")


def _home_tables(rep, scenario):
    """Tables (A2b counted as A2, its continuation) holding ``scenario`` as a row of its own."""
    return {("A2" if t.key == "A2b" else t.key) for t in rep.tables for r in t.rows
            if r.source is not None and not r.reference and r.scenario == scenario}


def test_builds_from_the_committed_ledger(rep, ledger):
    assert rep.problems == []
    assert [t.key for t in rep.tables] == ["A1", "A2", "A2b", "B", "C1", "C2", "D", "E"]
    assert all(t.rows and t.note for t in rep.tables)
    a = rep.accounting
    assert a["rows"] == len(ledger)
    assert a["in_tables"] + a["not_shown"] + a["single_seed"] + a["superseded"] + a["no_number"] == len(ledger)
    hashes = {(r["git_hash"], r["date"]) for r in ledger}
    source = next(line for line in rep.header if line.startswith("Source:"))
    assert all(h in source and d in source for h, d in hashes)
    assert any("placeholder" in line for line in rep.header)


def test_every_scenario_has_exactly_one_home(rep, ledger):
    scenarios = {r["scenario"] for r in ledger}
    assert set(rep.home) == scenarios
    listed = {s for s, _, _ in rep.not_shown}
    for s in scenarios:
        tables = _home_tables(rep, s)
        if rep.home[s] == "not shown":
            assert tables == set() and s in listed, s
        else:
            assert tables == {rep.home[s]} and s not in listed, s
    text, md = report.render_terminal(rep), report.render_markdown(rep)
    for s in scenarios:
        assert s in text and f"`{s}`" in md, s


def test_rows_repeated_for_comparison_are_only_the_switch_off_rows(rep):
    for t in rep.tables:
        for r in t.rows:
            if r.reference:
                assert t.key == "B" and "+" not in r.scenario
                assert rep.home[r.scenario] != "B"


def test_pass_fail_agrees_with_thresholds(rep):
    assert set(report.LIMIT_SOURCE) == set(THRESHOLDS)
    seen = set()
    for t in rep.tables:
        for r in t.rows:
            for c in r.checks:
                if c.limit_name not in THRESHOLDS:
                    continue
                seen.add(c.limit_name)
                metric = report.LIMIT_SOURCE[c.limit_name][1]
                column = "diverged__frac" if metric == "diverged" else f"{metric}__{c.stat}"
                value = r.source[column]
                assert c.value == value or (c.value != c.value and value != value)
                assert c.limit == THRESHOLDS[c.limit_name]
                assert c.passed == bool(value <= THRESHOLDS[c.limit_name]), (r.scenario, c)
    assert seen == set(THRESHOLDS)          # every limit is held against at least one row


def test_tuned_controller_verdicts_equal_rl_tune_criteria(rep, ledger):
    """The same verdicts from the raw CSV through rl_tune.criteria, per tuned file, seed set and statistic."""
    by_key = {(r["scenario"], r["seed_set"]): r for r in ledger}
    fine_rows = {(r.scenario, r.source["seed_set"]): [] for r in rep.table("A2").rows}
    for key in ("A2", "A2b"):
        for r in rep.table(key).rows:
            fine_rows[(r.scenario, r.source["seed_set"])] += r.checks
    assert rep.table("A1").rows and len(rep.table("A1").rows) == len(fine_rows)
    for acq in rep.table("A1").rows:
        seeds = acq.source["seed_set"]
        suffix = re.sub(r"^s6_acq\d+", "", acq.scenario)                  # _<rawmag|calmag>_<file>
        (fine_name,) = {name for name, s in fine_rows if s == seeds and re.sub(r"^s6_fine\d+", "", name) == suffix}
        raw = {"acq": by_key[(acq.scenario, seeds)], "fine": by_key[(fine_name, seeds)]}
        for stat in ("med", "p95"):
            metrics = {}
            for name, (episode, metric) in report.LIMIT_SOURCE.items():
                column = "diverged__frac" if metric == "diverged" else f"{metric}__{stat}"
                metrics[name] = _cell(raw[episode], column)
            expected = {name: passed for name, _, _, passed in criteria(metrics)}
            for c in acq.checks + fine_rows[(fine_name, seeds)]:
                if c.stat in (stat, "frac"):
                    assert c.passed == expected[c.limit_name], (acq.scenario, seeds, stat, c)


def test_knowledge_limit_is_read_from_the_plan(rep):
    limit = report.plan_knowledge_limit()
    assert limit is not None
    checks = [c for t in rep.tables for r in t.rows for c in r.checks if c.limit_name == report.KNOWLEDGE_LIMIT]
    assert checks
    for c in checks:
        assert c.limit == limit and c.passed == bool(c.value <= limit)
    assert report.plan_knowledge_limit(os.path.join(ROOT, "no-such-plan.md")) is None


def test_the_words_pass_and_fail_follow_the_checks(rep):
    for t in rep.tables:
        for r in t.rows:
            words = [c.split()[-1] for c in r.cells if c.split() and c.split()[-1] in ("PASS", "FAIL")]
            assert words == ["PASS" if c.passed else "FAIL" for c in r.checks], (t.key, r.cells)


def test_crosscheck_table_is_the_json_as_is(rep):
    with open(report.CROSSCHECK) as f:
        runs = json.load(f)["runs"]
    expected = [(key, g["quantity"], bool(g["passed"])) for key, run in runs.items() for g in run["gate_rows_as_is"]]
    got = [(r.scenario, r.cells[0], r.cells[-1] == "PASS") for r in rep.table("D").rows]
    assert got == expected and expected


def test_terminal_text_fits_100_columns_without_colour(rep):
    text = report.render_terminal(rep)
    assert max(len(line) for line in text.splitlines()) <= report.WIDTH == 100
    assert "\x1b" not in text and "\t" not in text
    order = [text.index(f"\n{t.key}. ") for t in rep.tables]
    assert order == sorted(order)


def _ledger_with(tmp_path, ledger, new_rows):
    path = tmp_path / "ledger.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ledger[0]))
        w.writeheader()
        w.writerows(ledger + new_rows)
    return str(path)


def test_a_new_switch_tag_and_a_new_scenario_show_up_without_a_code_change(tmp_path, ledger):
    base = next(r for r in ledger if r["scenario"] == "e4D_acq120_mekf" and r["seed_set"] == "0-39")
    tuned = next(r for r in ledger if r["scenario"].startswith("s6_fine") and r["seed_set"] == "0-39")
    new = [dict(base, scenario="e4D_acq120_mekf+zz9_neverseen"),
           dict(tuned, scenario=tuned["scenario"] + "+zz9_neverseen_with_a_long_name"),
           dict(base, scenario="zz_brand_new_closed_loop"),
           dict(base, seed_set="0-7", diverged__frac="0.125")]          # same run, another seed set
    rep2 = report.build(ledger_path=_ledger_with(tmp_path, ledger, new))
    assert rep2.home["e4D_acq120_mekf+zz9_neverseen"] == "B"
    assert rep2.home[tuned["scenario"] + "+zz9_neverseen_with_a_long_name"] == "B"
    assert rep2.home["zz_brand_new_closed_loop"] == "F"
    assert set(rep2.home) == {r["scenario"] for r in ledger + new}
    b = rep2.table("B")
    group = next(r.group for r in b.rows if r.scenario == "e4D_acq120_mekf+zz9_neverseen")
    members = [r for r in b.rows if r.group == group]
    assert [r.reference for r in members] == [True, False] and members[0].scenario == "e4D_acq120_mekf"
    text = report.render_terminal(rep2)
    assert max(len(line) for line in text.splitlines()) <= report.WIDTH
    assert "1/8 FAIL" in text
    for r in new:
        assert r["scenario"] in text


def test_a_rerun_replaces_the_earlier_row(tmp_path, ledger):
    base = next(r for r in ledger if r["scenario"] == "e4D_acq120_mekf" and r["seed_set"] == "0-39")
    rerun = dict(base, git_hash="feedbee", diverged__frac="0")
    rep2 = report.build(ledger_path=_ledger_with(tmp_path, ledger, [rerun]))
    (row,) = [r for r in rep2.table("C1").rows if r.scenario == "e4D_acq120_mekf"]
    assert row.source["git_hash"] == "feedbee" and row.cells[2] == "0/40 PASS"
    assert rep2.accounting["superseded"] == 1


def test_missing_files_do_not_raise(tmp_path):
    rep2 = report.build(ledger_path=str(tmp_path / "none.csv"), crosscheck_path=str(tmp_path / "none.json"))
    assert len(rep2.problems) == 2 and rep2.tables == []
    assert "could not be read" in report.render_terminal(rep2)
    assert "could not be read" in report.render_markdown(rep2)


def test_results_md_is_what_the_command_writes(rep):
    with open(report.RESULTS_MD, encoding="utf-8") as f:
        committed = f.read()
    assert committed == report.render_markdown(rep), \
        "docs/RESULTS.md is stale: run `python3 scripts/report.py --write` and commit the result"


class _Reporter:
    def __init__(self):
        self.lines = []

    def write_line(self, text, **kw):
        self.lines.append(text)

    def write_sep(self, sep, title=None, **kw):
        self.lines.append(f"{sep * 3} {title}")


class _Config:
    def __init__(self, **options):
        self.options = options

    def getoption(self, name, default=None):
        return self.options.get(name, default)


def test_pytest_hook_prints_the_scoreboard_and_never_raises(rep, monkeypatch):
    conftest = _load("scalar_conftest_under_test", os.path.join(ROOT, "conftest.py"))
    out = _Reporter()
    conftest.pytest_terminal_summary(out, 0, _Config())
    assert "\n".join(out.lines[1:]) + "\n" == report.render_terminal(rep)

    off = _Reporter()
    conftest.pytest_terminal_summary(off, 0, _Config(**{"--no-scoreboard": True}))
    assert off.lines == []

    monkeypatch.setattr(conftest, "_REPORT", os.path.join(ROOT, "scripts", "no_such_report.py"))
    broken = _Reporter()
    conftest.pytest_terminal_summary(broken, 0, _Config())
    assert len(broken.lines) == 1 and "not shown" in broken.lines[0]
