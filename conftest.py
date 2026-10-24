"""pytest configuration: print the headline of the results scoreboard at the end of every test run.

The headline is the top of ``python3 scripts/report.py`` (``report.headline_text``), followed by
one line saying where the full tables are; it reads results/ledger.csv and hardware.json and
runs no simulation. It never changes the outcome of the test run; if it cannot be built, one
line says why. ``--no-scoreboard`` leaves it out.
"""
import importlib.util
import os
import sys

_REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "report.py")


def pytest_addoption(parser):
    parser.addoption("--no-scoreboard", action="store_true", default=False,
                     help="do not print the results scoreboard (scripts/report.py) after the tests")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if config.getoption("--no-scoreboard") or config.getoption("collectonly", False):
        return
    try:
        spec = importlib.util.spec_from_file_location("_scalar_report", _REPORT)
        report = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = report            # dataclasses look the module up by name
        spec.loader.exec_module(report)
        text = report.headline_text()
    except Exception as e:  # noqa: BLE001  (the scoreboard must never fail a test run)
        terminalreporter.write_line(f"results scoreboard not shown ({type(e).__name__}: {e})")
        return
    terminalreporter.write_sep("=", "results headline (python3 scripts/report.py)")
    terminalreporter.write_line(text.rstrip("\n"))
