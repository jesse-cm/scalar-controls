"""``--legacy`` on scripts/lqr_tune.py and scripts/closed_loop_nn.py reaches every closed-loop run.

These two scripts produced experiments 4 and 5 of docs/learned-estimation-and-control.md.
Since 97ebd9e the closed loop defaults to the corrected plant step and error timing, so
the scripts print the notes' numbers only with ``--legacy``, which passes
``tune.LEGACY_LOOP`` to each run. No orbit is simulated here: the simulator, the gain
schedule and the network training are replaced by stubs that record their arguments.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from scalar.control import tune  # noqa: E402
from scalar.control.tune import LEGACY_LOOP, ClosedLoopResult  # noqa: E402

# script -> closed-loop runs it makes: lqr_tune.py [A] 4 + [B] 9 + [C] 6 + [D] 1; closed_loop_nn.py 2 x 2 x 3
SCRIPTS = {"lqr_tune.py": 20, "closed_loop_nn.py": 12}


def _load(script):
    spec = importlib.util.spec_from_file_location("_script_" + script[:-3], os.path.join(_ROOT, "scripts", script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(monkeypatch, script, argv):
    """Run the script's main(argv) with stubs; return the keyword arguments of every closed-loop call."""
    mod = _load(script)
    calls = []

    def fake_sim(*args, **kw):
        calls.append(kw)
        n = 10
        return ClosedLoopResult(np.arange(n, dtype=float), np.ones(n), np.ones(n), np.ones(n), np.zeros((n, 3)), 0.0)

    monkeypatch.setattr(mod, "simulate_closed_loop_estimated", fake_sim)
    monkeypatch.setattr(tune, "simulate_closed_loop_estimated", fake_sim)       # tune_lqr_weights' runs ([A])
    if hasattr(mod, "periodic_lqr_schedule"):
        monkeypatch.setattr(mod, "periodic_lqr_schedule", lambda *a, **k: (np.zeros((512, 3, 6)), None))
    if hasattr(mod, "train_sun_mlp"):
        monkeypatch.setattr(mod, "train_sun_mlp", lambda *a, **k: (None, 0.0))
    monkeypatch.setitem(sys.modules, "matplotlib", None)                       # lqr_tune.py: skip the plot
    mod.main(argv)
    return calls


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_arguments(script):
    mod = _load(script)
    assert mod.parse_args([]).legacy is False
    assert mod.parse_args(["--legacy"]).legacy is True
    with pytest.raises(SystemExit):
        mod.parse_args(["--no-such-flag"])


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_legacy_reaches_every_run(monkeypatch, capsys, script):
    calls = _run(monkeypatch, script, ["--legacy"])
    assert len(calls) == SCRIPTS[script]
    for kw in calls:
        assert {k: kw.get(k) for k in LEGACY_LOOP} == LEGACY_LOOP
    assert "legacy" in capsys.readouterr().out.splitlines()[0]


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_default_is_the_corrected_loop(monkeypatch, script):
    calls = _run(monkeypatch, script, [])
    assert len(calls) == SCRIPTS[script]
    for kw in calls:                       # nothing passed: the simulator's own (corrected) defaults apply
        assert not set(kw) & set(LEGACY_LOOP)
