"""The row scripts select their own scenarios (no simulation runs here).

scripts/mag_realism.py, gyro_realism.py and plant_step_rows.py each hand
scripts/matrix.py's ``main`` their own scenario list. Twice that hand-over was
done by patching a name inside matrix.py, and when matrix.py started reading a
different name (select_specs instead of baseline_specs, commit 1fdd122) the
patch went nowhere: mag_realism.py --list printed the 62 baseline scenarios and
plant_step_rows.py stopped. They now pass the list as ``main(argv, select=...)``.

Each script's documented --list command is run as a subprocess (--list runs no
simulation) and must print its own tagged scenarios only, none of the baseline
ones. A new script that calls matrix.main must be added to ROW_SCRIPTS, and no
script may assign to a name in the matrix module.
"""
import glob
import os
import re
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from scalar.harness.baseline import select_specs  # noqa: E402

# script: (--list command arguments, tag every listed scenario carries, expected count or None)
ROW_SCRIPTS = {
    "mag_realism.py": (["--list"], "+mag1c_", 4),            # the four rows of its docstring
    "gyro_realism.py": (["--list"], "+gyro1h", 3),           # the three rows of its docstring
    "plant_step_rows.py": (["--plant-hold", "dipole", "--plant-substeps", "2", "--match", "e4D*", "--list"],
                           "+plant_dipole_x2", None),         # every closed-loop e4D baseline scenario
}


def _listed(script, args):
    out = subprocess.run([sys.executable, os.path.join(_ROOT, "scripts", script), *args],
                         capture_output=True, text=True, cwd=_ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    n = int(lines[-1].split()[0])
    keys = [line.split(None, 2)[2] for line in lines[:-1]]
    assert n == len(keys), lines[-1]
    return keys


@pytest.mark.parametrize("script", sorted(ROW_SCRIPTS))
def test_list_shows_the_scripts_own_scenarios(script):
    args, tag, count = ROW_SCRIPTS[script]
    keys = _listed(script, args)
    baseline = {s.key for s in select_specs()}
    if count is None:
        count = sum(1 for s in select_specs() if s.kind == "closed_loop" and s.name.startswith("e4D"))
    assert keys and len(keys) == count, keys
    assert all(tag in k for k in keys), [k for k in keys if tag not in k]
    assert not baseline & set(keys), sorted(baseline & set(keys))


def test_every_script_that_drives_matrix_is_covered_and_none_patches_it():
    drivers = set()
    for path in glob.glob(os.path.join(_ROOT, "scripts", "*.py")):
        name = os.path.basename(path)
        if name == "matrix.py":
            continue
        with open(path) as f:
            src = f.read()
        assert not re.search(r"\bmatrix\.\w+\s*=(?!=)", src), f"{name} assigns into the matrix module"
        if "matrix.main(" in src:
            drivers.add(name)
    assert drivers == set(ROW_SCRIPTS), drivers ^ set(ROW_SCRIPTS)
