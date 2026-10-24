#!/usr/bin/env python3
"""Ledger rows with the plant switches on: scripts/matrix.py over the baseline scenarios, plant changed.

    python3 scripts/plant_step_rows.py --match 'e4D*' --seeds 0-39 --workers 4
    python3 scripts/plant_step_rows.py --plant-hold dipole --plant-substeps 4 --match 'e5*' --seeds 0-7 --no-ledger

``--plant-hold`` and ``--plant-substeps`` set ScenarioSpec.plant_hold and
.plant_substeps on every selected baseline scenario; every other argument goes
to scripts/matrix.py unchanged, and matrix.py does the run and writes the rows.
Only closed-loop scenarios have the switches, so the others are left out.

The ledger has no column for the plant, so the scenario name carries it:
``e4D_acq120_mekf+plant_dipole_x2``. The row with the plain name is the
baseline plant (torque held, one substep).

Default substep count: 2, the smallest count whose doubling moves the truth
attitude of the cross-check's detumble orbit by less than 0.01 deg
(tests/test_plant_step.py, CONVERGED_SUBSTEPS).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matrix  # noqa: E402  (scripts/matrix.py; it puts the repo root on sys.path)

from scalar.dynamics import PLANT_HOLDS  # noqa: E402
from scalar.harness.baseline import select_specs  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant-hold", default="dipole", choices=PLANT_HOLDS)
    ap.add_argument("--plant-substeps", type=int, default=2)
    a, rest = ap.parse_known_args(argv)
    if "--gate" in rest:
        ap.error("--gate compares the baseline plant with the doc; use scripts/matrix.py for it")

    def switched_specs(experiments=None):
        return [s.replace(name=f"{s.name}+plant_{a.plant_hold}_x{a.plant_substeps}",
                          plant_hold=a.plant_hold, plant_substeps=a.plant_substeps)
                for s in select_specs(experiments) if s.kind == "closed_loop"]

    return matrix.main(rest, select=switched_specs)


if __name__ == "__main__":
    sys.exit(main())
