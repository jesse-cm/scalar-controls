#!/usr/bin/env python3
"""Ledger rows for plan row 1c (magnetometer realism), written through scripts/matrix.py.

    python3 scripts/mag_realism.py --list
    python3 scripts/mag_realism.py --seeds 0-39 --workers 4              # the four rows
    python3 scripts/mag_realism.py --experiments 1c-parts --seeds 0-39 --no-ledger

This file only supplies the scenario list; argument parsing, the run, the
aggregation and the append to results/ledger.csv are scripts/matrix.py's
``main``, unchanged. (--gate is refused: these scenarios have no doc numbers.)

The four rows (switch on; the switch-off rows are the baseline rows of the same
scenario name without the suffix):

    e2_alongtrack_0km+mag1c_uncal   open loop, experiment 2, zero reference error
    e2_alongtrack_0km+mag1c_cal     same, rl_tune.calibration_pass before the run
    e4D_acq120_mekf+mag1c_uncal     closed loop, experiment 4 block D
    e4D_acq120_mekf+mag1c_cal       same, calibration pass before the run

``--experiments 1c-parts`` selects the open-loop scenario with one term on at a
time (no calibration), to show which term carries the effect.

WHERE THE NUMBERS COME FROM
    bias, soft iron      rl_tune.Plant defaults, scalar/control/rl_tune.py lines 75-77
                         (the "1.5 uT bias, 5 % soft iron" plant of the notes, section 6)
    quantisation         0.15 uT, docs/verification-loop-plan.md row 1c
    temperature drift    hardware.json (magnetometer.temp_coeff and the board_temperature entries),
                         read through scalar/hardware.py; as shipped both are "assumed", and each
                         entry's note says which bench test replaces it. Editing the file changes
                         what this script runs; the rows it already wrote stay as they are.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scalar import hardware  # noqa: E402
from scalar.control.rl_tune import default_plant  # noqa: E402
from scalar.harness import baseline_specs  # noqa: E402
from scalar.sensors.magnetometer import LIS2MDL_LSB_T  # noqa: E402

_HW = hardware.load()

# From hardware.json (one home per number); the names are kept for the tests that use them.
# The board temperature is a swing (amplitude, phase) or a logged profile; with a log, the two
# swing names are None and realism_fields passes the log.
ASSUMED_TEMP_COEFF_T_PER_K = _HW.require("magnetometer", "temp_coeff", "scripts/mag_realism.py")
ASSUMED_TEMP_AMP_K = _HW.value("board_temperature", "swing_amplitude")
ASSUMED_TEMP_PHASE_DEG = _HW.value("board_temperature", "swing_phase")


def realism_fields(bias_soft_iron=True, quantisation=True, temperature=True):
    """ScenarioSpec keyword arguments that switch the chosen row-1c terms on."""
    kw = {}
    if bias_soft_iron:
        p = default_plant()
        kw["mag_bias_T"] = tuple(float(x) for x in p.mag_bias)
        kw["mag_soft_iron"] = tuple(tuple(float(x) for x in row) for row in p.mag_soft_iron)
    if quantisation:
        kw["mag_quant_T"] = LIS2MDL_LSB_T
    if temperature:
        kw.update(hardware.temperature_fields(_HW, ("mag",), required=True)[0])
    return kw


def _base(name):
    (s,) = [s for s in baseline_specs() if s.name == name]
    return s


def open_loop_base():
    return _base("e2_alongtrack_0km")


def closed_loop_base():
    return _base("e4D_acq120_mekf")


def realism_specs():
    """The four ledger rows of row 1c, then the one-term-at-a-time open-loop scenarios."""
    out = []
    for base in (open_loop_base(), closed_loop_base()):
        for tag, cal in (("uncal", False), ("cal", True)):
            out.append(base.replace(name=f"{base.name}+mag1c_{tag}", mag_calibrated=cal, **realism_fields()))
    ol = open_loop_base()
    for tag, kw in (("bias_softiron", dict(quantisation=False, temperature=False)),
                    ("quant", dict(bias_soft_iron=False, temperature=False)),
                    ("temp", dict(bias_soft_iron=False, quantisation=False))):
        out.append(ol.replace(name=f"{ol.name}+mag1c_only_{tag}", experiment="1c-parts", **realism_fields(**kw)))
    return out


def _specs_for_matrix(experiments=None):
    specs = realism_specs()
    if experiments is None:
        return specs
    want = {str(e) for e in experiments}
    return [s for s in specs if s.experiment in want]


def main(argv=None):
    import matrix
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--gate" in argv:
        sys.exit("mag_realism.py: --gate does not apply (no doc numbers for these scenarios)")
    return matrix.main(argv, select=_specs_for_matrix)


if __name__ == "__main__":
    sys.exit(main())
