#!/usr/bin/env python3
"""Ledger rows for plan row 1h (gyro realism), written through scripts/matrix.py.

    python3 scripts/gyro_realism.py --list
    python3 scripts/gyro_realism.py --seeds 0-39 --workers 4             # the three rows
    python3 scripts/gyro_realism.py --experiments 1h-parts --seeds 0-39 --no-ledger
    python3 scripts/gyro_realism.py --bias-table --seeds 0-39 --workers 4

This file only supplies the scenario list; argument parsing, the run, the
aggregation and the append to results/ledger.csv are scripts/matrix.py's
``main``, unchanged. (--gate is refused: these scenarios have no doc numbers.)

The three rows (all four terms on; the switch-off rows are the rows of the same
scenario name without the suffix, already in the ledger):

    e2_alongtrack_0km+gyro1h                   open loop, experiment 2, zero reference error
    s6_acq120_calmag_lqr_tuned_3mode+gyro1h    closed loop, experiment 6, acquisition episode
    s6_fine20_calmag_lqr_tuned_3mode+gyro1h    closed loop, experiment 6, fine-pointing episode

``--experiments 1h-parts`` selects the open-loop scenario with one term on at a
time, to show which term carries the effect. ``--experiments 1h-notemp``
selects the two closed-loop scenarios with the three terms that do not rest on
the ASSUMED temperature values (scale factor, misalignment, quantisation).
Neither set is a ledger row unless it is run without --no-ledger.

``--bias-table`` is not a ledger route. The MEKF's gyro-bias estimate error has
no column in the fixed metric set, so it cannot be a ledger row; this prints it
(with the knowledge error of the same runs) for the open-loop scenario, off,
one term at a time and all together, as median / p95 across the seed set.

WHERE THE NUMBERS COME FROM
    scale factor     1 %, docs/verification-loop-plan.md row 1h. Which axes and which sign: ASSUMED.
    misalignment     0.5 deg, same row. About which axis: ASSUMED.
    quantisation     ASSUMED. The row says "at the FS in use"; the full-scale range the flight
                     code sets is not in this repo, and neither is the LSB that goes with it.
    temperature      ASSUMED coefficient. The board temperature is the magnetometer row's
                     (scripts/mag_realism.py, itself ASSUMED), so both sensors see one board.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mag_realism import ASSUMED_TEMP_AMP_K, ASSUMED_TEMP_PHASE_DEG  # noqa: E402  (one board temperature)
from scalar.harness.baseline import select_specs  # noqa: E402

# From the plan, row 1h: "1 % scale factor".
PLAN_SCALE_FACTOR = 0.01
# ASSUMED. The plan gives the size only. Here: the same sign on all three axes, so the sensor
# reads every rate 1 % high. Replaced by a rate-table run of the flight board (each axis at a
# known rate, both directions); phase 5 of the plan lists no such item yet.
ASSUMED_SCALE_FACTOR_SIGNS = (1.0, 1.0, 1.0)

# From the plan, row 1h: "0.5 deg misalignment".
PLAN_MISALIGN_DEG = 0.5
# ASSUMED. The plan gives the angle only. Here: the sensor triad turned as a whole about the body
# diagonal, so no axis is spared. Replaced by the same rate-table run (cross-axis response).
ASSUMED_MISALIGN_AXIS = (1.0, 1.0, 1.0)

# ASSUMED. Size of one output step [deg/s]. The row says "quantisation at the FS in use": the
# full-scale range the flight code writes to the gyro is not in this repo, and no file here gives
# the step for any range. The value is the author's recollection of the LSM6DSO sensitivity at
# its +-250 deg/s range, NOT checked against the datasheet or the flight code. Replaced by: the
# range setting read from the flight driver, and the step measured on the bench (the 2 h static
# log of phase 5 shows it directly as the spacing of the raw output values).
# The model rounds the one sample it produces per step; if the flight code averages several
# raw samples per estimator step the effective step is finer than this.
ASSUMED_GYRO_LSB_DEG_S = 8.75e-3

# ASSUMED. Drift of the zero-rate output per kelvin, per axis [deg/s/K]. No file in this repo
# states it; the size is the author's recollection of the datasheet's typical figure, unchecked,
# and the signs are a choice. Replaced by the plan's phase-5 item "gyro and magnetometer logs
# across a temperature ramp with the TMP112s": the slope of each axis' static output against
# TMP112 temperature.
ASSUMED_GYRO_TEMP_COEFF_DEG_S_PER_K = (0.010, -0.010, 0.010)

TERMS = ("temperature", "scale_factor", "misalignment", "quantisation")
LEDGER_BASES = ("e2_alongtrack_0km", "s6_acq120_calmag_lqr_tuned_3mode", "s6_fine20_calmag_lqr_tuned_3mode")
SUFFIX = "+gyro1h"


def realism_fields(temperature=True, scale_factor=True, misalignment=True, quantisation=True):
    """ScenarioSpec keyword arguments that switch the chosen row-1h terms on."""
    kw = {}
    if temperature:
        kw["gyro_temp_coeff_deg_s_per_K"] = ASSUMED_GYRO_TEMP_COEFF_DEG_S_PER_K
        kw["gyro_temp_amp_K"] = ASSUMED_TEMP_AMP_K
        kw["gyro_temp_phase_deg"] = ASSUMED_TEMP_PHASE_DEG
    if scale_factor:
        kw["gyro_scale_factor"] = tuple(PLAN_SCALE_FACTOR * s for s in ASSUMED_SCALE_FACTOR_SIGNS)
    if misalignment:
        axis = np.array(ASSUMED_MISALIGN_AXIS, float)
        kw["gyro_misalign_deg"] = tuple(float(x) for x in PLAN_MISALIGN_DEG * axis / np.linalg.norm(axis))
    if quantisation:
        kw["gyro_quant_deg_s"] = ASSUMED_GYRO_LSB_DEG_S
    return kw


def only(term):
    """realism_fields with exactly one term on."""
    return realism_fields(**{t: (t == term) for t in TERMS})


def _base(name):
    (s,) = [s for s in select_specs(["2", "6"]) if s.name == name]
    return s


def open_loop_base():
    return _base(LEDGER_BASES[0])


def closed_loop_bases():
    return [_base(n) for n in LEDGER_BASES[1:]]


def realism_specs():
    """The three ledger rows of row 1h, then the one-term-at-a-time and the no-temperature scenarios."""
    out = [b.replace(name=b.name + SUFFIX, **realism_fields()) for b in (open_loop_base(), *closed_loop_bases())]
    ol = open_loop_base()
    out += [ol.replace(name=f"{ol.name}{SUFFIX}_only_{t}", experiment="1h-parts", **only(t)) for t in TERMS]
    out += [b.replace(name=f"{b.name}{SUFFIX}_no_temperature", experiment="1h-notemp",
                      **realism_fields(temperature=False)) for b in closed_loop_bases()]
    return out


def _specs_for_matrix(experiments=None):
    specs = realism_specs()
    if experiments is None:
        return specs
    want = {str(e) for e in experiments}
    return [s for s in specs if s.experiment in want]


# ---------------------------------------------------------------------------
# the bias-estimate error (no ledger column): open loop only
# ---------------------------------------------------------------------------


def open_loop_bias_error(spec, seed):
    """One open-loop run with the harness's own builders, also scoring the MEKF's bias estimate.

    The bias error at a sample is |b_est - b_true| [deg/s], b_true the additive
    bias the gyro applied at that sample (random walk plus temperature drift;
    scale factor and misalignment are not a bias, so whatever part of them the
    filter books as bias counts as error). Scored over the harness's window
    (after the first fifth of the run). ``know_med`` / ``know_p95`` are the
    numbers run_case reports for the same spec and seed.
    """
    from scalar.constants import DEG
    from scalar.dynamics import RigidBody
    from scalar.estimator.mekf import run_mekf
    from scalar.harness import runner
    from scalar.scenario import Scenario
    if (spec.kind != "estimation" or spec.sun_front != "analytic" or spec.mag_calibrated
            or spec.ref_along_track_error_km != 0.0):
        raise ValueError(f"{spec.name}: open_loop_bias_error covers the plain open-loop run only")
    orbit, gyro = runner._orbit(spec), runner._gyro(spec)
    res = Scenario(epoch=runner._epoch(spec), orbit=orbit, body=RigidBody(runner._inertia(spec)),
                   q0=np.array(spec.q0, float), omega0=np.array(spec.omega0_deg_s, float) * DEG,
                   dt=spec.dt, duration=spec.n_orbits * orbit.period, gyro=gyro,
                   magnetometer=runner._magnetometer(spec), sun_sensor=runner._sun_sensor(spec), seed=seed,
                   albedo_coeff=spec.albedo_coeff, field_model_error_deg=spec.field_model_error_deg,
                   field_truth=spec.field_truth).run()
    out = run_mekf(res, runner._mekf(spec))
    b_true = res.gyro_bias_true + np.array([gyro.temperature_offset(t) for t in res.t])
    db = np.degrees(np.linalg.norm(out.b_est - b_true, axis=1))
    w = len(db) // 5
    return {"bias_err_med": float(np.median(db[w:])), "bias_err_p95": float(np.percentile(db[w:], 95)),
            "know_med": float(np.median(out.err_deg[w:])), "know_p95": float(np.percentile(out.err_deg[w:], 95))}


def _bias_task(task):
    label, kw, seed = task
    return label, seed, open_loop_bias_error(open_loop_base().replace(name=f"bias_{label}", **kw), seed)


def bias_table(seeds, workers=4):
    """{label: {metric: (median, p95) across seeds}} for off, each term alone, all on."""
    import multiprocessing as mp
    configs = [("off", {})] + [(t, only(t)) for t in TERMS] + [("all", realism_fields())]
    tasks = [(label, kw, int(sd)) for label, kw in configs for sd in seeds]
    if workers <= 1:
        done = [_bias_task(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(min(workers, len(tasks))) as pool:
            done = pool.map(_bias_task, tasks, chunksize=1)
    table = {}
    for label, _ in configs:
        rows = [r for lab, _, r in done if lab == label]
        table[label] = {k: (float(np.median([r[k] for r in rows])), float(np.percentile([r[k] for r in rows], 95)))
                        for k in rows[0]}
    return table


def _print_bias_table(argv):
    import argparse
    from scalar.harness.ledger import parse_seed_set
    ap = argparse.ArgumentParser(description="bias-estimate error, open loop (not a ledger route)")
    ap.add_argument("--bias-table", action="store_true")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    seeds = parse_seed_set(a.seeds)
    table = bias_table(seeds, a.workers)
    print(f"{open_loop_base().name}, seeds {a.seeds}: median / p95 across seeds")
    print(f"{'terms on':<14}{'bias err med [deg/s]':>24}{'bias err p95 [deg/s]':>24}"
          f"{'know med [deg]':>20}{'know p95 [deg]':>20}")
    for label, row in table.items():
        cells = [f"{row[k][0]:.5f} / {row[k][1]:.5f}" for k in ("bias_err_med", "bias_err_p95")]
        cells += [f"{row[k][0]:.3f} / {row[k][1]:.3f}" for k in ("know_med", "know_p95")]
        print(f"{label:<14}{cells[0]:>24}{cells[1]:>24}{cells[2]:>20}{cells[3]:>20}")
    return 0


def main(argv=None):
    import matrix
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--gate" in argv:
        sys.exit("gyro_realism.py: --gate does not apply (no doc numbers for these scenarios)")
    if "--bias-table" in argv:
        return _print_bias_table(argv)
    if "--experiments" not in argv:
        argv = ["--experiments", "2,6", *argv]        # the three ledger rows
    return matrix.main(argv, select=_specs_for_matrix)


if __name__ == "__main__":
    sys.exit(main())
