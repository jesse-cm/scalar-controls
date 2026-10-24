"""Baseline specs for experiments 1-5 of docs/learned-estimation-and-control.md.

Each function mirrors one script, case for case, with the same settings:

    experiment_1()  scripts/nn_sun_sensor.py
    experiment_2()  scripts/absolute_without_gps.py
    experiment_3()  scripts/sysid_magnetorquer.py     (the eight burst campaigns)
    experiment_4()  scripts/lqr_tune.py               (blocks A-D)
    experiment_5()  scripts/closed_loop_nn.py         (both controller designs)

``SCRIPT_SEEDS`` records the seed each script hard-codes for a case; that seed,
not the plan's assumed seed 2, is what produced the number in the doc.
``DOC_CHECKS`` lists every number the doc states for these experiments with its
line, so the phase-0 gate is a table printed by ``scripts/matrix.py --gate``.

Not mirrored: the 300 s inertia-shape check in sysid_magnetorquer.py (doc line
75) and the storage sizes of the gain schedule (doc lines 115-116); neither is
a run of the simulators the harness wraps.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..attitude import quat_from_rotvec
from .spec import ScenarioSpec

__all__ = ["SEEDS", "PLAN_GATE_SEED", "DocCheck", "baseline_specs", "SCRIPT_SEEDS", "DOC_CHECKS",
           "projected_weight_draws", "rl_tune_specs", "TUNED_FILES", "tuned_controller_specs",
           "select_specs"]

SEEDS = tuple(range(40))        # rule 4 of the verification-loop plan
PLAN_GATE_SEED = 2              # the seed the plan's gate assumes

# Experiment 6: the tuner outputs of sections 6-7 of the notes, in the order they were produced.
TUNED_FILES = ("lqr_tuned", "lqr_tuned2", "lqr_tuned_cal", "lqr_tuned_3mode")
_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")

_EPOCH_2024 = (2024, 3, 20, 12, 0, 0)       # Scenario() default epoch; jd 2460390.0
_M_RES = (2e-3, -1e-3, 0.5e-3)              # residual dipole used by scripts 3, 4, 5 [A m^2]
_Q_FAR, _W_FAR = (0.5, 0.5, 0.5, 0.5), (1.0, -0.6, 0.8)                       # 120 deg / 1 deg/s
_Q_NEAR = tuple(float(x) for x in quat_from_rotvec(np.deg2rad(20) * np.array([0.6, 0.8, 0.0])))
_W_NEAR = (0.1, -0.05, 0.08)                                                    # 20 deg / 0.1 deg/s


@dataclass(frozen=True)
class DocCheck:
    """One number (or range) the doc states, and where."""

    key: str                    # ScenarioSpec.key
    seed: int                   # the seed the script used for it
    metric: str                 # Metrics field or Metrics.extra key
    doc: str                    # the value as printed in the doc
    line: int                   # line in docs/learned-estimation-and-control.md
    kind: str = "value"         # "value" | "range" | "approx" (worded loosely; no yes/no)
    lo: Optional[float] = None
    hi: Optional[float] = None


def projected_weight_draws(n=4, seed=0):
    """The (q, w, r) weight sets tune.tune_lqr_weights(n_candidates=n, seed=seed) draws, in draw order."""
    rng = np.random.default_rng(seed)
    return [(float(10 ** rng.uniform(-1, 2)), float(10 ** rng.uniform(1, 5)),
             float(10 ** rng.uniform(7, 12))) for _ in range(n)]


def experiment_1():
    specs, seeds, checks = [], {}, []
    unit = dict(experiment="1", sun_unit="flight_like")
    table = {  # doc lines 32-34: albedo -> front -> (median, p95)
        0.0: (32, {"analytic": ("3.0", "6.0"), "mlp_a": ("1.8", "3.9"), "mlp_b": ("2.5", "6.7")}),
        0.3: (33, {"analytic": ("4.2", "11.6"), "mlp_a": ("2.6", "8.4"), "mlp_b": ("2.2", "5.3")}),
        0.5: (34, {"analytic": ("5.2", "18.5"), "mlp_a": ("3.3", "14.4"), "mlp_b": ("2.3", "5.6")}),
    }
    for a in (0.0, 0.15, 0.3, 0.5):
        for front in ("analytic", "mlp_a", "mlp_b"):
            s = ScenarioSpec(name=f"e1_sunvec_alb{a:g}", kind="sun_vector", sun_front=front,
                             dataset_albedo_max=a, **unit)
            specs.append(s)
            seeds[s.key] = 7
            if a in table:
                line, row = table[a]
                checks += [DocCheck(s.key, 7, "sun_err_med", row[front][0], line),
                           DocCheck(s.key, 7, "sun_err_p95", row[front][1], line)]
    mekf = {"analytic": ("0.77", "10.6"), "mlp_a": ("0.56", "2.8"), "mlp_b": ("0.50", "1.3")}   # line 37
    for front in ("analytic", "mlp_a", "mlp_b"):
        s = ScenarioSpec(name="e1_mekf_alb0.3", kind="estimation", epoch=_EPOCH_2024, inclination_deg=51.6,
                         omega0_deg_s=(4.0, -3.0, 5.0), albedo_coeff=0.3, sun_front=front,
                         knowledge_window="after_first_fifth", **unit)
        specs.append(s)
        seeds[s.key] = 3
        checks += [DocCheck(s.key, 3, "know_med_all", mekf[front][0], 37),
                   DocCheck(s.key, 3, "know_p95_all", mekf[front][1], 37)]
    return specs, seeds, checks


def experiment_2():
    specs, seeds, checks = [], {}, []
    table = {  # doc lines 57-62: km -> (line, clock, field dir, median, p95)
        0.0: (57, "0", "0.00", "0.42", "0.63"), 20.0: (58, "2.6", "0.30", "0.51", "0.92"),
        50.0: (59, "6.6", "0.75", "0.83", "1.53"), 100.0: (60, "13", "1.5", "1.5", "2.7"),
        300.0: (61, "40", "4.5", "4.2", "7.3"), 1000.0: (62, "132", "15", "14", "25"),
    }
    for km in (0.0, 5.0, 20.0, 50.0, 100.0, 300.0, 1000.0):
        s = ScenarioSpec(name=f"e2_alongtrack_{km:g}km", experiment="2", kind="estimation",
                         epoch=_EPOCH_2024, inclination_deg=98.61, j2=True, omega0_deg_s=(4.0, -3.0, 5.0),
                         ref_along_track_error_km=km, knowledge_window="after_first_fifth")
        specs.append(s)
        seeds[s.key] = 5
        if km in table:
            line, clock, fld, med, p95 = table[km]
            checks += [DocCheck(s.key, 5, "clock_equiv_s", clock, line),
                       DocCheck(s.key, 5, "field_dir_err_med", fld, line),
                       DocCheck(s.key, 5, "know_med_all", med, line),
                       DocCheck(s.key, 5, "know_p95_all", p95, line)]
    return specs, seeds, checks


def experiment_3():
    specs, seeds, checks = [], {}, []
    table = {  # doc lines 85-88: (pulse s, Hz) at 90 pulses -> (line, per-sample, impulse, x, y, z)
        (0.32, 1.0): (85, "0.50", "0.33", "0.27", "0.11", "0.52"),
        (0.32, 10.0): (86, "1.36", "0.28", "0.12", "0.12", "0.50"),
        (2.0, 1.0): (87, "0.26", "0.048", "0.018", "0.017", "0.086"),
        (2.0, 10.0): (88, "0.19", "0.066", "0.035", "0.021", "0.116"),
    }
    for dur in (0.32, 2.0):
        for hz in (1.0, 10.0):
            for n_p in (30, 90):
                s = ScenarioSpec(name=f"e3_pulse{dur:g}s_{hz:g}Hz_n{n_p}", experiment="3", kind="sysid",
                                 epoch=_EPOCH_2024, inclination_deg=51.6, omega0_deg_s=(1.0, -0.7, 0.5),
                                 m_res=_M_RES, pulse_duration_s=dur, sample_hz=hz, n_pulses=n_p)
                specs.append(s)
                seed = int(hz * 10 + n_p)           # the script's own seed formula
                seeds[s.key] = seed
                if n_p == 90:
                    line, ps, imp, cx, cy, cz = table[(dur, hz)]
                    checks += [DocCheck(s.key, seed, "M_rel_err_per_sample", ps, line),
                               DocCheck(s.key, seed, "M_rel_err_impulse", imp, line),
                               DocCheck(s.key, seed, "M_col_err_x", cx, line),
                               DocCheck(s.key, seed, "M_col_err_y", cy, line),
                               DocCheck(s.key, seed, "M_col_err_z", cz, line)]
                if (dur, hz, n_p) == (2.0, 10.0, 90):   # the script's closing run, seed 1
                    checks.append(DocCheck(s.key, 1, "m_res_err_mAm2", "~0.3", 92, kind="approx"))
    return specs, seeds, checks


def experiment_4():
    specs, seeds, checks = [], {}, []
    far = dict(experiment="4", q0=_Q_FAR, omega0_deg_s=_W_FAR)
    near = dict(experiment="4", q0=_Q_NEAR, omega0_deg_s=_W_NEAR)

    # [A] time-invariant LQR + projection, truth feedback, no residual dipole, 3 orbits
    for q, w, r in projected_weight_draws(4, 0):
        s = ScenarioSpec(name="e4A_acq120_truth", estimator="truth", controller="projected_lqr",
                         proj_q_weight=q, proj_w_weight=w, proj_r_weight=r, n_orbits=3, **far)
        specs.append(s)
        seeds[s.key] = 0
        checks.append(DocCheck(s.key, 0, "point_mean", "90–110", 107, kind="range", lo=90, hi=110))

    # [B] periodic LQR near the target, truth feedback, 4 orbits
    for rho in (0.1, 0.3, 1.0):
        for bins in (None, 256, 64):
            s = ScenarioSpec(name="e4B_near20_truth", estimator="truth", controller="periodic_lqr",
                             rho=rho, schedule_bins=bins, n_orbits=4, m_res=_M_RES, **near)
            specs.append(s)
            seeds[s.key] = 0
            if bins != 64:
                checks.append(DocCheck(s.key, 0, "point_mean", "0.2–0.5", 114, kind="range", lo=0.2, hi=0.5))
            else:
                checks.append(DocCheck(s.key, 0, "diverged", "1" if rho <= 0.3 else "0", 115))

    # [C] acquisition from 120 deg / 1 deg/s, 256 bins, truth feedback, 6 orbits
    for rho in (0.1, 0.3, 1.0):
        for ctrl in ("periodic_lqr", "two_mode"):
            s = ScenarioSpec(name="e4C_acq120_truth", estimator="truth", controller=ctrl, rho=rho,
                             n_orbits=6, m_res=_M_RES, **far)
            specs.append(s)
            seeds[s.key] = 0
            if ctrl == "periodic_lqr":
                checks += [DocCheck(s.key, 0, "diverged", "1", 118),
                           DocCheck(s.key, 0, "max_rate_deg_s", "to 20", 118, kind="approx")]
            else:
                checks += [DocCheck(s.key, 0, "settle_h", "3.1–4.8", 120, kind="range", lo=3.1, hi=4.8),
                           DocCheck(s.key, 0, "point_mean", "0.2–0.5", 120, kind="range", lo=0.2, hi=0.5)]

    # [D] two-mode, rho 0.3, 256 bins, MEKF in the loop, 6 orbits
    s = ScenarioSpec(name="e4D_acq120_mekf", controller="two_mode", n_orbits=6, m_res=_M_RES, **far)
    specs.append(s)
    seeds[s.key] = 2
    checks += [DocCheck(s.key, 2, "point_mean", "3.0", 124), DocCheck(s.key, 2, "point_p95", "8", 124),
               DocCheck(s.key, 2, "know_med_all", "0.6", 124)]
    return specs, seeds, checks


def experiment_5():
    specs, seeds, checks = [], {}, []
    table = {  # doc lines 136-141, stiff design: (front, albedo) -> (line, point mean, p95, know med, p95)
        ("analytic", 0.0): (136, "5.9", "12.4", "1.3", "3.9"), ("analytic", 0.3): (137, "116", "173", "9.6", "14.4"),
        ("analytic", 0.5): (138, "117", "174", "18", "22"), ("mlp_b", 0.0): (139, "5.9", "16.8", "1.2", "5.4"),
        ("mlp_b", 0.3): (140, "5.1", "11.6", "0.7", "3.9"), ("mlp_b", 0.5): (141, "10.4", "24.9", "1.0", "9.6"),
    }
    for th, wm, rho in ((10.0, 0.2, 0.3), (10.0, 1.0, 1.0)):       # "stiff", "low rate weight"
        for front in ("analytic", "mlp_b"):
            for a in (0.0, 0.3, 0.5):
                s = ScenarioSpec(name=f"e5_fine20_alb{a:g}", experiment="5", sun_unit="flight_like",
                                 sun_front=front, albedo_coeff=a, controller="periodic_lqr",
                                 theta_max_deg=th, omega_max_deg_s=wm, rho=rho, n_orbits=4,
                                 m_res=_M_RES, q0=_Q_NEAR, omega0_deg_s=_W_NEAR)
                specs.append(s)
                seeds[s.key] = 2
                if (th, wm, rho) == (10.0, 0.2, 0.3):
                    line, pm, pp, km, kp = table[(front, a)]
                    checks += [DocCheck(s.key, 2, "point_mean", pm, line), DocCheck(s.key, 2, "point_p95", pp, line),
                               DocCheck(s.key, 2, "know_med_all", km, line), DocCheck(s.key, 2, "know_p95_all", kp, line)]
                    if a > 0 and front == "analytic":
                        checks.append(DocCheck(s.key, 2, "diverged", "1", line))
    return specs, seeds, checks


_EXPERIMENTS = {"1": experiment_1, "2": experiment_2, "3": experiment_3, "4": experiment_4, "5": experiment_5}


def _collect():
    specs, seeds, checks = [], {}, []
    for fn in _EXPERIMENTS.values():
        s, sd, c = fn()
        specs += s
        seeds.update(sd)
        checks += c
    return specs, seeds, checks


_SPECS, SCRIPT_SEEDS, DOC_CHECKS = _collect()


def baseline_specs(experiments=None):
    """Baseline ScenarioSpecs, optionally only for the given experiment ids ('1'..'5')."""
    if experiments is None:
        return list(_SPECS)
    want = {str(e) for e in experiments}
    return [s for s in _SPECS if s.experiment in want]


def rl_tune_specs(params, mag_calibrated=True, n_orbits_acq=4, n_orbits_fine=3, coil_scale=1.0,
                  inertia_scale=1.0):
    """The two episodes of rl_tune.evaluate(params, default_plant(...)) as specs: (acquisition, fine).

    Not part of the experiment 1-5 baseline; this is the section 6-7 setup
    (mildly distorted magnetometer, three-mode controller, last-orbit scoring).
    """
    from ..control.rl_tune import default_plant
    p = default_plant()
    th, wm, rho, enter, exit_, k_damp, fth, fwm, frho = (float(x) for x in params)
    common = dict(experiment="6", controller="three_mode", theta_max_deg=th, omega_max_deg_s=wm, rho=rho,
                  enter_deg_s=enter, exit_deg_s=exit_, k_damp=k_damp, fine_theta_max_deg=fth,
                  fine_omega_max_deg_s=fwm, fine_rho=frho, m_res=tuple(float(x) for x in p.m_res),
                  mag_bias_T=tuple(float(x) for x in p.mag_bias),
                  mag_soft_iron=tuple(tuple(float(x) for x in row) for row in p.mag_soft_iron),
                  mag_calibrated=mag_calibrated, coil_scale=coil_scale, inertia_scale=inertia_scale,
                  actuation_window="last_orbit")
    acq = ScenarioSpec(name="s6_acq120", q0=_Q_FAR, omega0_deg_s=_W_FAR, n_orbits=n_orbits_acq, **common)
    fine = ScenarioSpec(name="s6_fine20", q0=_Q_NEAR, omega0_deg_s=_W_NEAR, n_orbits=n_orbits_fine,
                        start_mode=3, **common)
    return acq, fine


def tuned_controller_specs(files=TUNED_FILES, results_dir=_RESULTS):
    """Experiment 6: every tuned-controller file as the two rl_tune.evaluate episodes, (acq, fine) per file.

    Each file is run the way it says it was produced:
      - parameters through rl_tune.load_params, so a six-parameter file gets a
        fine schedule equal to its acquisition schedule;
      - the calibration pass is on exactly when the file's stored
        ``train_metrics`` has ``cal_dir_err_med`` (evaluate writes that key only
        when the plant has mag_calibrated=True).
    The magnetometer treatment is in neither the estimator nor the controller
    label, and lqr_tuned2 / lqr_tuned_cal hold the same parameters, so the
    scenario name carries both: ``s6_acq120_<rawmag|calmag>_<file>``.

    Not a baseline: no doc check, no script seed, never part of ``--gate``.
    """
    from ..control.rl_tune import load_params
    specs = []
    for stem in files:
        params, meta = load_params(os.path.join(results_dir, stem + ".json"))
        cal = "cal_dir_err_med" in meta["train_metrics"]
        tag = f"{'calmag' if cal else 'rawmag'}_{stem}"
        specs += [s.replace(name=f"{s.name}_{tag}") for s in rl_tune_specs(params, mag_calibrated=cal)]
    return specs


def select_specs(experiments=None):
    """baseline_specs(experiments), followed by tuned_controller_specs() when id '6' is asked for.

    ``None`` is the baseline only (experiments 1-5); experiment 6 is never selected by default.
    """
    specs = baseline_specs(experiments)
    if experiments is not None and "6" in {str(e) for e in experiments}:
        specs += tuned_controller_specs()
    return specs
