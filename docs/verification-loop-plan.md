# Verification loop plan — SCALAR estimation and control

Written 2026-09-12. Companion to `learned-estimation-and-control.md`, whose
conclusions are the things under test. What has been executed so far is in
[Status](#status) at the end of this file.

## Rules of the loop

1. **The ledger is the truth.** `results/ledger.csv` — one row per (scenario,
   estimator, controller, seed-set, git hash, date) with fixed metric columns.
   A conclusion is a ledger row that reproduces with one command. Prose without
   a row is an opinion.
2. **Pre-registered thresholds.** Written before the run, not after:
   knowledge p95 ≤ 15° (S4's budget, provenance still open — capture Q3);
   pointing mean ≤ 5° and p95 ≤ 15° in fine mode (placeholder until the
   mission states one); coil power ≤ 0.3 W average in fine mode (placeholder,
   power budget missing — not-done D); settle < 6 h from 120° / 1°/s.
3. **A model is added with a test that shows its effect**, never as a switch
   nobody has exercised. Each fidelity item below ships with one assertion of
   the form "with this on, metric X moves by at least Y".
4. **Monte Carlo or it did not happen.** 40 seeds, same seed list everywhere
   (`SEEDS = range(40)`), report median / p95 across seeds, not one run.
5. **Root cause must predict.** An explanation for a failure counts only when
   it predicts a new result that is then confirmed by a run.
6. **Agents add rows, not conclusions.** The orchestrator compares rows
   against the baseline and the thresholds and writes the conclusion.
7. **Nothing is committed by an agent.** Untracked work, reviewed, then one
   commit per phase by a person.

## Phase 0 — harness (one agent, blocking)

Deliverable: `scalar/harness/` with
- `ScenarioSpec` dataclass: every physics/noise/sampling switch with a default
  that reproduces today's runs exactly (so baseline rows equal the current doc).
- `run_case(spec, seed) -> Metrics` and `run_matrix(specs, seeds) -> ledger rows`.
- Fixed metric set: knowledge median/p95 (all, lit, eclipse), pointing
  mean/p95 (last orbit), settle time, saturation fraction, coil duty per
  channel, coil power W, max rate, divergence flag (pointing > 90° in last orbit).
- `scripts/matrix.py` that writes ledger rows; `results/ledger.csv` created
  with the baseline rows for experiments 1–5 at 40 seeds.
Gate: baseline rows for seed 2 match the numbers in
`learned-estimation-and-control.md` to two decimals.

## Phase 1 — fidelity (parallel agents, each independent, each one switch)

| # | Switch | What it adds | Effect test | Files touched |
|---|---|---|---|---|
| 1a | `field_truth=igrf` in closed loop | real IGRF truth, onboard reference stays dipole or truncated IGRF (order as a parameter) | knowledge median rises vs baseline | `control/tune.py`, `environment/igrf.py` |
| 1b | disturbance torques | gravity gradient (needs inertia), aero at 550 km (Cd·A as params), SRP | nonzero steady dipole in fine mode | `dynamics.py`, new `environment/disturbance.py` |
| 1c | magnetometer realism | bias, soft iron, 0.15 µT quantisation, temperature drift vs a TMP112 profile | knowledge degrades; ellipsoid calibration recovers most of it | `sensors/magnetometer.py`, `calibration/mag_onorbit.py` |
| 1d | coil–magnetometer coupling + duty cycle | B_meas += C·u during torque; measure-then-torque cycle with duty d; authority scales by d | sysid biased without coupling term; pointing power rises 1/d | `sysid/magnetorquer_id.py`, `control/tune.py` |
| 1e | sampling and latency | estimator dt and control dt swept over {1, 5, 10, 30, 60} s; gyro either continuous-integrated on the RP2350 or sampled at the same dt; link latency 0–2 s; ZOH on the dipole | table of every metric vs dt — this answers the 30 s question | `control/tune.py`, `estimator/mekf.py` (rate-as-state variant for the no-gyro-integration case) |
| 1f | VEML6031 sensor model | 7 sensors, normals as parameters (unknown → sweep), digital lux output, integration time, full-sun saturation, angular response from datasheet | analytic reconstruction breaks under saturation; MLP retrained on this model | `sensors/sun_sensor.py`, `learning/sun_mlp.py` |
| 1g | schedule ageing | multi-day run with J2 and Earth rotation; schedule computed at t0 and reused; then recomputed every N orbits | pointing error vs schedule age | `control/periodic_lqr.py` |
| 1h | gyro realism | temperature bias, 1 % scale factor, 0.5° misalignment, quantisation at the FS in use | bias-estimate error and knowledge shift | `sensors/gyro.py` |

Each agent brief carries: the exact file, the ScenarioSpec field to add, the
effect assertion, and the ledger rows to produce (baseline seed list, switch
on vs off).

## Phase 2 — independent cross-check (one agent, parallel with phase 1)

Set up Basilisk (fallback: NASA 42) with the same orbit, inertia, sensor suite
and coil set; run one orbit of free tumble and one of B-dot detumble; compare
truth attitude, B_body, eclipse timing and sun-sensor currents against our sim
at 1 Hz. Gate: direction errors < 0.5° and eclipse edges within 5 s, or a
written discrepancy list with a cause for each. This is the only line of
defence against the Python sim and the conclusions sharing a mistake.

## Phase 3 — re-run the five experiments on the upgraded harness (after 1)

Full matrix, 40 seeds, all phase-1 switches at their realistic settings. Output
is a table: which of the doc's conclusions survive, which flip, which are
inside the seed spread. Two root-cause agents run alongside, under rule 5:
- Why does the 1°/s rate-weight design diverge with ideal sensors?
- Why does the 64-bin schedule diverge and 256 not?

## Phase 4 — redesign only where phase 3 failed

Tuning is done by `scripts/rl_tune_lqr.py` (cross-entropy method over the six
LQR weights and mode thresholds, reward derived from the criteria in
`scalar/control/rl_tune.THRESHOLDS`, train seeds 0-1 only) and accepted only
when `scripts/verify_controller.py` passes on held-out seeds 100-102 and the
±10 % plant variants. Re-tune after every phase-1 switch lands; the criteria
do not move.

Candidates, each its own agent, each producing rows against the same matrix:
deadband / slower outer loop for the 1 Hz jitter; LVLH (nadir) target variant
of the periodic plant; online magnetometer calibration gating fine mode; MLP-B
cold start and int8 quantisation; schedule recomputation on the RP2350 with a
timing estimate in C.

## Phase 5 — replace assumed noise with measured noise (people, not agents)

Bench with the V5e board and the driver board:
- 2 h static logs of gyro and magnetometer at ≥ 50 Hz → Allan deviation →
  ARW, bias instability, RRW. Fill `LSM6DSO.rateNoiseDensity` and
  `LIS2MDL.noiseStdDev` in gaps.csv with measured values.
- Same log with each coil driven at full current, and with the UHF radio
  transmitting → coupling matrix C and RF-induced offsets.
- Board rotated through all orientations in Earth's field → hard/soft iron by
  ellipsoid fit; store as the flight unit's calibration.
- VEML6031 on a rotary stage under a lamp: angular response, saturation
  threshold, integration-time setting that survives full sun.
- Gyro and magnetometer logs across a temperature ramp with the TMP112s.
Each measured number replaces a `ScenarioSpec` default and re-runs phase 3.

## Order and parallelism

Phase 0 first, alone. Phases 1a–1h and 2 in parallel (9 agents). Phase 3 after
1 completes. Phase 4 after 3. Phase 5 whenever hardware time exists; it feeds
back into 3. Estimated agent count: 1 + 9 + 3 + up to 5 = 18 over four rounds.

## Status

Dated entries, newest last. Each entry names the rows or the file it rests on.

### 2026-10-24 — phase 0 (harness) done

- Code: `scalar/harness/`, `scripts/matrix.py`, `tests/test_harness.py`.
  Ledger: `results/ledger.csv`, 174 rows: 62 baseline scenarios over seeds
  0–39, plus single-seed rows. All rows carry git hash `b5217a8+dirty`.
- Gate, re-run with `python3 scripts/matrix.py --gate --no-ledger`: of the 123
  numbers in sections 1–5 of the notes, 117 reproduce at the seed the script
  hard-codes, 2 do not, and 4 are worded too loosely to score.
  - Not reproduced: section 2, 100 km row, p95 (notes 2.7, harness 2.65);
    section 4, "90–110°" for the projected LQR (one of four weight sets gives
    89.11).
  - Not scored: "residual dipole to ~0.3 mA m²" (0.59 at the script's seed)
    and "rates to 20°/s" (21.1 to 28.9).
- The plan's assumption that seed 2 produced the notes holds only for
  experiments 4 and 5. Experiments 1–3 use other seeds (7, 3, 5, 100, 190); at
  seed 2, 49 of their numbers do not match.

What the 40-seed rows say (ledger rows with seed set `0-39`; "diverged" is the
plan's flag, pointing above 90° in the last orbit):

| Scenario (section of the notes) | Notes, one seed | Diverged, of 40 | Pointing mean, median / p95 across seeds |
|---|---|---|---|
| Two-mode controller with the MEKF in the loop (4) | 3.0° | 11 | 5.75° / 129.5° |
| Fine pointing, analytic sun front end, albedo 0 (5) | 5.9° | 3 | 6.75° / 92.3° |
| Fine pointing, analytic, albedo 0.3 and 0.5 (5) | diverged | 40 and 40 | 109° and 115° |
| Fine pointing, MLP-B, albedo 0 (5) | 5.9° | 4 | 8.26° / 121.9° |
| Fine pointing, MLP-B, albedo 0.3 (5) | 5.1° | 4 | 7.97° / 105.5° |
| Fine pointing, MLP-B, albedo 0.5 (5) | 10.4° | 27 | 45.7° / 124.5° |

Conclusions against rule 4 ("Monte Carlo or it did not happen"):

1. The single-seed closed-loop numbers in sections 4 and 5 of the notes are
   draws from the good part of the spread. The hand-picked September design
   loses pointing on roughly a tenth to a quarter of seeds even in its best
   cases.
2. Section 5's claim that the learned front end lets the loop survive albedo
   holds at albedo 0.3 (4 of 40 diverge, against 40 of 40 for the analytic
   front end) and does not hold at albedo 0.5 (27 of 40 diverge).
3. These rows are the September hand-picked design, not the tuned three-mode
   controller of sections 6–8. That controller has no 40-seed row yet.

Not yet ledger rows, so not conclusions under rule 1: the sun-vector errors
of experiment 1 and the identification errors of experiment 3 have no column
in the fixed metric set. `python3 scripts/matrix.py --per-seed PATH` prints
them.

### 2026-10-24 — phase 2 (independent cross-check) done

Basilisk 2.12.2, same orbit, inertia, sensors and coils. Result, gate table
and every discrepancy with its confirming run:
[crosscheck-basilisk.md](crosscheck-basilisk.md); numbers in
`results/crosscheck_basilisk.json`; one command, `python3
scripts/crosscheck_basilisk.py`. Outcome in one line: conventions, signs,
orbit and sensor scale agree; the gate fails on the detumble orbit and on the
README-default free tumble, for three causes in our plant (torque held instead
of dipole over a step, second-order attitude step, frame labelled J2000 but
of-date). The first two enter phase 1 as switches; the first must land before
row 1e is run.

### Not started

Phases 1, 3 and 4. Phase 5 needs the hardware.

### Open

Acquisition results for `results/lqr_tuned_3mode.json` do not reconcile
between its stored `train_metrics`, `results/lqi_fine.json` and a run today
(seeds 0 and 1 reproduce the stored numbers; seed 101 diverges in
acquisition). See section 10 of the notes. Cause not investigated.
