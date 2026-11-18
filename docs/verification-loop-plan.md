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

### 2026-10-24 — tuned controllers at 40 seeds (ledger experiment 6)

Rows `s6_acq120_*` and `s6_fine20_*`, seed sets `0-39` and `100-114`
(`python3 scripts/matrix.py --experiments 6 --seeds 0-39`). Seeds 0 and 1
reproduce the stored `train_metrics` of `results/lqr_tuned_3mode.json` to the
last digit.

| Tuned file | Magnetometer | Acquisition diverged, of 40 / of 15 | Fine diverged | Fine pointing mean, median across seeds, 0–39 / 100–114 | Acquisition settle, median |
|---|---|---|---|---|---|
| `lqr_tuned` | raw | 11 / 5 | 0 / 0 | 7.93° / 9.70° | 6.02 h / 6.25 h |
| `lqr_tuned2` | raw | 10 / 2 | 0 / 0 | 7.66° / 9.43° | 6.02 h / 5.60 h |
| `lqr_tuned_cal` | calibrated | 5 / 5 | 0 / 0 | 5.52° / 6.89° | 5.18 h / 5.21 h |
| `lqr_tuned_3mode` | calibrated | 6 / 1 | 0 / 0 | 5.44° / 6.73° | 5.12 h / 5.15 h |

Conclusions against the thresholds in rule 2:

1. Fine pointing never diverges for any tuned file on any of the 55 seeds.
   The tuner did fix what the hand-picked design got wrong there.
2. The fine-pointing mean stays above the 5° placeholder on the median seed
   for every file; the calibrated files come closest. This agrees with
   sections 6–8 of the notes.
3. "Acquisition diverged = 0" fails for every file: 5 to 11 of 40 seeds are
   flagged. Calibration roughly halves the count.
4. Before conclusion 3 is trusted, the two acquisition measures must be
   reconciled. For the three-mode file, seeds 5, 9, 28, 32, 37 and 101 are
   flagged diverged yet have a finite settle time, and seed 35 never settles
   yet is not flagged. Hypothesis, not run: the same mismatch between the two
   definitions is behind the open acquisition item below.

### 2026-10-24 — phase 1, row 1c (magnetometer realism)

Rows `e2_alongtrack_0km+mag1c_*` and `e4D_acq120_mekf+mag1c_*`, seeds 0–39
(`python3 scripts/mag_realism.py --seeds 0-39`). Switches are off by default;
the gate is unchanged with them off.

| Scenario | Off | On, uncalibrated | On, calibrated |
|---|---|---|---|
| Open loop, knowledge median (median across seeds) | 0.417° | 1.464° | 0.599° |
| Closed loop, two-mode September design: diverged of 40 | 11 | 23 | 15 |
| Closed loop: pointing mean, median across seeds | 5.75° | 52.5° | 7.92° |

1. The row's assertion holds: knowledge degrades, and the ellipsoid
   calibration recovers most of it.
2. Quantisation at 0.15 µT is invisible next to the sensor noise (it moves
   the open-loop knowledge median by less than 0.001°; asserted in
   `tests/test_mag_realism.py`).
3. The temperature drift rests on two ASSUMED values with no source in this
   repo (offset coefficient and the once-per-orbit temperature swing, both in
   `scripts/mag_realism.py`). Not a ledger row: at ten times the assumed drift
   the plain ellipsoid fit recovers almost nothing (8 seeds). The phase-5
   temperature-ramp log replaces both values.

### 2026-10-24 — phase 1, plant-step row (added after the cross-check)

Switches `plant_hold` and `plant_substeps`, off by default; rows
`*+plant_dipole_x2` and one `+plant_dipole_x4`, seeds 0–39
(`python3 scripts/plant_step_rows.py --plant-hold dipole --plant-substeps 2
--seeds 0-39`). Effect tests in `tests/test_plant_step.py`.

1. Holding the dipole instead of the torque moves the truth attitude by 1.86°
   over the cross-check's detumble orbit; two substeps are converged (×2
   against ×4 differs by 0.004°). This is discrepancy D of the cross-check,
   reproduced without Basilisk.
2. At scenario level the counts barely move (two-mode September design: 11 of
   40 lost with the switch off, 12 on, 14 at four substeps).
3. Which seeds are lost does move: between two and four substeps the flag
   differs on 12 of 40 seeds. A change of 0.004° in the plant reshuffles
   individual acquisition outcomes, so a lost-seed count from 40 seeds is a
   rate with a wide spread, not a list of bad seeds.

### 2026-10-24 — phase 1, row 1h (gyro realism)

Rows `*+gyro1h`, seeds 0–39 (`python3 scripts/gyro_realism.py --seeds 0-39`).
All four terms on: temperature-dependent bias, 1 % scale factor, 0.5°
misalignment, output quantisation.

| Scenario | Off | On |
|---|---|---|
| Open loop, knowledge median (median across seeds) | 0.417° | 1.071° |
| Tuned three-mode, acquisition: lost of 40 | 6 | 39 |
| Tuned three-mode, fine pointing: lost of 40 | 0 | 40 |

1. The row's assertion holds: bias-estimate error and knowledge shift.
2. The closed-loop loss is the temperature term alone. Not ledger rows: with
   the other three terms on and temperature off, fine pointing loses 0 of 40
   and is within 0.15° per seed; with the temperature coefficient scaled to
   3 % of the ASSUMED value, 0 of 8 are lost.
3. The temperature coefficient and the board temperature swing are ASSUMED
   (`scripts/gyro_realism.py`, `scripts/mag_realism.py`); no value for either
   is in the repo. Until the phase-5 temperature ramp is logged, this row says
   that the loop has no tolerance to a gyro bias that follows board
   temperature, not that the flight gyro has one of this size.
4. The filter cannot follow the drift by raising its bias process noise alone
   (open loop, 4 seeds). Candidates, none run: subtract a ground-calibrated
   temperature term using the TMP112 reading; a temperature-slope state.

### 2026-10-24 — phase 1, row 1f (VEML6031 sun sensor) and Earth albedo

Rows `*+veml*`, `*+alb*` and `*_L-*`
(`python3 scripts/sun_realism.py --experiments 1f,1f-normals,1f-albedo`).
The `+alb*` and `_L-*` rows were run on the WashU cluster at commit `28c434b`;
see the note on platforms below.

Tuned three-mode controller, fine pointing, seeds 0–39, lost of 40 and
pointing mean (median across seeds):

| Sun sensor, front end | No albedo | Albedo 0.3 | Albedo 0.5 |
|---|---|---|---|
| Cube, analytic | 0, 5.44° | 0, 12.84° | 40, lost |
| Cube, MLP-B | — | 0, 5.90° | 0, 7.34° |
| Cube, sun sensor unused | — | 0, 11.18° | 0, 11.18° |
| VEML 6.25 ms, analytic | 0, 5.50° | 0, 12.85° | 39, lost |
| VEML 6.25 ms, MLP-B | 0, 7.36° | 0, 7.10° | 7, 9.08° |
| VEML 25 ms, analytic | 37, lost | — | — |
| VEML 25 ms, MLP-B | 0, 9.93° | — | — |

1. The row's assertion holds: at 25 ms every sunlit sample clips and the
   analytic reconstruction loses pointing on 37 of 40; at 6.25 ms the real
   sensor matches the ideal cube.
2. With albedo, the analytic front end is not usable for pointing even with
   the tuned controller. MLP-B holds fine pointing at both albedo levels on
   the cube and at 0.3 on the VEML array; on the VEML array at 0.5 it loses 7
   of 40.
3. Not using the sun sensor is immune to albedo and costs about 6° against
   the no-albedo result.
4. Acquisition with albedo (cube): analytic 3 of 40 flagged at 0.3 and 40 at
   0.5; MLP-B 2 and 9. Subject to the acquisition-measure item under Open.
5. Sensor-direction sweep (seed set 0–7, open loop, knowledge median): with
   the side faces mis-assigned the analytic front end goes from 0.42° to
   between 1.2° and 69°; a 2° mounting tilt costs 0.02°. A network trained on
   the layout flown stays at 0.41–0.48° on every layout.
6. Every network here is trained and tested on this simulator's sensor and
   albedo models. These rows show the method can work, not that a given
   network will.

### 2026-10-24 — where runs are made, and what that does to the rows

From this date simulation runs are made on the WashU engineering cluster
(Linux, Python 3.9), not on the laptop that produced the earlier rows. The
same commit gives slightly different numbers on the two: fine-pointing means
agree to about 1e-6°, and acquisition results differ visibly (for the
three-mode file on seeds 0–1, settle 3.63 h against 3.56 h), consistent with
conclusion 3 of the plant-step row. The ledger has no platform column. Rows
with git hash `28c434b` and later are cluster rows; a lost-seed count or an
acquisition figure should be compared only with rows from the same platform.

### 2026-10-24 — tuning on eight seeds (`results/lqr_tuned_ms.json`)

Cross-entropy search from `lqr_tuned_3mode`, population 12, 5 generations,
training seeds 0–7, same criteria and reward
(`python3 scripts/rl_tune_lqr.py 12 5 results/lqr_tuned_3mode.json
results/lqr_tuned_ms.json --seeds 0-7`; log in `results/rl_tune_ms.log`). Both
files were then run on the cluster at commit `6fed91d`: rows `s6_*_lqr_tuned_ms`
and `s6_*_lqr_tuned_3mode` for seed sets `0-39` and `100-114`, and the 6-orbit
acquisition rows `s6_acq120x6orb_*` (`--experiments 6L`). The earlier laptop
rows of the three-mode file are superseded by these on the scoreboard.

| | `lqr_tuned_3mode` | `lqr_tuned_ms` |
|---|---|---|
| Fine pointing mean, median across seeds, 0–39 / 100–114 | 5.44° / 6.73° | 5.47° / 6.81° |
| Fine pointing lost, of 40 / of 15 | 0 / 0 | 0 / 0 |
| Acquisition flagged lost, 4 orbits, of 40 / of 15 | 5 / 2 | 0 / 0 |
| Acquisition flagged lost, 6 orbits, of 40 | 0 | 0 |
| Acquisition settle, median, 4 orbits / 6 orbits | 5.12 h / 6.71 h | 4.96 h / 6.29 h |

1. Tuning on four times as many seeds does not move fine pointing. This
   agrees with section 6 of the notes: the fine-pointing error is set by the
   estimator, not by the controller weights.
2. The acquisition flags go away, including on seeds the tuner did not see.
   Not ledger rows: on seeds 8–39 the count is 4 for the three-mode file and
   0 for the new one.
3. With a 6-orbit episode neither file loses a seed. The seeds flagged at
   4 orbits were slow, not lost; what the new weights buy is speed.
4. The settle time is not comparable across episode lengths. It is the first
   moment after which the error is under 10° for at least 90 % of the rest of
   the run (`ClosedLoopResult.settle_time`, `scalar/control/tune.py`), so the
   same controller reports about 5 h in a 4-orbit run and 6.3–6.7 h in a
   6-orbit run. The 6 h limit cannot be judged until the measure uses a fixed
   window.
5. One seed (29) does not meet the settle condition in the 6-orbit run for
   either file. Not investigated.

### 2026-11-18 — two fixes in the closed loop

1. Errors are recorded at matching instants (`error_timing="aligned"` in
   `simulate_closed_loop_estimated`). The simulated trajectory is unchanged;
   only what is recorded moves. `tests/test_error_timing.py` fails on the old
   record with a knowledge error of 1.41° for an exact estimator.
2. The corrected plant step of the plant-step row (coil dipole held, two
   substeps) is now the default of the closed loop, the tuner and the
   harness.
3. The scenarios that reproduce the notes (the closed-loop scenarios of
   experiments 4 and 5, and `matrix.py --gate`) pin the old loop through
   `LEGACY_LOOP` in `scalar/control/tune.py`. Proof that they did not move:
   the gate's per-seed output on the cluster is identical before and after
   the change, apart from run times.
4. Rows made with the old loop and now superseded by re-runs: experiments 6,
   6L and hw, and the tuned-controller rows of `gyro_realism.py` and
   `sun_realism.py`. The stored `train_metrics` inside `results/lqr_tuned*.json`
   were produced with the old loop and reproduce only with `LEGACY_LOOP`. The
   tuned files have not been re-tuned against the corrected loop.
5. On the cluster the gate reads 108 of 123 matching and 11 not, against 117
   and 2 on the laptop that produced the notes, before and after these fixes
   alike. The two that fail on both are the ones named in the phase-0 entry.
   The nine extra are all in experiment 5, stiff design, with albedo, at
   seed 2 (cluster; laptop ledger row of 2026-10-24; notes):
   - analytic front end, albedo 0.3 (line 137): pointing mean 113.9; 116.0;
     116. Knowledge median 14.5; 9.62; 9.6. Knowledge p95 17.8; 14.4; 14.4.
   - analytic, albedo 0.5 (line 138): pointing mean 103.0; 117.0; 117.
     Pointing p95 171.6; 173.7; 174. Knowledge median 19.95; 18.28; 18.
     Knowledge p95 24.8; 22.05; 22.
   - MLP-B front end, albedo 0.3 (line 140): pointing p95 11.51; 11.57; 11.6.
     Knowledge p95 3.985; 3.926; 3.9.

   The seven analytic ones are runs that lose pointing on both machines (the
   notes say so too); no network is involved. The two MLP-B ones hold
   pointing; they move by 0.5 % and 1.5 %, across the rounding of the notes'
   figure. The same split holds for every closed-loop row at the same seed:
   runs that hold pointing without a network agree to the ledger's six digits
   in pointing and knowledge; runs with a trained network (experiments 1 and
   5) differ by up to 1.5 %; runs that lose pointing differ by up to tens of
   degrees, truth-feedback runs of experiment 4 included (the projected-LQR
   weight set that fails on both reads 88.93 and 89.11; "rates to 20°/s"
   reads 12.2–23.2 and 21.1–28.9). So a lost run lets any small difference
   grow, and the network, which each machine trains for itself with torch,
   is the small difference that shows in runs that hold. What makes the
   first difference is not established: the laptop rows come from older code
   (`b5217a8+dirty`) as well as another machine, Python and libraries.
   Hypothesis: torch training and floating-point rounding differ between the
   machines. A gate run on the laptop at the cluster's commit would separate
   machine from code.
6. Also fixed: the frame is labelled as the mean equator and equinox of date
   in the code's comments (discrepancy A); `results/estimator_levers.json`
   records the seeds it holds; `matrix.py --gate` refuses every experiment
   that has no numbers in the notes; `outputs/detumble_lqr.png` is
   regenerated after the sign fix.

### 2026-11-18 — tuned-controller rows re-run with the corrected loop

All rows built on the tuned controllers were run again on the cluster at
commit `5058c86`: experiments 6, 6L and hw, and the tuned-controller rows of
`gyro_realism.py` and `sun_realism.py` (99 rows; each replaces the earlier row
of the same scenario and seed set on the scoreboard). The numbers quoted in
the dated entries above are from the old loop and stay as the record of that
day; [RESULTS.md](RESULTS.md) shows the current rows.

1. Fine pointing does not depend on the fixes. For both tuned files the
   fine-pointing rows move by less than 1 % in every column, and no seed is
   lost before or after.
2. Acquisition counts move by a seed or two, as the plant-step entry
   predicted: the three-mode file goes from 5 to 4 flagged of 40 and from 2
   to 3 of 15; the eight-seed file stays at 0 of 40 and 0 of 15.
3. The conclusions about albedo, the VEML6031 array, the gyro temperature
   term and the as-built rows are unchanged: the same front ends hold or
   lose pointing, with counts within a few seeds of the earlier rows.
4. The full suite passes on the cluster at the same commit.

### Not started

Phase 1 rows 1a, 1b, 1d, 1e and 1g; phases 3 and 4. Phase 5 needs the
hardware.

### Open

- Acquisition results for `results/lqr_tuned_3mode.json` do not reconcile
  between its stored `train_metrics`, `results/lqi_fine.json` and later runs
  (seeds 0 and 1 reproduce the stored numbers; seed 101 diverges in
  acquisition). See section 10 of the notes. Cause not confirmed; the two
  items below are the candidates.
- The two acquisition measures overlap. The episode is 4 orbits (6.38 h), the
  lost flag looks at the last orbit (from 4.78 h) and the settle limit is 6 h.
  Of the 7 flagged seeds of the three-mode file (55 seeds), 6 met the settle
  condition during the last orbit. The 6-orbit rows (entry above) show those
  seeds were slow, not lost. Still open: the lost flag and the settle time
  both depend on the episode length and need definitions tied to a fixed
  window.
- Fixed 2026-11-18 (entry above): `scalar/control/tune.py` recorded knowledge
  and pointing error one step late (the estimate at t_k against the truth at
  t_k + dt), about |rate| x dt of extra knowledge error, 1.4° at the start of
  an acquisition and negligible in fine pointing.
- `scripts/estimator_levers.py` and `scripts/mechanism_gyro_bias.py` carry
  their own copies of the closed loop and do not get the plant switches.
