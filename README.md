# SCALAR attitude determination and control simulation

A Python simulation of two jobs for SCALAR, the WashU/PROVES 1U CubeSat:
working out which way the spacecraft is pointing (its attitude) from its gyro,
magnetometer and sun sensors, and turning it with magnetic coils using that
estimate. It is a study tool that runs on a laptop. None of it is flight
software and none of it has been checked against hardware.

Results: [docs/RESULTS.md](docs/RESULTS.md). Map of every file and every open
question: [docs/HANDOFF.md](docs/HANDOFF.md).

## 1. Working with the code

Needs Python 3.9 or later (checked with 3.13; the WashU cluster runs it with
3.9). Run everything from the repository root.

```bash
pip install -r requirements.txt       # numpy, scipy, matplotlib, ppigrf (Earth's magnetic field model)
pip install -r requirements-dev.txt   # pytest, and torch (only for the sun-sensor network)
python3 -m pytest tests -q            # the tests; ends with the scoreboard's headline (--no-scoreboard leaves it out)
python3 scripts/report.py             # headline, then every table, in under a second; runs no simulation
python3 scripts/report.py --write     # the same, and regenerates docs/RESULTS.md
python3 scripts/matrix.py --list      # the scenarios a simulation run can add rows for; runs nothing
```

- **Adding a result.** `scripts/matrix.py` runs scenarios and appends one row
  per scenario and set of seeds to `results/ledger.csv` (a seed is one repeat
  of a run with different random noise). Then run `report.py --write` and
  commit both files; a test fails while they disagree. The rules, and the
  commands behind the existing rows: [docs/verification-loop-plan.md](docs/verification-loop-plan.md).
- **Entering hardware results.** Bench numbers go in `hardware.json` at the
  repository root, one entry per number: `value` (null if unknown), `units`
  (fixed; convert the number), `source` and `note`. Source is one of three
  words: `assumed`, `datasheet`, `measured`. The orbit and start date are in
  the same file, in the group `mission`. Then run
  `python3 -m pytest tests/test_hardware.py -q` (checks the file), the as-built
  rows `python3 scripts/matrix.py --experiments hw --seeds 0-39`, and
  `report.py --write`. What follows the file: the as-built rows (table G) and
  the realism scripts (`scripts/*_realism.py`). What stays frozen: experiments
  1-6 and `--gate`, which never read it.
- **Other scripts.** Each starts with a comment saying what it tests and how
  to call it. Re-run on 2026-10-24 without error, ten minutes or less each:
  `run_mekf`, `monte_carlo`, `realism_sweep`, `calibration_compare`,
  `detumble_lqr`, `lqr_tune`, `verify_controller`, `nn_sun_sensor`,
  `sysid_magnetorquer`, `closed_loop_nn`, `absolute_without_gps`. Not re-run
  (many simulations each, and each overwrites a file in `results/`):
  `rl_tune_lqr`, `ablation_pointing`, `lqi_fine`, `estimator_levers`,
  `veml6031_baseline`.
- **Conventions.** Defined in `scalar/attitude.py` only: the quaternion is
  scalar-last `[x, y, z, w]`, inertial frame to body frame. The attitude error
  vector shrinks at the body rate, so a controller feeding it back negates it.

## 2. Where to read

| Where | What is there |
|---|---|
| `scalar/*.py` | frames and quaternions (`attitude.py`); how a rigid body rotates (`dynamics.py`); one estimator run put together, and many compared (`scenario.py`, `montecarlo.py`); `hardware.json` read and checked (`hardware.py`) |
| `scalar/environment/` | the simulated world: orbit, Sun and eclipse (Earth's shadow), Earth's magnetic field as a simple dipole and as IGRF, the standard published model |
| `scalar/sensors/` | gyro (rotation rate), magnetometer (magnetic field), sun sensor (a generic six-face one and the VEML6031 light sensors), star tracker |
| `scalar/estimator/` | attitude estimators: TRIAD (attitude from two measured directions at one instant) and the MEKF, the main one, a Kalman filter that blends gyro, magnetometer and sun sensor |
| `scalar/calibration/` | three ways of measuring and removing magnetometer and sun-sensor errors |
| `scalar/actuators/` | the magnetic coils (magnetorquers): Earth's field pushes on an energised coil and turns the spacecraft |
| `scalar/control/` | controllers: LQR (feedback gains from a cost on error and effort), periodic LQR (a different gain for each point of the orbit), mode switching, LQI (adds accumulated past error); the closed-loop simulation in `tune.py`; the automatic tuner and the pass/fail limits in `rl_tune.py` |
| `scalar/learning/` | a small neural network that turns raw sun-sensor readings into a Sun direction |
| `scalar/sysid/` | working out each coil's strength and direction from recorded motion |
| `scalar/harness/` | one record per scenario, one runner, the fixed set of metrics, the ledger format |
| `scripts/report.py` | the results scoreboard |
| `scripts/matrix.py`, `scripts/*_realism.py` | run scenarios and append ledger rows; the realism scripts run the gyro, magnetometer and sun-sensor error scenarios |
| `scripts/run_mekf.py`, `scripts/lqr_tune.py` | one orbit through the MEKF; periodic LQR holding a target, then reaching it from a large error |
| `scripts/rl_tune_lqr.py`, `scripts/verify_controller.py` | the tuner; the pass/fail check of a tuned controller on seeds the tuner never saw |
| `scripts/crosscheck_basilisk.py` | compares this simulator with Basilisk, an independent one (needs `pip install bsk`) |
| `tests/` | one file per area, named after what it tests |

File-by-file map, with every script and result file:
[docs/HANDOFF.md](docs/HANDOFF.md#where-things-are). Dated experiment notes
behind the control half: [docs/learned-estimation-and-control.md](docs/learned-estimation-and-control.md).

## 3. What the scoreboard shows

`scripts/report.py` prints a headline grid, then the tables below.
[docs/RESULTS.md](docs/RESULTS.md) is the same saved; the numbers live there,
not here.

| Table | What it shows |
|---|---|
| Headline | one line per scenario: the tuned controller (`lqr_tuned_3mode`) plain, with each model error switched on, and as built; then the second tuned controller (`lqr_tuned_ms`) plain. One column per sun front end (the step that turns sun-sensor readings into a Sun direction). Each cell: seeds lost, typical pointing error, PASS or FAIL |
| A1 | controllers found by the automatic tuner, starting far from the target and tumbling: how many seeds lost pointing, and how long settling took |
| A1L | the same for two of those controllers, with a run of 6 orbits instead of 4 |
| A2, A2b | the same controllers starting near the target: seeds lost and pointing error (angle between where the spacecraft points and the target); then top rotation rate, share of time a coil is at full output, and coil power |
| B | one model error switched off, on, and on with its fix: magnetometer errors, the plant step (how the truth is moved forward in time), gyro errors, the real sun sensor (VEML6031) and its integration time, sunlight reflected by Earth (albedo), and wrong guesses of which way each sun sensor faces |
| C1, C2 | the controller designs picked by hand in September, with the estimator in the loop and with the true attitude fed in |
| D | this simulator against Basilisk, one row per compared quantity and run |
| E | the estimator alone: knowledge error (angle between estimated and true attitude) |
| G | the tuned controller with the values of `hardware.json` applied, all of them and only the non-assumed ones |
| H | every entry of `hardware.json`: value, units, source |
| Not shown | scenarios whose result has no column in the ledger, listed by name |

- Each row gives the typical seed (median across seeds) and the bad seed (95th
  percentile). The last column is the scenario's name in the ledger.
- PASS means the value is at or under its limit, and the scoreboard says no
  more than that. The limits are `THRESHOLDS` in `scalar/control/rl_tune.py`
  and the knowledge limit in rule 2 of [the plan](docs/verification-loop-plan.md#rules-of-the-loop).
  What the rows are taken to mean: [the plan's Status section](docs/verification-loop-plan.md#status).
- Older results, not in the ledger: estimator comparison and sensor
  calibration in [docs/determination-results.md](docs/determination-results.md);
  control, learning and coil identification in the notes.

## 4. Hardware numbers and other inputs

Every hardware number a bench test, a CAD model or a datasheet can replace,
and the mission numbers (orbit, start date), is an entry of
[hardware.json](hardware.json). Each entry's note says where the number comes
from and what replaces it; table H lists them. Almost all are assumed today.

Hardware numbers that are not in the file, where they live, and why
(`NOT_IN_FILE` in `scalar/hardware.py` lists these, with what the file cannot
change in the estimator):

- Coil current and radio leaking into the magnetometer: no model of it exists yet.
- The VEML6031 gain setting and its brightness ceilings: a datasheet table in
  `scalar/sensors/veml6031.py`. The file picks a row of it through
  `sun_sensor.integration_time`; only one gain setting is coded.
- The generic six-photodiode sun sensor (`scalar/sensors/sun_sensor.py`) and
  its "flight-like" variant (`flight_like_sensor` in `scalar/learning/sun_mlp.py`):
  models of a generic sensor used by older experiments, not of the VEML6031 array.
- The coil drive voltage: inside `coils.max_current` (voltage over coil
  resistance). The coil-power figure has its own copy in
  `scalar/harness/metrics.py` (`COIL_VOLTAGE`) and `scalar/control/rl_tune.py`
  (`evaluate`); a new voltage means editing all three.
- Star tracker numbers (`scalar/sensors/star_tracker.py`): SCALAR has no star
  tracker; it is used only by the ground calibration comparison.
- The two-rods-and-one-coil set (`scalar/control/lqr.py`, `scripts/detumble_lqr.py`):
  an early demonstration set that no source matches.

Inputs that are not hardware numbers:

- Pass/fail limits: `THRESHOLDS` in `scalar/control/rl_tune.py`, and the
  knowledge limit in [plan rule 2](docs/verification-loop-plan.md#rules-of-the-loop).
- Estimator noise settings: the `MEKF` defaults in `scalar/estimator/mekf.py`.
  The closed loop builds its filter in `scalar/control/tune.py` with fixed
  settings; they do not follow `hardware.json`.
- Sensor, estimator and control rate: `dt` in `scalar/harness/spec.py`, a
  flight-software setting.
- Start attitude and rate, albedo level and run length of each scenario: test
  cases, in `scalar/harness/baseline.py`.
- Controller weights: tuner outputs, `results/lqr_tuned*.json`.
- Physical constants (`scalar/constants.py`) and the constants of the field
  and Sun models (`scalar/environment/`).
- The orbits of experiments 1-6: each keeps the one it was published with.
  The `mission` entries set the as-built rows' orbit only.

## 5. Limitations

Every known limitation and open problem, one line each, with where it is
discussed.

**What the simulated world leaves out**

- Circular orbit, no air drag, no gravity terms beyond J2 (Earth's flattening); the Sun's position comes from a short formula (`scalar/environment/`).
- No disturbance torques: gravity gradient, air drag, sunlight pressure ([plan row 1b](docs/verification-loop-plan.md#phase-1--fidelity-parallel-agents-each-independent-each-one-switch), not started).
- The control studies use the simple dipole field for both the truth and the controller's model; it is far from the real field ([plan row 1a](docs/verification-loop-plan.md#phase-1--fidelity-parallel-agents-each-independent-each-one-switch), not started; [cross-check, model differences](docs/crosscheck-basilisk.md#model-differences-not-passfail)).
- Earth's shadow is a cylinder with no half-shadow (penumbra), and Earth is a sphere, not flattened; both move the eclipse edges by seconds ([discrepancy B](docs/crosscheck-basilisk.md#b-eclipse-cylinder-against-cone-with-penumbra--model-difference), [model differences](docs/crosscheck-basilisk.md#model-differences-not-passfail)).
- The inertial frame is the equator and equinox of the date, not J2000; an attitude or orbit taken into or out of the simulation in J2000 needs a correction for the slow turning of Earth's axis (precession) ([discrepancy A](docs/crosscheck-basilisk.md#a-our-eci-is-the-equator-and-equinox-of-the-date-not-j2000--ours-labelling)). <!-- verify after p2-bugs -->
- Earth's rotation differs slightly from Basilisk's; which is closer is not settled ([discrepancy E](docs/crosscheck-basilisk.md#e-earth-rotation-differs-by-0086-after-a-is-removed--basilisks-model)).
- A controller gain table computed once and reused for days has not been tested ([plan row 1g](docs/verification-loop-plan.md#phase-1--fidelity-parallel-agents-each-independent-each-one-switch), not started).

**What the spacecraft model leaves out** (the plant: the simulated spacecraft and its coils)

- The true attitude is moved forward with a low-order step whose error grows with rotation rate and step size; the plant substeps reduce it, they do not remove it ([discrepancy C](docs/crosscheck-basilisk.md#c-truth-attitude-is-advanced-with-a-second-order-step--ours-numerical)).
- The plant holds the coil dipole over each control step, in two substeps, so the torque follows the field as in a real coil; the old step, which held the torque, is kept only for the runs that reproduce the notes ([discrepancy D](docs/crosscheck-basilisk.md#d-the-plant-freezes-the-torque-for-a-whole-control-step--ours-physical), [plant-step row](docs/verification-loop-plan.md#2026-10-24--phase-1-plant-step-row-added-after-the-cross-check)). <!-- verify after p2-bugs -->
- No coil electrical behaviour, no coil field at the magnetometer, no measure-then-torque cycle in `scalar/` ([plan row 1d](docs/verification-loop-plan.md#phase-1--fidelity-parallel-agents-each-independent-each-one-switch), not started; [magnetorquer-defects.md](docs/magnetorquer-defects.md)).
- No temperature effects other than the gyro and magnetometer offset drift (table B, off by default).
- Coil saturation was never reached in the cross-check, so that path is unchecked against Basilisk ([not covered](docs/crosscheck-basilisk.md#not-covered)).

**What the sensors and estimator leave out**

- Bench values enter as uncorrected sensor errors: the estimator does not compensate a measured gyro scale factor or misalignment (`NOT_IN_FILE` in `scalar/hardware.py`).
- Measured sun-sensor normals change the simulated truth only; the analytic reconstruction still assumes the nominal layout (`scalar/harness/spec.py`, sun-sensor notes).
- The closed-loop estimator's noise settings are fixed, not retuned to measured noise (`scalar/control/tune.py`).
- The as-built rows use the analytic sun front end (a formula that compares opposite faces) and the VEML6031 array at its default integration time, at which it clips in direct sun ([notes section 12](docs/learned-estimation-and-control.md#12-the-real-sun-sensor-veml6031-counts-and-saturation-2026-10-24), [row 1f](docs/verification-loop-plan.md#2026-10-24--phase-1-row-1f-veml6031-sun-sensor-and-earth-albedo)).
- Gyro clipping at full scale is not modelled (`scalar/sensors/gyro.py`).
- The loop has no tolerance to a gyro bias that follows board temperature; raising the filter's bias noise does not help, and temperature compensation is not built ([row 1h](docs/verification-loop-plan.md#2026-10-24--phase-1-row-1h-gyro-realism)).
- A magnetometer offset drift ten times the assumed one defeats the plain ellipsoid calibration (the magnetometer correction fitted over one orbit) ([row 1c](docs/verification-loop-plan.md#2026-10-24--phase-1-row-1c-magnetometer-realism)).
- Which way each sun sensor faces is unknown, and the face numbering disagrees across three files; with the side faces mis-assigned the analytic front end fails ([notes section 9](docs/learned-estimation-and-control.md#9-the-sun-sensor-geometry-is-assumed-not-known-2026-09-13), [row 1f](docs/verification-loop-plan.md#2026-10-24--phase-1-row-1f-veml6031-sun-sensor-and-earth-albedo)).
- On the one seed tried, acquisition (turning from far off the target) without the sun sensor misses its settle limit ([notes section 9](docs/learned-estimation-and-control.md#9-the-sun-sensor-geometry-is-assumed-not-known-2026-09-13)).
- Not covered for the VEML6031: its datasheet gain tolerance, rebuilding clipped readings with albedo present, and the real face normals; that rebuild needs the Sun's brightness to be known ([notes section 12](docs/learned-estimation-and-control.md#12-the-real-sun-sensor-veml6031-counts-and-saturation-2026-10-24)).
- With albedo the analytic front end cannot be used for pointing; the learned one still loses some seeds on the VEML6031 at high albedo ([row 1f](docs/verification-loop-plan.md#2026-10-24--phase-1-row-1f-veml6031-sun-sensor-and-earth-albedo)).
- Every network is trained and tested on this simulator's own sensor and albedo models: the rows show the method can work, not that a given network will ([row 1f, point 6](docs/verification-loop-plan.md#2026-10-24--phase-1-row-1f-veml6031-sun-sensor-and-earth-albedo)).
- The learned front end combined with the faster filter settings loses pointing on some seeds; cause not investigated ([notes section 11](docs/learned-estimation-and-control.md#11-estimator-side-levers-on-the-attitude-estimate-half-2026-10-24)).
- The best estimator-side change found needs a faster sensor rate than the flight software has ([notes section 11](docs/learned-estimation-and-control.md#11-estimator-side-levers-on-the-attitude-estimate-half-2026-10-24)).

**How results are measured**

- Knowledge and pointing error are recorded at matching instants; ledger rows made before that fix have the estimate one step late, which inflates knowledge error at high rotation rates ([plan, Open](docs/verification-loop-plan.md#open)). <!-- verify after p2-bugs -->
- A lost-seed count from 40 seeds is a rate with a wide spread: a tiny plant change reshuffles which seeds are lost ([plant-step row](docs/verification-loop-plan.md#2026-10-24--phase-1-plant-step-row-added-after-the-cross-check)).
- Rows made on the laptop and on the cluster are not digit-identical, and the ledger has no column saying which is which; compare lost-seed counts and acquisition figures only within one platform ([where runs are made](docs/verification-loop-plan.md#2026-10-24--where-runs-are-made-and-what-that-does-to-the-rows)).
- The two acquisition measures (lost flag, settle time) overlap and both depend on the run length; the settle limit cannot be judged until they use a fixed window ([plan, Open](docs/verification-loop-plan.md#open), [eight-seed entry](docs/verification-loop-plan.md#2026-10-24--tuning-on-eight-seeds-resultslqr_tuned_msjson)).
- The sun-vector errors of experiment 1 and the coil identification errors of experiment 3 have no ledger column (table "Not shown").
- `scripts/estimator_levers.py` and `scripts/mechanism_gyro_bias.py` carry their own copies of the closed loop and do not get the plant switches ([plan, Open](docs/verification-loop-plan.md#open)).
- The cross-check against Basilisk does not cover sensor noise, the gyro, the magnetometer, the estimator, the controllers, the residual dipole or disturbance torques ([not covered](docs/crosscheck-basilisk.md#not-covered)).
- The notes' single-seed closed-loop numbers (sections 4 and 5) come from the good part of the spread; the plan's assumed seed holds only for experiments 4 and 5 ([phase 0](docs/verification-loop-plan.md#2026-10-24--phase-0-harness-done)).
- Four statements in the notes did not reproduce; each has a dated correction ([phase 0](docs/verification-loop-plan.md#2026-10-24--phase-0-harness-done)). The integral-term recommendation of notes sections 7 and 8 is withdrawn ([notes section 10](docs/learned-estimation-and-control.md#10-lqi-in-fine-mode-tested-it-does-not-cancel-the-rate-path-2026-10-24)).
- The older determination tables do not reproduce exactly ([determination-results.md](docs/determination-results.md)).

**Known problems not yet fixed**

- Every tuned controller misses the fine-pointing limit on the typical seed in its own runs; the notes trace the error to the estimator, mainly the gyro bias drift ([notes sections 7-8](docs/learned-estimation-and-control.md#nine-parameter-result-and-the-gyro-requirement-2026-09-13), [experiment 6](docs/verification-loop-plan.md#2026-10-24--tuned-controllers-at-40-seeds-ledger-experiment-6)).
- Acquisition results for `lqr_tuned_3mode` disagree between its stored metrics, `results/lqi_fine.json` and later runs; cause not confirmed ([plan, Open](docs/verification-loop-plan.md#open), [notes section 10](docs/learned-estimation-and-control.md#10-lqi-in-fine-mode-tested-it-does-not-cancel-the-rate-path-2026-10-24)).
- One seed never meets the settle condition in the 6-orbit run, for either tuned file; not investigated ([eight-seed entry](docs/verification-loop-plan.md#2026-10-24--tuning-on-eight-seeds-resultslqr_tuned_msjson)).
- The integral-term test left gaps: weaker settings untried on the isolated path, the other weights not re-tuned with it, and no test file for `scalar/control/lqi.py` ([notes section 10](docs/learned-estimation-and-control.md#10-lqi-in-fine-mode-tested-it-does-not-cancel-the-rate-path-2026-10-24)).
- Not started: plan rows 1a, 1b, 1d, 1e, 1g and phases 3 and 4; phase 5 needs the hardware ([plan, Not started](docs/verification-loop-plan.md#not-started)).

**Decisions not yet made**

- Pointing target: every run holds a fixed inertial attitude; an Earth-facing variant is a candidate, not built ([plan phase 4](docs/verification-loop-plan.md#phase-4--redesign-only-where-phase-3-failed)).
- Which coil set is real: the sources disagree, the runs use one choice, and no gain should be trusted until the coils are measured ([notes, hardware](docs/learned-estimation-and-control.md#what-the-hardware-actually-is), [where things should run](docs/learned-estimation-and-control.md#where-things-should-run)).
- The 5 degree pointing limit and the coil-power limit are placeholders, and the knowledge limit's origin is open ([plan rule 2](docs/verification-loop-plan.md#rules-of-the-loop)).
- Sensor and control rate, including the 30 s question: the sweep that answers it is not run ([plan row 1e](docs/verification-loop-plan.md#phase-1--fidelity-parallel-agents-each-independent-each-one-switch)).
- Where the control law runs: the sources say the payload processor; the notes argue for the main flight computer ([where things should run](docs/learned-estimation-and-control.md#where-things-should-run)).
- The orbit and start date: the `mission` entries of `hardware.json` are assumed.
