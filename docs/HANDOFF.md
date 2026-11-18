# Handoff: where things are

For a teammate picking this work up cold. This page is a map. It says where
each thing lives and which document owns each fact; it does not repeat the
facts. If this page and a linked document disagree, the linked document wins.

Written 2026-10-24.

## What this is and what state it is in

This repository simulates attitude determination and magnetic attitude control
for SCALAR, the WashU/PROVES 1U CubeSat, in Python on a laptop. The
determination half (estimators, sensor models, sensor calibration) is the
older and more settled part; its results are in
[determination-results.md](determination-results.md). The control half (magnetorquer
control, tuning, a learned sun-sensor front end, coil identification) is a
set of dated experiments written up in
[learned-estimation-and-control.md](learned-estimation-and-control.md), with
corrections added beside claims that later runs contradicted. All of it is
simulation: nothing is flight software, nothing has been run on hardware, and
several inputs are assumed rather than measured (see
[What to do next](#what-to-do-next)).
Every hardware number, with its source, and where bench results are typed
in: [`hardware.json`](../hardware.json) ([README, section 1](../README.md#1-working-with-the-code));
the hardware numbers that are not in it, and where they live:
[LIMITATIONS.md](LIMITATIONS.md#inputs-not-in-hardwarejson).

## What to do next

Written 2026-11-18. The jobs left once the work is handed over, in three
groups so different people can take them: bench measurements, decisions the
team has to make, and software work. In each group the most valuable job
comes first. The first three bench jobs are in order of the effect measured so
far ([row 1h](verification-loop-plan.md#2026-10-24--phase-1-row-1h-gyro-realism),
[notes sections 7-8](learned-estimation-and-control.md#nine-parameter-result-and-the-gyro-requirement-2026-09-13),
[row 1f](verification-loop-plan.md#2026-10-24--phase-1-row-1f-veml6031-sun-sensor-and-earth-albedo));
the order after that is a judgement made during the handover, not a measured ranking.
Every job comes from [LIMITATIONS.md](LIMITATIONS.md), a note in
[hardware.json](../hardware.json) or [the plan](verification-loop-plan.md);
the linked place has the detail.

Each job says what to do, why (the result that shows it), what you need,
where the result goes, and what to run afterwards. "The routine" is
[After you change a number](#after-you-change-a-number). Heavy runs:
[Where heavy runs go](#where-heavy-runs-go).

No code in the repository reads a recorded bench log. Where a job needs such
code it says so, and software job 2 lists it all.

### Bench measurements

On the V5e flight board and the driver board
([plan phase 5](verification-loop-plan.md#phase-5--replace-assumed-noise-with-measured-noise-people-not-agents)).

1. **Gyro drift with temperature.**
   - Do: log the gyro and the magnetometer at rest while the board's temperature is ramped, with the TMP112 temperature sensors logged alongside; fit each axis' output against temperature with a straight line.
   - Why: with the assumed drift the tuned controller loses pointing on nearly every seed, and the other gyro errors barely matter ([row 1h](verification-loop-plan.md#2026-10-24--phase-1-row-1h-gyro-realism)).
   - Needs: something that heats and cools the board. Nothing in the repository fits the slope from a log: someone has to write it (software job 2).
   - Goes to: `gyro.temp_coeff` and `magnetometer.temp_coeff`. A board temperature logged over an orbit (from a thermal model, later from flight) goes in `board_temperature.log_csv`, with `swing_amplitude` and `swing_phase` set to null; the CSV format is in that entry's note.
   - Then: `python3 scripts/gyro_realism.py --seeds 0-39` and `python3 scripts/mag_realism.py --seeds 0-39` (both read the file), then the routine.
2. **Gyro and magnetometer noise: a two-hour static log.**
   - Do: log the gyro and the magnetometer at rest for 2 h at 50 Hz or faster, and compute the Allan deviation of each gyro axis (the standard plot of noise against averaging time, which separates white noise from bias drift).
   - Why: the gyro's bias drift sets most of the fine-pointing error; the notes call this the most valuable bench number ([notes sections 7-8](learned-estimation-and-control.md#nine-parameter-result-and-the-gyro-requirement-2026-09-13)).
   - Needs: the board and a logger. There is no Allan-deviation code in the repository: someone has to write it (software job 2).
   - Goes to: `gyro.noise_density`, `gyro.bias_random_walk`, `gyro.initial_bias` (the log's mean), `gyro.lsb` (the step in the log, at the range the flight driver sets), `magnetometer.noise_std`, `magnetometer.lsb`.
   - Then: the routine. The closed-loop estimator keeps its own fixed noise settings, so this changes the simulated sensor, not the filter (software job 3).
3. **Light-sensor integration time, and the flight-software setting.**
   - Do: put the VEML6031 light sensors on a rotary stage under a lamp and find the integration time that survives full sun. Then have the flight software set it: this is a setting to change on the spacecraft, not only a number to enter here.
   - Why: at the default integration time every sunlit sample clips and the formula front end loses pointing; a shorter time does not clip ([row 1f](verification-loop-plan.md#2026-10-24--phase-1-row-1f-veml6031-sun-sensor-and-earth-albedo), [notes section 12](learned-estimation-and-control.md#12-the-real-sun-sensor-veml6031-counts-and-saturation-2026-10-24)).
   - Needs: a rotary stage, a lamp, the board; the flight-software team for the driver setting ([proves-core-reference](https://github.com/washu-satellite/proves-core-reference)).
   - Goes to: `sun_sensor.integration_time` (one of the values its note lists).
   - Then: the routine (the as-built rows use this value).
4. **Which way each light sensor faces.**
   - Do: first ask: the mechanical CAD and the flight-board layout give each sensor's direction. Failing that, sweep the assembled unit on the rotary stage under the lamp and record every sensor against angle; the same sweep gives each sensor's gain and noise ([notes section 9](learned-estimation-and-control.md#9-the-sun-sensor-geometry-is-assumed-not-known-2026-09-13), options 1 and 2).
   - Why: the face numbering disagrees across three files, and with the side faces mis-assigned the formula front end fails ([notes section 9](learned-estimation-and-control.md#9-the-sun-sensor-geometry-is-assumed-not-known-2026-09-13), [row 1f](verification-loop-plan.md#2026-10-24--phase-1-row-1f-veml6031-sun-sensor-and-earth-albedo)).
   - Needs: CAD and layout access, or the sweep. Nothing reads a sweep file and nothing fits a direction from one: someone has to write the loader and the fit (software job 2). `scalar/calibration/sun_onground.py` fits gain and roll-off from a sweep, but for the generic sensor and from arrays.
   - Goes to: `sun_sensor.normals` (one row per sensor, devicetree faces 0, 1, 2, 3, 5, 6, 7), `sun_sensor.gains`, `sun_sensor.noise`.
   - Then: the routine. The formula front end still assumes the nominal directions (software job 3); the sweep is also the network's training set (software job 4).
5. **Coils.**
   - Do: measure each coil channel's strength (dipole per amp), its resistance and the driver's output voltage. Phase 5 also asks for the static log of job 2 repeated with each coil at full current and with the radio transmitting, to see how much they disturb the magnetometer.
   - Why: every controller gain is for a coil set no source confirms, and no gain should be trusted until the coils are measured ([notes, where things should run](learned-estimation-and-control.md#where-things-should-run)).
   - Needs: the coils on the driver board, and decision 1 (which coil set is real).
   - Goes to: `coils.dipole_per_amp` and `coils.max_current` (driver voltage over coil resistance). A different voltage must also be edited in two places in code ([LIMITATIONS.md](LIMITATIONS.md#hardware-numbers-not-in-the-file-where-they-live-and-why)). The disturbance of the magnetometer has nowhere to go yet: there is no model of it (software job 6, row 1d).
   - Then: the routine. In flight the coils can also be measured from torque pulses ([notes section 3](learned-estimation-and-control.md#3-learning-the-magnetorquer-configuration), `identify_dipole_matrix_impulse` in `scalar/sysid/magnetorquer_id.py`); nothing reads a recorded pulse log yet.
6. **Magnetometer hard and soft iron.**
   - Do: rotate the board through all orientations in Earth's field and fit an ellipsoid to the readings.
   - Why: uncorrected, these errors degrade knowledge; the fit recovers most of it ([row 1c](verification-loop-plan.md#2026-10-24--phase-1-row-1c-magnetometer-realism)).
   - Needs: the board and a logger. The fit exists (`fit_magnetometer_ellipsoid` in `scalar/calibration/mag_onorbit.py`, from arrays); nothing reads a log file (software job 2).
   - Goes to: `magnetometer.hard_iron_bias`, `magnetometer.soft_iron`.
   - Then: the routine.
7. **Inertia and residual magnetism of the spacecraft.**
   - Do: get the inertia from a CAD mass model or a pendulum test of the flight unit; measure the assembled unit's own magnetic dipole with the coils off.
   - Why: the inertia's scale cannot be learned in flight, and an inertia error becomes the same error in the identified coil strength ([notes section 3](learned-estimation-and-control.md#3-learning-the-magnetorquer-configuration)).
   - Needs: CAD, or a pendulum rig; a magnetic-cleanliness setup.
   - Goes to: `body.inertia`, `body.residual_dipole`.
   - Then: the routine.
8. **Gyro scale factor and misalignment.**
   - Do: run the flight board on a rate table, each axis in both directions.
   - Why: last because its measured effect is small ([row 1h](verification-loop-plan.md#2026-10-24--phase-1-row-1h-gyro-realism), point 2).
   - Needs: a rate table.
   - Goes to: `gyro.scale_factor`, `gyro.misalignment`.
   - Then: `python3 scripts/gyro_realism.py --seeds 0-39`, then the routine. The estimator does not correct these (software job 3).

### Decisions the team has to make

1. **Which coil set is real.** The sources disagree; the runs use one choice ([notes, hardware](learned-estimation-and-control.md#what-the-hardware-actually-is), [magnetorquer-defects.md](magnetorquer-defects.md)). Goes to: the `coils` entries and their notes. Then: bench job 5, and a re-tune (software job 8).
2. **Pointing target: a fixed direction in space, or Earth-facing.** Every run holds a fixed inertial attitude (the target in `scalar/harness/runner.py`); an Earth-facing variant is a candidate, not built ([plan phase 4](verification-loop-plan.md#phase-4--redesign-only-where-phase-3-failed)). Goes to: software job 8.
3. **Pass/fail limits.** The 5 degree pointing limit and the coil-power limit are placeholders; where the knowledge limit comes from is open ([plan rule 2](verification-loop-plan.md#rules-of-the-loop)). Goes to: `THRESHOLDS` in `scalar/control/rl_tune.py`, and rule 2 of the plan for the knowledge limit. Then: `python3 scripts/report.py --write` (the verdicts change; no run needed). The tuner scores against the same limits (software job 8).
4. **Sensor and control rate, including the 30 s question.** The flight software reads sensors once a second; faster sensing helps a little ([notes section 11](learned-estimation-and-control.md#11-estimator-side-levers-on-the-attitude-estimate-half-2026-10-24)). Needs: the sweep of software job 6 (row 1e) first. Goes to: the flight software, and `dt` in `scalar/harness/spec.py`.
5. **Which processor runs the control law.** The sources say the payload processor; the notes argue for the main flight computer ([where things should run](learned-estimation-and-control.md#where-things-should-run)). Goes to: the flight software; nothing in this repository changes.
6. **Orbit and start date.** Once launch is known. Goes to: the `mission` entries of `hardware.json`. Then: the routine.

### Software work

1. **Check the two fixes in progress.** Knowledge and pointing error recorded one step late, and the corrected plant step becoming the default: their state is in [LIMITATIONS.md](LIMITATIONS.md#how-results-are-measured) (the second under [the spacecraft model](LIMITATIONS.md#what-the-spacecraft-model-leaves-out)). Both must be in before row 1e is run ([plan, Open](verification-loop-plan.md#open), [phase 2 entry](verification-loop-plan.md#2026-10-24--phase-2-independent-cross-check-done)). Then: `python3 -m pytest tests -q`.
2. **Tools that turn bench logs into numbers.** None exist. Needed: an Allan-deviation script for the static log (bench 2); a slope fit against temperature (bench 1); a reader for a lamp-sweep file with a fit for each sensor's direction and gain (bench 4); readers that feed a rotation log to `fit_magnetometer_ellipsoid` (bench 6) and a coil-pulse log to `identify_dipole_matrix_impulse` (bench 5). New scripts under `scripts/`; the file formats are for the writer to choose, stated at the top of each script. Reported during the handover, not written up elsewhere.
3. **Make the estimator use what is measured** ([LIMITATIONS.md](LIMITATIONS.md#what-the-sensors-and-estimator-leave-out)). Most valuable first: gyro temperature compensation from the TMP112 reading (the two candidates in [row 1h](verification-loop-plan.md#2026-10-24--phase-1-row-1h-gyro-realism), point 4, neither built); the closed-loop filter's noise settings matched to the measured noise (fixed in `scalar/control/tune.py`, defaults in `scalar/estimator/mekf.py`); the measured light-sensor directions in the formula front end (`scalar/sensors/veml6031.py`); the measured gyro scale factor and misalignment corrected. Then: the row script of the sensor concerned, and the routine.
4. **Train the network on real readings.** Today it learns only from this simulator's readings ([notes section 1](learned-estimation-and-control.md#1-a-neural-network-for-state-estimation--where-it-actually-helps)). Train it on the bench-4 sweep plus the simulated albedo model ([notes section 9](learned-estimation-and-control.md#9-the-sun-sensor-geometry-is-assumed-not-known-2026-09-13), option 2). `train_sun_mlp` in `scalar/learning/sun_mlp.py` builds its data from a simulated sensor only: a path from a sweep file has to be written, and a ledger row for the result added. `python3 scripts/nn_sun_sensor.py` exports the weights to C.
5. **Replay a recorded sensor log through the estimator.** Nothing does this: the estimator runs only on simulated readings (`scalar/scenario.py`, `run_mekf` in `scalar/estimator/mekf.py`). Reported during the handover, not written up elsewhere; needed before any estimator result is checked on real data.
6. **The plan rows not started** ([plan phase 1](verification-loop-plan.md#phase-1--fidelity-parallel-agents-each-independent-each-one-switch); the table names the files for each). In order of need: 1e, the sweep of sensor and control step that answers decision 4 (after job 1); 1a, the real field model in the closed loop; 1d, coil disturbance of the magnetometer and the measure-then-torque cycle (uses bench 5); 1b, disturbance torques; 1g, a gain table reused for days. Each row comes with a test showing its effect (plan rule 3) and adds ledger rows.
7. **Known problems** ([LIMITATIONS.md](LIMITATIONS.md#known-problems-not-yet-fixed), [plan, Open](verification-loop-plan.md#open)): tie the acquisition measures to a fixed window (`settle_time` in `scalar/control/tune.py`, the lost flag in `scalar/harness/metrics.py`); find why `lqr_tuned_3mode`'s acquisition results disagree; seed 29 in the 6-orbit run; give `scripts/estimator_levers.py` and `scripts/mechanism_gyro_bias.py` the shared closed loop; a ledger column saying which machine made a row; the gaps of the integral-term test and a test file for `scalar/control/lqi.py`.
8. **Plan phases 3 and 4.** After job 6 and the bench numbers: re-run every experiment with the new models at realistic settings ([phase 3](verification-loop-plan.md#phase-3--re-run-the-five-experiments-on-the-upgraded-harness-after-1)), then redesign only where it fails ([phase 4](verification-loop-plan.md#phase-4--redesign-only-where-phase-3-failed), which lists the candidates, the Earth-facing target among them). Re-tune with `scripts/rl_tune_lqr.py` and accept only what `scripts/verify_controller.py` passes. Heavy.

### After you change a number

1. Edit the entry in `hardware.json`: the value in the entry's units, `source` set to `measured` or `datasheet`, and the note saying what was measured, when and how.
2. `python3 -m pytest tests/test_hardware.py -q` checks the file; a mistake names the entry.
3. `python3 scripts/matrix.py --experiments hw --seeds 0-39` runs the as-built rows ([table G](RESULTS.md#g-as-built-the-tuned-controller-with-the-values-of-hardwarejson)). Heavy: see below.
4. `python3 scripts/report.py --write` regenerates `docs/RESULTS.md`.
5. Commit `hardware.json`, `results/ledger.csv` and `docs/RESULTS.md` together; a test fails while the ledger and `RESULTS.md` disagree.

### Where heavy runs go

A 40-seed closed-loop row is many simulations. The owner ran them on the
WashU engineering cluster (Slurm) with two scripts in `scripts/cluster/`:
`remote_run.sh` pushes the committed worktree to a bare repository in your
cluster home, builds a Python environment on a compute node, runs one command
there and prints the job id; `remote_wait.sh` waits until that job ends and
copies its output folder and log back. Set `HOST`, `PARTITION` and `ACCOUNT`
at the top of `remote_run.sh` and `HOST` in `remote_wait.sh`; the header of
each says how to call it. `remote_run.sh` runs only a committed, clean
worktree, so each row carries a real commit: commit the edited `hardware.json`
first, then the ledger rows and `docs/RESULTS.md` once they are back.

```bash
scripts/cluster/remote_run.sh . hw-rows 8 08:00:00 0 \
  python3 scripts/matrix.py --experiments hw --seeds 0-39 --workers 8 --ledger '$REMOTE_OUT/rows.csv'
scripts/cluster/remote_wait.sh hw-rows <job id> ~/scalar-out/hw-rows
tail -n +2 ~/scalar-out/hw-rows/rows.csv >> results/ledger.csv    # the rows, without the header line
```

- Limits on the student queue the owner used: 8 cores per job, 4 jobs at once, 8 h per job.
- Measured during the handover: one closed-loop case took about five times longer per core there than on a recent laptop.
- Numbers from two machines differ in the last digits, and acquisition results visibly; compare rows from one machine only ([where runs are made](verification-loop-plan.md#2026-10-24--where-runs-are-made-and-what-that-does-to-the-rows)).

## Where things are

### Code: `scalar/`

| Thing | Path |
|---|---|
| Frame and quaternion conventions, the one definition everything else follows | `scalar/attitude.py` |
| Physical constants and units | `scalar/constants.py` |
| Time conversions | `scalar/timeutil.py` |
| Rigid-body rotation, free and with applied torque | `scalar/dynamics.py` |
| One determination run assembled: truth, environment, sensor readings | `scalar/scenario.py` |
| Monte Carlo comparison of estimators | `scalar/montecarlo.py` |
| Orbit, Sun and eclipse, magnetic field (dipole and IGRF) | `scalar/environment/` |
| Gyro, magnetometer, sun sensor (generic and VEML6031), star tracker | `scalar/sensors/` |
| TRIAD, MEKF, USQUE, TRIAD-only baseline | `scalar/estimator/` |
| Magnetometer and sun-sensor calibration, three methods | `scalar/calibration/` |
| Magnetorquer model | `scalar/actuators/` |
| Fixed-gain LQR, periodic LQR and mode switching, LQI, closed-loop simulation, tuner | `scalar/control/` |
| Neural-network sun-sensor front end | `scalar/learning/` |
| Coil and inertia identification | `scalar/sysid/` |

Every module starts with a comment block that says what it models and why.

### Scripts: `scripts/`

The right-hand column is the first line of each script's own description.

| Script | What it answers |
|---|---|
| `run_mekf.py` | Run one determination scenario through the MEKF and report accuracy. |
| `monte_carlo.py` | Monte-Carlo estimator comparison for the SCALAR determination sim. |
| `realism_sweep.py` | Expected MEKF accuracy vs systematic (unmodeled) error level. |
| `calibration_compare.py` | Compare sensor-calibration techniques and their effect on MEKF knowledge error. |
| `detumble_lqr.py` | Closed-loop LQR magnetorquer detumble demo. |
| `nn_sun_sensor.py` | Experiment 1: a learned sun-sensor front end for the MEKF. |
| `absolute_without_gps.py` | Experiment 2: absolute attitude with no GPS — how good must position/time be? |
| `sysid_magnetorquer.py` | Experiment 3: learn the magnetorquer configuration from torque-pulse bursts. |
| `lqr_tune.py` | Experiment 4: LQR for the identified coils, with the field inside the design. |
| `closed_loop_nn.py` | Experiment 5: the whole chain — learned sun front end -> MEKF -> periodic LQR. |
| `rl_tune_lqr.py` | Tune LQR weights + mode thresholds by cross-entropy method against the criteria. |
| `verify_controller.py` | Verify a tuned controller on HELD-OUT seeds and robustness variants. Exit 1 on any FAIL. |
| `ablation_pointing.py` | Ablation of the fine-pointing error: who owns the 7.4 deg? |
| `mechanism_gyro_bias.py` | Causal test: is fine-pointing error gyro-bias error x the controller's rate gain? |
| `lqi_fine.py` | Does an integral state on attitude error (fine-mode LQI) close the gyro-bias rate path? |
| `estimator_levers.py` | Estimator-side levers on the gyro-bias "attitude-estimate path". |
| `veml6031_baseline.py` | Config B with the real sun sensor (7x VEML6031X00) instead of the ideal cube. |
| `cluster/remote_run.sh`, `cluster/remote_wait.sh` | Submit one simulation command to the WashU engineering cluster; wait for it and fetch its outputs ([Where heavy runs go](#where-heavy-runs-go)). |

Which section of the notes reports each script's result: the
[Files list](learned-estimation-and-control.md#files) in the notes.

### Result files: `results/`

| File | Written by |
|---|---|
| `lqr_tuned.json` | `scripts/rl_tune_lqr.py`, its default output name |
| `lqr_tuned2.json`, `lqr_tuned_cal.json`, `lqr_tuned_3mode.json` | `scripts/rl_tune_lqr.py`, with the output name given as its fourth argument |
| `rl_tune.log`, `rl_tune2.log`, `rl_tune_cal.log`, `rl_tune_3mode.log` | the screen output of those four tuner runs, saved by hand; the last line of each names the JSON file it goes with |
| `ablation_pointing.csv` | `scripts/ablation_pointing.py` |
| `mechanism_gyro_bias.json` | `scripts/mechanism_gyro_bias.py` |
| `lqi_fine.json` | `scripts/lqi_fine.py` |
| `estimator_levers.json` | `scripts/estimator_levers.py`; each run adds to the file instead of replacing it |
| `veml6031_baseline.json` | `scripts/veml6031_baseline.py` |

`lqr_tuned_3mode.json` is the controller every later study loads
(`ablation_pointing`, `mechanism_gyro_bias`, `lqi_fine`, `estimator_levers`,
and through it `veml6031_baseline`).

### Plots and exports: `outputs/`

| File | Written by |
|---|---|
| `mekf_error.png` | `scripts/run_mekf.py` |
| `monte_carlo.png` | `scripts/monte_carlo.py` |
| `detumble_lqr.png` | `scripts/detumble_lqr.py` |
| `lqr_two_mode.png` | `scripts/lqr_tune.py` |
| `sun_mlp_weights.h` | `scripts/nn_sun_sensor.py` |
| `lqr_estimated_loop.png` | no script in the repository writes a file of this name |

`scripts/calibration_compare.py` also writes a table and a plot here; they are
not in the repository. Git ignores new files in `outputs/` and new `.png` and
`.csv` files (`.gitignore`), so the files above were added on purpose.
Re-running a script overwrites the file of the same name, and git then shows
it as modified.

### Documents: `docs/`

| File | What it is |
|---|---|
| `HANDOFF.md` | this page |
| `learned-estimation-and-control.md` | the experiment notes for the control half: sections 1 to 12, dated, with corrections in place. The home of every control, learning and identification number |
| `learned-estimation-and-control.html` | a one-page formatted version of sections 1 to 5 of the notes as they stood on 2026-09-11. Not updated since; the `.md` is the source |
| `magnetorquer-defects.md` | what can go wrong with hand-built coils, how big, and what absorbs it |
| `verification-loop-plan.md` | the plan for checking the notes' conclusions: rules, pass/fail limits, phases |
| `LIMITATIONS.md` | every known limitation, fix needed and undecided question, one line each with a link; and the inputs that are not in `hardware.json` |
| `RESULTS.md` | the results scoreboard: the ledger rows as tables, each against its pass/fail limit. Written by `python3 scripts/report.py --write`; not edited by hand |
| `determination-results.md` | the determination results (estimator accuracy, estimator comparison, sensor calibration), moved unchanged from the README |

### Tests: `tests/`

One file per area, named after what it tests (`test_mekf.py`,
`test_periodic_lqr.py`, `test_veml6031.py`, and so on); the first line of each
file says what it covers. There is no test file for `scalar/control/lqi.py`.
Current test status is in the notes, at the end of
[section 6](learned-estimation-and-control.md#6-tuning-by-reward-verifying-by-criteria-added-2026-09-12).

## How to run

From the repository root:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python3 -m pytest tests -q
python3 scripts/report.py
python3 scripts/run_mekf.py
python3 scripts/lqr_tune.py
python3 scripts/verify_controller.py results/lqr_tuned_3mode.json
```

The first two lines install the packages; `pytest` and `torch` are in
`requirements-dev.txt`. The third runs the tests. The fourth prints the results
scoreboard: every result row in `results/ledger.csv` as a table, with PASS or
FAIL against its limit, under a short headline; it runs no simulation, and the
test run ends with the headline. Then: one determination run, the
periodic-LQR demonstration, and the pass/fail check of the tuned controller
(it exits with status 1 when a limit is missed). What each line needs, the
rest of the scripts, and which of them were re-run on 2026-10-24 are in the
[README](../README.md#1-working-with-the-code).

## What is established and what is not

Pointers only. The section named owns the claim and its numbers.

Determination, in [determination-results.md](determination-results.md):

- how well the MEKF knows attitude, with and without the errors that do not
  average out: [Result](determination-results.md#result)
- how the estimators differ and when it matters:
  [Estimators](determination-results.md#estimators--what-the-comparison-shows)
- what each calibration method buys:
  [Sensor calibration](determination-results.md#sensor-calibration)

Control, learning and identification, in the notes:

- where a neural network helps the estimator and where it does not:
  [section 1](learned-estimation-and-control.md#1-a-neural-network-for-state-estimation--where-it-actually-helps)
- how good position and time must be without GNSS:
  [section 2](learned-estimation-and-control.md#2-absolute-attitude-without-gps)
- how the coil configuration can be learned in flight, and what cannot be:
  [section 3](learned-estimation-and-control.md#3-learning-the-magnetorquer-configuration)
- which controller points and which does not, and the sign error found in the
  first one:
  [section 4](learned-estimation-and-control.md#4-lqr-parameters-that-point)
- the full chain with the network in front:
  [section 5](learned-estimation-and-control.md#5-the-whole-chain-nn---mekf---periodic-lqr)
- the pass/fail limits, the tuner, and the result on seeds it never saw:
  [section 6](learned-estimation-and-control.md#6-tuning-by-reward-verifying-by-criteria-added-2026-09-12)
- magnetometer calibration in the loop, and the gyro requirement that follows:
  [section 7](learned-estimation-and-control.md#7-magnetometer-calibration-in-the-loop-added-2026-09-13)
- which error source owns the fine-pointing error, and by what path:
  [section 8](learned-estimation-and-control.md#8-ablation-and-mechanism-2026-09-13-two-opus-agents-scriptsablation_pointingpy-scriptsmechanism_gyro_biaspy)

Tried and did not work, or did not close the gap:

- the integral term (LQI) proposed in sections 7 and 8:
  [section 10](learned-estimation-and-control.md#10-lqi-in-fine-mode-tested-it-does-not-cancel-the-rate-path-2026-10-24).
  The two earlier claims carry dated corrections where they stand.
- changes on the estimator side (filter setting, faster sampling, the network):
  [section 11](learned-estimation-and-control.md#11-estimator-side-levers-on-the-attitude-estimate-half-2026-10-24)

Assumed, not known:

- sun-sensor geometry:
  [section 9](learned-estimation-and-control.md#9-the-sun-sensor-geometry-is-assumed-not-known-2026-09-13);
  what the real sensor's saturation does:
  [section 12](learned-estimation-and-control.md#12-the-real-sun-sensor-veml6031-counts-and-saturation-2026-10-24)
- the hardware list the runs are built on, and what is disputed in it:
  [What the hardware actually is](learned-estimation-and-control.md#what-the-hardware-actually-is)
- what the simulation leaves out, every known open problem, and the
  decisions not yet made:
  [LIMITATIONS.md](LIMITATIONS.md)

How the conclusions are meant to be checked:
[verification-loop-plan.md](verification-loop-plan.md).

### Verification status

Owned by the [Status section of the plan](verification-loop-plan.md#status),
which is dated and names the result rows behind each statement. As of
2026-10-24: the harness and its 40-seed baseline exist (phase 0), the
independent cross-check against Basilisk is done (phase 2), and the fidelity
switches and the full re-run (phases 1, 3, 4) are not.

| Thing | Path |
|---|---|
| Harness: one scenario record, one runner, fixed metrics | `scalar/harness/` |
| Runs the scenario matrix and appends result rows | `scripts/matrix.py` |
| The result ledger, one row per scenario and seed set | `results/ledger.csv` |
| Cross-check against Basilisk, one command | `scripts/crosscheck_basilisk.py` |
| Cross-check numbers | `results/crosscheck_basilisk.json` |
| Cross-check gate table and discrepancy list | [crosscheck-basilisk.md](crosscheck-basilisk.md) |

The cross-check needs Basilisk (`pip install bsk`), which is not in
`requirements.txt`.

## Related places and people

- Team flight-software repository:
  [github.com/washu-satellite/proves-core-reference](https://github.com/washu-satellite/proves-core-reference)
- Jesse Myoung: owner of this repository.
- Ben Cook: the sensor calibration suite (commit `19f5d75`).
