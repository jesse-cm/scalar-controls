# Handoff: where things are

For a teammate picking this work up cold. This page is a map. It says where
each thing lives and which document owns each fact; it does not repeat the
facts. If this page and a linked document disagree, the linked document wins.

Written 2026-10-24.

## What this is and what state it is in

This repository simulates attitude determination and magnetic attitude control
for SCALAR, the WashU/PROVES 1U CubeSat, in Python on a laptop. The
determination half (estimators, sensor models, sensor calibration) is the
older and more settled part; its results are in the
[README](../README.md#determination-results). The control half (magnetorquer
control, tuning, a learned sun-sensor front end, coil identification) is a
set of dated experiments written up in
[learned-estimation-and-control.md](learned-estimation-and-control.md), with
corrections added beside claims that later runs contradicted. All of it is
simulation: nothing is flight software, nothing has been run on hardware, and
several inputs are assumed rather than measured (see
[Open items](#open-items-that-need-hardware-or-a-decision)). Terms are defined
in the [README](../README.md#words-used-here).

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
pip install pytest torch
python3 -m pytest tests -q
python3 scripts/run_mekf.py
python3 scripts/lqr_tune.py
python3 scripts/verify_controller.py results/lqr_tuned_3mode.json
```

The first two lines install the packages; `pytest` and `torch` are not in
`requirements.txt`. The third runs the tests. Then: one determination run, the
periodic-LQR demonstration, and the pass/fail check of the tuned controller
(it exits with status 1 when a limit is missed). What each line needs, the
rest of the scripts, and which of them were re-run on 2026-10-24 are in the
[README](../README.md#install-and-run).

## What is established and what is not

Pointers only. The section named owns the claim and its numbers.

Determination, in the README:

- how well the MEKF knows attitude, with and without the errors that do not
  average out: [Result](../README.md#result)
- how the estimators differ and when it matters:
  [Estimators](../README.md#estimators--what-the-comparison-shows)
- what each calibration method buys:
  [Sensor calibration](../README.md#sensor-calibration)

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
- what the simulation leaves out:
  [README](../README.md#what-is-modelled-and-what-is-not)

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

## Open items that need hardware or a decision

- **Gyro Allan-deviation log.** A long static log of the flight gyro, to replace the assumed bias drift that the fine-pointing result hangs on: [notes, gyro requirement](learned-estimation-and-control.md#nine-parameter-result-and-the-gyro-requirement-2026-09-13), [plan phase 5](verification-loop-plan.md#phase-5--replace-assumed-noise-with-measured-noise-people-not-agents).
- **Sun-sensor face normals.** Which way each light sensor points is assumed in every run: [notes section 9](learned-estimation-and-control.md#9-the-sun-sensor-geometry-is-assumed-not-known-2026-09-13), [plan phase 5](verification-loop-plan.md#phase-5--replace-assumed-noise-with-measured-noise-people-not-agents).
- **Pointing target, inertial or nadir.** A nadir-pointing variant is listed as a candidate, not built: [plan phase 4](verification-loop-plan.md#phase-4--redesign-only-where-phase-3-failed).
- **Actuator set unconfirmed.** The coil set behind every gain is not confirmed by any source: [notes, hardware](learned-estimation-and-control.md#what-the-hardware-actually-is), [notes, where things should run](learned-estimation-and-control.md#where-things-should-run), [magnetorquer-defects.md](magnetorquer-defects.md).
- **The 30 s sampling question.** The sweep of estimator and control step that would answer it, including 30 s, is [plan phase 1, row 1e](verification-loop-plan.md#phase-1--fidelity-parallel-agents-each-independent-each-one-switch).

The full bench list is
[plan phase 5](verification-loop-plan.md#phase-5--replace-assumed-noise-with-measured-noise-people-not-agents).

## Related places and people

- Team flight-software repository:
  [github.com/washu-satellite/proves-core-reference](https://github.com/washu-satellite/proves-core-reference)
- Jesse Myoung: owner of this repository.
- Ben Cook: the sensor calibration suite (commit `19f5d75`).
