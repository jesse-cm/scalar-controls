# Limitations and fixes needed

What the simulation leaves out, what still needs fixing or deciding, and the inputs that are not
hardware numbers. Hardware numbers live in [hardware.json](../hardware.json) ([README, section 4](../README.md#4-hardware-numbers-and-other-inputs)).

One line per item, with where it is discussed. Paths in backticks are from the repository root.

## Limitations

### What the simulated world leaves out

- Circular orbit, no air drag, no gravity terms beyond J2 (Earth's flattening); the Sun's position comes from a short formula (`scalar/environment/`).
- No disturbance torques: gravity gradient, air drag, sunlight pressure ([plan row 1b](verification-loop-plan.md#phase-1--fidelity-parallel-agents-each-independent-each-one-switch), not started).
- The control studies use the simple dipole field for both the truth and the controller's model; it is far from the real field ([plan row 1a](verification-loop-plan.md#phase-1--fidelity-parallel-agents-each-independent-each-one-switch), not started; [cross-check, model differences](crosscheck-basilisk.md#model-differences-not-passfail)).
- Earth's shadow is a cylinder with no half-shadow (penumbra), and Earth is a sphere, not flattened; both move the eclipse edges by seconds ([discrepancy B](crosscheck-basilisk.md#b-eclipse-cylinder-against-cone-with-penumbra--model-difference), [model differences](crosscheck-basilisk.md#model-differences-not-passfail)).
- The inertial frame is the equator and equinox of the date, not J2000; an attitude or orbit taken into or out of the simulation in J2000 needs a correction for the slow turning of Earth's axis (precession) ([discrepancy A](crosscheck-basilisk.md#a-our-eci-is-the-equator-and-equinox-of-the-date-not-j2000--ours-labelling)).
- Earth's rotation differs slightly from Basilisk's; which is closer is not settled ([discrepancy E](crosscheck-basilisk.md#e-earth-rotation-differs-by-0086-after-a-is-removed--basilisks-model)).
- A controller gain table computed once and reused for days has not been tested ([plan row 1g](verification-loop-plan.md#phase-1--fidelity-parallel-agents-each-independent-each-one-switch), not started).

### What the spacecraft model leaves out

The plant is the simulated spacecraft and its coils.

- The true attitude is moved forward with a low-order step whose error grows with rotation rate and step size; the plant substeps reduce it, they do not remove it ([discrepancy C](crosscheck-basilisk.md#c-truth-attitude-is-advanced-with-a-second-order-step--ours-numerical)).
- The plant holds the coil dipole over each control step, in two substeps, so the torque follows the field as in a real coil. The old step, which held the torque, is kept only for the runs that reproduce the notes (experiments 4 and 5) ([discrepancy D](crosscheck-basilisk.md#d-the-plant-freezes-the-torque-for-a-whole-control-step--ours-physical), [plant-step row](verification-loop-plan.md#2026-10-24--phase-1-plant-step-row-added-after-the-cross-check)).
- No coil electrical behaviour, no coil field at the magnetometer, no measure-then-torque cycle in `scalar/` ([plan row 1d](verification-loop-plan.md#phase-1--fidelity-parallel-agents-each-independent-each-one-switch), not started; [magnetorquer-defects.md](magnetorquer-defects.md)).
- No temperature effects other than the gyro and magnetometer offset drift (table B, off by default).
- Coil saturation was never reached in the cross-check, so that path is unchecked against Basilisk ([not covered](crosscheck-basilisk.md#not-covered)).

### What the sensors and estimator leave out

- Bench values enter as uncorrected sensor errors: the estimator does not compensate a measured gyro scale factor or misalignment (`NOT_IN_FILE` in `scalar/hardware.py`).
- Measured sun-sensor normals change the simulated truth only; the analytic reconstruction still assumes the nominal layout (`scalar/harness/spec.py`, sun-sensor notes).
- The closed-loop estimator's noise settings are fixed, not retuned to measured noise (`scalar/control/tune.py`).
- The as-built rows use the analytic sun front end (a formula that compares opposite faces) and the VEML6031 array at its default integration time, at which it clips in direct sun ([notes section 12](learned-estimation-and-control.md#12-the-real-sun-sensor-veml6031-counts-and-saturation-2026-10-24), [row 1f](verification-loop-plan.md#2026-10-24--phase-1-row-1f-veml6031-sun-sensor-and-earth-albedo)).
- Gyro clipping at full scale is not modelled (`scalar/sensors/gyro.py`).
- The loop has no tolerance to a gyro bias that follows board temperature; raising the filter's bias noise does not help, and temperature compensation is not built ([row 1h](verification-loop-plan.md#2026-10-24--phase-1-row-1h-gyro-realism)).
- A magnetometer offset drift ten times the assumed one defeats the plain ellipsoid calibration (the magnetometer correction fitted over one orbit) ([row 1c](verification-loop-plan.md#2026-10-24--phase-1-row-1c-magnetometer-realism)).
- Which way each sun sensor faces is unknown, and the face numbering disagrees across three files; with the side faces mis-assigned the analytic front end fails ([notes section 9](learned-estimation-and-control.md#9-the-sun-sensor-geometry-is-assumed-not-known-2026-09-13), [row 1f](verification-loop-plan.md#2026-10-24--phase-1-row-1f-veml6031-sun-sensor-and-earth-albedo)).
- On the one seed tried, acquisition (turning from far off the target) without the sun sensor misses its settle limit ([notes section 9](learned-estimation-and-control.md#9-the-sun-sensor-geometry-is-assumed-not-known-2026-09-13)).
- Not covered for the VEML6031: its datasheet gain tolerance, rebuilding clipped readings with albedo present, and the real face normals; that rebuild needs the Sun's brightness to be known ([notes section 12](learned-estimation-and-control.md#12-the-real-sun-sensor-veml6031-counts-and-saturation-2026-10-24)).
- With albedo the analytic front end cannot be used for pointing; the learned one still loses some seeds on the VEML6031 at high albedo ([row 1f](verification-loop-plan.md#2026-10-24--phase-1-row-1f-veml6031-sun-sensor-and-earth-albedo)).
- Every network is trained and tested on this simulator's own sensor and albedo models: the rows show the method can work, not that a given network will ([row 1f, point 6](verification-loop-plan.md#2026-10-24--phase-1-row-1f-veml6031-sun-sensor-and-earth-albedo)).
- The learned front end combined with the faster filter settings loses pointing on some seeds; cause not investigated ([notes section 11](learned-estimation-and-control.md#11-estimator-side-levers-on-the-attitude-estimate-half-2026-10-24)).
- The best estimator-side change found needs a faster sensor rate than the flight software has ([notes section 11](learned-estimation-and-control.md#11-estimator-side-levers-on-the-attitude-estimate-half-2026-10-24)).

### How results are measured

- Knowledge and pointing error are recorded at matching instants. The runs that reproduce the notes (experiments 4 and 5) keep the old record, one step late, which inflates knowledge error at high rotation rates ([plan, Open](verification-loop-plan.md#open)).
- A lost-seed count from 40 seeds is a rate with a wide spread: a tiny plant change reshuffles which seeds are lost ([plant-step row](verification-loop-plan.md#2026-10-24--phase-1-plant-step-row-added-after-the-cross-check)).
- Rows made on the laptop and on the cluster are not digit-identical, and the ledger has no column saying which is which; compare lost-seed counts and acquisition figures only within one platform ([where runs are made](verification-loop-plan.md#2026-10-24--where-runs-are-made-and-what-that-does-to-the-rows)). The differences are large in runs that lose pointing and up to about 1.5 % in runs that use the trained sun network; on the cluster nine of the notes' section 5 figures therefore do not reproduce ([two fixes, point 5](verification-loop-plan.md#2026-11-18--two-fixes-in-the-closed-loop)).
- The two acquisition measures (lost flag, settle time) overlap and both depend on the run length; the settle limit cannot be judged until they use a fixed window ([plan, Open](verification-loop-plan.md#open), [eight-seed entry](verification-loop-plan.md#2026-10-24--tuning-on-eight-seeds-resultslqr_tuned_msjson)).
- The sun-vector errors of experiment 1 and the coil identification errors of experiment 3 have no ledger column (table "Not shown").
- `scripts/estimator_levers.py` and `scripts/mechanism_gyro_bias.py` carry their own copies of the closed loop and do not get the plant switches ([plan, Open](verification-loop-plan.md#open)).
- The cross-check against Basilisk does not cover sensor noise, the gyro, the magnetometer, the estimator, the controllers, the residual dipole or disturbance torques ([not covered](crosscheck-basilisk.md#not-covered)).
- The notes' single-seed closed-loop numbers (sections 4 and 5) come from the good part of the spread; the plan's assumed seed holds only for experiments 4 and 5 ([phase 0](verification-loop-plan.md#2026-10-24--phase-0-harness-done)).
- Four statements in the notes did not reproduce; each has a dated correction ([phase 0](verification-loop-plan.md#2026-10-24--phase-0-harness-done)). The integral-term recommendation of notes sections 7 and 8 is withdrawn ([notes section 10](learned-estimation-and-control.md#10-lqi-in-fine-mode-tested-it-does-not-cancel-the-rate-path-2026-10-24)).
- The older determination tables do not reproduce exactly ([determination-results.md](determination-results.md)).

## Fixes needed

### Known problems not yet fixed

- Every tuned controller misses the fine-pointing limit on the typical seed in its own runs; the notes trace the error to the estimator, mainly the gyro bias drift ([notes sections 7-8](learned-estimation-and-control.md#nine-parameter-result-and-the-gyro-requirement-2026-09-13), [experiment 6](verification-loop-plan.md#2026-10-24--tuned-controllers-at-40-seeds-ledger-experiment-6)).
- Acquisition results for `lqr_tuned_3mode` disagree between its stored metrics, `results/lqi_fine.json` and later runs; cause not confirmed ([plan, Open](verification-loop-plan.md#open), [notes section 10](learned-estimation-and-control.md#10-lqi-in-fine-mode-tested-it-does-not-cancel-the-rate-path-2026-10-24)).
- One seed never meets the settle condition in the 6-orbit run, for either tuned file; not investigated ([eight-seed entry](verification-loop-plan.md#2026-10-24--tuning-on-eight-seeds-resultslqr_tuned_msjson)).
- The integral-term test left gaps: weaker settings untried on the isolated path, the other weights not re-tuned with it, and no test file for `scalar/control/lqi.py` ([notes section 10](learned-estimation-and-control.md#10-lqi-in-fine-mode-tested-it-does-not-cancel-the-rate-path-2026-10-24)).

### Work not started

- Not started: plan rows 1a, 1b, 1d, 1e, 1g and phases 3 and 4; phase 5 needs the hardware ([plan, Not started](verification-loop-plan.md#not-started)).

### Decisions not yet made

- Pointing target: every run holds a fixed inertial attitude; an Earth-facing variant is a candidate, not built ([plan phase 4](verification-loop-plan.md#phase-4--redesign-only-where-phase-3-failed)).
- Which coil set is real: the sources disagree, the runs use one choice, and no gain should be trusted until the coils are measured ([notes, hardware](learned-estimation-and-control.md#what-the-hardware-actually-is), [where things should run](learned-estimation-and-control.md#where-things-should-run)).
- The 5 degree pointing limit and the coil-power limit are placeholders, and the knowledge limit's origin is open ([plan rule 2](verification-loop-plan.md#rules-of-the-loop)).
- Sensor and control rate, including the 30 s question: the sweep that answers it is not run ([plan row 1e](verification-loop-plan.md#phase-1--fidelity-parallel-agents-each-independent-each-one-switch)).
- Where the control law runs: the sources say the payload processor; the notes argue for the main flight computer ([where things should run](learned-estimation-and-control.md#where-things-should-run)).
- The orbit and start date: the `mission` entries of `hardware.json` are assumed.

## Inputs not in hardware.json

### Hardware numbers not in the file, where they live, and why

`NOT_IN_FILE` in `scalar/hardware.py` lists these, with what the file cannot change in the estimator.

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

### Inputs that are not hardware numbers

- Pass/fail limits: `THRESHOLDS` in `scalar/control/rl_tune.py`, and the
  knowledge limit in [plan rule 2](verification-loop-plan.md#rules-of-the-loop).
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
