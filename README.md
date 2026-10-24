# SCALAR attitude determination and control simulation

A Python simulation of two jobs for SCALAR, the WashU/PROVES 1U CubeSat:
working out which way the spacecraft is pointing (its attitude) from its gyro,
magnetometer and sun sensors, and turning it with magnetic coils using that
estimate. It is a study tool that runs on a laptop. None of it is flight
software and none of it has been checked against hardware.

Results: [docs/RESULTS.md](docs/RESULTS.md). Map of every file and every open
question: [docs/HANDOFF.md](docs/HANDOFF.md).

## 1. Working with the code

Needs Python 3 (checked with 3.13). Run everything from the repository root.

```bash
pip install -r requirements.txt     # numpy, scipy, matplotlib, ppigrf (Earth's magnetic field model)
pip install pytest torch            # not in requirements.txt; torch is only for the sun-sensor network
python3 -m pytest tests -q          # the tests; ends with the scoreboard's headline (--no-scoreboard leaves it out)
python3 scripts/report.py           # headline, then every table, in under a second; runs no simulation
python3 scripts/report.py --write   # the same, and regenerates docs/RESULTS.md
python3 scripts/matrix.py --list    # the scenarios a simulation run can add rows for; runs nothing
```

- **Adding a result.** `scripts/matrix.py` runs scenarios and appends one row
  per scenario and set of seeds to `results/ledger.csv` (a seed is one repeat
  of a run with different random noise). Then run `report.py --write` and
  commit both files; a test fails while they disagree. The rules, and the
  commands behind the existing rows: [docs/verification-loop-plan.md](docs/verification-loop-plan.md).
- **Entering hardware results.** Bench numbers go in `hardware.json` at the
  repository root, one entry per number: `value` (null if unknown), `units`
  (fixed; convert the number), `source` and `note`. Source is one of three
  words: `assumed`, `datasheet`, `measured`. Then run
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
| `scalar/*.py` | frames and quaternions (`attitude.py`); how a rigid body rotates (`dynamics.py`); one estimator run put together, and many compared (`scenario.py`, `montecarlo.py`) |
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
| `scripts/matrix.py`, `scripts/mag_realism.py` | run scenarios and append ledger rows; the second is for the magnetometer-error scenarios |
| `scripts/run_mekf.py`, `scripts/lqr_tune.py` | one orbit through the MEKF; periodic LQR holding a target, then reaching it from a large error |
| `scripts/rl_tune_lqr.py`, `scripts/verify_controller.py` | the tuner; the pass/fail check of a tuned controller on seeds the tuner never saw |
| `scripts/crosscheck_basilisk.py` | compares this simulator with Basilisk, an independent one (needs `pip install bsk`) |
| `tests/` | one file per area, named after what it tests |

File-by-file map, with every script and result file:
[docs/HANDOFF.md](docs/HANDOFF.md#where-things-are). Dated experiment notes
behind the control half: [docs/learned-estimation-and-control.md](docs/learned-estimation-and-control.md).

## 3. What the scoreboard shows

The tables `scripts/report.py` prints, under a headline: the tuned controller
plain, with each switch, and as built, one line each. [docs/RESULTS.md](docs/RESULTS.md) is
the same saved; the numbers live there, not here.

| Table | What it shows |
|---|---|
| A1 | controllers found by the automatic tuner, starting far from the target and tumbling: how many seeds lost pointing, and how long settling took |
| A2, A2b | the same controllers starting near the target: seeds lost and pointing error (angle between where the spacecraft points and the target); then top rotation rate, share of time a coil is at full output, and coil power |
| B | one model error switched off, on, and on with its fix. Today: magnetometer errors, without and with calibration |
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
  and the knowledge limit in rule 2 of [the plan](docs/verification-loop-plan.md),
  which calls the pointing and coil-power limits placeholders. What the rows
  are taken to mean: [the plan's Status section](docs/verification-loop-plan.md#status).
- Older results, not in the ledger: estimator comparison and sensor
  calibration in [docs/determination-results.md](docs/determination-results.md);
  control, learning and coil identification in the notes.

## 4. Assumptions

Inputs that come from neither a SCALAR source nor a measurement:

- The default orbit, the inertia of the 1U body, and which way each sun-sensor
  face points ([notes §9](docs/learned-estimation-and-control.md#9-the-sun-sensor-geometry-is-assumed-not-known-2026-09-13)).
- The coil set. `scripts/detumble_lqr.py` uses two rods and one flat coil, the
  later control studies use a different three-coil set, and no source confirms
  either ([notes, hardware section](docs/learned-estimation-and-control.md#what-the-hardware-actually-is);
  the comment at the top of `scalar/control/tune.py`).
- Sensor noise levels: assumed values, not bench measurements
  ([plan, phase 5](docs/verification-loop-plan.md#phase-5--replace-assumed-noise-with-measured-noise-people-not-agents)).
- Every sensor, coil and body number with its source, including the
  temperature drift and board temperature swing: [hardware.json](hardware.json)
  (table H of the scoreboard).

## 5. Limitations

- The orbit is circular, with no air drag and no gravity terms beyond J2 (the
  slow drift of the orbit plane caused by Earth's flattening). The Sun's
  position comes from a short analytic formula. No disturbance torques from
  gravity gradient, air drag or sunlight pressure.
- The control studies use the simple dipole field for both the truth and the
  controller's model.
- The coils' own magnetic field at the magnetometer is not modelled in
  `scalar/`. [docs/magnetorquer-defects.md](docs/magnetorquer-defects.md)
  lists this and the other ways hand-built coils can misbehave.
- No coil electrical behaviour, and no temperature effects other than the
  magnetometer offset drift of table B (off by default).

Found by the cross-check against Basilisk:

- The inertial frame the code calls J2000 is the equator and equinox of the
  date. It matters when a number leaves the simulation
  ([discrepancy A](docs/crosscheck-basilisk.md#a-our-eci-is-the-equator-and-equinox-of-the-date-not-j2000--ours-labelling)).
- The true attitude is advanced with a low-order step; its error grows with
  rotation rate and step size
  ([discrepancy C](docs/crosscheck-basilisk.md#c-truth-attitude-is-advanced-with-a-second-order-step--ours-numerical)).
- The torque is held fixed for a whole control step; in a real coil the
  current is held and the torque keeps following the field
  ([discrepancy D](docs/crosscheck-basilisk.md#d-the-plant-freezes-the-torque-for-a-whole-control-step--ours-physical)).
