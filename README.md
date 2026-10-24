# SCALAR attitude determination and control simulation

This repository is a Python simulation of two jobs for SCALAR, the
WashU/PROVES 1U CubeSat:

1. **working out which way the spacecraft is pointing**, from its gyro,
   magnetometer and sun sensors, and
2. **turning it** with magnetic coils, using that estimate.

Everything runs on a laptop. It is a study tool. None of it is flight software
and none of it has been checked against hardware.

**Picking this up for the first time? Read [docs/HANDOFF.md](docs/HANDOFF.md)
first.** It is the map: where each piece of code, each result file and each
open question lives.

## Words used here

- **Attitude**: the orientation of the spacecraft body relative to the stars.
- **Attitude determination**: estimating the attitude from sensor readings.
  The **knowledge error** is the angle between the estimated and the true
  attitude.
- **Attitude control**: turning the spacecraft to a wanted attitude. The
  **pointing error** is the angle between the true attitude and the target.
- **Truth**: the simulated real world (true attitude, true magnetic field).
  The estimator never sees it; it only sees simulated sensor readings.
- **Gyro**: measures how fast the body is rotating. Its reading has a slowly
  drifting offset, the **gyro bias**.
- **Magnetometer**: measures the direction and strength of Earth's magnetic
  field in the body frame.
- **Sun sensor**: light sensors on the faces; comparing them gives the
  direction to the Sun. It gives nothing in **eclipse** (Earth's shadow), and
  sunlight reflected off Earth (**albedo**) pulls its answer toward Earth.
- **TRIAD**: a formula that gives attitude from two measured directions (Sun
  and magnetic field) at one instant, with no memory.
- **MEKF** (multiplicative extended Kalman filter): the main estimator. It
  keeps a running attitude estimate, pushes it forward with the gyro, and
  corrects it with the magnetometer and sun sensor. It also estimates the
  gyro bias. **IMEKF** is the same filter repeating its correction step a few
  times; **USQUE** is a different filter of the same family (an unscented
  Kalman filter).
- **IGRF**: the standard published model of Earth's magnetic field. A
  **dipole** is the simple bar-magnet approximation of it.
- **Magnetorquer**: a coil (or a rod with a coil around it) that acts as an
  electromagnet. Earth's field pushes on it and turns the spacecraft. It can
  never push around the field direction itself, only across it.
- **Detumble**: slowing an unwanted spin.
- **LQR** (linear quadratic regulator): a standard recipe that turns "how much
  do I care about error versus actuator effort" into feedback gains.
  **Periodic LQR** computes a different gain for each point around the orbit,
  because the magnetic field direction changes around the orbit. **LQI** adds
  a term that accumulates past error.
- **System identification**: working out a property of the hardware (here, how
  strong each coil is and which way it points) from recorded motion.
- **Monte Carlo**: repeating a run many times with different random numbers
  (**seeds**) and reporting the spread, not one lucky run.

## What is in the repository

| Part | What it does | Code | Where its results are |
|---|---|---|---|
| Simulated world | orbit, Sun and eclipse, Earth's magnetic field (dipole and IGRF), how a rigid body rotates | `scalar/environment/`, `scalar/dynamics.py`, `scalar/attitude.py` | — |
| Sensors | gyro, magnetometer, sun sensor (a generic six-face one and a model of the VEML6031 light sensors), star tracker | `scalar/sensors/` | — |
| Determination | TRIAD, MEKF and IMEKF, USQUE, and a TRIAD-only baseline, compared by Monte Carlo | `scalar/estimator/`, `scalar/scenario.py`, `scalar/montecarlo.py` | [Determination results](#determination-results) below |
| Sensor calibration | three ways of measuring and removing magnetometer and sun-sensor errors | `scalar/calibration/` | [Sensor calibration](#sensor-calibration) below |
| Actuators and control | magnetorquer model; fixed-gain LQR, periodic LQR with a mode switch, LQI | `scalar/actuators/`, `scalar/control/` | [notes §4](docs/learned-estimation-and-control.md#4-lqr-parameters-that-point), [§10](docs/learned-estimation-and-control.md#10-lqi-in-fine-mode-tested-it-does-not-cancel-the-rate-path-2026-10-24) |
| Tuning and verification | automatic search for controller settings against fixed pass/fail limits, then a separate check on seeds the search never saw | `scalar/control/rl_tune.py`, `scripts/rl_tune_lqr.py`, `scripts/verify_controller.py` | [notes §6](docs/learned-estimation-and-control.md#6-tuning-by-reward-verifying-by-criteria-added-2026-09-12), [§7](docs/learned-estimation-and-control.md#7-magnetometer-calibration-in-the-loop-added-2026-09-13) |
| Learned sun-sensor front end | a small neural network that turns raw sun-sensor readings into a Sun direction | `scalar/learning/` | [notes §1](docs/learned-estimation-and-control.md#1-a-neural-network-for-state-estimation--where-it-actually-helps), [§5](docs/learned-estimation-and-control.md#5-the-whole-chain-nn---mekf---periodic-lqr) |
| Magnetorquer identification | recovers coil strength and direction from recorded torque pulses | `scalar/sysid/` | [notes §3](docs/learned-estimation-and-control.md#3-learning-the-magnetorquer-configuration) |
| Error studies | scripts that take the pointing error apart and try fixes | `scripts/` | [notes §8](docs/learned-estimation-and-control.md#8-ablation-and-mechanism-2026-09-13-two-opus-agents-scriptsablation_pointingpy-scriptsmechanism_gyro_biaspy) to [§12](docs/learned-estimation-and-control.md#12-the-real-sun-sensor-veml6031-counts-and-saturation-2026-10-24) |

"Notes" means [docs/learned-estimation-and-control.md](docs/learned-estimation-and-control.md). The file-by-file
map, including every script and every result file, is in
[docs/HANDOFF.md](docs/HANDOFF.md#where-things-are).

## Install and run

Needs Python 3 (checked with Python 3.13). Run everything from the repository
root.

```bash
pip install -r requirements.txt     # numpy, scipy, matplotlib, ppigrf
pip install pytest torch            # not in requirements.txt; see below
python3 -m pytest tests -q          # the test suite
```

- The tests need `ppigrf` (the package that supplies the IGRF field). It is in
  `requirements.txt`. Without it the tests in `tests/test_igrf.py` and
  `tests/test_calibration_compare.py` fail.
- `pytest` runs the tests. `torch` is needed only by
  `scalar/learning/sun_mlp.py`: the two tests in `tests/test_sun_mlp.py` and
  the scripts that train the network.
- Use pytest. Running a test file directly (`python3 tests/<file>.py`) does
  not work for every file.

Scripts that finish in seconds to about ten minutes:

```bash
# determination
python3 scripts/run_mekf.py                 # one orbit through the MEKF, prints the knowledge error
python3 scripts/monte_carlo.py 40           # estimator comparison over 40 random runs
python3 scripts/realism_sweep.py 40         # the same with albedo and field-model error switched on
python3 scripts/calibration_compare.py 12   # the three calibration methods compared

# control, learning, identification
python3 scripts/detumble_lqr.py             # fixed-gain LQR slowing a tumble
python3 scripts/lqr_tune.py                 # periodic LQR: near-target pointing, then acquisition from a large error
python3 scripts/verify_controller.py results/lqr_tuned_3mode.json   # pass/fail table for a tuned controller
python3 scripts/nn_sun_sensor.py            # trains the sun-sensor network and compares it (needs torch)
python3 scripts/sysid_magnetorquer.py       # coil identification from torque pulses
python3 scripts/closed_loop_nn.py           # network -> MEKF -> periodic LQR in one loop (needs torch)
python3 scripts/absolute_without_gps.py     # how wrong the position and clock may be
```

`verify_controller.py` exits with status 1 when any limit is missed; that is
its report, not a crash.

Every command above was run on 2026-10-24 with Python 3.13 and finished
without error (`verify_controller.py` with status 1, as just described). Two
things to know about that check:

- The two `pip` lines found every package already installed, so they did not
  test an install from nothing.
- The scripts were run in a copy of the repository, so that the plots and
  result files kept in git were not overwritten.

Time each one took, on a laptop that was busy with other work: `run_mekf` 3 s,
`monte_carlo 40` 9 min, `realism_sweep 40` 4 min, `calibration_compare 12`
1 min, `detumble_lqr` 2 s, `lqr_tune` 3 min, `verify_controller` 2 min,
`nn_sun_sensor` 24 s, `sysid_magnetorquer` 1 min, `closed_loop_nn` 3 min,
`absolute_without_gps` 8 s.

The remaining scripts each run many closed-loop simulations on several
processor cores and overwrite a file in `results/`. They were not re-run for
this README; the result files in the repository are from the dates given in
the notes:

```bash
python3 scripts/rl_tune_lqr.py              # not run
python3 scripts/ablation_pointing.py        # not run
python3 scripts/lqi_fine.py                 # not run
python3 scripts/estimator_levers.py         # not run
python3 scripts/veml6031_baseline.py        # not run
```

Each script starts with a comment block saying what it tests and how to call
it.

## Where results live

- **Determination and calibration results**: the tables in this README,
  below. The scripts that produce them print their numbers to the screen and
  save plots to `outputs/`.
- **Control, learning and identification results**: the notes,
  [docs/learned-estimation-and-control.md](docs/learned-estimation-and-control.md). They are dated, and
  where a later run contradicted an earlier claim the correction sits beside
  the original text.
- **`results/`**: files written by the tuning and study scripts: tuned
  controller settings, the tuner's logs, and tables of per-run numbers.
  Sections 10 to 12 of the notes name the file and the entry behind each of
  their numbers; the earlier sections name the script.
- **`outputs/`**: plots, and the network weights exported as a C header. Git
  ignores new files here, and new `.png` and `.csv` files anywhere
  (`.gitignore`); the ones in the repository were added on purpose. Re-running
  a script overwrites the plot of the same name, and git then shows that file
  as modified.
- **Which script wrote which file**:
  [docs/HANDOFF.md](docs/HANDOFF.md#where-things-are).

## Determination results

The three subsections below are the determination results as written when
that half of the work was done. Their text and tables are unchanged.

The tables have not been regenerated since. On 2026-10-24 the three scripts
behind them were run again. `scripts/calibration_compare.py 12` printed the
calibration table again except for the last digit of two entries.
`scripts/monte_carlo.py 40` and `scripts/realism_sweep.py 40` printed values
close to the two tables under Result but not identical to them. Treat the
tables as a record of the earlier runs until someone regenerates them.

### Result

Reported two ways, because they answer different questions.

**1. Validation baseline** — perfect environment model (truth == estimator
model), so the only error is zero-mean sensor noise the filter averages away.
This shows the filter and observability are sound; it is **not** an in-orbit
prediction. Over one 550 km / 51.6° orbit (~37% eclipse), 40-run Monte-Carlo:

| Estimator | Median | p95 | Eclipse | Budget met |
|---|---|---|---|---|
| **MEKF** | **0.34°** | 0.90° | 0.43° | **100%** |
| MEKF cold-start | 0.35° | 0.91° | 0.45° | 98% |
| TRIAD-only (gyro-free) | 3.69° | 163° | 104° | 0% |

The MEKF wins; TRIAD init prevents rare cold-start divergence; gyro fusion is
essential (the gyro-free baseline drifts to ~100° through eclipse).

**2. Expected in-orbit accuracy** — the two dominant real-world errors for this
sensor suite are *systematic* (Earth-albedo on the coarse sun sensors,
IGRF-vs-dipole field error), so they enter attitude error ~1:1 instead of being
filtered. Adding them (`scripts/realism_sweep.py`, 40-run sweep):

| Systematics | Median | p95 | Eclipse | Budget <15° |
|---|---|---|---|---|
| none (validation) | 0.34° | 0.90° | 0.43° | 100% |
| moderate (albedo 0.3, field 5°) | **5.5°** | 12.2° | 5.9° | 78% |
| worst-case (albedo 0.5, field 8°) | 8.9° | 20.9° | 9.6° | 0% |

So the **defensible expectation is ~5–9° median knowledge error**, consistent
with flown CubeSats using the same suite (e.g. UVSQ-SAT: <3° sunlit, ~10°
eclipse). Under worst-case systematics the p95 (~21°) exceeds the 15° budget —
which is exactly the headroom the 10–15° target reserves. This sweep uses the
synthetic dipole field error; with the **real IGRF** field as the onboard
reference and sensor **calibration** applied, the residual sensor error falls to
~1–3° (see [Sensor calibration](#sensor-calibration) below).

One caveat from the audit: the sub-degree *eclipse* number depends on the bus
tumbling fast enough (≳2°/s) for the magnetometer to be observable about all
axes. (A second — coarse-dt divergence — was traced to the first-order
covariance transition and fixed; see below.)

### Estimators & what the comparison shows

Five estimators (`scalar/estimator/`, compared in `scripts/monte_carlo.py`):
**MEKF** (multiplicative EKF, exact covariance transition), **IMEKF** (same, with
an iterated update — `MEKF(n_iter>1)`), **USQUE** (unscented, GRP sigma points),
**MEKF cold-start** (no TRIAD init), and **TRIAD-only** (gyro-free baseline).

The differences are **entirely regime-dependent**: a warm-started, near-linear
run makes every filter look identical, and only the strongly nonlinear regime
separates them. (This bit me twice during the exploration — dismissing variants
on warm-started tests before checking the cold regime.) By regime:

- **Nominal** (TRIAD init, update every step): MEKF = IMEKF = USQUE, all
  sub-degree. Measurement-dominated and near-linear, so filter choice is moot.
- **Coarse dt / high tumble**: the first-order transition `I−[ω]dt` diverges
  (dt 2–4 s, ~60% of seeds); the exact `expm(−[ω]dt)` (now default) and USQUE do
  not. This was an implementation gap, not an EKF-vs-UKF gap.
- **Systematic error (albedo/IGRF)**: accuracy is floored by the systematic
  magnitude (~5–9°), identical for every filter. Adaptive-R and GRP-tuning don't
  help (inflating R can't remove a bias; adaptive-R is slightly worse).
- **Cold acquisition (large initial error)**: filters diverge hard. From a 150°
  cold start (6 seeds):

  | filter | err @ step 5 | converge (<2°) |
  |---|---|---|
  | **IMEKF** | **0.9°** | **3 steps** |
  | USQUE | 22.8° | 8 steps |
  | MEKF | 33.8° | never (in-window) |

  The **iterated update is the standout** — relinearizing drives it to the
  two-vector (QUEST-like) solution each step, so it locks on in ~3 steps from any
  prior, at negligible cost and no sigma points. USQUE also converges (sigma
  points span the wide uncertainty); shrinking its spread α→0 reproduces the MEKF
  exactly, a consistency check that both are correct.

**Bottom line:** gyro fusion is essential (vs TRIAD-only); the exact transition
is a free must-have; for steady-state ops the plain MEKF suffices; if the bus can
start from an unknown tumbling attitude, **the iterated MEKF is the best and
cheapest choice** (USQUE second). No estimator beats the systematic-error floor.

### Sensor calibration

The field is now the **real IGRF** (via `ppigrf`), not a dipole stand-in, so
the estimator can carry IGRF as its onboard reference and the residual attitude
error is driven by *sensor* imperfections — magnetometer soft/hard iron and
sun-sensor gain/roll-off — rather than a field-model mismatch. Three calibration
techniques are implemented (`scalar/calibration/`) and compared for their effect
on downstream MEKF knowledge error (`scripts/calibration_compare.py`, 12-orbit
Monte-Carlo with representative distortions):

| Calibration regime | Recovers | Median error | vs raw |
|---|---|---|---|
| none (raw) | — | 8.7° | 1.0× |
| on-ground sun | per-diode gains (0.4%) | 7.9° | 1.1× |
| on-orbit mag (ellipsoid) | hard-iron + symmetric soft-iron, up to rotation (295 nT scalar) | 2.9° | 3.0× |
| on-ground + on-orbit | both of the above | 1.7° | 5.2× |
| star-tracker-referenced | **full** soft-iron matrix + bias incl. rotation (D to 7e-4, bias to 8 nT) | **1.1°** | **7.7×** |

Key findings:

- The **magnetometer distortion dominates** — sun-sensor calibration alone barely
  moves the error, because the magnetometer drives the MEKF at every step while
  the sun sensor only helps when sunlit.
- The **on-orbit ellipsoid fit** is attitude-*independent*: it exploits the known
  IGRF field magnitude as a scalar reference and recovers the distortion without
  ever knowing attitude. It removes hard-iron and the symmetric part of soft-iron
  but leaves an unobservable rotation (a scalar reference cannot fix direction).
- The **star-tracker-referenced fit** closes that gap. When the tracker has a fix
  (~83% of the orbit here), attitude is known, so a linear regression of the raw
  field against the IGRF body-frame field recovers the *full* soft-iron matrix
  including its rotation — the most accurate regime.
- The on-orbit ellipsoid fit requires the field magnitude to **vary along the
  orbit** (18–46 µT over one LEO pass); a constant-field Helmholtz-cage sweep is
  rank-degenerate and cannot solve the 9-parameter fit.

## Control and learning results

These are not summarised here, so that each number has one home. Read the
notes, [docs/learned-estimation-and-control.md](docs/learned-estimation-and-control.md); a list of
what they establish and what they do not, section by section, is in
[docs/HANDOFF.md](docs/HANDOFF.md#what-is-established-and-what-is-not).

The short version, without numbers: slowing a tumble works; holding a target
attitude works when the controller is told the true attitude; with the
estimator in the loop and realistic sensor errors the fine-pointing limit set
for these studies is missed on most seeds, and the fixes tried so far have not
closed the gap.

## Conventions

- The attitude quaternion `q` is Hamilton, **scalar-last** `[x, y, z, w]`
  (the same layout as `scipy`). It takes a vector from the inertial frame to
  the body frame: `v_body = C(q) v_eci`. `scalar/attitude.py` is the one place
  this is defined.
- `attitude_error_rotvec(q, target)` shrinks at the body rate: its time
  derivative is minus the angular velocity. A controller that feeds it back
  must negate it. Two tests pin this: `test_error_vector_derivative_sign` in
  `tests/test_periodic_lqr.py` and `test_attitude_feedback_is_restoring` in
  `tests/test_control.py`.
- SI units throughout; angles are in radians unless a name ends in `_deg`.
- Time is carried as Julian Date for astronomy and as seconds from the start
  for propagation.

## What is modelled and what is not

Modelled:

- **Orbit**: a circular low Earth orbit whose plane drifts slowly because
  Earth is not a perfect sphere (the J2 effect). The defaults are in
  `scalar/environment/orbit.py`.
- **Sun**: direction, brightness and eclipse.
- **Magnetic field**: the dipole approximation, and the real IGRF field
  through `ppigrf`.
- **Rotation**: rigid-body motion, free or with an applied torque.
- **Gyro** (LSM6DSO): noise and a wandering bias.
- **Magnetometer** (LIS2MDL): noise, plus the constant offset and distortion
  that nearby metal causes (hard and soft iron).
- **Sun sensor**: by default a generic sensor with one light detector on each
  of six faces, with per-detector gain, cover-glass roll-off and tilt errors.
  `scalar/sensors/veml6031.py` models the seven VEML6031 light sensors the
  notes list for SCALAR, including the point where they saturate.
- **Star tracker**: a camera that returns attitude directly. It is here for
  the calibration comparison; the notes list no star tracker on the
  spacecraft.
- **Two errors that do not average out**, both off by default so that the
  default run is the clean baseline: Earth albedo on the sun sensor, and a
  mismatch between the true field and the field model the estimator uses.
- **Magnetorquers**: each one a magnetic dipole along a fixed axis with a
  maximum strength; torque is dipole cross field. The control studies add a
  constant leftover magnetic dipole of the spacecraft itself.
- **The control loop** as the flight computer could run it: the controller
  sees only the estimator's output, once per second.

Simplified or left out:

- The Sun's position comes from a short analytic formula used directly in the
  inertial frame. The orbit is circular, with no air drag and no gravity terms
  beyond J2.
- No disturbance torques from gravity gradient, air drag or sunlight pressure.
- The control studies use the dipole field for both the truth and the
  controller's model.
- The magnetic field of the coils at the magnetometer is not modelled in
  `scalar/`. One script, `scripts/estimator_levers.py`, adds a
  measure-then-torque cycle. [docs/magnetorquer-defects.md](docs/magnetorquer-defects.md)
  lists this and the other ways hand-built coils can misbehave.
- No coil electrical behaviour and no temperature effects.

Assumed, not taken from a SCALAR source:

- the default orbit, the 1U inertia, and which way each sun-sensor face points
  ([notes §9](docs/learned-estimation-and-control.md#9-the-sun-sensor-geometry-is-assumed-not-known-2026-09-13));
- the coil set: `scripts/detumble_lqr.py` uses two rods and one flat coil, the
  later control studies use a different three-coil set, and neither is
  confirmed by a source
  ([notes, hardware section](docs/learned-estimation-and-control.md#what-the-hardware-actually-is);
  the comment at the top of `scalar/control/tune.py`);
- sensor noise levels are assumed values, not bench measurements
  ([verification plan, phase 5](docs/verification-loop-plan.md#phase-5--replace-assumed-noise-with-measured-noise-people-not-agents)).
