# SCALAR — CubeSat Attitude-Determination Simulation

A Python simulation of **3D attitude determination** for the WashU/PROVES 1U
CubeSat (SCALAR). A freely tumbling spacecraft is observed through a realistic
sensor suite (gyro + magnetometer + coarse sun sensors) and its attitude is
reconstructed by a **Multiplicative Extended Kalman Filter (MEKF)**, then
benchmarked against a 10–15° attitude-knowledge budget.

Determination only — **no control, no actuator, no detumble** (descoped).

## Result

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

## Estimators & what the comparison shows

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

## Sensor calibration

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

## Quick start

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib, ppigrf
python3 scripts/run_mekf.py              # one orbit + error plot -> outputs/
python3 scripts/monte_carlo.py 40        # estimator comparison -> outputs/
python3 scripts/realism_sweep.py 40      # expected accuracy vs systematic error
python3 scripts/calibration_compare.py 12  # calibration-technique comparison -> outputs/
for f in tests/test_*.py; do python3 "$f"; done   # 110 tests
```

## Layout

```
scalar/
  attitude.py         frame + quaternion conventions (q_{B<-I}, Hamilton scalar-last)
  constants.py        physical constants, SI unit conventions
  timeutil.py         Julian Date, GMST
  dynamics.py         torque-free rigid-body tumble (Euler + exp-map quaternion)
  scenario.py         assembles truth + environment + sensors into one SimResult
  environment/
    sun.py            ECI sun vector, irradiance, cylindrical eclipse
    orbit.py          550 km / 51.6° circular LEO with J2 secular precession
    magfield.py       tilted rotating dipole (estimator reference model)
    igrf.py           real IGRF-13/14 field (ppigrf), vectorised ECI batch
  sensors/
    gyro.py           LSM6DSO: ARW noise + random-walk bias
    magnetometer.py   LIS2MDL: body-frame field + noise + soft/hard iron
    sun_sensor.py     6-face photodiodes + per-diode gain, FOV roll-off, misalignment
    star_tracker.py   camera: absolute quaternion, anisotropic noise, keep-out gating
  estimator/
    triad.py          deterministic two-vector attitude (init)
    mekf.py           the MEKF (state [dtheta, gyro-bias], vector + attitude updates)
    algebraic.py      TRIAD-only baseline
  calibration/
    mag_onorbit.py    attitude-independent ellipsoid fit (|B_IGRF| scalar reference)
    sun_onground.py   pre-flight sun-sensor response-curve calibration
    startracker_ref.py  star-tracker-referenced full soft-iron + bias recovery
  montecarlo.py       randomized multi-run comparison harness
scripts/              runnable demos (incl. calibration_compare.py)
tests/                110 physics-validated tests
```

## Conventions

- Attitude quaternion `q` is Hamilton, **scalar-last** `[x,y,z,w]` (matches
  `scipy`), representing `q_{B<-I}`: `v_body = C(q) v_eci`.
- Strict SI; angles in radians unless a name ends in `_deg`.
- Time carried as Julian Date (astronomy) and seconds-from-epoch (propagation).

## Modeling scope & honest limitations

Real hardware is modeled where it drives the estimator: **LSM6DSO** gyro,
**LIS2MDL** magnetometer (both confirmed against the PROVES FC_V5d schematics),
face-board CMOS photodiodes (part unconfirmed — generic model). The two dominant
systematic errors are modeled but **off by default**, so the default run is the
clean validation baseline:

- **Earth albedo** on the sun sensor (`albedo_coeff`) — reflected light from
  nadir biases the reconstructed sun vector; the dominant sunlit error in orbit.
- **Field-model error** — selectable truth field: the tilted centered dipole with
  an optional synthetic direction offset (`field_model_error_deg`), or the **real
  IGRF** (`field_truth="igrf"`). With IGRF truth, the dipole-vs-real discrepancy
  is physical (~10° median direction error); pointing the estimator's reference
  at the same IGRF removes it and exposes the sensor-calibration floor.

Now modeled (previously limitations): the **real IGRF** field (`ppigrf`);
**J2 secular precession** of the orbit (node regresses −4.6°/day at 550 km /
51.6°); sun-sensor **cover-glass FOV roll-off**, **per-diode gain mismatch** and
**normal misalignment**; magnetometer **soft- and hard-iron** distortion; a
**star-tracker** sensor with anisotropic noise and Sun/Earth keep-out gating; and
on-orbit / on-ground / star-tracker-referenced **sensor calibration**.

Still simplified / omitted: sun ephemeris uses the ~0.01° analytic series taken
directly as ECI; orbit is circular (J2 secular rates applied, no drag or higher
zonal harmonics).

With systematics off (default) the estimator shares the truth's environment
models, so only sensor noise drives the error — the sub-degree validation
baseline. Turning them on (truth ≠ model) gives the realistic ~5–9° estimate and
is what the 10–15° budget's headroom is for. **Not reference-derived** (my
assumptions): orbit 550 km / 51.6°, 1U inertia, and the ideal 6-face sun-sensor
placement.
