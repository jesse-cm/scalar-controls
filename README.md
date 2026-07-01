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
which is exactly the headroom the 10–15° target reserves.

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

## Quick start

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib
python3 scripts/run_mekf.py              # one orbit + error plot -> outputs/
python3 scripts/monte_carlo.py 40        # estimator comparison -> outputs/
python3 scripts/realism_sweep.py 40      # expected accuracy vs systematic error
for f in tests/test_*.py; do python3 "$f"; done   # 62 tests
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
    orbit.py          550 km / 51.6° circular LEO
    magfield.py       tilted rotating dipole (IGRF stand-in)
  sensors/
    gyro.py           LSM6DSO: ARW noise + random-walk bias
    magnetometer.py   LIS2MDL: body-frame field + noise
    sun_sensor.py     6-face cosine photodiodes + sun-vector reconstruction
  estimator/
    triad.py          deterministic two-vector attitude (init)
    mekf.py           the MEKF (state [dtheta, gyro-bias], vector updates)
    algebraic.py      TRIAD-only baseline
  montecarlo.py       randomized multi-run comparison harness
scripts/              runnable demos
tests/                56 physics-validated tests
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
- **IGRF-vs-dipole field error** (`field_model_error_deg`) — the field is a
  tilted centered dipole; the ~5–10° non-dipole direction error is injected as a
  systematic between truth and the estimator's reference. A full IGRF is
  drop-in replaceable.

Still simplified / omitted: sun-sensor cover-glass FOV roll-off and diode
misalignment; sun ephemeris uses the ~0.01° analytic series taken directly as
ECI; orbit is Keplerian circular (no J2 over the short runs).

With systematics off (default) the estimator shares the truth's environment
models, so only sensor noise drives the error — the sub-degree validation
baseline. Turning them on (truth ≠ model) gives the realistic ~5–9° estimate and
is what the 10–15° budget's headroom is for. **Not reference-derived** (my
assumptions): orbit 550 km / 51.6°, 1U inertia, and the ideal 6-face sun-sensor
placement.
