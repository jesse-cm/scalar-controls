# Determination results

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
