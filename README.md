# SCALAR — CubeSat Attitude-Determination Simulation

A Python simulation of **3D attitude determination** for the WashU/PROVES 1U
CubeSat (SCALAR). A freely tumbling spacecraft is observed through a realistic
sensor suite (gyro + magnetometer + coarse sun sensors) and its attitude is
reconstructed by a **Multiplicative Extended Kalman Filter (MEKF)**, then
benchmarked against a 10–15° attitude-knowledge budget.

Determination only — **no control, no actuator, no detumble** (descoped).

## Result

Over one 550 km / 51.6° orbit (~37% eclipse), the MEKF holds attitude-knowledge
error to **~0.3° median** — far inside the 10–15° budget under the modeled
sensor noise. A 40-orbit Monte-Carlo:

| Variant | Median | p95 | Eclipse | Budget met |
|---|---|---|---|---|
| **MEKF** | **0.34°** | 0.90° | 0.43° | **100%** |
| MEKF cold-start | 0.35° | 0.91° | 0.45° | 98% |
| TRIAD-only (gyro-free) | 3.69° | 163° | 104° | 0% |

Takeaways: the MEKF is the most accurate; TRIAD initialization prevents rare
cold-start divergence; and gyro fusion is essential — the gyro-free baseline
drifts badly through eclipse.

## Quick start

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib
python3 scripts/run_mekf.py              # one orbit + error plot -> outputs/
python3 scripts/monte_carlo.py 40        # estimator comparison -> outputs/
for f in tests/test_*.py; do python3 "$f"; done   # 56 tests
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
**LIS2MDL** magnetometer, face-board CMOS photodiodes. Deliberately simplified
(documented in-module), and why the real budget is 10–15° while this sim shows
sub-degree:

- **Geomagnetic field** is a tilted centered dipole, not full IGRF (~5–10° local
  direction error omitted). Drop-in replaceable.
- **Sun sensor** models per-face current noise but not Earth-albedo
  contamination, cover-glass FOV roll-off, or diode misalignment.
- **Sun ephemeris** is the low-precision analytic series (~0.01°), used directly
  as ECI (sub-degree mean-equinox-of-date offset).
- Orbit is Keplerian circular (no J2 drift over the short runs).

The estimator uses the *same* environment models as the truth, so only sensor
noise drives the error here; unmodeled systematics above are the headroom the
10–15° budget reserves.
