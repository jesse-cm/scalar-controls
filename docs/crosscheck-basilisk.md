# Phase 2 — independent cross-check against Basilisk

2026-10-24. Phase 2 of `verification-loop-plan.md` (lines 60–67). Every number
here is in `results/crosscheck_basilisk.json` and comes from one command:

    python3 scripts/crosscheck_basilisk.py        # 2 min 10-20 s wall, deterministic

(Python 3.13, Basilisk 2.12.2. Three consecutive runs gave identical JSON apart
from the wall-time field.) Nothing in the existing code was changed.

## What was compared

Two set-ups, because the repo uses two:

| name | orbit | epoch (UTC) | start | our step | where it comes from | orbits run |
|---|---|---|---|---|---|---|
| control | 550 km, 98.61° | 2026-09-11 12:00 | q = [.5,.5,.5,.5], ω = [1, −0.6, 0.8] °/s | 1 s | `scripts/lqr_tune.py:46-49`, `scalar/control/rl_tune.py:72-73` | free tumble, detumble |
| readme | 550 km, 51.6° | 2024-03-20 12:00 | q = identity, ω = [4, −3, 5] °/s | 0.5 s | `scalar/scenario.py:76-83`, `scalar/environment/orbit.py:51` | free tumble |

Each orbit is 5739 s, 5740 samples at 1 Hz.

| | our sim (unchanged repo code) | Basilisk |
|---|---|---|
| orbit | `CircularOrbit`, two-body | `Spacecraft` + Earth point mass, same μ, RK4 |
| rigid body | `RigidBody()` default inertia, `simulate_free_tumble` / `simulate_controlled` | same inertia in the hub, RK4 at 0.1 s (free tumble) or 0.02 s (detumble) |
| sun | `sun_direction_eci` (almanac series) | SPICE, de430 |
| Earth rotation | `gmst_rad` | SPICE, `IAU_EARTH` from pck00010 |
| field | `magnetic_field_eci` tilted dipole | `MagneticFieldCenteredDipole` given our B0, tilt, longitude and radius as g10/g11/h11 |
| eclipse | `eclipse_factor`, cylinder | `Eclipse`, cone with penumbra |
| sun sensors | `SunSensorArray().currents`, six faces, no noise | six `CoarseSunSensor`, same normals and full-scale current, no noise |
| coils | `s3_coil_matrix` → `actuators_from_matrix` | `MtbEffector`, same axes and dipole limits |
| detumble law | `TwoModeController` held in mode 1, m = sat(k ω×B), k = 5e3, 1 Hz, truth rate and field | the same law object, fed Basilisk's own rate and field at 1 Hz |

The damping law is an input on both sides, so the detumble orbit tests the
plant, not the law. Our detumble path reproduces
`tune.simulate_closed_loop_estimated(use_truth=True)` exactly (rate and dipole
histories differ by 0.0).

Gate, fixed before the first run: every direction error < 0.5°, every eclipse
edge within 5 s. One threshold is mine, not the plan's: face current within
0.873 % of full scale, which is what a 0.5° direction error can do to one face.
"Eclipse edge" is our edge against the instant Basilisk's illumination crosses
50 %; our cylinder has no penumbra to compare a start or end against.

## Gate tables — our sim as it stands

"Same numbers in both inertial frames": our position, velocity and attitude
are handed to Basilisk unchanged.

**control / free tumble — all rows pass**

| quantity | max | median | threshold | result |
|---|---|---|---|---|
| truth attitude | 0.0054° | 0.0027° | 0.5° | PASS |
| B_body direction | 0.436° | 0.179° | 0.5° | PASS |
| sun direction in body | 0.366° | 0.365° | 0.5° | PASS |
| sun direction rebuilt from currents (lit) | 0.366° | 0.365° | 0.5° | PASS |
| face current, outside eclipse edges | 0.63 % FS | 0.47 % FS | 0.873 % FS | PASS |
| eclipse edge | 1.98 s | 1.55 s | 5 s | PASS |

**control / detumble — fails on attitude and everything that depends on it**

| quantity | max | median | threshold | result |
|---|---|---|---|---|
| truth attitude | 0.705° | 0.323° | 0.5° | FAIL |
| B_body direction | 0.712° | 0.393° | 0.5° | FAIL |
| sun direction in body | 0.926° | 0.546° | 0.5° | FAIL |
| sun direction rebuilt from currents (lit) | 0.926° | 0.554° | 0.5° | FAIL |
| face current, outside eclipse edges | 1.59 % FS | 0.62 % FS | 0.873 % FS | FAIL |
| eclipse edge | 1.98 s | 1.55 s | 5 s | PASS |

Both sims end the orbit at the same rate (ours 0.00622 °/s, Basilisk
0.00619 °/s); the commanded dipoles differ by at most 2.1e-5 A·m² (peak command
1.9e-3). The attitude number understates the plant difference: with the frame
offset removed it is 1.64°, and with the field made identical it is 1.81°
(discrepancy D). The 0.705° is two causes partly cancelling.

**readme / free tumble — fails on sun direction**

| quantity | max | median | threshold | result |
|---|---|---|---|---|
| truth attitude | 0.251° | 0.125° | 0.5° | PASS |
| B_body direction | 0.186° | 0.111° | 0.5° | PASS |
| sun direction in body | 0.555° | 0.445° | 0.5° | FAIL |
| sun direction rebuilt from currents (lit) | 0.555° | 0.407° | 0.5° | FAIL |
| face current, outside eclipse edges | 0.96 % FS | 0.50 % FS | 0.873 % FS | FAIL |
| eclipse edge | 4.76 s | 4.72 s | 5 s | PASS (0.24 s margin) |

So the first branch of the gate is met for one orbit of three. The rest is the
second branch: the discrepancy list.

## Discrepancy list

Each cause below made a prediction before its run; the run is named.

### A. Our "ECI" is the equator and equinox of the date, not J2000 — ours, labelling

`scalar/environment/sun.py:12-14` says the sun vector is "mean equator/equinox
of date and used directly as ECI". `scalar/timeutil.py:38-44` (GMST) is
referred to the same equinox. `scalar/attitude.py:9` calls the frame
"J2000/GCRF". The two differ by precession since 2000.

- Prediction, from the IAU 1976 precession matrix alone: sun direction off by
  0.373° (2026) and 0.338° (2024); eclipse edges shifted by 1.58 s (control,
  mean of the two edges) and 4.76 s (readme).
- Run: `confirm_frame`, and "confirmation A" — Basilisk restarted from our
  initial state rotated from of-date to J2000, i.e. the same physical scenario.
- Result: sun offset measured 0.364° and 0.338°. After the rotation:

| free tumble, after rotation | control max | readme max |
|---|---|---|
| sun direction in body | 0.0105° | 0.218° (all attitude, see C) |
| B_body direction | 0.029° | 0.199° (all attitude, see C) |
| face current, outside edges | 0.018 % FS | 0.37 % FS |
| eclipse edge | 0.086 s | 0.075 s |

  Edge shift measured 1.58 s mean (2.07 and 1.10) and 4.76 s. Confirmed.

Inside our sim the sun, the Earth rotation and the orbit are consistent with
each other, so nothing computed so far is wrong because of this. It matters
when a number leaves the sim: an attitude quoted as "J2000" is 0.37° off, and
orbit elements taken from a J2000 source would be placed 0.37° off. The readme
orbit happens to put almost all of the shift in the orbit plane, which is why
its eclipse edge sits at 4.76 s of a 5 s gate.

### B. Eclipse: cylinder against cone with penumbra — model difference

- Prediction: our cylinder edge is the middle of Basilisk's penumbra; penumbra
  lasts about 8.5 s for a sun line lying in the orbit plane.
- Run: `confirm_eclipse_model` — our `eclipse_factor` fed Basilisk's own
  positions and sun vector.
- Result: it flips within 0.08 s of Basilisk's 50 % point on all four edges.
  Basilisk's penumbra lasts 8.8 s (control) and 8.6 s (readme). Confirmed. The
  same run shows Basilisk's Earth is a sphere of the equatorial radius, as
  ours is (a 7 km smaller radius would move the edges about 2.4 s).

With the frame offset (A) removed, our edge is 4.2–4.4 s after the start of
Basilisk's penumbra and 4.3–4.4 s before its end. As the sim stands, the worst
figures against the penumbra limits are 6.4 s (control) and 9.1 s (readme),
so a gate defined on the start of the penumbra rather than its 50 % point
would fail. During the penumbra seconds the face currents differ by up to
49–55 % of full scale (control) and 80 % (readme, where A adds 4.8 s of edge
shift); 17–18 samples per orbit are affected.

### C. Truth attitude is advanced with a second-order step — ours, numerical

`scalar/dynamics.py:100` and `:129`, `scalar/control/tune.py:158`: the rate is
integrated with RK4 but the quaternion is advanced with the average of the
start and end rates. The docstring says "~2nd order"; this is its size.

- Prediction: attitude difference to Basilisk falls 4× each time our step is
  halved, and Basilisk does not move when its own step is halved.
- Run: `confirm_integrator`.

| our step | control (1.4 °/s) max after one orbit | readme (7.1 °/s) max after one orbit |
|---|---|---|
| 1 s | 0.0054° | 1.002° |
| 0.5 s | 0.0014° | 0.251° |
| 0.25 s | 0.00034° | 0.063° |
| 0.125 s | 0.000085° | 0.016° |
| Basilisk, step halved | 5e-11° | 7e-8° |

  Ratios are 4.00 throughout. Confirmed. The error grows linearly through the
  orbit (maximum at the last sample).

At the control runs' rates it is negligible. At the Scenario defaults
(7 °/s, 0.5 s) it is 0.25° per orbit and passes the gate for one orbit only.
Hypothesis, not run: because the gyro model reads the accurate rate while the
truth attitude drifts from it, the estimator sees this as a gyro bias of about
0.16 °/h in fast-tumble runs.

### D. The plant freezes the torque for a whole control step — ours, physical

`scalar/dynamics.py:122-128` (`simulate_controlled`) and
`scalar/control/tune.py:153-157`: the torque m × B_body is computed once at
the start of the step and held fixed in the body frame until the next step. A
real coil holds its dipole; the torque keeps following B_body as the body
turns and the spacecraft moves along the orbit. Basilisk does the second.

- Prediction: with the same field on both sides, our law with the dipole held
  lands on Basilisk; with the torque held it reproduces the difference, at any
  plant step.
- Run: `confirm_hold`. Basilisk's dipole module was given our Earth rotation,
  so the two fields agree to 1e-9°. Our variants use the repo's
  `simulate_controlled` and the same 1 Hz commands; only what is held changes.

| our plant | max attitude difference to Basilisk | max rate difference |
|---|---|---|
| as it stands (torque held, 1 s step) | 1.81° | 0.0014 °/s |
| torque held, plant step 0.05 s | 1.83° | 0.0014 °/s |
| dipole held, plant step 0.02 s | 0.0013° | — |
| Basilisk against itself, step doubled to 0.04 s | 0.038° | — |

  Confirmed. The second row rules out C as the cause.

The size is 0.1 % of the rotation accumulated during the detumble, so settle
times and pointing statistics at 1 Hz are not at risk; a point-by-point truth
comparison is. Hypothesis, not run: the error grows about in proportion to the
step. Phase 1e sweeps the step to 60 s through this same loop, where the plant
would also be integrated in 60 s steps; the plant step needs to be separated
from the control step before that table is read.

### E. Earth rotation differs by 0.086° after A is removed — Basilisk's model

- Run: `confirm_earth_orientation`, `confirm_dipole_formula`.
- Result: our Rz(GMST) against Basilisk's Earth matrix differs by 0.453°
  (2026) as it stands and 0.086° after precession (0.081° in 2024). Basilisk's
  matrix equals the linear `IAU_EARTH` formula in SPICE pck00010 (W = 190.147°
  + 360.9856235°/day) to 7e-6°. Our dipole expression fed Basilisk's position
  and Earth matrix reproduces Basilisk's field to 1e-13 in direction and
  magnitude, so the whole remaining field difference (0.029° max) is this
  rotation and none of it is the dipole formula or the coefficient conversion.
- Hypothesis, not confirmed by a run: `IAU_EARTH` is the less accurate of the
  two (NAIF documents it as a low-accuracy frame); settling it needs a
  high-precision Earth kernel that Basilisk does not ship.

Well under the gate either way.

## Model differences (not pass/fail)

| row | result |
|---|---|
| Our IGRF (`scalar/environment/igrf.py`, ppigrf) against Basilisk's WMM2025, control orbit, same Earth rotation on both sides | direction 0.029° max, 0.007° median; 17 nT max, 6 nT median. For scale, our dipole against WMM is 33.7° max, 7.3° median. `igrf.py`'s frame handling has no sign or axis error at this level. |
| Flattened Earth, which neither sim has | Our own cylinder test on the WGS-84 ellipsoid moves the edges by 7.2 s and 6.9 s (control, edges near the poles) and 4.4 s (readme), shortening the eclipse. Computed with our code only, so it is an estimate of a shared assumption, not a cross-checked number. |

## What agreed without comment

- Attitude convention: q as body-from-inertial, ω in body axes, and the sense
  of the kinematics. A sign error would have shown in the first seconds.
- Free-tumble rate: 1.6e-11 °/s (control), 3.4e-8 °/s (readme).
- Orbit position: 1 mm over the orbit.
- Coil torque sign and per-axis limits: τ = m × B, same in both.
- Sun-sensor scale: irradiance within 0.011–0.013 %.
- Residual sun direction after A: 0.0105°, the accuracy `sun.py` states.

## Not covered

Sensor noise, gyro, magnetometer, MEKF, the LQR and its schedule, residual
dipole, disturbance torques. No coil ever saturated in the detumble orbit
(peak 1.9 mA·m² against limits of 58 / 58 / 8.7), so the saturation path was
not exercised; the comment "saturates anyway" at
`scalar/control/periodic_lqr.py:131` does not hold at 1.4 °/s.

## Notes on the Basilisk side

- With the coils on, Basilisk's attitude is only first-order in its dynamics
  step: 0.038° between 0.04 s and 0.02 s steps (in the JSON); probe runs gave
  0.095° between 0.1 s and 0.05 s and 0.019° between 0.02 s and 0.01 s.
  Detumble runs use 0.02 s for that reason, so Basilisk's own detumble
  attitude carries about 0.04° of step error. Hypothesis, not confirmed: the
  coil effector uses a field sample that is refreshed once per step.
- I first blamed that step dependence on the coil effector picking the
  command up one step late. A run with the effector ordered before and after
  the controller gave identical attitudes (difference 0.0), so that
  explanation is withdrawn.
- Fetched on first use into `~/Library/Caches/bsk_support_data`: `de430.bsp`
  (120 MB), `naif0012.tls`, `pck00010.tpc`, `de-403-masses.tpc` from
  naif.jpl.nasa.gov, and `WMM2025.COF` from Basilisk's GitHub repository
  (tag v2.12.2). The wheel ships no examples or module documentation, so each
  module's behaviour was measured with a probe run rather than read.
