# Learned estimation and control on SCALAR's actual hardware — experiment notes

Date: 2026-09-11. Untracked experiment on top of the `adcs-sim` tree. Nothing
here is committed or flight code. All numbers are from the scripts named below
and reproduce with `python3 scripts/<name>.py`.

## What the hardware actually is

Gyro (LSM6DSO), magnetometer (LIS2MDL), seven VEML6031 ambient-light sensors,
an RTC, no GNSS, no star tracker. Flight software reads sensors at 1 Hz. The
RP2350 (Cortex-M33, FPU, 520 KB SRAM) has firmware; the payload STM32L031
(Cortex-M0+, no FPU, 8 KB RAM, 32 KB flash) is stated to host the control law
and has no firmware. Actuators are disputed (model conflict C-09); the runs
below use the S3 coil parameters paired onto S6's three H-bridge channels:
0.058 A m^2 on X and Y, 0.0087 A m^2 on Z at 3.3 V.

## 1. A neural network for state estimation — where it actually helps

`scripts/nn_sun_sensor.py`, `scalar/learning/sun_mlp.py`

The MEKF already averages noise optimally; the 5–9° floor in the README is
*systematic*: Earth albedo on the sun sensor, plus per-diode gain, cover-glass
roll-off and face tilt. A 1.5k-parameter MLP (6 currents [+ predicted nadir in
body frame + dayside factor] -> 32 -> 32 -> unit vector) trained for one
flight-like unit replaces the opposing-pair reconstruction and feeds the MEKF
unchanged.

Sun-vector error, sunlit, random geometry (median / p95, deg):

| albedo | analytic | MLP-A (currents) | MLP-B (+ predicted nadir) |
|---|---|---|---|
| 0.0 | 3.0 / 6.0 | 1.8 / 3.9 | 2.5 / 6.7 |
| 0.3 | 4.2 / 11.6 | 2.6 / 8.4 | 2.2 / 5.3 |
| 0.5 | 5.2 / 18.5 | 3.3 / 14.4 | 2.3 / 5.6 |

MEKF knowledge error, one orbit at 1 Hz, albedo 0.3, ideal magnetometer:
analytic 0.77° median / **10.6° p95**; MLP-A 0.56 / 2.8; MLP-B 0.50 / **1.3**.

Cost: 1440 MACs per sample. Microseconds on the RP2350, ~30 µs in soft float
on the STM32. Weights export to `outputs/sun_mlp_weights.h` (6 KB float32).

Where a network does **not** help: magnetometer soft/hard iron (the ellipsoid
fit already solves it analytically), and the propagation step (gyro + exact
covariance transition is already right).

## 2. Absolute attitude without GPS

`scripts/absolute_without_gps.py`

Mag + sun + gyro gives absolute inertial attitude *if* the estimator knows
position and time: uplinked TLE + RTC + onboard propagator. Sweep of
along-track position error in the estimator's reference field, 550 km /
98.61°, MEKF at 1 Hz:

| along-track error | clock equivalent | field direction error | MEKF median | p95 |
|---|---|---|---|---|
| 0 km | 0 s | 0.00° | 0.42° | 0.63° |
| 20 km | 2.6 s | 0.30° | 0.51° | 0.92° |
| 50 km | 6.6 s | 0.75° | 0.83° | 1.53° |
| 100 km | 13 s | 1.5° | 1.5° | 2.7° |
| 300 km | 40 s | 4.5° | 4.2° | 7.3° |
| 1000 km | 132 s | 15° | 14° | 25° |

So: a TLE a week old (few km) and an RTC within ~5 s are fine. Attitude error
tracks field-direction error roughly 1:1 beyond 50 km. Sun-direction error
from clock error is ~1°/day, negligible.

## 3. Learning the magnetorquer configuration

`scripts/sysid_magnetorquer.py`, `scalar/sysid/magnetorquer_id.py`

Model: I w_dot + w x (I w) = (M u + m_res) x B. Linear in [vec(M); m_res] once
inertia is known, so plain least squares from the A9 burst record (coil
current, body field, body rate). Inertia *shape* is checkable in flight from
any torque-free segment (recovered to 3 decimals from 300 s at 10 Hz); the
*scale* is not — motion data fixes M/|I| only, so inertia scale comes from
CAD/ground measurement and a 10 % inertia error is a 10 % dipole error.

Differentiating the gyro is the wrong formulation (error grows with sample
rate). Integrating across each pulse (impulse form, 1 s gyro averages before
and after) works. Relative error of the recovered 3x3 M:

| pulse | rate | pulses | per-sample fit | impulse fit | per column x / y / z |
|---|---|---|---|---|---|
| 0.32 s | 1 Hz | 90 | 0.50 | 0.33 | 0.27 / 0.11 / 0.52 |
| 0.32 s | 10 Hz | 90 | 1.36 | 0.28 | 0.12 / 0.12 / 0.50 |
| 2 s | 1 Hz | 90 | 0.26 | **0.048** | 0.018 / 0.017 / 0.086 |
| 2 s | 10 Hz | 90 | 0.19 | 0.066 | 0.035 / 0.021 / 0.116 |

Pulse *length* matters more than sample rate: the 320 ms detumble pulse is not
an identification pulse. Ninety 2 s pulses (15 min) recover X/Y to 2–3 %, Z to
~10 %, and the residual spacecraft dipole (2.3 mA m^2 truth) to ~0.3 mA m^2.
Implication for A9: sample at ≥ 5 Hz so the windows can be placed, but
command 2 s pulses for the identification campaign.

## 4. LQR parameters that point

`scripts/lqr_tune.py`, `scalar/control/periodic_lqr.py`, `scalar/control/tune.py`

Two findings about the existing `lqr.py`:

- **Sign bug.** With this repo's `q_{B<-I}` convention,
  d/dt `attitude_error_rotvec(q, target)` = **−ω**. Both LQR designs had
  positive feedback on attitude; the demo only ever damped rate. Fixed in the
  new controllers (error vector negated), test `test_error_vector_derivative_sign`.
- **Projection cannot point.** Even sign-corrected, the time-invariant LQR +
  ⊥B projection ends at 90–110° from a 120° start, and a low-gain
  averaging-theory PD is out-muscled by a 2.3 mA m^2 residual dipole.

What works: **periodic LQR** — the field is inside the Riccati design; dipole
is the input; a gain schedule K(orbit phase) is the output.

Near target (20°, 0.1°/s), residual dipole, truth feedback, 4 orbits:
0.2–0.5° pointing for Bryson rho 0.1–1. Schedule storage: full (404 KB) and
256-bin (18 KB float32) behave identically; **64-bin (4.5 KB) diverges** for
rho ≤ 0.3. 256 bins is the flight number.

Acquisition from 120° / 1°/s: the schedule alone diverges (rates to 20°/s).
Rate damping (m = k ω×B, the gyro form of B-dot) until 0.3°/s, then the
schedule: settles to <10° in 3.1–4.8 h and holds 0.2–0.5°. This two-mode
supervisor is the C-26 structure — DetumbleManager as mode 1, pointing law as
mode 2.

With the MEKF in the loop at 1 Hz, ideal sensors: 3.0° mean, 8° p95, knowledge
0.6°. The 1 Hz loop chatters at full dipole on Y in fine pointing
(`outputs/lqr_two_mode.png`); a deadband or 2–5 Hz control would fix it and
is a data-link (C-27) question.

## 5. The whole chain: NN -> MEKF -> periodic LQR

`scripts/closed_loop_nn.py`. Fine pointing from 20°, flight-like sun unit,
residual dipole, 1 Hz, controller sees only the estimate. Last-orbit numbers:

| front end | albedo | pointing mean / p95 | knowledge median / p95 |
|---|---|---|---|
| analytic | 0.0 | 5.9° / 12.4° | 1.3° / 3.9° |
| analytic | 0.3 | **116° / 173° (diverged)** | 9.6° / 14.4° |
| analytic | 0.5 | 117° / 174° (diverged) | 18° / 22° |
| MLP-B | 0.0 | 5.9° / 16.8° | 1.2° / 5.4° |
| MLP-B | 0.3 | **5.1° / 11.6°** | 0.7° / 3.9° |
| MLP-B | 0.5 | 10.4° / 24.9° | 1.0° / 9.6° |

The network's value is not accuracy; it is that the pointing loop survives
albedo at all. A stiff controller fed a sun vector that jumps toward nadir at
every terminator crossing slews the spacecraft after the jump.

## Where things should run

- Estimator + NN front end on the RP2350 (FPU, memory, owns the sensors).
- The STM32L031 should receive [δθ, ω, B] or a dipole command over UART at the
  loop rate and do current control with a link-loss timeout. A 256-bin
  schedule (18 KB float32, 9 KB int16) fits its 32 KB flash only with a
  minimal firmware; carrying the gain on the RP2350 and sending dipole
  commands is the safer split.
- All of this presumes the actuator matrix is measured (§3) before any gain
  is trusted. The gains above are for a coil set no source confirms.

## Files

    scalar/learning/sun_mlp.py         MLP front end, training, C export
    scalar/sysid/magnetorquer_id.py    inertia shape, impulse-form dipole ID, burst sim
    scalar/control/periodic_lqr.py     periodic Riccati schedule, quantisation, two-mode supervisor
    scalar/control/tune.py             estimate-in-the-loop closed-loop sim, weight sweeps
    scripts/{nn_sun_sensor,absolute_without_gps,sysid_magnetorquer,lqr_tune,closed_loop_nn}.py
    tests/{test_sun_mlp,test_sysid,test_periodic_lqr}.py   8 tests, all pass

## 6. Tuning by reward, verifying by criteria (added 2026-09-12)

`scalar/control/rl_tune.py`, `scripts/rl_tune_lqr.py`, `scripts/verify_controller.py`

Six parameters (Bryson theta_max, omega_max, rho; mode enter/exit thresholds;
damping gain) are searched by the cross-entropy method against a reward
derived from eight pre-registered pass/fail criteria (`THRESHOLDS`). Training
uses seeds 0–1 on a plant with a residual dipole and a mildly uncalibrated
magnetometer (1.5 µT bias, 5 % soft iron). Verification is one command on
held-out seeds 100–102 plus ±10 % coil/inertia variants; exit code 1 on any
FAIL. Two runs of 16 × 6 (7 min each on this laptop, 14 s per evaluation):

| criterion | threshold | hand-picked, seed 0 | tuned, train seeds | tuned, held-out 100–102 |
|---|---|---|---|---|
| acquisition settle < 10° | ≤ 6 h | 5.6 | 4.6 | 4.0–6.4 (one fail; one plant variant diverged) |
| acquisition diverged | 0 | 0 | 0 | 0 nominal; 1 on coils +10 % |
| fine pointing mean | ≤ 5° | 72 (diverged) | 7.7 | **9.4–12.2** |
| fine pointing p95 | ≤ 15° | 174 | 14.8 | 19–31 |
| fine max rate | ≤ 0.5°/s | 1.56 | 0.13 | 0.12–0.27 |
| fine saturation | ≤ 0.2 | 0.04 | 0.00 | 0.00 |
| fine coil power | ≤ 0.3 W | 0.10 | 0.05 | 0.05 |
| fine diverged | 0 | 1 | 0 | 0 |

What the loop established, in order of importance:

1. **The tuner fixed what tuning can fix.** The hand-picked design diverged in
   fine pointing on seed 0 with the mild magnetometer error; the tuned design
   never diverges, saturates 0 % of the time and uses 50 mW. Tuned values:
   theta_max ≈ 22–27°, omega_max ≈ 0.17–0.25°/s, rho ≈ 0.6–0.8, enter ≈
   0.15–0.24°/s, exit ≈ 1.0–1.2°/s, k_damp ≈ 3e4. The reward plateaued at
   +2.2 in the second run: the search is done.
2. **The 5° criterion is estimator-limited, not controller-limited.** Halving
   the coil dipole doubles power and changes pointing by 0.01°. With a
   perfect magnetometer the same design reaches 6.5° (knowledge 1.0°); with the
   mild distortion 10.7° (knowledge 3.6°). About 4° of the floor is the
   magnetometer bias becoming a pointing bias; the remaining ~6° is the 1 Hz
   loop amplifying estimate jitter (truth feedback gives 0.3°).
3. **Overfitting shows up exactly where the verifier looks.** The second run
   tightened the mode-enter threshold to 0.15°/s to shave settle time on the
   training seeds, and a +10 % coil variant then failed to acquire on a
   held-out seed. The first run's looser 0.24°/s did not. Keep both files;
   prefer `results/lqr_tuned.json`.

Consequence: no re-tuning will pass the fine criterion. The next change is in
the estimator path — on-orbit magnetometer calibration gating fine mode (the
ellipsoid fit in `calibration/mag_onorbit.py` already exists), and a slower
outer loop or deadband so the controller stops chasing 1 Hz jitter. Re-run
the tuner after each; the thresholds do not move.

Pre-existing test failures unrelated to this work: `test_igrf`,
`test_calibration_compare` need `ppigrf` installed; `test_mag_calibration`,
`test_star_tracker`, `test_sun_ground_cal` lack the `sys.path` shim and only
run under pytest. All 16 other files pass, including the 4 new ones.

## 7. Magnetometer calibration in the loop (added 2026-09-13)

`calibration_pass()` in `rl_tune.py`: one orbit of raw magnetometer logging
against |B_model|, ellipsoid fit (`calibration/mag_onorbit.py`), applied to
every measurement before the MEKF; pointing mode is only allowed after it.
Sequence assumed: detumble → one calibration orbit → pointing.

| | raw magnetometer | calibrated |
|---|---|---|
| field direction error after correction | several deg | 0.85° median |
| knowledge median, fine mode, seed 100 | 3.6° | 1.5° |
| fine pointing mean, seed 100, tuned gains | 10.7° | 7.5° |
| re-tuned, train seeds | 7.7° | 6.7° |
| re-tuned, held-out 100–102 | 9.4–12.2° | 6.9–9.5° |

Still fails the 5° criterion. Two diagnostics locate the rest:

- Sunlit vs eclipse (seed 100, last orbit): pointing 6.3° / 9.6° mean;
  knowledge 1.1° / 3.4° median. Eclipse is worse but sunlight alone fails.
- Gyro: a perfect gyro (no bias, no random walk) gives 2.6° pointing with the
  same gains; removing only the initial bias changes nothing. **The bias
  random walk is the driver** — the estimator's bias error enters the
  controller's rate term and becomes a pointing offset. The sim's bias
  random walk (1e-5 rad/s/√s) is a datasheet guess; the real LSM6DSO Allan
  deviation is the single most valuable bench number for this loop.

Softening only the fine-mode rate weight by hand (omega_max 2°/s) gave 7.1°,
so the fix is not a one-parameter change. Fine mode now has its own three
weights (ThreeModeController: damp → acquisition schedule → fine schedule at
< 15° estimated error); the tuner searches nine parameters. Result: see
`results/lqr_tuned_3mode.json` and the log beside it.

### Nine-parameter result and the gyro requirement (2026-09-13)

`results/lqr_tuned_3mode.json`: train fine mean 6.6°, held-out 6.7–9.3°,
acquisition 3.6 h. The tuner put the fine weights (24°, 0.27°/s, rho 0.77)
next to the acquisition weights; the search is exhausted at the same floor.

Sweeping the gyro bias random walk with the filter matched to it (seed 100):

| bias RW [rad/s/√s] | × datasheet guess | fine pointing mean | p95 | knowledge |
|---|---|---|---|---|
| 3e-5 | 3 | 16.3° | 43° | 1.5° |
| 1e-5 | 1 | 7.4° | 18° | 1.5° |
| 3e-6 | 0.3 | **4.5°** | 7.8° | 1.4° |
| 1e-6 | 0.1 | 3.1° | 4.9° | 1.2° |
| 3e-7 | 0.03 | 1.8° | 2.9° | 1.0° |

Knowledge barely moves; pointing scales with the bias random walk. The path
is the controller's rate term, not the attitude estimate. Two consequences:

1. **The 5° criterion is a gyro specification:** bias random walk ≤ 3e-6
   rad/s/√s, three times better than the datasheet-class guess in the sim.
   Whether the flight LSM6DSO meets it is a two-hour Allan-deviation log,
   nobody knows today, and it is now the highest-value bench measurement.
2. **A controller-side lever exists and is untested:** an integral state on
   attitude error in fine mode (LQI). A slowly walking bias error is a
   near-constant offset the integrator would cancel. Nine-state Riccati, same
   schedule machinery, same criteria. Recommended next change if the measured
   gyro is worse than 3e-6.

## 8. Ablation and mechanism (2026-09-13, two Opus agents; scripts/ablation_pointing.py, scripts/mechanism_gyro_bias.py)

One-at-a-time removal from the realistic baseline, seeds 100–102, fine
pointing mean (± spread across seeds): baseline 7.82 ±2.61°.

| removed | pointing mean | change |
|---|---|---|
| gyro bias random walk (F1) | 1.18 ±0.22° | **−6.64°** |
| sun-sensor noise (F5) | 6.78° | −1.04° (inside spread) |
| residual magnetometer distortion after calibration (F4) | 7.07° | −0.75° (inside spread) |
| gyro white noise, magnetometer white noise, residual dipole | 7.8–8.0° | 0 |
| controller on truth (F8) | 1.04° | disturbance-rejection floor |
| all six removed | 0.02° | — |

Cumulative build-up from ideal: F1 alone gives 6.33°; the six factors are
additive to within 4 % and explain 99.7 % of the error. The sunlit/eclipse
difference seen on seed 100 reverses on seeds 101 and 102: it was noise.

Mechanism (seed 100): the bias random walk reaches pointing through TWO paths
of roughly equal size, not the rate term alone as first claimed.
- Rate term: bias estimation error (median 0.009°/s, p95 0.018°/s) times the
  fine schedule's rate/attitude gain ratio (median 300 s) predicts 3.55°;
  feeding the controller the true attitude with the estimated rate gives
  3.65° observed. Quantitatively confirmed.
- Attitude estimate: the same bias error doubles knowledge error (0.76° →
  1.54° median) and the controller amplifies that 2.6×; true rate with the
  estimated attitude gives 3.97°.
- Both paths removed: 1.08°. Sum of the two ≈ 7.6° vs 7.4° observed.

Consequences: an integral state (LQI) can cancel the rate-term half (~3.6°)
but not the attitude-estimate half (~4°); only a better gyro or better bias
observability moves that. The 5° criterion stands as a gyro requirement
(bias RW ≤ 3e-6 rad/s/√s) or needs both fixes together.

## 9. The sun-sensor geometry is assumed, not known (2026-09-13)

Every run uses S4's six-face cube (`scalar/sensors/sun_sensor.py`
CUBE_FACE_NORMALS: ±x, ±y, ±z) with an ideal cosine response. SCALAR has
seven VEML6031 ambient-light sensors on devicetree faces 0,1,2,3,5,6,7; the
model's `VEML6031.normalVector` is unset for every one (no source gives it),
the face numbering is inconsistent across three files (capture Q7), and
S5 puts panels on five of six faces. Nothing here was geometrically
simulated against the real spacecraft.

What the estimator loses if the geometry stays unknown (sun sensor unusable,
magnetometer + gyro only, seed 100):

| episode | with the assumed cube | no sun sensor |
|---|---|---|
| fine, pointing mean / p95 | 7.4° / 18.0° | 9.1° / 24.2° |
| fine, knowledge median | 1.5° | 2.2° |
| acquisition, pointing mean | 10.6° | 14.8° |
| acquisition, knowledge median | 1.5° | 4.8° |
| acquisition, settle | 5.6 h | 6.3 h (fails the 6 h criterion) |

So the design converges without the sun sensor — the field rotating through
the body frame over an orbit makes attitude observable from magnetometer +
gyro alone — but slower and ~2° worse, and acquisition misses its criterion.
The sun sensor is worth having only once its normals are known.

How the normals get known, in order of preference:
1. Ask. Board placement is in the mechanical CAD and the flight-board
   layout; this is a gaps.csv row, not an algorithm.
2. Ground sweep of the assembled unit on a rotary stage under a lamp: gives
   each sensor's normal, gain and angular response directly, and is the
   training set the MLP front end needs anyway (the network then learns the
   currents→sun map without needing the normals stated).
3. On-orbit self-calibration: with the magnetometer-only attitude as a
   coarse reference, regress each sensor's current against the predicted
   body-frame sun vector, I_i = g_i·max(0, n_i·s); linear per sensor, needs
   a few sunlit orbits. Same structure as the star-tracker-referenced
   magnetometer fit already in `calibration/`.
Add the sensor layout (count, normals, response) as a ScenarioSpec field in
phase 1f of the verification loop; the cube is one entry in its candidate set.
