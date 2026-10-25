# Learned estimation and control on SCALAR's actual hardware — experiment notes

Date: 2026-09-11, with dated additions through 2026-10-24. Committed on the
`adcs-sim` branch (it was an untracked experiment when these notes were first
written). Simulation only; none of it is flight code. All numbers are from the
scripts named below and reproduce with `python3 scripts/<name>.py`. Where a
later run contradicted an earlier claim, the earlier text is left in place
with a dated correction beside it.

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

> **Where the training data comes from (added 2026-10-24).** From this
> simulator; no real sensor data is used anywhere. `make_dataset` in
> `scalar/learning/sun_mlp.py` draws random situations (a Sun direction, an
> Earth direction, an albedo level between 0 and 0.5, a dayside factor), asks
> the simulated sensor for its readings in each, noise included, and uses the
> Sun direction it drew as the answer to learn. `train_sun_mlp` uses 200,000
> such examples plus 20,000 held back for checking, 20 passes, and a fixed
> seed, and is run again at the start of every script that needs a network.
> The simulated sensor is one "flight-like" unit (`flight_like_sensor`): each
> detector has a gain error (5 % spread), a tilt (2° spread) and a roll-off at
> shallow angles. For the VEML6031 array the Sun's brightness is also varied
> over its yearly range. So the network is graded on the same simulated sensor
> and albedo model it learned from; see
> [LIMITATIONS.md](LIMITATIONS.md#what-the-sensors-and-estimator-leave-out).

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

> **Correction, added 2026-10-24.** The 100 km row's p95 does not reproduce.
> The harness, at the seed this script uses, gives 2.65°, against 2.7° in the
> table. The rest of the table reproduces. See the
> [phase 0 entry of the plan's Status](verification-loop-plan.md#2026-10-24--phase-0-harness-done).

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

> **Correction, added 2026-10-24.** "To ~0.3 mA m^2" for the residual dipole
> does not hold at the seed this script uses: the harness recovers it to
> 0.59 mA m^2. The claim is worded too loosely to score as pass or fail. See
> the [phase 0 entry of the plan's Status](verification-loop-plan.md#2026-10-24--phase-0-harness-done).

## 4. LQR parameters that point

`scripts/lqr_tune.py`, `scalar/control/periodic_lqr.py`, `scalar/control/tune.py`

Two findings about the existing `lqr.py`:

- **Sign bug.** With this repo's `q_{B<-I}` convention,
  d/dt `attitude_error_rotvec(q, target)` = **−ω**. Both LQR designs had
  positive feedback on attitude; the demo only ever damped rate. Fixed in the
  new controllers (error vector negated), test `test_error_vector_derivative_sign`.
  *Added 2026-10-24:* `lqr.py` itself now negates the error vector in
  `desired_torque`; `test_attitude_feedback_is_restoring` in
  `tests/test_control.py` fails on the old sign. The next bullet still holds.
- **Projection cannot point.** Even sign-corrected, the time-invariant LQR +
  ⊥B projection ends at 90–110° from a 120° start, and a low-gain
  averaging-theory PD is out-muscled by a 2.3 mA m^2 residual dipole.

  > **Correction, added 2026-10-24.** "90–110°" is not the whole range. Of
  > the four weight sets this script draws, one ends at 89.11°, just under
  > 90°; the other three end between 91.4° and 107.9° (scoreboard table C2).
  > The conclusion, that the projected LQR does not point, stands. See the
  > [phase 0 entry of the plan's Status](verification-loop-plan.md#2026-10-24--phase-0-harness-done).

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

> **Correction, added 2026-10-24.** "Rates to 20°/s" understates the rates.
> The three schedule-alone runs (rho 0.1, 0.3 and 1) reach top rates of 21.1
> to 28.9°/s. The claim is worded too loosely to score as pass or fail. See
> the [phase 0 entry of the plan's Status](verification-loop-plan.md#2026-10-24--phase-0-harness-done).

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

Updated 2026-10-24 to cover the later sections as well.

    scalar/learning/sun_mlp.py         MLP front end, training, C export
    scalar/sysid/magnetorquer_id.py    inertia shape, impulse-form dipole ID, burst sim
    scalar/actuators/magnetorquer.py   saturable torquers, ActuatorSet
    scalar/control/lqr.py              time-invariant LQR + projection (§4)
    scalar/control/periodic_lqr.py     periodic Riccati schedule, quantisation, two- and three-mode supervisor
    scalar/control/tune.py             estimate-in-the-loop closed-loop sim, weight sweeps
    scalar/control/rl_tune.py          criteria, reward, cross-entropy tuner, calibration_pass (§6, §7)
    scalar/control/lqi.py              nine-state periodic LQI for fine mode (§10)
    scalar/sensors/veml6031.py         seven VEML6031X00 sensors: counts, clipping, clip modes (§12)
    scripts/{nn_sun_sensor,absolute_without_gps,sysid_magnetorquer,lqr_tune,closed_loop_nn}.py   §1–§5
    scripts/{rl_tune_lqr,verify_controller}.py           §6, §7
    scripts/{ablation_pointing,mechanism_gyro_bias}.py   §8
    scripts/lqi_fine.py                                  §10
    scripts/estimator_levers.py                          §11
    scripts/veml6031_baseline.py                         §12 (imports estimator_levers.run_episode)
    results/lqr_tuned*.json, results/rl_tune*.log        tuner outputs and logs (§6, §7)
    results/ablation_pointing.csv, results/mechanism_gyro_bias.json   §8
    results/{lqi_fine,estimator_levers,veml6031_baseline}.json        §10, §11, §12
    outputs/lqr_two_mode.png           written by scripts/lqr_tune.py
    outputs/sun_mlp_weights.h          written by scripts/nn_sun_sensor.py
    tests/{test_sun_mlp,test_sysid,test_periodic_lqr}.py   8 tests, all pass
    tests/{test_actuators,test_control,test_rl_tune,test_veml6031}.py   22 tests, all pass
    (no test file covers scalar/control/lqi.py)

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

Test status, re-checked 2026-10-24 (this replaces the 2026-09-12 note on
pre-existing failures, which named the same five files). Under pytest all 25
tracked test files pass, 148 tests. `test_igrf` and `test_calibration_compare`
still need `ppigrf`; it is in `requirements.txt` and with it installed both
pass, under pytest and as scripts. `test_mag_calibration`, `test_star_tracker`
and `test_sun_ground_cal` still lack the `sys.path` shim: `python3
tests/<file>.py` stops with `ModuleNotFoundError: No module named 'scalar'`,
and they pass under pytest. `test_veml6031` has the shim but no `__main__`
runner, so as a script it runs nothing; under pytest its 6 tests pass. The
other 21 files pass both ways.

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

   > **Correction, added 2026-10-24.** This was tested on 2026-10-24 and did
   > not hold: see §10. The integrator did not cancel the error, and the
   > recommendation above is withdrawn. The premise was the part that failed:
   > on the isolated rate path the error is not a near-constant offset.

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

> **Correction, added 2026-10-24.** The first clause was tested on 2026-10-24
> and failed: LQI does not cancel the rate-term half (§10). The rate-term half
> itself reproduced there. "Both fixes together" therefore has no working
> controller-side fix in it. The estimator-side levers tried for the
> attitude-estimate half are in §11.

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

## 10. LQI in fine mode: tested, it does not cancel the rate path (2026-10-24)

`scalar/control/lqi.py`, `scripts/lqi_fine.py`, `results/lqi_fine.json`

The test of the lever proposed at the end of §7 and repeated in §8: an
integral state on attitude error in fine mode. Nine-state periodic Riccati,
the same 256-bin quantisation, the nine tuned parameters of
`results/lqr_tuned_3mode.json` unchanged, the fine slot of the three-mode
controller replaced, and one new weight: the integral time scale T_i, with
Q_z = I/(theta_max_fine·T_i)², so a larger T_i is a weaker integrator.
Anti-windup: integrate only inside a 10° gate and while no channel is
saturated, clamp at theta_max_fine·T_i, reset on every entry into fine mode.
Written up 2026-10-24. Every number in this section is from
`results/lqi_fine.json`; the key is named with each table.

Baseline first (`baseline_reproduction`): fine pointing mean 7.40 / 6.73 /
9.34° on seeds 100 / 101 / 102, the §8 baseline.

T_i grid on train seeds 0–1 (`train_grid.<T_i>.reward` and `.metrics`; the
metrics are the two-seed mean, the diverged flag the maximum):

| T_i [s] | reward | fine pointing mean | fine diverged | acquisition settle |
|---|---|---|---|---|
| none (baseline) | −5.10 | 6.58° | 0 | never |
| 200 | −15.58 | 66.99° | 1 | never |
| 500 | −5.50 | 7.10° | 0 | never |
| 1000 | −5.28 | 6.81° | 0 | never |
| 2000 | −5.17 | 6.63° | 0 | never |
| 5000 | −2.80 | 6.61° | 0 | 4.12 h |
| 10000 | −2.78 | 6.59° | 0 | 4.07 h |

The script takes the highest reward, T_i = 10000 s (`chosen_Ti_s`): the
weakest integrator in the grid. No T_i has a lower train fine-pointing mean
than the baseline. The reward step between 2000 s and 5000 s is in the
acquisition settle time of seed 0, which becomes finite
(`train_grid.5000.per_seed.0.acq_settle_h` = 6.33 h), not in fine pointing.
At T_i = 200 s fine mode diverged on seed 1
(`train_grid.200.per_seed.1.fine_point_mean` = 125.6°).

Held-out seeds and the ±10 % coil/inertia variants, T_i = 10000 s
(`held_out.<case>.baseline|lqi.metrics`):

| case | fine mean, baseline → LQI | fine p95, baseline → LQI |
|---|---|---|
| nominal, seed 100 | 7.40° → 7.32° | 17.98° → 18.33° |
| nominal, seed 101 | 6.73° → 6.74° | 12.52° → 12.59° |
| nominal, seed 102 | 9.34° → 9.36° | 21.61° → 22.10° |
| coils −10 %, inertia +10 %, seed 100 | 7.40° → 7.32° | 17.98° → 18.32° |
| coils +10 %, inertia −10 %, seed 100 | 7.40° → 7.32° | 17.98° → 18.32° |

`all_pass` is false in all ten entries, and in each case the criteria that
fail are the same for the baseline and for LQI
(`held_out.<case>.*.criteria`). The 5° fine-pointing criterion fails in all
ten.

Why, on seed 100 (`isolated_rate_path_seed100.<controller>/<feed>`). DC is the
length of the last-orbit mean error vector (`err_dc_deg`), AC the mean
deviation from that vector (`err_ac_deg`):

| what the controller is fed | controller | pointing mean | p95 | DC | AC |
|---|---|---|---|---|---|
| true attitude, estimated rate | baseline | 3.65° | 8.03° | 0.28° | 3.60° |
| | LQI, T_i 10000 s | 3.71° | 8.20° | 0.25° | 3.67° |
| | LQI, T_i 5000 s | 3.75° | 8.26° | 0.20° | 3.73° |
| | LQI, T_i 2000 s | 3.83° | 8.33° | 0.10° | 3.81° |
| | LQI, T_i 1000 s | 3.87° | 8.45° | 0.10° | 3.88° |
| estimated attitude and rate (the full loop) | baseline | 7.40° | 17.98° | 2.07° | 7.25° |
| | LQI, T_i 10000 s | 7.32° | 18.33° | 2.38° | 7.18° |
| true attitude, perfect gyro, constant fake rate | baseline | 3.82° | 6.05° | 3.09° | 2.71° |
| | LQI, T_i 10000 s | 2.92° | 4.17° | 0.83° | 2.83° |
| | LQI, T_i 5000 s | 3.01° | 4.64° | 0.23° | 3.00° |
| | LQI, T_i 2000 s | 3.48° | 5.72° | 0.03° | 3.48° |
| | LQI, T_i 1000 s | 4.23° | 7.03° | 0.02° | 4.23° |

What the numbers support:

1. **The rate-term half is real and it is not an offset.** The isolated rate
   path gives 3.65° again (§8), but its mean error vector is 0.28° against a
   3.60° deviation about it. The bias error's 1/e correlation time on that
   run is 346 / 70 / 153 s per axis (`base/true_att.db_corr_time_s`). §7's
   "near-constant offset the integrator would cancel" was wrong about the
   offset.
2. **The integrator works on the case it was built for.** With a constant fake
   rate it takes the offset from 3.09° to 0.02–0.83°. The mean pointing error
   in that case still only moves from 3.82° to between 2.92° and 4.23°,
   because the deviation part stays (2.71° → 2.83–4.23°).
3. **On the real error it does nothing useful.** On the isolated rate path
   every T_i tried is slightly worse than the baseline (3.71–3.87° against
   3.65°), more so for the stronger integrators. On the full loop the
   held-out fine mean moves by at most 0.08° and p95 is higher in all five
   cases.

Not established, and one thing that does not reconcile:

- T_i below 1000 s was not run on the isolated path, and T_i was the only
  parameter searched; the nine tuned parameters were not re-tuned with the
  integrator in place.
- The acquisition columns of this file disagree with §7. For the same
  parameter file and train seeds, §7 reports "acquisition 3.6 h"; here the
  baseline row has `acq_diverged` = 1 and no settle time
  (`train_grid.baseline.metrics`), and held-out seed 101 diverges in
  acquisition under both controllers (`held_out.nominal/101`). The
  fine-pointing numbers do reproduce. The T_i choice above rests on that
  acquisition term, so it should not be read as evidence that T_i = 10000 s
  helps. Cause not investigated. Re-running `scripts/verify_controller.py
  results/lqr_tuned_3mode.json` on 2026-10-24 reproduced the fine-pointing
  numbers and gave a third, different set of acquisition values (screen
  output only, not in a results file).
- No test file covers `scalar/control/lqi.py`.

## 11. Estimator-side levers on the attitude-estimate half (2026-10-24)

`scripts/estimator_levers.py`, `results/estimator_levers.json`

§8 left an attitude-estimate half of the gyro-bias error that no controller
change reaches. This run changes only the estimator side and leaves the tuned
three-mode controller and its 1 Hz update alone. Baseline `B` is §8's config
B: fine pointing from 20°, calibrated magnetometer, residual dipole, three
orbits, last-orbit metrics. Levers (from the script docstring):

- **L1** the MLP-B sun front end of §1, on the flight-like sun unit at albedo
  0.3 and on the ideal unit;
- **L2** the filter's bias random-walk parameter sigma_u scaled by 0.3, 3 and
  10 while the plant stays at 1e-5;
- **L3** gyro, magnetometer and sun sampled at 2 or 4 Hz, each selectable;
- **D** the same with a measure-then-torque duty cycle of 0.6: coils off for
  the first 0.4 s of each control second, magnetometer read only then, so
  control authority is 0.6 of the command. This duty cycle exists only in
  this script, not in `scalar/`;
- **X** combinations of the above, passed on the command line; their settings
  are in `runs.<name>.config`.

Written up 2026-10-24. Every run in the file has 15 seeds, 100–114, although
`meta.seeds` still lists only 100–102. All numbers below are computed from
`runs.<name>.seeds.<seed>`: the mean over the 15 seeds of `point_mean`,
`know_med` and `dbias_med_dps`; Δ is the mean of the per-seed difference from
`B` and SE is the sample standard deviation of those differences over √15.
"Worst seed" is the largest per-seed `point_mean`.

| run (`runs.<name>`) | pointing mean | Δ vs B ± SE | seeds better than B | seeds < 5° | worst seed | knowledge median | bias error median [°/s] |
|---|---|---|---|---|---|---|---|
| `B` | 6.99° | — | — | 2 | 10.6° | 1.35° | 0.0095 |
| `L1-ref analytic flight-unit a0.3` | 14.12° | +7.13 ± 0.32 | 0 | 0 | 18.1° | 2.96° | 0.0132 |
| `L1 MLP-B flight-unit a0.3` | 7.85° | +0.86 ± 0.18 | 1 | 0 | 11.2° | 1.48° | 0.0104 |
| `L1 MLP-B ideal-unit a0` | 8.91° | +1.92 ± 0.12 | 0 | 0 | 12.4° | 2.18° | 0.0109 |
| `L2 su x0.3` | 8.38° | +1.39 ± 0.23 | 1 | 1 | 13.0° | 1.58° | 0.0117 |
| `L2 su x3` | 6.38° | −0.61 ± 0.23 | 11 | 3 | 10.4° | 1.31° | 0.0087 |
| `L2 su x10` | 14.29° | +7.30 ± 3.94 | 2 | 0 | 69.3° | 1.41° | 0.0122 |
| `L3 all 2Hz` | 6.29° | −0.70 ± 0.58 | 10 | 3 | 10.1° | 1.30° | 0.0091 |
| `L3 all 4Hz` | 5.67° | −1.32 ± 0.59 | 12 | 5 | 8.8° | 1.27° | 0.0081 |
| `L3 gyro 4Hz` | 6.24° | −0.75 ± 0.63 | 8 | 3 | 8.7° | 1.30° | 0.0092 |
| `L3 mag 4Hz` | 5.71° | −1.29 ± 0.61 | 9 | 6 | 10.2° | 1.27° | 0.0086 |
| `L3 sun 4Hz` | 6.24° | −0.75 ± 0.68 | 10 | 3 | 8.8° | 1.23° | 0.0091 |
| `D B duty` | 7.17° | +0.17 ± 0.03 | 1 | 1 | 10.7° | 1.35° | 0.0095 |
| `D all 2Hz` | 6.42° | −0.57 ± 0.72 | 9 | 2 | 9.5° | 1.29° | 0.0091 |
| `D all 4Hz` | 6.31° | −0.68 ± 0.86 | 10 | 5 | 14.0° | 1.27° | 0.0087 |
| `D gyro 4Hz` | 6.43° | −0.56 ± 0.61 | 8 | 0 | 8.7° | 1.30° | 0.0092 |
| `D mag 4Hz` | 6.07° | −0.92 ± 0.52 | 12 | 2 | 8.1° | 1.36° | 0.0090 |
| `D sun 4Hz` | 6.46° | −0.53 ± 0.67 | 10 | 1 | 9.2° | 1.23° | 0.0091 |
| `X su3+all4Hz` | 5.23° | −1.76 ± 0.62 | 11 | 8 | 8.6° | 1.24° | 0.0076 |
| `X su3+mag4+sun4` | 5.94° | −1.05 ± 0.81 | 9 | 8 | 10.0° | 1.21° | 0.0078 |
| `X su3+all4Hz duty` | 5.98° | −1.02 ± 0.87 | 12 | 5 | 13.9° | 1.23° | 0.0081 |
| `X MLP a0.3 su3+all4Hz` | 35.40° | +28.41 ± 13.16 | 6 | 0 | 139.7° | 1.44° | 0.0099 |
| `X MLP a0.3 su3+all4Hz duty` | 29.62° | +22.63 ± 11.99 | 5 | 0 | 135.8° | 1.53° | 0.0105 |
| `X su3+all2Hz duty` | 6.13° | −0.87 ± 0.72 | 9 | 3 | 9.0° | 1.25° | 0.0082 |
| `X MLP a0.3 sun4Hz` | 7.90° | +0.91 ± 0.76 | 6 | 0 | 11.2° | 1.68° | 0.0112 |
| `X MLP a0.3 su3+mag4Hz` | 7.31° | +0.31 ± 0.60 | 5 | 0 | 11.1° | 1.49° | 0.0093 |
| `X MLP a0.3 su3+gyro4+mag4` | 7.36° | +0.37 ± 0.52 | 6 | 0 | 9.2° | 1.36° | 0.0093 |
| `X MLP a0.3 su3+gyro4+mag4 duty` | 8.55° | +1.56 ± 0.78 | 4 | 0 | 12.2° | 1.72° | 0.0100 |
| `X su3+gyro4+mag4 duty` | 6.63° | −0.36 ± 0.83 | 7 | 3 | 11.4° | 1.35° | 0.0083 |
| `X MLP a0.3 su3` | 7.73° | +0.73 ± 0.22 | 3 | 0 | 11.2° | 1.45° | 0.0099 |

`B` on seeds 100–102 alone is 7.40 / 6.73 / 9.34° (mean 7.82°, as in §8);
over 15 seeds it is 6.99°.

What the numbers support:

1. **No run brings the 15-seed mean to 5°.** The lowest is `X su3+all4Hz`
   (sigma_u × 3 and all three sensors at 4 Hz, no duty cycle) at 5.23°,
   −1.76 ± 0.62° from `B`, with 8 of 15 seeds under 5°. `B` has 2 of 15.
2. **Filter sigma_u has a narrow optimum above the true value.** × 3 gives
   −0.61 ± 0.23° and is better on 11 of 15 seeds; × 0.3 gives +1.39 ± 0.23°;
   × 10 gives +7.30 ± 3.94° with one seed at 69.3° (seed 111).
3. **Faster sampling helps by about a degree at most, and barely clears the
   noise.** The five L3 runs are −0.70 to −1.32° with SE 0.58–0.68°; the two
   largest are all sensors at 4 Hz (−1.32 ± 0.59°) and the magnetometer alone
   at 4 Hz (−1.29 ± 0.61°).
4. **The duty cycle alone costs little, and every sampling gain is smaller
   with it.** `D B duty` is +0.17 ± 0.03° (worse on 14 of 15 seeds). With it the
   five faster-sampling runs are −0.53 to −0.92° with SE 0.52–0.86°, each
   smaller than its L3 counterpart, though by less than one SE. The off
   window also halves the magnetometer rate (`mag_reads_per_s`): 4 Hz becomes
   2 reads per second, 2 Hz becomes 1.
5. **MLP-B is not a lever on this error.** On the ideal unit without albedo it
   is worse than the analytic front end on every seed (+1.92 ± 0.12°). On the
   flight-like unit at albedo 0.3 it takes the analytic 14.12° down to 7.85°,
   still +0.86 ± 0.18° from `B`.
6. **MLP-B combined with sigma_u × 3 and all sensors at 4 Hz loses pointing on
   some seeds.** `X MLP a0.3 su3+all4Hz` has 4 of 15 seeds above 75° (103, 107,
   112, 113); with the duty cycle, 3 of 15 (100, 112, 114). The other five
   MLP combinations stay between 7.31° and 8.55°, worst seed 12.2°. Cause not
   investigated.
7. **Knowledge and bias error hardly move.** Across the 20 runs with the
   analytic front end on the ideal unit, the knowledge median is 1.21–1.58°
   (`B` 1.35°) and the bias-error median 0.0076–0.0122 °/s (`B` 0.0095). The
   run with the lowest pointing error has the lowest bias error.

So the largest measured gain from the estimator side is under 2° on a 7°
baseline, and it needs a 4 Hz sensor rate on top of the filter setting. The
hardware section at the top of these notes has the flight software reading
sensors at 1 Hz.

## 12. The real sun sensor: VEML6031 counts and saturation (2026-10-24)

`scalar/sensors/veml6031.py`, `scripts/veml6031_baseline.py`,
`results/veml6031_baseline.json`, `tests/test_veml6031.py`

§9 said every run used an ideal six-face cube. This replaces the cube in
config `B` with a model of the seven VEML6031X00 sensors: illuminance in lux,
gain and dark offset, rounding to 16-bit counts, clipping at 65535, and the
two −Z sensors averaged before the unchanged opposing-pair reconstruction. It
reuses `estimator_levers.run_episode`, so everything except the sun unit is
`B`. Still assumed, as in §9: which side face is ±X or ±Y. The run uses unit
gains and `B`'s noise level (`meta.sensor`).

Two integration times are compared. From `meta.sensor`: 0.8704 lx/count at
25 ms and 3.4816 lx/count at 6.25 ms, against 133,300 lx face-on at 1 AU. At
65535 counts that is a ceiling of 57,042 lx at 25 ms, 43 % of face-on sun,
and 228,167 lx at 6.25 ms, above it. The module default, and what the test
file calls the flight setting, is 25 ms.

Three ways of handling clipped samples at 25 ms: feed the clipped counts as
they are; `clip-drop`, discard any sample with a clipped sensor; `clip-fill`,
rebuild the clipped axis from the unit-norm constraint with the irradiance
known.

Written up 2026-10-24. 15 seeds, 100–114 (`meta.seeds`). All numbers are
`summary.<run>.<metric>.mean`, ± `.se`; Δ is `.delta` ± `.delta_se`, paired by
seed against the cube run at the same albedo. Sun metrics are over sunlit
samples of the last orbit.

| run | clipped fraction | sun vector error med / p95 | sun valid | knowledge median | pointing mean | Δ pointing |
|---|---|---|---|---|---|---|
| `B` (cube) | 0 | 1.60° / 3.50° | 1.00 | 1.35° | 6.99 ± 0.46° | — |
| `VEML 25ms` | 1.00 | 11.85° / 19.72° | 1.00 | 7.29° | 89.42 ± 7.70° | +82.43 ± 7.70° |
| `VEML 25ms clip-drop` | 1.00 | none valid | 0.00 | 3.80° | 13.10 ± 1.25° | +6.11 ± 1.35° |
| `VEML 25ms clip-fill` | 1.00 | 1.62° / 3.50° | 1.00 | 1.53° | 7.24 ± 0.48° | +0.25 ± 0.04° |
| `VEML 6.25ms` | 0 | 1.59° / 3.40° | 1.00 | 1.37° | 7.05 ± 0.47° | +0.05 ± 0.01° |
| `B a0.3` (cube, albedo 0.3) | 0 | 6.01° / 11.31° | 1.00 | 2.73° | 13.72 ± 0.46° | — |
| `VEML 25ms a0.3` | 1.00 | 16.89° / 39.26° | 0.98 | 8.50° | 101.56 ± 7.44° | +87.85 ± 7.41° |
| `VEML 6.25ms a0.3` | 0 | 5.97° / 11.17° | 1.00 | 2.74° | 13.71 ± 0.47° | −0.00 ± 0.01° |

What the numbers support:

1. **At 25 ms the sensor is saturated on every sunlit sample.** The clipped
   fraction is 1.00 with zero spread across the 15 seeds (`clip_frac.se` = 0):
   at least one sensor is at 65535 whenever the spacecraft is lit.
2. **Fed as they are, clipped counts break the loop.** Sun-vector error goes
   from 1.60° to 11.85° median, knowledge from 1.35° to 7.29°, and fine
   pointing from 6.99° to 89.42°.
3. **Dropping clipped samples is the same as having no sun sensor.** No sunlit
   sample survives (`sun_valid_frac` = 0). On seed 100 the run gives 9.12°
   mean, 24.16° p95 and 2.18° knowledge
   (`runs.VEML 25ms clip-drop.seeds.100`), which is §9's no-sun-sensor row.
   Over the 15 seeds it is 13.10°, knowledge 3.80°.
4. **Either fix recovers the cube result.** 6.25 ms does not clip and is
   +0.05 ± 0.01° from `B`; rebuilding the clipped axis at 25 ms is
   +0.25 ± 0.04°. With albedo 0.3 the 6.25 ms run matches the cube at the same
   albedo (−0.00 ± 0.01°).
5. **Count quantisation, the dark offset and the doubled −Z sensor are not
   what limits anything here**: with clipping out of the way the VEML runs are
   within 0.25° of the cube.

`tests/test_veml6031.py` pins the model, not the loop: face-on sun clips at
25 ms and fits at 6.25 ms, where the clip cone ends, the direction bias that
clipping causes, the `clip-fill` recovery for one clipped axis, and the −Z
pair averaging.

Not covered: the ±10 % datasheet gain tolerance (the module accepts per-sensor
gains; this run used unit gains), `clip-fill` with albedo, and the real face
normals. `clip-fill` needs the irradiance to be known.
