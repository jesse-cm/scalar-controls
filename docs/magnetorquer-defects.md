# Hand-built magnetorquer: what can go wrong, how big, and what absorbs it

2026-09-13. Numbers are order-of-magnitude estimates against the S3 coil set
(X/Y 0.058 A m^2, Z 0.0087 A m^2 at 3.3 V) unless a run is cited.

| Defect | Expected size | Effect | Absorbed by | Verified how |
|---|---|---|---|---|
| Effective area / turns error (bulge, non-planar winding) | 5–15 % dipole gain | gain error only | sysid recovers X/Y to 2–3 %; LQR verified insensitive to ±10 % | bench dipole measurement; harness gain sweep |
| Coil axis tilt (mounting, non-planar) | 2–5° -> 3–9 % cross-coupling | off-diagonal M; per-axis saturation wrong | full 3x3 M from sysid + least-squares dipole allocation (to write) | harness tilt sweep 0–10° |
| Copper tempco | +0.39 %/°C -> ±16 % current over 80 °C at fixed voltage | dipole drifts with temperature | current control on the STM32 (ISENSE + NTC exist) | log I(T) on bench |
| Shorted / partial turns | step drop in N and R | dipole loss, hotspot | detect: R = V/I vs nominal; re-identify | R, L per coil before integration |
| Open coil / dead channel | one axis lost | 2-coil control: slower, degraded fine pointing | schedule redesigned with a zero column; sysid detects | harness lost-axis case |
| Pair wired anti-phase (X+ and X- in series) | axis cancels to ~0 | axis lost | polarity check with a magnetometer | ground, before closing the stack |
| Residual spacecraft dipole (screws, battery tabs, DRV/H-bridge) | 1–5 mA m^2 typical; 10–50 with a magnet | steady torque; > Z authority (8.7 mA m^2) kills Z-axis fine pointing | identify (sysid: 0.3 mA m^2) and feed-forward cancel (to write) | harness m_res sweep to 20 mA m^2 |
| Sun-correlated harness/panel dipole (current loops) | 0.5 A × 50 cm^2 = 2.5 mA m^2, varies with illumination | time-varying disturbance the constant m_res term cannot fit | twist/route wiring; regress m_res on INA219 panel currents | harness: m_res(t) model |
| Coil field at the LIS2MDL | 430 µT at 3 cm, 12 µT at 10 cm (Earth: 25–47 µT) | magnetometer unusable while torquing | measure-then-torque duty cycle (authority × duty); sysid uses only pre/post windows (fix needed) | bench coupling matrix C; harness duty sweep |
| Current-sense offset/gain | 1 % gain, mA offsets | M scaled by the sense error (self-consistent); offset = fake residual dipole | sysid absorbs into M and m_res | shunt/amp calibration on bench |
| Coil self-heating | 0.84 W in the X pair at full drive | R rises, current falls, thermal budget | current control; duty limit on NTC | thermal-vac or bench soak |
| Hysteresis material (rods?) | mA m^2, history-dependent | damps rate (good), unmodeled torque in fine pointing (bad) | none if present; decide whether rods are fitted (model: HysteresisRod unset) | ask; then harness |
| PWM / H-bridge EMI at the magnetometer | unknown | noise on B at the switching rate | ODR/filter choice; measure | bench log with coils driven |
| Inertia scale | 5–10 % (CAD) | sysid dipole scale off by the same | benign for control (verified) | mass-properties measurement |

## The three that can actually stop pointing

1. Residual dipole above Z authority. Anything magnetic in the build can do
   it. Feed-forward cancellation works only while |m_res| < m_max.
2. Coil–magnetometer coupling without a duty cycle. Not a defect, a geometry
   fact: the magnetometer must never be read with a coil on.
3. Hysteresis or other history-dependent magnetics. No identification
   algorithm handles it cleanly; it is a build decision.

## Non-trivial tuning that follows from this

- Tune against a distribution of plants, not one: sample gain ±20 %, tilt
  0–8°, m_res 0–10 mA m^2, duty 0.6–1.0 per episode (domain randomisation),
  keep the same eight criteria. Parameters that pass are robust by
  construction; parameters that only pass on the nominal plant are not.
- Re-identify M and m_res periodically in flight and re-derive the schedule
  on the RP2350; treat a change in a column norm > 20 % as a fault flag.
- Ground campaign before any of it: per-coil R, L, dipole at known distance,
  polarity, coupling matrix at the magnetometer, I(T).
