"""Effect tests for plan row 1c, magnetometer realism (docs/verification-loop-plan.md, rule 3).

The plan's assertion: "knowledge degrades; ellipsoid calibration recovers most
of it". Made numeric here on the open-loop scenario e2_alongtrack_0km (free
tumble, MEKF, zero reference-field error), seeds 0-3, metric = median over the
scoring window of the MEKF-vs-truth angle, averaged over the seeds:

    all terms on, no calibration   worse than baseline by at least 0.5 deg
    all terms on, calibrated       at least 60 % of that loss recovered

Measured when the margins were written (seeds 0-3): baseline 0.43, on 1.48,
calibrated 0.60 deg; loss 1.05 deg, 83 % recovered. The margins are about half
the measured loss and three quarters of the measured recovery: wide enough for
the seed spread (the per-seed loss ranges 0.99-1.14 deg), tight enough to fail
if the bias / soft-iron plumbing or the calibration is lost.

Each term also has its own assertion, because a switch nobody has exercised is
what rule 3 forbids:

    quantisation   output on the 0.15 uT grid, error RMS = LSB / sqrt(12) within 10 %;
                   on knowledge it moves the seed-mean by less than 0.02 deg (measured
                   0.0004): at 0.3 uT noise the 0.15 uT step is not visible, and the test
                   pins that finding instead of pretending an effect
    temperature    offset follows k_T (T(t) - T_ref) exactly; with the ASSUMED
                   coefficient and profile the knowledge median rises by at least
                   0.03 deg (measured 0.065)

ASSUMED values (scripts/mag_realism.py): temperature coefficient
(30, -20, 10) nT/K and a 15 K once-per-orbit sinusoid. Neither is in the repo;
the plan's phase-5 temperature ramp with the TMP112s replaces both.
Bias and soft iron are the rl_tune.Plant defaults (scalar/control/rl_tune.py
lines 75-77); the 0.15 uT step is the plan's.
"""
import functools
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import mag_realism as mr  # noqa: E402
from scalar.attitude import IDENTITY_QUAT, to_body  # noqa: E402
from scalar.control.rl_tune import default_plant  # noqa: E402
from scalar.harness import ScenarioSpec, baseline_specs, run_case  # noqa: E402
from scalar.harness import runner  # noqa: E402
from scalar.sensors.magnetometer import (LIS2MDL_LSB_T, Magnetometer, sinusoid_temperature,  # noqa: E402
                                         tabulated_temperature)

SEEDS = (0, 1, 2, 3)
MIN_LOSS_DEG = 0.5          # all terms on, uncalibrated, vs baseline
MIN_RECOVERED = 0.6         # share of that loss the ellipsoid calibration takes back
QUANT_MAX_SHIFT_DEG = 0.02  # quantisation alone: no visible knowledge effect
TEMP_MIN_LOSS_DEG = 0.03    # temperature drift alone, ASSUMED coefficient and profile

B_ECI = np.array([2.0e-5, -1.3e-5, 3.1e-5])


@functools.lru_cache(maxsize=None)
def _know(config):
    """Seed-mean of the knowledge median and p95 [deg] for one open-loop configuration."""
    kw = {
        "off": {},
        "quant": mr.realism_fields(bias_soft_iron=False, temperature=False),
        "temp": mr.realism_fields(bias_soft_iron=False, quantisation=False),
        "on_uncal": mr.realism_fields(),
        "on_cal": dict(mr.realism_fields(), mag_calibrated=True),
    }[config]
    ms = [run_case(mr.open_loop_base().replace(name=f"t_{config}", **kw), sd) for sd in SEEDS]
    return (float(np.mean([m.know_med_all for m in ms])), float(np.mean([m.know_p95_all for m in ms])))


# ---------------------------------------------------------------------------
# defaults: nothing moves
# ---------------------------------------------------------------------------


def test_switches_default_to_off():
    s = ScenarioSpec(name="x")
    assert s.mag_quant_T is None and s.mag_temp_coeff_T_per_K is None and s.mag_temp_amp_K is None
    assert s.mag_temp_phase_deg == 0.0 and s.mag_calibrated is False
    m = runner._magnetometer(s)
    assert m.quantisation is None and m.temp_coeff is None and m.temperature is None
    for b in baseline_specs():
        m = runner._magnetometer(b)
        assert m.quantisation is None and m.temp_coeff is None and not b.mag_calibrated


def test_default_measurement_is_the_old_arithmetic_bit_for_bit():
    """Realism off: measure() == D B_body + b + noise, same generator draws, every call."""
    p = default_plant()
    q = np.array([0.5, -0.5, 0.5, 0.5])
    mag = Magnetometer(bias=p.mag_bias, soft_iron=p.mag_soft_iron)
    rng_a, rng_b = np.random.default_rng(11), np.random.default_rng(11)
    for _ in range(5):
        want = p.mag_soft_iron @ to_body(q, B_ECI) + p.mag_bias + rng_b.normal(0.0, mag.noise_std, size=3)
        assert np.array_equal(mag.measure(B_ECI, q, rng_a), want)
    mag.reset()
    assert np.array_equal(mag.measure(B_ECI, q, np.random.default_rng(11)),
                          Magnetometer(bias=p.mag_bias, soft_iron=p.mag_soft_iron).measure(
                              B_ECI, q, np.random.default_rng(11)))


# ---------------------------------------------------------------------------
# each term at the sensor
# ---------------------------------------------------------------------------


def test_quantisation_puts_output_on_the_lsb_grid_with_uniform_error():
    assert LIS2MDL_LSB_T == 0.15e-6                     # plan row 1c
    rng = np.random.default_rng(0)
    ideal, quant = Magnetometer(noise_std=0.0), Magnetometer(noise_std=0.0, quantisation=LIS2MDL_LSB_T)
    err = []
    for _ in range(4000):
        B = rng.normal(0.0, 3e-5, size=3)
        out = quant.measure(B, IDENTITY_QUAT, rng)
        counts = out / LIS2MDL_LSB_T
        assert np.allclose(counts, np.round(counts), atol=1e-6)
        err.append(out - ideal.measure(B, IDENTITY_QUAT, rng))
    err = np.array(err)
    assert np.max(np.abs(err)) <= 0.5 * LIS2MDL_LSB_T * (1 + 1e-9)
    rms = float(np.sqrt(np.mean(err ** 2)))
    assert rms == pytest.approx(LIS2MDL_LSB_T / np.sqrt(12.0), rel=0.10)    # 0.043 uT


def test_temperature_drift_is_coeff_times_temperature_on_the_sample_clock():
    period, amp, k = 400.0, 15.0, np.array(mr.ASSUMED_TEMP_COEFF_T_PER_K)
    mag = Magnetometer(noise_std=0.0, temp_coeff=k, sample_dt=2.0,
                       temperature=sinusoid_temperature(0.0, amp, period))
    rng = np.random.default_rng(0)
    base = to_body(IDENTITY_QUAT, B_ECI)
    drift = np.array([mag.measure(B_ECI, IDENTITY_QUAT, rng) - base for _ in range(200)])   # one period
    t = 2.0 * np.arange(200)
    assert np.allclose(drift, np.outer(amp * np.sin(2 * np.pi * t / period), k), atol=1e-15)
    assert np.allclose(drift[50], amp * k)                      # quarter period: the full swing, 0.45 uT on x
    assert np.max(np.abs(drift[:, 0])) == pytest.approx(0.45e-6)
    # reset rewinds the clock; an explicit time overrides it; a reference temperature shifts it
    mag.reset()
    assert np.allclose(mag.measure(B_ECI, IDENTITY_QUAT, rng), base)
    assert np.allclose(mag.measure(B_ECI, IDENTITY_QUAT, rng, t=100.0), base + amp * k)
    warm = Magnetometer(noise_std=0.0, temp_coeff=k, temp_ref=25.0, temperature=lambda t: 35.0)
    assert np.allclose(warm.measure(B_ECI, IDENTITY_QUAT, rng), base + 10.0 * k)
    assert not Magnetometer(temp_coeff=k).temperature_offset(123.0).any()       # no profile -> no drift


def test_tabulated_profile_replays_a_logged_orbit():
    """A TMP112 log (time, deg C) drives the same drift as the function it was sampled from."""
    period = 5700.0
    f = sinusoid_temperature(12.0, 15.0, period, 0.4)
    t_log = np.arange(0.0, period, 10.0)
    g = tabulated_temperature(t_log, [f(t) for t in t_log], period_s=period)
    for t in (0.0, 123.4, 5699.0, 5700.0 + 123.4, 3 * period + 4000.0):
        assert g(t) == pytest.approx(f(t), abs=2e-3)
    held = tabulated_temperature([0.0, 100.0], [10.0, 20.0])
    assert held(50.0) == pytest.approx(15.0) and held(1e4) == 20.0


# ---------------------------------------------------------------------------
# plumbing: spec -> sensor -> run
# ---------------------------------------------------------------------------


def test_spec_fields_reach_the_sensor():
    s = mr.open_loop_base().replace(name="x", **mr.realism_fields())
    p = default_plant()
    m = runner._magnetometer(s)
    assert np.array_equal(m.bias, p.mag_bias) and np.array_equal(m.soft_iron, p.mag_soft_iron)
    assert m.quantisation == 0.15e-6
    assert np.array_equal(m.temp_coeff, mr.ASSUMED_TEMP_COEFF_T_PER_K) and m.temp_ref == 0.0
    period = runner._orbit(s).period
    assert m.temperature(period / 4) == pytest.approx(mr.ASSUMED_TEMP_AMP_K)
    assert m.temperature(period) == pytest.approx(0.0, abs=1e-9)
    assert m.temperature(0.0) == 0.0
    shifted = runner._magnetometer(s.replace(mag_temp_phase_deg=90.0))
    assert shifted.temperature(0.0) == pytest.approx(mr.ASSUMED_TEMP_AMP_K)


def test_validate_rejects_what_cannot_be_honoured():
    ol = mr.open_loop_base()
    ol.replace(name="x", mag_calibrated=True, **mr.realism_fields()).validate()
    mr.closed_loop_base().replace(name="x", mag_calibrated=True, **mr.realism_fields()).validate()
    with pytest.raises(ValueError):
        ol.replace(name="x", mag_temp_coeff_T_per_K=(1e-9, 0.0, 0.0)).validate()        # no profile
    with pytest.raises(ValueError):
        ol.replace(name="x", mag_temp_amp_K=10.0).validate()                            # no coefficient
    with pytest.raises(ValueError):
        ol.replace(name="x", mag_temp_phase_deg=30.0).validate()                        # nothing to phase
    with pytest.raises(ValueError):
        ol.replace(name="x", mag_calibrated=True, field_truth="igrf").validate()
    sysid = [s for s in baseline_specs(["3"])][0]
    sunvec = [s for s in baseline_specs(["1"]) if s.kind == "sun_vector"][0]
    for other in (sysid, sunvec):
        for kw in (dict(mag_quant_T=0.15e-6), dict(mag_calibrated=True),
                   dict(mag_temp_coeff_T_per_K=(1e-9, 0.0, 0.0), mag_temp_amp_K=10.0)):
            with pytest.raises(ValueError):
                other.replace(name="x", **kw).validate()


def test_the_four_ledger_scenarios():
    specs = [s for s in mr.realism_specs() if s.experiment in ("2", "4")]
    assert [s.name for s in specs] == ["e2_alongtrack_0km+mag1c_uncal", "e2_alongtrack_0km+mag1c_cal",
                                       "e4D_acq120_mekf+mag1c_uncal", "e4D_acq120_mekf+mag1c_cal"]
    assert [s.mag_calibrated for s in specs] == [False, True, False, True]
    for s in specs:
        s.validate()
        base = mr.open_loop_base() if s.kind == "estimation" else mr.closed_loop_base()
        on = dict(mr.realism_fields(), mag_calibrated=s.mag_calibrated)
        assert s == base.replace(name=s.name, **on)                 # nothing else differs from the baseline row


def test_calibration_in_the_open_loop_run_is_the_rl_tune_pass():
    """estimation + mag_calibrated: the MEKF sees cal.apply(raw), cal from rl_tune.calibration_pass."""
    from scalar.control.rl_tune import calibration_pass
    from scalar.estimator.mekf import MEKF, run_mekf
    from scalar.scenario import Scenario
    from scalar.dynamics import RigidBody
    from scalar.timeutil import julian_date
    from scalar.constants import DEG
    s = mr.open_loop_base().replace(name="x", n_orbits=0.05, mag_calibrated=True, **mr.realism_fields())
    got = run_case(s, 3)

    orbit, mag = runner._orbit(s), runner._magnetometer(s)
    cal, dir_err = calibration_pass(runner._plant(s), mag, orbit, julian_date(runner._epoch(s)), 3)
    mag.reset()
    res = Scenario(epoch=runner._epoch(s), orbit=orbit, body=RigidBody(runner._inertia(s)),
                   q0=np.array(s.q0, float), omega0=np.array(s.omega0_deg_s, float) * DEG, dt=s.dt,
                   duration=s.n_orbits * orbit.period, magnetometer=mag, seed=3).run()
    res = res.__class__(**{**res.__dict__, "mag_body": cal.apply(res.mag_body)})
    err = run_mekf(res, MEKF(n_iter=s.mekf_n_iter)).err_deg
    assert got.know_med_all == float(np.median(err[len(err) // 5:]))
    assert got.extra["cal_dir_err_med"] == dir_err


# ---------------------------------------------------------------------------
# rule 3: the effect on MEKF knowledge
# ---------------------------------------------------------------------------


def test_quantisation_alone_does_not_move_knowledge():
    off, quant = _know("off")[0], _know("quant")[0]
    assert quant != off                                 # the switch is live ...
    assert abs(quant - off) < QUANT_MAX_SHIFT_DEG       # ... and invisible next to 0.3 uT noise


def test_temperature_drift_alone_degrades_knowledge():
    assert _know("temp")[0] - _know("off")[0] >= TEMP_MIN_LOSS_DEG


def test_realism_degrades_knowledge_and_calibration_recovers_most_of_it():
    (off, off95), (on, on95), (cal, cal95) = _know("off"), _know("on_uncal"), _know("on_cal")
    loss = on - off
    assert loss >= MIN_LOSS_DEG, (off, on)
    assert (on - cal) / loss >= MIN_RECOVERED, (off, on, cal)
    assert cal > off                                    # not all of it: the fit cannot see a rotation or the drift
    # the same statement holds for the tail
    assert on95 - off95 >= MIN_LOSS_DEG and (on95 - cal95) / (on95 - off95) >= MIN_RECOVERED, (off95, on95, cal95)
