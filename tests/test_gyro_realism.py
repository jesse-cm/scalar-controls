"""Effect tests for plan row 1h, gyro realism (docs/verification-loop-plan.md, rule 3).

The plan's assertion: "bias-estimate error and knowledge shift". Made numeric
here on the open-loop scenario e2_alongtrack_0km (free tumble at 7.1 deg/s,
MEKF, zero reference-field error), seeds 0-3. Two metrics, each the mean over
the seeds of a median over the scoring window:

    bias error   |b_est - b_true| [deg/s], b_true = the additive bias the gyro applied
                 (random walk + temperature drift)
    knowledge    MEKF-vs-truth angle [deg], the harness's know_med_all

Shift against the switch-off run, measured when the margins were written, and
the margin asserted (about half the measured shift, as in the row-1c test):

    term            bias error [deg/s]            knowledge median [deg]
                    measured   asserted           measured   asserted
    off             0.0100 (the level)            0.426 (the level)
    temperature     +0.0250    >= +0.012          +0.211     >= +0.10
    scale factor    +0.0827    >= +0.040          +0.289     >= +0.14
    misalignment    +0.0027    +0.001 .. +0.006   +0.291     >= +0.14
    quantisation    +0.00004   |shift| < 0.0005   -0.0007    |shift| < 0.02
    all four        +0.0902    >= +0.045          +0.644     >= +0.32

Two terms do not behave like the others and the test pins that instead of
claiming an effect that is not there:

    misalignment   moves knowledge as much as the scale factor does but hardly moves the
                   bias estimate: its rate error is at right angles to the rate and turns
                   with the tumble, so little of it looks like a constant bias
    quantisation   at the ASSUMED step (8.75e-3 deg/s) nothing measurable moves; the
                   per-seed knowledge shift is +-0.02 deg of either sign

ASSUMED values (scripts/gyro_realism.py; none is in a file of this repo):
    output step                8.75e-3 deg/s. The plan says "at the FS in use"; the range the
                               flight code sets is not in the repo. Replaced by the range
                               setting in the flight driver and the step seen in the phase-5
                               static log.
    temperature coefficient    (0.010, -0.010, 0.010) deg/s/K. Replaced by the phase-5
                               temperature ramp with the TMP112s.
    board temperature          15 K once-per-orbit sinusoid, the magnetometer row's value
                               (scripts/mag_realism.py), so both sensors see one board.
                               Replaced by a TMP112 log over an orbit.
    scale-factor signs         +1 % on all three axes (the plan gives the size only).
    misalignment axis          the body diagonal (1, 1, 1) (the plan gives the angle only).
The 1 % and the 0.5 deg are the plan's.
"""
import functools
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import gyro_realism as gr  # noqa: E402
import mag_realism as mr  # noqa: E402
from scalar.attitude import quat_from_rotvec, to_body  # noqa: E402
from scalar.constants import DEG  # noqa: E402
from scalar.harness import ScenarioSpec, baseline_specs, run_case  # noqa: E402
from scalar.harness import runner  # noqa: E402
from scalar.harness.baseline import tuned_controller_specs  # noqa: E402
from scalar.sensors.gyro import Gyro  # noqa: E402
from scalar.sensors.magnetometer import sinusoid_temperature  # noqa: E402

SEEDS = (0, 1, 2, 3)

# shift of the seed-mean against the switch-off run: (bias error [deg/s], knowledge median [deg])
TEMP_MIN = (0.012, 0.10)
TEMP_MIN_P95_DEG = 0.7              # temperature drift shows mostly in the knowledge tail (measured +1.54)
SCALE_MIN = (0.040, 0.14)
MISALIGN_BIAS_RANGE = (0.001, 0.006)
MISALIGN_MIN_KNOW_DEG = 0.14
QUANT_MAX = (0.0005, 0.02)          # ASSUMED step: no visible effect
ALL_MIN = (0.045, 0.32)
ALL_MIN_P95_DEG = 0.8               # measured +1.59

NEW_FIELDS = ("gyro_scale_factor", "gyro_misalign_deg", "gyro_quant_deg_s",
              "gyro_temp_coeff_deg_s_per_K", "gyro_temp_amp_K")
W = np.array([4.0, -3.0, 5.0]) * DEG        # the open-loop scenario's initial rate


def _quiet(**kw):
    """A gyro with no noise, no random walk and no bias: only the realism terms act."""
    return Gyro(rate_noise_density=0.0, bias_rw=0.0, initial_bias=np.zeros(3), **kw)


@functools.lru_cache(maxsize=None)
def _ol(config):
    """Seed-mean (bias-error median [deg/s], knowledge median [deg], knowledge p95 [deg]), open loop."""
    kw = {"off": {}, "all": gr.realism_fields(), **{t: gr.only(t) for t in gr.TERMS}}[config]
    rows = [gr.open_loop_bias_error(gr.open_loop_base().replace(name=f"t_{config}", **kw), sd) for sd in SEEDS]
    return tuple(float(np.mean([r[k] for r in rows])) for k in ("bias_err_med", "know_med", "know_p95"))


def _shift(config):
    return tuple(a - b for a, b in zip(_ol(config), _ol("off")))


# ---------------------------------------------------------------------------
# defaults: nothing moves
# ---------------------------------------------------------------------------


def test_switches_default_to_off():
    s = ScenarioSpec(name="x")
    assert all(getattr(s, f) is None for f in NEW_FIELDS) and s.gyro_temp_phase_deg == 0.0
    for spec in [s, *baseline_specs(), *tuned_controller_specs()]:
        assert all(getattr(spec, f) is None for f in NEW_FIELDS)
        g = runner._gyro(spec)
        assert g.scale_factor is None and g.misalignment is None and g.quantisation is None
        assert g.temp_coeff is None and g.temperature is None
        assert np.array_equal(g.sensing_matrix(), np.eye(3)) and not g.temperature_offset(1234.0).any()


def test_default_measurement_is_the_old_arithmetic_bit_for_bit():
    """Realism off: measure() == omega + bias + white, then the bias walks; same generator draws."""
    g, dt = Gyro(), 1.0
    rng_a, rng_b = np.random.default_rng(11), np.random.default_rng(11)
    bias = np.asarray(g.initial_bias, float).copy()
    for _ in range(5):
        want = W + bias + rng_b.normal(0.0, g.rate_noise_density / np.sqrt(dt), size=3)
        assert np.array_equal(g.measure(W, dt, rng_a), want)
        bias = bias + rng_b.normal(0.0, g.bias_rw * np.sqrt(dt), size=3)
        assert np.array_equal(g.bias, bias)
    g.reset()
    assert np.array_equal(g.bias, g.initial_bias)


def test_realism_terms_use_the_same_random_draws():
    """Switching the terms on changes no draw: seed k with and without them is a paired comparison."""
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    plain, real = Gyro(), runner._gyro(gr.open_loop_base().replace(name="x", **gr.realism_fields()))
    for _ in range(20):
        plain.measure(W, 1.0, rng_a)
        real.measure(W, 1.0, rng_b)
    assert np.array_equal(plain.bias, real.bias)                    # the random walk is untouched
    assert rng_a.bit_generator.state == rng_b.bit_generator.state


# ---------------------------------------------------------------------------
# each term at the sensor
# ---------------------------------------------------------------------------


def test_scale_factor_scales_each_axis():
    rng = np.random.default_rng(0)
    assert gr.PLAN_SCALE_FACTOR == 0.01                             # plan row 1h
    assert np.allclose(_quiet(scale_factor=0.01).measure(W, 1.0, rng), 1.01 * W, rtol=0, atol=1e-18)
    per_axis = _quiet(scale_factor=np.array([0.01, -0.01, 0.0])).measure(W, 1.0, rng)
    assert np.allclose(per_axis, W * [1.01, 0.99, 1.0], rtol=0, atol=1e-18)
    # 1 % of the 7.07 deg/s tumble: a 0.0707 deg/s rate error along the rate
    err = _quiet(scale_factor=0.01).measure(W, 1.0, rng) - W
    assert np.degrees(np.linalg.norm(err)) == pytest.approx(0.0707, abs=1e-4)
    assert np.allclose(np.cross(err, W), 0.0, atol=1e-18)


def test_misalignment_turns_the_rate_by_half_a_degree():
    assert gr.PLAN_MISALIGN_DEG == 0.5                              # plan row 1h
    rng = np.random.default_rng(0)
    phi = 0.5 * DEG * np.array([0.0, 0.0, 1.0])
    g = _quiet(misalignment=phi)
    x = g.measure([1.0, 0.0, 0.0], 1.0, rng)
    assert np.allclose(x, [np.cos(0.5 * DEG), np.sin(0.5 * DEG), 0.0], atol=1e-15)      # 0.87 % leaks into y
    assert np.array_equal(g.measure([0.0, 0.0, 1.0], 1.0, rng), [0.0, 0.0, 1.0])        # along the axis: untouched
    out = g.measure(W, 1.0, rng)
    assert np.allclose(out, to_body(quat_from_rotvec(phi), W), atol=1e-18)              # attitude.py's convention
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(W), rel=1e-12)           # a rotation: no scale change
    # the row's value: 0.5 deg about the ASSUMED axis, whatever the rate direction at right angles to it
    row = runner._gyro(gr.open_loop_base().replace(name="x", **gr.only("misalignment")))
    assert np.degrees(np.linalg.norm(row.misalignment)) == pytest.approx(0.5)
    v = np.cross(row.misalignment, [1.0, 0.0, 0.0])
    cosang = (row.sensing_matrix() @ v) @ v / (v @ v)
    assert np.degrees(np.arccos(cosang)) == pytest.approx(0.5, rel=1e-6)


def test_scale_factor_and_misalignment_compose():
    s, phi = np.array([0.01, 0.01, 0.01]), 0.5 * DEG * np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    g = _quiet(scale_factor=s, misalignment=phi)
    want = (1.0 + s) * to_body(quat_from_rotvec(phi), W)
    assert np.allclose(g.measure(W, 1.0, np.random.default_rng(0)), want, atol=1e-18)


def test_quantisation_puts_output_on_the_lsb_grid_with_uniform_error():
    lsb = gr.ASSUMED_GYRO_LSB_DEG_S * DEG
    rng = np.random.default_rng(0)
    ideal, quant = _quiet(), _quiet(quantisation=lsb)
    err = []
    for _ in range(4000):
        w = rng.normal(0.0, 5.0 * DEG, size=3)
        out = quant.measure(w, 1.0, rng)
        counts = out / lsb
        assert np.allclose(counts, np.round(counts), atol=1e-6)
        err.append(out - ideal.measure(w, 1.0, rng))
    err = np.array(err)
    assert np.max(np.abs(err)) <= 0.5 * lsb * (1 + 1e-9)
    assert float(np.sqrt(np.mean(err ** 2))) == pytest.approx(lsb / np.sqrt(12.0), rel=0.10)
    # bias and noise are quantised too: the step is applied to the output, not to the rate
    g = Gyro(quantisation=lsb)
    counts = g.measure(W, 1.0, rng) / lsb
    assert np.allclose(counts, np.round(counts), atol=1e-6)


def test_temperature_drift_is_coeff_times_temperature_on_the_gyro_clock():
    period, amp, k = 400.0, 15.0, np.array(gr.ASSUMED_GYRO_TEMP_COEFF_DEG_S_PER_K) * DEG
    g = _quiet(temp_coeff=k, temperature=sinusoid_temperature(0.0, amp, period))
    rng = np.random.default_rng(0)
    drift = np.array([g.measure(W, 2.0, rng) - W for _ in range(200)])       # one period at dt = 2 s
    t = 2.0 * np.arange(200)
    assert np.allclose(drift, np.outer(amp * np.sin(2 * np.pi * t / period), k), atol=1e-15)
    assert np.allclose(drift[50], amp * k)                                   # quarter period: the full swing
    assert np.degrees(np.max(np.abs(drift[:, 0]))) == pytest.approx(0.15)    # 0.15 deg/s on x
    assert not g.bias.any()                                 # ``bias`` stays the random-walk state alone
    # reset rewinds the clock; an explicit time overrides it; a reference temperature shifts it
    g.reset()
    assert np.allclose(g.measure(W, 2.0, rng), W)
    assert np.allclose(g.measure(W, 2.0, rng, t=100.0), W + amp * k)
    warm = _quiet(temp_coeff=k, temp_ref=25.0, temperature=lambda t: 35.0)
    assert np.allclose(warm.measure(W, 1.0, rng), W + 10.0 * k)
    assert not _quiet(temp_coeff=k).temperature_offset(123.0).any()          # no profile -> no drift


# ---------------------------------------------------------------------------
# plumbing: spec -> sensor -> run
# ---------------------------------------------------------------------------


def test_spec_fields_reach_the_sensor():
    s = gr.open_loop_base().replace(name="x", **gr.realism_fields())
    g = runner._gyro(s)
    assert np.allclose(g.scale_factor, 0.01)
    assert np.allclose(g.misalignment, 0.5 * DEG * np.ones(3) / np.sqrt(3.0))
    assert g.quantisation == pytest.approx(gr.ASSUMED_GYRO_LSB_DEG_S * DEG)
    assert np.allclose(g.temp_coeff, np.array(gr.ASSUMED_GYRO_TEMP_COEFF_DEG_S_PER_K) * DEG) and g.temp_ref == 0.0
    period = runner._orbit(s).period
    assert g.temperature(period / 4) == pytest.approx(mr.ASSUMED_TEMP_AMP_K)
    assert g.temperature(0.0) == 0.0
    shifted = runner._gyro(s.replace(gyro_temp_phase_deg=90.0))
    assert shifted.temperature(0.0) == pytest.approx(mr.ASSUMED_TEMP_AMP_K)
    # the terms that were there before are untouched
    old = runner._gyro(gr.open_loop_base())
    assert g.rate_noise_density == old.rate_noise_density and g.bias_rw == old.bias_rw
    assert np.array_equal(g.initial_bias, old.initial_bias)


def test_gyro_and_magnetometer_see_one_board_temperature():
    both = gr.open_loop_base().replace(
        name="x", **gr.realism_fields(), **mr.realism_fields(bias_soft_iron=False, quantisation=False)).validate()
    g, m = runner._gyro(both), runner._magnetometer(both)
    for t in (0.0, 311.0, 1434.7, 5000.0, 12345.6):
        assert g.temperature(t) == m.temperature(t)
    with pytest.raises(ValueError):
        both.replace(gyro_temp_amp_K=mr.ASSUMED_TEMP_AMP_K + 1.0).validate()
    with pytest.raises(ValueError):
        both.replace(gyro_temp_phase_deg=30.0).validate()


def test_the_run_starts_the_gyro_clock_at_zero():
    """In a harness run sample k is taken at k * dt: gyro output minus rate is bias + drift at res.t."""
    from scalar.dynamics import RigidBody
    from scalar.scenario import Scenario
    s = gr.open_loop_base().replace(name="x", n_orbits=0.02, gyro_noise_density=0.0, gyro_bias_rw=0.0,
                                    **gr.only("temperature"))
    g = runner._gyro(s)
    g.measure(W, 1.0, np.random.default_rng(0))             # a used sensor: the run must rewind it
    orbit = runner._orbit(s)
    res = Scenario(epoch=runner._epoch(s), orbit=orbit, body=RigidBody(runner._inertia(s)),
                   q0=np.array(s.q0, float), omega0=np.array(s.omega0_deg_s, float) * DEG, dt=s.dt,
                   duration=s.n_orbits * orbit.period, gyro=g, seed=0).run()
    want = res.gyro_bias_true + np.array([g.temperature_offset(t) for t in res.t])
    assert np.allclose(res.gyro_meas - res.omega_true, want, atol=1e-15)
    assert np.ptp(want[:, 0]) > 0.0                         # and the drift is really in there


def test_validate_rejects_what_cannot_be_honoured():
    ol = gr.open_loop_base()
    ol.replace(name="x", **gr.realism_fields()).validate()
    for cl in gr.closed_loop_bases():
        cl.replace(name="x", **gr.realism_fields()).validate()
    with pytest.raises(ValueError):
        ol.replace(name="x", gyro_temp_coeff_deg_s_per_K=(0.01, 0.0, 0.0)).validate()   # no profile
    with pytest.raises(ValueError):
        ol.replace(name="x", gyro_temp_amp_K=10.0).validate()                           # no coefficient
    with pytest.raises(ValueError):
        ol.replace(name="x", gyro_temp_phase_deg=30.0).validate()                       # nothing to phase
    sysid = [s for s in baseline_specs(["3"])][0]
    sunvec = [s for s in baseline_specs(["1"]) if s.kind == "sun_vector"][0]
    for other in (sysid, sunvec):
        for term in gr.TERMS:
            with pytest.raises(ValueError):
                other.replace(name="x", **gr.only(term)).validate()


def test_the_three_ledger_scenarios():
    specs = [s for s in gr.realism_specs() if s.experiment in ("2", "6")]
    assert [s.name for s in specs] == ["e2_alongtrack_0km+gyro1h",
                                       "s6_acq120_calmag_lqr_tuned_3mode+gyro1h",
                                       "s6_fine20_calmag_lqr_tuned_3mode+gyro1h"]
    assert gr._specs_for_matrix(["2", "6"]) == specs
    for s, base in zip(specs, [gr.open_loop_base(), *gr.closed_loop_bases()]):
        s.validate()
        assert s == base.replace(name=s.name, **gr.realism_fields())    # nothing else differs from the off row
        assert all(getattr(s, f) is not None for f in NEW_FIELDS)       # all four terms on
    parts = gr._specs_for_matrix(["1h-parts"])
    assert [p.name.rsplit("_only_", 1)[1] for p in parts] == list(gr.TERMS)
    for p, term in zip(parts, gr.TERMS):
        assert p == gr.open_loop_base().replace(name=p.name, experiment="1h-parts", **gr.only(term))
    notemp = gr._specs_for_matrix(["1h-notemp"])
    assert [p.name for p in notemp] == [b.name + "+gyro1h_no_temperature" for b in gr.closed_loop_bases()]
    for p in notemp:
        p.validate()
        assert p.gyro_temp_coeff_deg_s_per_K is None and p.gyro_temp_amp_K is None
        assert None not in (p.gyro_scale_factor, p.gyro_misalign_deg, p.gyro_quant_deg_s)


def test_bias_error_helper_runs_the_harness_scenario():
    """open_loop_bias_error scores the same run as run_case: identical knowledge numbers."""
    s = gr.open_loop_base().replace(name="x", n_orbits=0.05, **gr.realism_fields())
    got, m = gr.open_loop_bias_error(s, 3), run_case(s, 3)
    assert got["know_med"] == m.know_med_all and got["know_p95"] == m.know_p95_all
    assert 0.0 < got["bias_err_med"] <= got["bias_err_p95"]


# ---------------------------------------------------------------------------
# rule 3: the effect on the MEKF's bias estimate and on knowledge, one term at a time
# ---------------------------------------------------------------------------


def test_temperature_drift_shifts_bias_estimate_and_knowledge():
    d_bias, d_know, d_p95 = _shift("temperature")
    assert d_bias >= TEMP_MIN[0], (_ol("off"), _ol("temperature"))
    assert d_know >= TEMP_MIN[1], (_ol("off"), _ol("temperature"))
    assert d_p95 >= TEMP_MIN_P95_DEG, (_ol("off"), _ol("temperature"))


def test_scale_factor_shifts_bias_estimate_and_knowledge():
    d_bias, d_know, _ = _shift("scale_factor")
    assert d_bias >= SCALE_MIN[0], (_ol("off"), _ol("scale_factor"))
    assert d_know >= SCALE_MIN[1], (_ol("off"), _ol("scale_factor"))


def test_misalignment_shifts_knowledge_and_barely_the_bias_estimate():
    d_bias, d_know, _ = _shift("misalignment")
    assert d_know >= MISALIGN_MIN_KNOW_DEG, (_ol("off"), _ol("misalignment"))
    assert MISALIGN_BIAS_RANGE[0] <= d_bias <= MISALIGN_BIAS_RANGE[1], (_ol("off"), _ol("misalignment"))
    assert d_bias < 0.1 * _shift("scale_factor")[0]         # an order below the scale factor's


def test_quantisation_alone_moves_neither():
    d_bias, d_know, _ = _shift("quantisation")
    assert _ol("quantisation") != _ol("off")                # the switch is live ...
    assert abs(d_bias) < QUANT_MAX[0], (_ol("off"), _ol("quantisation"))     # ... and invisible at the
    assert abs(d_know) < QUANT_MAX[1], (_ol("off"), _ol("quantisation"))     # ASSUMED step


def test_all_terms_together_shift_bias_estimate_and_knowledge():
    d_bias, d_know, d_p95 = _shift("all")
    assert d_bias >= ALL_MIN[0], (_ol("off"), _ol("all"))
    assert d_know >= ALL_MIN[1], (_ol("off"), _ol("all"))
    assert d_p95 >= ALL_MIN_P95_DEG, (_ol("off"), _ol("all"))
    # larger than any single term
    for term in gr.TERMS:
        assert d_bias > _shift(term)[0] and d_know > _shift(term)[1]
