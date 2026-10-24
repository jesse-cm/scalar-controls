"""Effect tests for plan row 1f, the VEML6031 sun-sensor model (docs/verification-loop-plan.md, rule 3).

The plan's assertion: "analytic reconstruction breaks under saturation; MLP
retrained on this model". Made numeric at two levels. All numbers are with the
ideal unit and no albedo; "measured" is what the run gave when the margins
were written, and every margin is about half the measured effect.

A. AT THE SENSOR. 3000 random sun directions in full sun at 1 AU, through the
   harness's own builders (runner._sun_sensor, runner._sun_front). Metric: the
   angle between the reconstructed and the true sun vector, median.

                                        clipped   median    p95
       cube (SunSensorArray)             -         1.59     3.48 deg
       VEML 25 ms,   analytic            1.000    13.74    20.46
       VEML 6.25 ms, analytic            0.000     1.50     3.28
       VEML 25 ms,   analytic, "fill"    1.000     5.01    17.02
       VEML 25 ms,   MLP-B               1.000     7.82    18.17
       VEML 6.25 ms, MLP-B               0.000     2.60     6.89
       VEML 6.25 ms, side_ring, analytic 0.000    22.20    97.31
       VEML 6.25 ms, side_ring, MLP-B    0.000     2.42     6.17

   - breaks: at 25 ms every lit sample has a clipped sensor and the analytic
     median is at least 6 deg above the 6.25 ms one (measured 12.2);
   - does not break: at 6.25 ms nothing clips and the median is within 0.2 deg
     of the cube's (measured 0.08);
   - retrained MLP against the analytic front end on the same sensor: at 25 ms
     the MLP is better by at least 2.5 deg (measured 5.9) and still at least
     3 deg worse than the unclipped sensor (measured 6.3), so retraining takes
     back about half of what saturation costs; at 6.25 ms the MLP is WORSE than
     the analytic reconstruction by at least 0.4 deg (measured 1.1). The test
     pins both directions as they came out;
   - layout: with the side faces numbered round the body ("side_ring") the
     analytic median is above 10 deg (measured 22.2); the MLP trained on that
     layout is within 0.5 deg of the MLP on the assumed one (measured 0.19).

B. IN THE FILTER. Open loop, e2_alongtrack_0km (free tumble at about 7 deg/s,
   MEKF, zero reference error), seeds 0-3, metric = knowledge median averaged
   over the seeds.

       cube                         0.426 deg
       VEML 25 ms,   analytic       0.753        loss 0.328
       VEML 6.25 ms, analytic       0.425
       VEML 25 ms,   clip "fill"    0.467
       VEML 25 ms,   MLP-B          0.530
       VEML 6.25 ms, MLP-B          0.496
       VEML 6.25 ms, side_ring      1.311   (analytic, wrong side-face assignment)
       no sun sensor                0.453

   - breaks: 25 ms analytic at least 0.15 deg above 6.25 ms (measured 0.33);
   - does not break: 6.25 ms within 0.03 deg of the cube (measured 0.001);
   - MLP-B at 25 ms recovers at least 40 % of that loss (measured 68 %); at
     6.25 ms it is worse than the analytic front end by at least 0.03 deg
     (measured 0.07);
   - clip_mode="fill" recovers at least 60 % of the loss (measured 87 %);
     clip_mode="drop" at 25 ms is the same run as sun_front="none";
   - veml_normals: the "side_ring" layout flown under the cube assumption
     costs the analytic front end at least 0.4 deg (measured 0.89);
   - sun_front="none": in this fast tumble the sun sensor is worth little, the
     loss is between 0.01 and 0.1 deg (measured 0.027). So clipped counts fed
     as they are do more harm than no sun sensor at all: 25 ms analytic is at
     least 0.15 deg above the no-sun run (measured 0.30).

C. AGAINST THE STORED RUN. results/veml6031_baseline.json was produced by
   scripts/veml6031_baseline.py on another loop (estimator_levers.run_episode).
   The harness switch must give its seed-100 numbers for the same scenario
   (tuned three-mode controller, fine pointing): "VEML 25ms", and
   sun_front="none" against "VEML 25ms clip-drop" (no sunlit sample survives
   there, which the notes, section 12, identify with the no-sun-sensor row of
   section 9). In closed loop the sun sensor is worth more: without it the
   pointing mean is at least 0.8 deg above the stored cube run "B" (measured
   1.72: 9.12 against 7.40).

ASSUMED values: none enter these tests except through LAYOUTS["cube_tilt2"]
(scripts/sun_realism.py, ASSUMED_TILT_DEG), which is only checked for shape.
Sensor numbers are scalar/sensors/veml6031.py's.
"""
import functools
import json
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import sun_realism as sr  # noqa: E402
from scalar.constants import SOLAR_CONSTANT_1AU  # noqa: E402
from scalar.harness import ScenarioSpec, baseline_specs, run_case  # noqa: E402
from scalar.harness import runner  # noqa: E402
from scalar.harness.baseline import tuned_controller_specs  # noqa: E402
from scalar.learning.sun_mlp import (angular_error_deg, annual_irradiance_range, make_dataset,  # noqa: E402
                                     sensor_inputs)
from scalar.sensors.sun_sensor import SunSensorArray  # noqa: E402
from scalar.sensors.veml6031 import (COUNT_MAX, LUX_1AU, MAX_LUX_G05_PD4, VEML6031_NORMALS,  # noqa: E402
                                     VEML6031Array)

E = SOLAR_CONSTANT_1AU
SEEDS = (0, 1, 2, 3)
NC = sr.NON_CLIPPING_MS

# A. at the sensor [deg, median sun-vector error]
SAT_MIN_LOSS_DEG = 6.0          # analytic, 25 ms minus 6.25 ms
NOCLIP_MAX_SHIFT_DEG = 0.2      # analytic, 6.25 ms against the cube
MLP_SAT_MIN_GAIN_DEG = 2.5      # 25 ms: analytic minus MLP-B
MLP_SAT_MIN_RESIDUAL_DEG = 3.0  # 25 ms MLP-B minus 6.25 ms analytic
MLP_NOCLIP_MIN_LOSS_DEG = 0.4   # 6.25 ms: MLP-B minus analytic
# B. in the filter [deg, seed-mean of the knowledge median]
KNOW_SAT_MIN_LOSS_DEG = 0.15
KNOW_NOCLIP_MAX_SHIFT_DEG = 0.03
KNOW_MLP_MIN_RECOVERED = 0.4
KNOW_MLP_NOCLIP_MIN_LOSS_DEG = 0.03
KNOW_FILL_MIN_RECOVERED = 0.6
KNOW_LAYOUT_MIN_LOSS_DEG = 0.4
KNOW_NOSUN_LOSS_RANGE_DEG = (0.01, 0.1)     # no sun sensor minus cube
KNOW_SAT_WORSE_THAN_NOSUN_DEG = 0.15        # 25 ms analytic minus no sun sensor
POINT_NOSUN_MIN_LOSS_DEG = 0.8              # C: closed loop, seed 100, against the stored cube run

_CONFIGS = {
    "cube": {},
    "a25": sr.veml_fields(sr.FLIGHT_MS),
    "a6": sr.veml_fields(NC),
    "fill25": sr.veml_fields(sr.FLIGHT_MS, clip_mode="fill"),
    "drop25": sr.veml_fields(sr.FLIGHT_MS, clip_mode="drop"),
    "b25": dict(sr.veml_fields(sr.FLIGHT_MS), sun_front="mlp_b"),
    "b6": dict(sr.veml_fields(NC), sun_front="mlp_b"),
    "ring_a6": sr.veml_fields(NC, "side_ring"),
    "ring_b6": dict(sr.veml_fields(NC, "side_ring"), sun_front="mlp_b"),
    "none": {"sun_front": "none"},
}


def _spec(config):
    return sr.open_loop_base().replace(name=f"t_{config}", **_CONFIGS[config]).validate()


@functools.lru_cache(maxsize=None)
def _static(config, n=3000, seed=9):
    """(clipped fraction, valid fraction, median, p95) of the sun-vector error [deg], full sun, no albedo."""
    spec = _spec(config)
    sensor = runner._sun_sensor(spec, seed)
    front = runner._sun_front(spec, sensor)
    rng = np.random.default_rng(seed)
    sun = rng.standard_normal((n, 3))
    sun /= np.linalg.norm(sun, axis=1, keepdims=True)
    nadir = rng.standard_normal((n, 3))
    nadir /= np.linalg.norm(nadir, axis=1, keepdims=True)
    dayside = rng.uniform(0.0, 1.0, n)
    clipped, err = [], []
    for i in range(n):
        c = sensor.measure(sun[i], E, True, rng)
        clipped.append(spec.sun_array == "veml6031" and bool(np.any(c >= COUNT_MAX)))
        s_hat, ok = sensor.reconstruct(c, E) if front is None else front(c, E, nadir[i], dayside[i])
        if ok:
            err.append(float(angular_error_deg(s_hat, sun[i])))
    return (float(np.mean(clipped)), len(err) / n, float(np.median(err)), float(np.percentile(err, 95)))


@functools.lru_cache(maxsize=None)
def _know(config):
    """Seed-mean of the open-loop knowledge median [deg]."""
    return float(np.mean([run_case(_spec(config), sd).know_med_all for sd in SEEDS]))


# ---------------------------------------------------------------------------
# defaults: nothing moves
# ---------------------------------------------------------------------------


def test_switches_default_to_the_cube():
    s = ScenarioSpec(name="x")
    assert s.sun_array == "cube" and s.sun_front == "analytic"
    assert s.veml_integration_ms is None and s.veml_normals is None and s.veml_clip_mode is None
    assert type(runner._sun_sensor(s)) is SunSensorArray
    for b in baseline_specs() + tuned_controller_specs():
        assert b.sun_array == "cube" and b.sun_front != "none"
        assert type(runner._sun_sensor(b, 3)) is SunSensorArray


def test_cube_dataset_is_the_old_arithmetic_bit_for_bit():
    """make_dataset on the photodiode cube: same draws, same division as before row 1f."""
    sensor = SunSensorArray()
    Xc, Xe, y = make_dataset(sensor, 50, np.random.default_rng(4), albedo_max=0.5)
    rng = np.random.default_rng(4)                      # the function body at commit a11144e
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)  # noqa: E731
    s, nadir = unit(rng.standard_normal((50, 3))), unit(rng.standard_normal((50, 3)))
    coeff, dayside = rng.uniform(0.0, 0.5, 50), rng.uniform(0.0, 1.0, 50)
    X = np.empty((50, 6))
    for i in range(50):
        X[i] = sensor.measure(s[i], 1361.0, True, rng, albedo_dir_body=nadir[i],
                              albedo_irradiance=coeff[i] * 1361.0 * dayside[i])
    X /= sensor.full_scale_current(1361.0)
    assert np.array_equal(Xc, X.astype(np.float32)) and np.array_equal(y, s.astype(np.float32))
    assert np.array_equal(Xe, np.hstack([nadir, dayside[:, None]]).astype(np.float32))


def test_validate_refuses_what_it_cannot_honour():
    base = sr.open_loop_base()
    ok = base.replace(name="v", **sr.veml_fields(NC))
    ok.validate()
    bad = [
        base.replace(name="v", veml_integration_ms=25),                 # veml field on the cube
        base.replace(name="v", veml_clip_mode="fill"),
        base.replace(name="v", veml_normals=sr.LAYOUTS["side_ring"]),
        ok.replace(veml_integration_ms=10),                             # not in Table 10
        ok.replace(veml_normals=sr.LAYOUTS["cube"][:6]),                # six normals
        ok.replace(veml_clip_mode="guess"),
        ok.replace(veml_clip_mode="fill", sun_front="mlp_b"),           # fill is the analytic path's
        ok.replace(sun_unit="flight_like"),                             # roll-off, tilt, gains not plumbed
        ok.replace(sun_array="photodiode"),
        ScenarioSpec(name="v", kind="sun_vector", sun_array="veml6031"),
        ScenarioSpec(name="v", kind="sun_vector", sun_front="none"),
        ScenarioSpec(name="v", kind="sysid", sun_array="veml6031"),
    ]
    for s in bad:
        with pytest.raises(ValueError):
            s.validate()
    base.replace(name="v", sun_front="none").validate()


# ---------------------------------------------------------------------------
# the switch builds the sensor it names
# ---------------------------------------------------------------------------


def test_harness_builds_the_veml_array_as_specified():
    s = _spec("a6").replace(sun_noise_frac=0.01, veml_normals=sr.LAYOUTS["z_flip"], veml_clip_mode="drop")
    v = runner._sun_sensor(s.validate(), 7)
    assert isinstance(v, VEML6031Array) and v.integration_ms == NC and v.clip_mode == "drop"
    assert v.base.noise_frac == 0.01 and np.array_equal(v.gains, np.ones(7))
    assert np.array_equal(v.normals, np.array(sr.LAYOUTS["z_flip"])) and v.true_normals is v.normals
    d = runner._sun_sensor(_spec("a25"), 7)
    assert d.integration_ms == 25 and np.array_equal(d.normals, VEML6031_NORMALS)
    # the second -Z sensor's private noise stream follows the case seed
    a, b = runner._sun_sensor(_spec("a6"), 7), runner._sun_sensor(_spec("a6"), 8)
    assert a.aux_rng.normal() == runner._sun_sensor(_spec("a6"), 7).aux_rng.normal() != b.aux_rng.normal()


def test_non_clipping_time_and_candidate_layouts():
    face_on_max = LUX_1AU * annual_irradiance_range()[1] / SOLAR_CONSTANT_1AU
    assert sr.FLIGHT_MS == 25 and NC == 6.25
    assert MAX_LUX_G05_PD4[25] < 0.5 * face_on_max              # 25 ms clips below half of face-on sun
    assert MAX_LUX_G05_PD4[12.5] < face_on_max                  # the next one down still clips
    assert MAX_LUX_G05_PD4[NC] > 1.5 * face_on_max              # 6.25 ms: sun plus albedo 0.5 on one face fits
    assert list(sr.LAYOUTS)[0] == "cube" and np.array_equal(sr.LAYOUTS["cube"], VEML6031_NORMALS)
    for name, n in sr.LAYOUTS.items():
        n = np.array(n)
        assert n.shape == (7, 3) and np.allclose(np.linalg.norm(n, axis=1), 1.0), name
        if name.startswith("side_"):                            # side faces relabelled, Z faces as assumed
            assert sorted(map(tuple, n[:4].tolist())) == sorted(map(tuple, VEML6031_NORMALS[:4].tolist())), name
            assert np.array_equal(n[4:], VEML6031_NORMALS[4:]) and not np.array_equal(n, VEML6031_NORMALS), name
    assert np.array_equal(np.array(sr.LAYOUTS["z_flip"]), VEML6031_NORMALS * [1, 1, -1])
    tilt = np.degrees(np.arccos(np.sum(np.array(sr.LAYOUTS["cube_tilt2"]) * VEML6031_NORMALS, axis=1)))
    assert 0.0 < tilt.max() < 4 * sr.ASSUMED_TILT_DEG


def test_network_inputs_show_the_clip_plateau():
    v = VEML6031Array(integration_ms=25, base=SunSensorArray(noise_frac=0.0), dark_counts=0.0)
    x = sensor_inputs(v, v.measure([1.0, 0.0, 0.0], E, True, np.random.default_rng(0)), E)
    assert x.shape == (7,) and abs(x[0] - MAX_LUX_G05_PD4[25] / LUX_1AU) < 1e-4 and x[1] == 0.0
    lo, hi = annual_irradiance_range()
    assert lo < E < hi and 0.05 < (hi - lo) / E < 0.08          # the plateau moves by this much over a year
    Xc, Xe, _ = make_dataset(v, 20, np.random.default_rng(1), irradiance=(lo, hi))
    assert Xc.shape == (20, 7) and Xe.shape == (20, 4)
    plateau = Xc.max(axis=1)
    assert plateau.max() - plateau.min() > 0.01                 # ... and the training set sees it move


# ---------------------------------------------------------------------------
# A. at the sensor
# ---------------------------------------------------------------------------


def test_analytic_breaks_under_saturation_and_not_without_it():
    clip25, valid25, med25, _ = _static("a25")
    clip6, valid6, med6, _ = _static("a6")
    _, _, med_cube, _ = _static("cube")
    assert clip25 == 1.0 and clip6 == 0.0
    assert valid25 == 1.0 and valid6 == 1.0                      # clipped counts still pass the gate
    assert med25 - med6 >= SAT_MIN_LOSS_DEG, (med25, med6)
    assert abs(med6 - med_cube) <= NOCLIP_MAX_SHIFT_DEG, (med6, med_cube)


def test_retrained_mlp_against_analytic_on_the_same_sensor():
    med_a25, med_a6 = _static("a25")[2], _static("a6")[2]
    med_b25, med_b6 = _static("b25")[2], _static("b6")[2]
    # saturated: the retrained network is better than the analytic reconstruction ...
    assert med_a25 - med_b25 >= MLP_SAT_MIN_GAIN_DEG, (med_a25, med_b25)
    # ... and nowhere near the unclipped sensor: the information is gone, retraining cannot bring it back
    assert med_b25 - med_a6 >= MLP_SAT_MIN_RESIDUAL_DEG, (med_b25, med_a6)
    # not saturated, ideal unit, no albedo: the network is the worse of the two
    assert med_b6 - med_a6 >= MLP_NOCLIP_MIN_LOSS_DEG, (med_b6, med_a6)


def test_mlp_trained_on_the_flown_layout_does_not_need_the_normals():
    """Wrong side-face assignment: the analytic front end is off by tens of degrees, the MLP is not."""
    assert _static("ring_a6")[2] > 10.0
    assert abs(_static("ring_b6")[2] - _static("b6")[2]) < 0.5


# ---------------------------------------------------------------------------
# B. in the filter (open loop)
# ---------------------------------------------------------------------------


def test_knowledge_degrades_at_25ms_and_not_at_6p25ms():
    assert _know("a25") - _know("a6") >= KNOW_SAT_MIN_LOSS_DEG, (_know("a25"), _know("a6"))
    assert abs(_know("a6") - _know("cube")) <= KNOW_NOCLIP_MAX_SHIFT_DEG, (_know("a6"), _know("cube"))


def test_mlp_in_the_filter_recovers_part_of_the_saturation_loss_only():
    loss = _know("a25") - _know("a6")
    recovered = (_know("a25") - _know("b25")) / loss
    assert recovered >= KNOW_MLP_MIN_RECOVERED, recovered
    assert _know("b6") - _know("a6") >= KNOW_MLP_NOCLIP_MIN_LOSS_DEG, (_know("b6"), _know("a6"))


def test_clip_modes():
    loss = _know("a25") - _know("a6")
    assert (_know("a25") - _know("fill25")) / loss >= KNOW_FILL_MIN_RECOVERED
    # every lit sample is clipped at 25 ms, so "drop" leaves the filter without a sun sensor
    assert abs(_know("drop25") - _know("none")) < 1e-9


def test_wrong_layout_and_no_sun_sensor_both_cost_knowledge():
    assert _know("ring_a6") - _know("a6") >= KNOW_LAYOUT_MIN_LOSS_DEG, (_know("ring_a6"), _know("a6"))
    lo, hi = KNOW_NOSUN_LOSS_RANGE_DEG
    assert lo <= _know("none") - _know("cube") <= hi, (_know("none"), _know("cube"))
    assert _know("a25") - _know("none") >= KNOW_SAT_WORSE_THAN_NOSUN_DEG, (_know("a25"), _know("none"))


# ---------------------------------------------------------------------------
# C. against results/veml6031_baseline.json (closed loop, tuned three-mode controller, seed 100)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config, stored", [
    (dict(sr.veml_fields(25)), "VEML 25ms"),
    (dict(sun_front="none"), "VEML 25ms clip-drop"),
])
def test_closed_loop_reproduces_the_stored_veml_run(config, stored):
    with open(os.path.join(_ROOT, "results", "veml6031_baseline.json")) as f:
        runs = json.load(f)["runs"]
    ref, cube = runs[stored]["seeds"]["100"], runs["B"]["seeds"]["100"]
    m = run_case(sr.fine_base().replace(name="t_stored", **config), 100)
    assert abs(m.point_mean - ref["point_mean"]) < 1e-6
    assert abs(m.point_p95 - ref["point_p95"]) < 1e-6
    assert abs(m.know_med_all - ref["know_med"]) < 1e-6
    assert m.point_mean - cube["point_mean"] >= POINT_NOSUN_MIN_LOSS_DEG     # either way the sun sensor is lost
