"""Tests: hardware.json and scalar/hardware.py (no simulation runs here).

    - the shipped file validates, and holds every entry of the schema
    - each kind of mistake gives one HardwareFileError naming the entry
    - the realism scripts read their numbers from the file, and while an entry is still the
      assumed one they read exactly the constant it replaced (old literals pinned below)
    - the as-built specs (experiment "hw") follow the file; the baseline, experiment 6 and the
      --gate seeds do not change when the file changes, nor break when it is broken
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scalar import hardware  # noqa: E402
from scalar.constants import DEG  # noqa: E402
from scalar.harness import baseline  # noqa: E402

with open(hardware.HARDWARE_JSON, encoding="utf-8") as _f:
    SHIPPED = json.load(_f)


def _write(tmp_path, data, name="hardware.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data, indent=2))
    return str(path)


def _edited(tmp_path, edit):
    data = copy.deepcopy(SHIPPED)
    edit(data)
    return _write(tmp_path, data)


# ------------------------------------------------------------------ the shipped file
def test_shipped_file_validates_and_holds_every_entry():
    hw = hardware.load()
    assert list(hw.entries) == [f"{d}.{n}" for d, entries in hardware.SCHEMA.items() for n in entries]
    for e in hw.entries.values():
        assert e.source in hardware.SOURCES and e.units == hardware.SCHEMA[e.device][e.name][0]
        assert e.note.strip(), e.key
    # the one datasheet value the code cites, and everything else assumed; nothing measured yet
    assert {k for k, e in hw.entries.items() if e.source != "assumed"} == {"sun_sensor.dark_offset"}


def test_shipped_values_are_the_code_defaults():
    """The file was filled from the code as it stands: the values the reference runs use."""
    from scalar.control.rl_tune import default_plant
    from scalar.dynamics import default_inertia
    from scalar.sensors.gyro import Gyro
    from scalar.sensors.magnetometer import LIS2MDL_LSB_T, Magnetometer
    from scalar.sensors.sun_sensor import SunSensorArray
    from scalar.sensors.veml6031 import VEML6031_NORMALS, VEML6031Array
    from scalar.sysid.magnetorquer_id import s3_coil_matrix
    hw, g, m, p, M, i_max = hardware.load(), Gyro(), Magnetometer(), default_plant(), *s3_coil_matrix()
    v = hw.value
    assert v("gyro", "noise_density") * DEG == g.rate_noise_density
    assert v("gyro", "bias_random_walk") == g.bias_rw
    assert np.array_equal(np.array(v("gyro", "initial_bias")) * DEG, g.initial_bias)
    assert v("magnetometer", "noise_std") == m.noise_std and v("magnetometer", "lsb") == LIS2MDL_LSB_T
    assert np.array_equal(v("magnetometer", "hard_iron_bias"), p.mag_bias)
    assert np.array_equal(v("magnetometer", "soft_iron"), p.mag_soft_iron)
    assert np.array_equal(v("body", "residual_dipole"), p.m_res)
    assert np.array_equal(v("body", "inertia"), default_inertia())
    assert np.array_equal(v("coils", "dipole_per_amp"), M) and np.array_equal(v("coils", "max_current"), i_max)
    veml = VEML6031Array()
    assert v("sun_sensor", "integration_time") == veml.integration_ms
    assert np.array_equal(v("sun_sensor", "normals"), VEML6031_NORMALS)
    assert np.array_equal(v("sun_sensor", "gains"), veml.gains)
    assert v("sun_sensor", "dark_offset") == veml.dark_counts
    assert v("sun_sensor", "noise") == SunSensorArray().noise_frac
    # the mission numbers are the orbit and start date of every closed-loop run (spec and plant defaults)
    from scalar.harness.spec import ScenarioSpec
    spec, plant = ScenarioSpec(name="x"), default_plant()
    assert v("mission", "altitude") * 1e3 == spec.altitude_m
    assert v("mission", "inclination") == spec.inclination_deg == plant.inclination_deg
    assert v("mission", "raan") == spec.raan_deg
    assert hardware._epoch_tuple(v("mission", "epoch")) == spec.epoch
    assert hardware._epoch_tuple(v("mission", "epoch")) == plant.epoch.timetuple()[:6]


def test_not_in_file_names_real_files():
    """Each reason in NOT_IN_FILE that names a file of this repo names one that exists."""
    import re
    for item, why in hardware.NOT_IN_FILE:
        for path in re.findall(r"(?:scalar|scripts)/[\w/]+\.py", item + " " + why):
            assert os.path.isfile(os.path.join(ROOT, path)), (item, path)


# ------------------------------------------------------------------ mistakes
def _set(device, name, field, value):
    def edit(d):
        d[device][name][field] = value
    return edit


def _del(path):
    def edit(d):
        node = d
        for k in path[:-1]:
            node = node[k]
        del node[path[-1]]
    return edit


def _swing_and_log(d):
    d["board_temperature"]["log_csv"]["value"] = "log.csv"


MISTAKES = [   # (id, edit, words the message must contain)
    ("unknown device", lambda d: d.update(gyroscope={}), ["gyroscope", "unknown device"]),
    ("missing device", _del(["coils"]), ["coils", "device missing"]),
    ("unknown entry", lambda d: d["gyro"].update(noise={}), ["gyro.noise", "unknown entry"]),
    ("missing entry", _del(["gyro", "lsb"]), ["gyro.lsb", "entry missing"]),
    ("extra field", lambda d: d["gyro"]["lsb"].update(unit="deg/s"), ["gyro.lsb", "unknown field unit"]),
    ("missing field", _del(["gyro", "lsb", "note"]), ["gyro.lsb", "missing note"]),
    ("short vector", _set("gyro", "initial_bias", "value", [0.2, 0.1]), ["gyro.initial_bias", "3 numbers"]),
    ("matrix shape", _set("magnetometer", "soft_iron", "value", [[1, 0, 0], [0, 1, 0]]),
     ["magnetometer.soft_iron", "3 lists of 3"]),
    ("text for a number", _set("gyro", "lsb", "value", "0.00875"), ["gyro.lsb", "expected a number"]),
    ("boolean for a number", _set("body", "residual_dipole", "value", [True, 0, 0]),
     ["body.residual_dipole", "expected a number"]),
    ("infinite", _set("gyro", "lsb", "value", float("inf")), ["gyro.lsb", "not a finite number"]),
    ("not a number", _set("magnetometer", "temp_coeff", "value", [float("nan"), 0, 0]),
     ["magnetometer.temp_coeff", "not a finite number"]),
    ("bad source", _set("gyro", "lsb", "source", "estimated"), ["gyro.lsb", "assumed, datasheet, measured"]),
    ("wrong units", _set("gyro", "noise_density", "units", "mdps/sqrt(Hz)"),
     ["gyro.noise_density", "deg/s/sqrt(Hz)", "convert"]),
    ("negative noise", _set("magnetometer", "noise_std", "value", -3e-7), ["magnetometer.noise_std", "> 0"]),
    ("integration time", _set("sun_sensor", "integration_time", "value", 30), ["sun_sensor.integration_time", "6.25"]),
    ("inertia", _set("body", "inertia", "value", [[1e-3, 1e-4, 0], [0, 1e-3, 0], [0, 0, 1e-3]]),
     ["body.inertia", "symmetric"]),
    ("seed", _set("sun_sensor", "mounting_tilt_seed", "value", 1.5), ["sun_sensor.mounting_tilt_seed", "whole number"]),
    ("swing and log", _swing_and_log, ["board_temperature.log_csv", "replaces the swing"]),
    ("phase alone", _set("board_temperature", "swing_amplitude", "value", None),
     ["board_temperature.swing_phase", "nothing to phase"]),
    ("date format", _set("mission", "epoch", "value", "2026-09-11 12:00"), ["mission.epoch", "YYYY-MM-DDTHH:MM:SS"]),
    ("date as numbers", _set("mission", "epoch", "value", [2026, 9, 11, 12, 0, 0]), ["mission.epoch", "YYYY-MM-DDTHH:MM:SS"]),
    ("inclination", _set("mission", "inclination", "value", 198.6), ["mission.inclination", "between 0 and 180"]),
    ("altitude", _set("mission", "altitude", "value", -550.0), ["mission.altitude", "> 0"]),
    ("altitude in m", _set("mission", "altitude", "units", "m"), ["mission.altitude", "'km'", "convert"]),
]


@pytest.mark.parametrize("edit, words", [m[1:] for m in MISTAKES], ids=[m[0] for m in MISTAKES])
def test_each_mistake_names_the_entry(tmp_path, edit, words):
    path = _edited(tmp_path, edit)
    with pytest.raises(hardware.HardwareFileError) as e:
        hardware.load(path)
    msg = str(e.value)
    assert msg.startswith(path) and "\n" not in msg
    for w in words:
        assert w in msg, (w, msg)


@pytest.mark.parametrize("text, words", [
    ("{not json", ["not valid JSON"]),
    ('{"gyro": {}, "gyro": {}}', ["'gyro' appears twice"]),
    ("[1, 2]", ["one JSON object"]),
])
def test_unreadable_file_is_one_clear_error(tmp_path, text, words):
    path = tmp_path / "hardware.json"
    path.write_text(text)
    with pytest.raises(hardware.HardwareFileError) as e:
        hardware.load(str(path))
    assert all(w in str(e.value) for w in words), str(e.value)
    with pytest.raises(hardware.HardwareFileError, match="cannot read"):
        hardware.load(str(tmp_path / "none.json"))


def test_a_null_value_is_unknown_and_not_applied(tmp_path):
    hw = hardware.load(_edited(tmp_path, _set("gyro", "lsb", "value", None)))
    fields, skipped = hardware.as_built_fields(hw)
    assert "gyro_quant_deg_s" not in fields and skipped["gyro.lsb"] == "value is null"
    with pytest.raises(hardware.HardwareFileError, match="gyro.lsb is null"):
        hw.require("gyro", "lsb")


# ------------------------------------------------------------------ the realism scripts (decision 5)
# The constants the three scripts defined before they read hardware.json (commit 1785b73).
OLD_LITERALS = {
    "gyro.scale_factor": tuple(0.01 * s for s in (1.0, 1.0, 1.0)),           # PLAN_SCALE_FACTOR x SIGNS
    "gyro.misalignment": tuple(float(x) for x in 0.5 * np.array((1.0, 1.0, 1.0)) / np.linalg.norm((1.0, 1.0, 1.0))),
    "gyro.lsb": 8.75e-3,
    "gyro.temp_coeff": (0.010, -0.010, 0.010),
    "magnetometer.temp_coeff": (30e-9, -20e-9, 10e-9),
    "board_temperature.swing_amplitude": 15.0,
    "board_temperature.swing_phase": 0.0,
    "sun_sensor.mounting_tilt_sigma": 2.0,
    "sun_sensor.mounting_tilt_seed": 1,
}


def _scripts():
    import gyro_realism as gr
    import mag_realism as mr
    import sun_realism as sr
    return gr, mr, sr


def test_realism_scripts_read_the_file():
    gr, mr, sr = _scripts()
    hw = hardware.load()
    read = {
        "gyro.scale_factor": gr.GYRO_SCALE_FACTOR,
        "gyro.misalignment": gr.GYRO_MISALIGN_DEG,
        "gyro.lsb": gr.ASSUMED_GYRO_LSB_DEG_S,
        "gyro.temp_coeff": gr.ASSUMED_GYRO_TEMP_COEFF_DEG_S_PER_K,
        "magnetometer.temp_coeff": mr.ASSUMED_TEMP_COEFF_T_PER_K,
        "board_temperature.swing_amplitude": mr.ASSUMED_TEMP_AMP_K,
        "board_temperature.swing_phase": mr.ASSUMED_TEMP_PHASE_DEG,
        "sun_sensor.mounting_tilt_sigma": sr.ASSUMED_TILT_DEG,
        "sun_sensor.mounting_tilt_seed": sr.ASSUMED_TILT_SEED,
    }
    assert set(read) == set(OLD_LITERALS)
    for key, v in read.items():
        assert v == hw.entries[key].value, key
    on = gr.realism_fields()
    assert on["gyro_scale_factor"] == read["gyro.scale_factor"]
    assert on["gyro_misalign_deg"] == read["gyro.misalignment"]
    assert on["gyro_quant_deg_s"] == read["gyro.lsb"]
    assert mr.realism_fields()["mag_temp_coeff_T_per_K"] == read["magnetometer.temp_coeff"]


@pytest.mark.parametrize("key", sorted(OLD_LITERALS))
def test_while_assumed_the_scripts_read_the_old_constant(key):
    """Bit for bit, so the realism rows already in the ledger are what the scripts run today.

    Skipped once the entry holds a datasheet or measured value: the scripts then run that.
    """
    e = hardware.load().entries[key]
    if e.source != "assumed":
        pytest.skip(f"{key} is now {e.source}")
    assert e.value == OLD_LITERALS[key] and type(e.value) is type(OLD_LITERALS[key])
    if key == "sun_sensor.mounting_tilt_sigma":
        assert "cube_tilt2" in _scripts()[2].LAYOUTS


# ------------------------------------------------------------------ as-built specs (decisions 3, 4)
def _reference_specs():
    specs, seeds, checks = baseline._collect()
    return specs, seeds, checks, baseline.tuned_controller_specs(), baseline.select_specs(["1", "2", "3", "4", "5", "6"])


def test_editing_the_file_changes_the_as_built_specs_only(tmp_path, monkeypatch):
    before_ref = _reference_specs()
    before_hw = baseline.as_built_specs()

    def edit(d):
        d["gyro"]["lsb"].update(value=0.0175, source="measured")
        d["body"]["residual_dipole"]["value"] = [0.001, 0.0, 0.0]
        d["mission"]["inclination"]["value"] = 97.5
        d["mission"]["epoch"]["value"] = "2027-03-01T06:30:00"
    path = _edited(tmp_path, edit)
    after_hw = baseline.as_built_specs(path)
    assert after_hw != before_hw
    by = {s.name: s for s in after_hw}
    assert by["hw_fine20_all"].gyro_quant_deg_s == 0.0175 and by["hw_fine20_all"].m_res == (0.001, 0.0, 0.0)
    assert by["hw_acq120_all"].inclination_deg == 97.5 and by["hw_acq120_all"].epoch == (2027, 3, 1, 6, 30, 0)
    from scalar.harness import runner
    assert runner._plant(by["hw_acq120_all"]).inclination_deg == 97.5     # the controller is designed on it too
    assert by["hw_fine20_not_assumed"].gyro_quant_deg_s == 0.0175            # measured: in both variants
    assert by["hw_fine20_not_assumed"].m_res == baseline.tuned_controller_specs()[1].m_res   # assumed: not

    monkeypatch.setattr(hardware, "HARDWARE_JSON", path)            # the default file is now the edited one
    assert baseline.as_built_specs() == after_hw
    assert _reference_specs() == before_ref

    broken = tmp_path / "broken.json"
    broken.write_text("{")
    monkeypatch.setattr(hardware, "HARDWARE_JSON", str(broken))
    assert _reference_specs() == before_ref                         # a broken file cannot break them either
    with pytest.raises(hardware.HardwareFileError):
        baseline.select_specs(["hw"])


def test_as_built_specs_are_experiment_6_plus_the_file():
    acq, fine = baseline.tuned_controller_specs((baseline.AS_BUILT_FILE,))
    hw = hardware.load()
    specs = {s.name: s for s in baseline.as_built_specs()}
    assert list(specs) == ["hw_acq120_all", "hw_fine20_all", "hw_acq120_not_assumed", "hw_fine20_not_assumed"]
    for variant, sources in baseline.AS_BUILT_VARIANTS:
        fields, skipped = hardware.as_built_fields(hw, sources)
        for ref, ep in ((acq, "acq120"), (fine, "fine20")):
            s = specs[f"hw_{ep}_{variant}"]
            assert s == ref.replace(name=s.name, experiment="hw", plant_hold="dipole", plant_substeps=2, **fields)
            s.validate()
        # every entry is applied or said not to be, never both
        applied = set(hw.entries) - set(skipped)
        assert applied and all(hw.entries[k].value is not None for k in applied)
        if sources is not None:
            assert all(hw.entries[k].source in sources for k in applied)
    na = hardware.as_built_fields(hw, hardware.NOT_ASSUMED)[0]
    assert na == {"sun_array": "veml6031", "veml_dark_counts": 4.0}
    allf = hardware.as_built_fields(hw)[0]
    assert allf["gyro_noise_density"] == 3.8e-3 * DEG and allf["veml_integration_ms"] == 25
    assert allf["gyro_temp_amp_K"] == allf["mag_temp_amp_K"] == 15.0
    # the mission numbers, while assumed, give exactly the experiment-6 orbit and start date
    assert (allf["altitude_m"], allf["inclination_deg"], allf["raan_deg"], allf["epoch"]) == (
        fine.altitude_m, fine.inclination_deg, fine.raan_deg, fine.epoch)
    assert all(type(x) is int for x in allf["epoch"])


def test_a_temperature_log_replaces_the_swing(tmp_path):
    (tmp_path / "board.csv").write_text("time_s,temp_C\n0,20\n600,30\n1200,25\n1800,15\n")

    def edit(d):
        bt = d["board_temperature"]
        bt["swing_amplitude"]["value"] = bt["swing_phase"]["value"] = None
        bt["log_csv"].update(value="board.csv", source="measured")
    hw = hardware.load(_edited(tmp_path, edit))
    t, dT = hw.board_log
    assert t == (0.0, 600.0, 1200.0, 1800.0) and dT == (-2.5, 7.5, 2.5, -7.5)
    fields, skipped = hardware.as_built_fields(hw)
    assert fields["board_temp_log"] == hw.board_log
    assert "gyro_temp_amp_K" not in fields and "mag_temp_amp_K" not in fields
    assert "gyro_temp_coeff_deg_s_per_K" in fields and "mag_temp_coeff_T_per_K" in fields
    (fine,) = [s for s in baseline.as_built_specs(str(tmp_path / "hardware.json")) if s.name == "hw_fine20_all"]
    fine.validate()
    from scalar.harness import runner
    assert runner._gyro(fine).temperature(2100.0) == pytest.approx(-7.5 + 0.5 * (-2.5 + 7.5))   # wraps to the start
    # the measured log alone, with the coefficients assumed: nothing to drive, so not applied
    na, skipped = hardware.as_built_fields(hw, hardware.NOT_ASSUMED)
    assert "board_temp_log" not in na and skipped["board_temperature.log_csv"] == "no temperature coefficient applies"


@pytest.mark.parametrize("rows, words", [
    ("t,T\n0,20\n", "at least two rows"),
    ("t,T\n0,20\n0,21\n", "times must increase"),
    ("t,T\n0,20,1\n10,21,1\n", "3 columns"),
    ("t,T\n0,20\n10,warm\n", "not two numbers"),
])
def test_a_bad_temperature_log_names_the_entry(tmp_path, rows, words):
    (tmp_path / "board.csv").write_text(rows)

    def edit(d):
        bt = d["board_temperature"]
        bt["swing_amplitude"]["value"] = bt["swing_phase"]["value"] = None
        bt["log_csv"]["value"] = "board.csv"
    with pytest.raises(hardware.HardwareFileError, match="board_temperature.log_csv") as e:
        hardware.load(_edited(tmp_path, edit))
    assert words in str(e.value)


def test_matrix_lists_the_as_built_scenarios_and_not_by_default():
    run = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "matrix.py"), *a],  # noqa: E731
                                    capture_output=True, text=True, cwd=ROOT, timeout=120)
    out = run("--list", "--experiments", "hw")
    assert out.returncode == 0, out.stderr
    names = [line.split()[2] for line in out.stdout.strip().splitlines()[:-1]]
    assert names == ["hw_acq120_all", "hw_fine20_all", "hw_acq120_not_assumed", "hw_fine20_not_assumed"]
    assert not any(s.name.startswith("hw_") for s in baseline.select_specs())
    gate = run("--gate", "--experiments", "hw")
    assert gate.returncode != 0 and "no gate numbers" in gate.stderr
