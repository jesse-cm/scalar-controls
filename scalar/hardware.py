"""Hardware inputs: ``hardware.json`` at the repository root, read and checked.

One file holds every hardware number a bench test can replace. Each entry is

    {"value": ..., "units": "...", "source": "assumed" | "datasheet" | "measured", "note": "..."}

grouped by device (``SCHEMA`` below lists every entry, its units and its shape).
``value`` null means unknown: the entry is not applied and the code's own
default stands. The units of an entry are fixed: a bench result in other units
is converted before it is typed in, and a units string that differs from
``SCHEMA`` is refused rather than guessed at.

``load()`` reads the file and raises ``HardwareFileError`` (a ValueError) with
one message naming the entry for any mistake: an unknown or missing device or
entry, a missing or extra field, a wrong shape, a number that is not finite, a
``source`` that is not one of the three words, a units string that does not
match, or a value outside what the entry can be (a negative noise level, an
integration time the VEML6031 does not have, an inertia that is not symmetric
positive definite, an unreadable temperature log).

WHO READS IT
    as_built_fields()     the ScenarioSpec fields of the as-built rows (scalar/harness/baseline.py,
                          experiment "hw"): every applicable entry, or only those whose source is
                          not "assumed"
    temperature_fields()  the board-temperature fields, also used by scripts/gyro_realism.py and
                          scripts/mag_realism.py
    Hardware.require()    one value, for the realism scripts (scripts/*_realism.py)
The reference runs (experiments 1-6, --gate) never call this module, so editing
the file cannot change them.

WHAT IS NOT IN THE FILE, and why: see ``NOT_IN_FILE``.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import DEG

__all__ = ["HARDWARE_JSON", "SOURCES", "NOT_ASSUMED", "SCHEMA", "NOT_IN_FILE", "HardwareFileError",
           "Entry", "Hardware", "load", "as_built_fields", "temperature_fields"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARDWARE_JSON = os.path.join(ROOT, "hardware.json")
SOURCES = ("assumed", "datasheet", "measured")
NOT_ASSUMED = ("datasheet", "measured")
FIELDS = ("value", "units", "source", "note")
ABOUT = "about"                 # optional top-level key: free text (a string or a list of strings)

# device -> entry -> (units, shape). Shape: () one number, (n,) n numbers, (n, m) n rows of m numbers,
# "int" a whole number >= 0, "path" a file path (relative to the folder of the JSON file).
SCHEMA = OrderedDict([
    ("gyro", OrderedDict([
        ("noise_density", ("deg/s/sqrt(Hz)", ())),
        ("bias_random_walk", ("rad/s/sqrt(s)", ())),
        ("initial_bias", ("deg/s", (3,))),
        ("scale_factor", ("fraction", (3,))),
        ("misalignment", ("deg", (3,))),
        ("lsb", ("deg/s", ())),
        ("temp_coeff", ("deg/s/K", (3,))),
    ])),
    ("magnetometer", OrderedDict([
        ("noise_std", ("T", ())),
        ("hard_iron_bias", ("T", (3,))),
        ("soft_iron", ("-", (3, 3))),
        ("lsb", ("T", ())),
        ("temp_coeff", ("T/K", (3,))),
    ])),
    ("sun_sensor", OrderedDict([
        ("integration_time", ("ms", ())),
        ("normals", ("-", (7, 3))),
        ("gains", ("-", (7,))),
        ("dark_offset", ("counts", ())),
        ("noise", ("fraction of full scale", ())),
        ("mounting_tilt_sigma", ("deg", ())),
        ("mounting_tilt_seed", ("-", "int")),
    ])),
    ("board_temperature", OrderedDict([
        ("swing_amplitude", ("K", ())),
        ("swing_phase", ("deg", ())),
        ("log_csv", ("path", "path")),
    ])),
    ("coils", OrderedDict([
        ("dipole_per_amp", ("A m^2/A", (3, 3))),
        ("max_current", ("A", (3,))),
    ])),
    ("body", OrderedDict([
        ("inertia", ("kg m^2", (3, 3))),
        ("residual_dipole", ("A m^2", (3,))),
    ])),
])

# Entry -> (ScenarioSpec field, conversion from the file's units to the field's). The as-built
# rows set these; the temperature entries go through temperature_fields().
_SPEC_FIELD = OrderedDict([
    ("gyro.noise_density", ("gyro_noise_density", lambda v: v * DEG)),
    ("gyro.bias_random_walk", ("gyro_bias_rw", None)),
    ("gyro.initial_bias", ("gyro_initial_bias_deg_s", None)),
    ("gyro.scale_factor", ("gyro_scale_factor", None)),
    ("gyro.misalignment", ("gyro_misalign_deg", None)),
    ("gyro.lsb", ("gyro_quant_deg_s", None)),
    ("magnetometer.noise_std", ("mag_noise_std", None)),
    ("magnetometer.hard_iron_bias", ("mag_bias_T", None)),
    ("magnetometer.soft_iron", ("mag_soft_iron", None)),
    ("magnetometer.lsb", ("mag_quant_T", None)),
    ("sun_sensor.integration_time", ("veml_integration_ms", None)),
    ("sun_sensor.normals", ("veml_normals", None)),
    ("sun_sensor.gains", ("veml_gains", None)),
    ("sun_sensor.dark_offset", ("veml_dark_counts", None)),
    ("sun_sensor.noise", ("sun_noise_frac", None)),
    ("coils.dipole_per_amp", ("coil_matrix", None)),
    ("coils.max_current", ("coil_i_max", None)),
    ("body.inertia", ("inertia", None)),
    ("body.residual_dipole", ("m_res", None)),
])
# Entries in the file that the as-built rows do not apply, and why.
_STUDY_ONLY = {
    "sun_sensor.mounting_tilt_sigma": "used only by the cube_tilt layout of scripts/sun_realism.py; "
                                      "the as-built rows fly sun_sensor.normals",
    "sun_sensor.mounting_tilt_seed": "used only by the cube_tilt layout of scripts/sun_realism.py; "
                                     "the as-built rows fly sun_sensor.normals",
}

# Hardware numbers in the code that have no entry, and why (reported, not read).
NOT_IN_FILE = (
    ("coupling of coil current and radio into the magnetometer (phase-5 bench item)",
     "no model of it in scalar/; adding one is a change to sensor physics"),
    ("VEML6031 lux ceiling per integration time (MAX_LUX_G05_PD4, datasheet Table 10)",
     "a datasheet table in scalar/sensors/veml6031.py; the as-built rows choose from it through "
     "sun_sensor.integration_time"),
    ("the flight-like photodiode cube of experiments 1 and 5 (gain spread, roll-off, tilt)",
     "a model of a generic cube, not of the VEML6031 array the spacecraft flies"),
    ("SunSensorArray responsivity, area, dark floor, validity gate",
     "scale factors of the generic cube; the VEML6031 model does not use them"),
    ("calibration of the gyro scale factor and misalignment from bench values",
     "a bench value enters as an uncorrected sensor error; correcting it in the estimator is a "
     "change to estimator code"),
    ("sun-sensor normals known to the estimator",
     "measured normals enter the truth only; the analytic reconstruction still assumes "
     "veml6031.VEML6031_NORMALS (scalar/harness/spec.py, sun_array notes)"),
    ("MEKF noise settings matched to measured sensor noise",
     "the closed loop builds its own MEKF with fixed settings (scalar/control/tune.py)"),
)

_POSITIVE = {"gyro.noise_density", "gyro.lsb", "magnetometer.noise_std", "magnetometer.lsb",
             "sun_sensor.noise", "coils.max_current", "sun_sensor.gains"}
_NON_NEGATIVE = {"gyro.bias_random_walk", "sun_sensor.dark_offset", "sun_sensor.mounting_tilt_sigma",
                 "board_temperature.swing_amplitude"}
VEML_INTEGRATION_MS = (400, 200, 100, 50, 25, 12.5, 6.25)    # veml6031.MAX_LUX_G05_PD4 keys


class HardwareFileError(ValueError):
    """A mistake in hardware.json; the message names the file and the entry."""


@dataclass(frozen=True)
class Entry:
    device: str
    name: str
    value: Any                  # None, float, int, str (a path), or tuples of floats
    units: str
    source: str
    note: str

    @property
    def key(self):
        return f"{self.device}.{self.name}"


class Hardware:
    """A checked hardware.json. ``entries``: OrderedDict 'device.name' -> Entry, in file-schema order.
    ``board_log``: (times [s] from 0, temperature minus its mean [K]) of board_temperature.log_csv, or None."""

    def __init__(self, path, entries, board_log=None):
        self.path = path
        self.entries = entries
        self.board_log = board_log

    def entry(self, device, name):
        return self.entries[f"{device}.{name}"]

    def value(self, device, name):
        return self.entry(device, name).value

    def require(self, device, name, who="this script"):
        """The value of an entry; HardwareFileError if it is null."""
        v = self.value(device, name)
        if v is None:
            raise HardwareFileError(f"{self.path}: {device}.{name} is null, but {who} needs a value for it")
        return v


# ------------------------------------------------------------------ reading and checking
def _fail(path, key, msg):
    raise HardwareFileError(f"{path}: {key}: {msg}")


def _no_duplicates(pairs):
    seen = OrderedDict()
    for k, v in pairs:
        if k in seen:
            raise HardwareFileError(f"key {k!r} appears twice in one object")
        seen[k] = v
    return seen


def _number(path, key, x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        _fail(path, key, f"expected a number, got {json.dumps(x)}")
    if not math.isfinite(x):
        _fail(path, key, f"{x!r} is not a finite number")
    return float(x)


def _shaped(path, key, v, shape):
    """The JSON value as a float, a tuple of floats or a tuple of tuples, checked against ``shape``."""
    if shape == ():
        return _number(path, key, v)
    words = {1: "a list of {} numbers", 2: "a list of {} lists of {} numbers"}[len(shape)].format(*shape)
    if not isinstance(v, list) or len(v) != shape[0]:
        _fail(path, key, f"expected {words}, got {json.dumps(v)}")
    if len(shape) == 1:
        return tuple(_number(path, key, x) for x in v)
    rows = []
    for row in v:
        if not isinstance(row, list) or len(row) != shape[1]:
            _fail(path, key, f"expected {words}, got {json.dumps(v)}")
        rows.append(tuple(_number(path, key, x) for x in row))
    return tuple(rows)


def _read_log(path, key, csv_path):
    """(t from 0 [s], T - mean(T) [K]) from a two-column CSV with a header row."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    except OSError as e:
        _fail(path, key, f"cannot read {csv_path} ({e.strerror or e})")
    if rows:
        try:
            [float(c) for c in rows[0]]
        except ValueError:
            rows = rows[1:]                         # the header row
    if len(rows) < 2:
        _fail(path, key, f"{csv_path}: needs at least two rows of (time s, temperature C)")
    t, T = [], []
    for i, r in enumerate(rows, 1):
        if len(r) != 2:
            _fail(path, key, f"{csv_path}: data row {i} has {len(r)} columns; expected time s, temperature C")
        try:
            a, b = float(r[0]), float(r[1])
        except ValueError:
            _fail(path, key, f"{csv_path}: data row {i} is not two numbers: {r}")
        if not (math.isfinite(a) and math.isfinite(b)):
            _fail(path, key, f"{csv_path}: data row {i} has a number that is not finite")
        t.append(a)
        T.append(b)
    if any(t2 <= t1 for t1, t2 in zip(t, t[1:])):
        _fail(path, key, f"{csv_path}: times must increase from row to row")
    mean = float(np.mean(T))
    return tuple(x - t[0] for x in t), tuple(x - mean for x in T)


def _check_entry(path, key, value, units, source, note, shape, want_units):
    if source not in SOURCES:
        _fail(path, key, f"source {source!r} is not one of {', '.join(SOURCES)}")
    if units != want_units:
        _fail(path, key, f"units {units!r}, but this entry is in {want_units!r}: convert the number to "
                         f"{want_units!r} and write units {want_units!r}")
    if not isinstance(note, str):
        _fail(path, key, "note must be text")
    if value is None:
        return None
    if shape == "int":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail(path, key, f"expected a whole number >= 0, got {json.dumps(value)}")
        return value
    if shape == "path":
        if not isinstance(value, str) or not value.strip():
            _fail(path, key, f"expected a file path (text), got {json.dumps(value)}")
        return value
    v = _shaped(path, key, value, shape)
    flat = np.ravel(np.array(v, float))
    if key in _POSITIVE and not np.all(flat > 0):
        _fail(path, key, f"every number must be > 0, got {json.dumps(value)}")
    if key in _NON_NEGATIVE and not np.all(flat >= 0):
        _fail(path, key, f"every number must be >= 0, got {json.dumps(value)}")
    if key == "sun_sensor.integration_time" and v not in VEML_INTEGRATION_MS:
        _fail(path, key, f"{value!r} ms is not a VEML6031 integration time; one of "
                         f"{', '.join(f'{x:g}' for x in VEML_INTEGRATION_MS)}")
    if key == "sun_sensor.normals" and np.any(np.linalg.norm(np.array(v), axis=1) < 1e-9):
        _fail(path, key, "every normal needs a direction (a zero row was given)")
    if key == "body.inertia":
        J = np.array(v)
        if not np.allclose(J, J.T, rtol=0, atol=1e-12 * np.abs(J).max()) or np.any(np.linalg.eigvalsh(J) <= 0):
            _fail(path, key, "an inertia matrix must be symmetric with all principal moments > 0")
    if key in ("coils.dipole_per_amp", "magnetometer.soft_iron") and abs(np.linalg.det(np.array(v))) < 1e-12:
        _fail(path, key, "the matrix is singular")
    return v


def load(path=None):
    """Read and check ``path`` (default: hardware.json at the repository root). Returns a Hardware."""
    path = HARDWARE_JSON if path is None else os.path.abspath(path)
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f, object_pairs_hook=_no_duplicates, parse_constant=float)
    except OSError as e:
        raise HardwareFileError(f"{path}: cannot read ({e.strerror or e})") from None
    except HardwareFileError as e:
        raise HardwareFileError(f"{path}: {e}") from None
    except ValueError as e:
        raise HardwareFileError(f"{path}: not valid JSON ({e})") from None
    if not isinstance(raw, dict):
        raise HardwareFileError(f"{path}: the file must hold one JSON object, device name -> entries")

    unknown = [k for k in raw if k not in SCHEMA and k != ABOUT]
    if unknown:
        raise HardwareFileError(f"{path}: {unknown[0]}: unknown device; the devices are {', '.join(SCHEMA)}")
    about = raw.get(ABOUT, "")
    if not (isinstance(about, str) or (isinstance(about, list) and all(isinstance(x, str) for x in about))):
        raise HardwareFileError(f"{path}: {ABOUT}: must be text or a list of text lines")

    entries = OrderedDict()
    for device, spec in SCHEMA.items():
        if device not in raw:
            raise HardwareFileError(f"{path}: {device}: device missing; every device of the schema must be "
                                    f"present (an unknown value is written as null)")
        group = raw[device]
        if not isinstance(group, dict):
            raise HardwareFileError(f"{path}: {device}: expected an object of entries")
        for name in group:
            if name not in spec:
                _fail(path, f"{device}.{name}", f"unknown entry; {device} has {', '.join(spec)}")
        for name, (want_units, shape) in spec.items():
            key = f"{device}.{name}"
            if name not in group:
                _fail(path, key, "entry missing (an unknown value is written as null)")
            e = group[name]
            if not isinstance(e, dict):
                _fail(path, key, f"expected an object with {', '.join(FIELDS)}")
            missing = [k for k in FIELDS if k not in e]
            extra = [k for k in e if k not in FIELDS]
            if missing or extra:
                _fail(path, key, (f"missing {', '.join(missing)}" if missing else "")
                      + ("; " if missing and extra else "") + (f"unknown field {', '.join(extra)}" if extra else "")
                      + f" (each entry has exactly {', '.join(FIELDS)})")
            v = _check_entry(path, key, e["value"], e["units"], e["source"], e["note"], shape, want_units)
            entries[key] = Entry(device, name, v, e["units"], e["source"], e["note"])

    amp, phase, log = (entries[f"board_temperature.{n}"].value for n in ("swing_amplitude", "swing_phase", "log_csv"))
    if log is not None and (amp is not None or phase is not None):
        _fail(path, "board_temperature.log_csv", "a log replaces the swing: set swing_amplitude and swing_phase "
                                                 "to null, or log_csv to null")
    if phase is not None and amp is None:
        _fail(path, "board_temperature.swing_phase", "set, but swing_amplitude is null: nothing to phase")
    board_log = None
    if log is not None:
        csv_path = log if os.path.isabs(log) else os.path.join(os.path.dirname(path), log)
        board_log = _read_log(path, "board_temperature.log_csv", csv_path)
    return Hardware(path, entries, board_log)


# ------------------------------------------------------------------ hardware.json -> ScenarioSpec fields
def _usable(hw, key, sources):
    """(applies?, reason if not)."""
    e = hw.entries[key]
    if e.value is None:
        return False, "value is null"
    if sources is not None and e.source not in sources:
        return False, f"source is {e.source}"
    return True, ""


def temperature_fields(hw, sensors=("gyro", "mag"), sources=None, required=False):
    """ScenarioSpec fields that switch on the temperature drift of ``sensors`` ('gyro', 'mag').

    A sensor's drift is on when its temp_coeff applies and a board temperature applies: the swing
    (swing_amplitude, with swing_phase or the harness default 0) or the log (board_temp_log, one
    profile for both sensors). Returns (fields, {entry key: reason it was not applied}).
    ``sources``: None = every non-null entry, else only entries with these sources.
    ``required``: raise HardwareFileError instead of leaving a requested sensor's drift off.
    """
    coeff = {"gyro": ("gyro.temp_coeff", "gyro_temp_coeff_deg_s_per_K"),
             "mag": ("magnetometer.temp_coeff", "mag_temp_coeff_T_per_K")}
    prefix = {"gyro": "gyro_temp", "mag": "mag_temp"}
    skipped = {}
    use_amp, why_amp = _usable(hw, "board_temperature.swing_amplitude", sources)
    use_phase, why_phase = _usable(hw, "board_temperature.swing_phase", sources)
    use_log, why_log = _usable(hw, "board_temperature.log_csv", sources)
    profile = "log" if use_log else "swing" if use_amp else None
    fields, on = {}, []
    for s in sensors:
        key, field = coeff[s]
        ok, why = _usable(hw, key, sources)
        if ok and profile is None:
            ok, why = False, ("no board temperature applies (board_temperature.swing_amplitude: "
                              f"{why_amp}; board_temperature.log_csv: {why_log})")
        if not ok:
            if required:
                raise HardwareFileError(f"{hw.path}: {key}: the temperature drift needs it and a board "
                                        f"temperature, but {why}")
            skipped[key] = why
            continue
        on.append(s)
        fields[field] = hw.entries[key].value
        if profile == "swing":
            fields[f"{prefix[s]}_amp_K"] = hw.entries["board_temperature.swing_amplitude"].value
            if use_phase:
                fields[f"{prefix[s]}_phase_deg"] = hw.entries["board_temperature.swing_phase"].value
    if profile == "log" and on:
        fields["board_temp_log"] = hw.board_log
    board = {"board_temperature.swing_amplitude": (use_amp, why_amp),
             "board_temperature.swing_phase": (use_phase, why_phase),
             "board_temperature.log_csv": (use_log, why_log)}
    for key, (ok, why) in board.items():
        if not ok:
            skipped[key] = why
        elif not on:
            skipped[key] = "no temperature coefficient applies"
        elif profile == "log" and key != "board_temperature.log_csv":
            skipped[key] = "board_temperature.log_csv is used instead"
    return fields, skipped


def as_built_fields(hw, sources=None):
    """(ScenarioSpec fields of the as-built rows, {entry key: reason it was not applied}).

    ``sources`` None applies every non-null entry; a tuple (e.g. NOT_ASSUMED) applies only entries
    whose source is in it. An entry not applied leaves its field at the value the reference run has.
    The sun sensor is always the VEML6031 array (sun_array="veml6031"), the part the spacecraft flies;
    a sun_sensor entry not applied leaves VEML6031Array's own default.
    """
    fields, skipped = {"sun_array": "veml6031"}, {}
    for key, (field, conv) in _SPEC_FIELD.items():
        ok, why = _usable(hw, key, sources)
        if not ok:
            skipped[key] = why
            continue
        v = hw.entries[key].value
        fields[field] = conv(v) if conv else v
    t_fields, t_skipped = temperature_fields(hw, ("gyro", "mag"), sources)
    fields.update(t_fields)
    skipped.update(t_skipped)
    for key, why in _STUDY_ONLY.items():
        skipped[key] = why
    order = list(hw.entries)
    return fields, OrderedDict(sorted(skipped.items(), key=lambda kv: order.index(kv[0])))
