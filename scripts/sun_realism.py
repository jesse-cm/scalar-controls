#!/usr/bin/env python3
"""Ledger rows for plan row 1f (VEML6031 sun-sensor model), written through scripts/matrix.py.

    python3 scripts/sun_realism.py --list --experiments 1f,1f-normals,1f-albedo
    python3 scripts/sun_realism.py --seeds 0-39 --workers 4                            # 1f
    python3 scripts/sun_realism.py --experiments 1f-normals --seeds 0-7 --workers 4    # the sweep
    python3 scripts/sun_realism.py --experiments 1f-albedo --seeds 0-39 --workers 4

This file only supplies the scenario list; argument parsing, the run, the
aggregation and the append to results/ledger.csv are scripts/matrix.py's
``main``, unchanged. (--gate is refused: these scenarios have no doc numbers.
--experiments defaults to 1f here.)

Two base scenarios, both with the ideal sun unit and albedo 0, so the rows of
the same name without a suffix are the switch-off rows:

    s6_fine20_calmag_lqr_tuned_3mode   closed loop, tuned three-mode controller, fine pointing
    e2_alongtrack_0km                  open loop, free tumble, MEKF, zero reference error

EXPERIMENT 1f. Each base with the VEML array in its nominal layout:

    +veml25         25 ms, the module default: clips whenever the spacecraft is lit
    +veml6.25       6.25 ms, see NON_CLIPPING_MS
    +veml25_fill    25 ms with VEML6031Array's clip_mode="fill" (analytic front end only)

``+veml25`` and ``+veml6.25`` are run with three front ends, told apart by the
ledger's estimator column: mekf+analytic, mekf+mlp_a, mekf+mlp_b. Both MLPs are
retrained on the VEML array of that row (scalar/learning/sun_mlp.py).

EXPERIMENT 1f-normals. The physical normals of the seven sensors are unknown
(notes, section 9). Every entry of LAYOUTS is flown as the truth at
NON_CLIPPING_MS, on both bases, 8 seeds each:

    <base>+veml6.25_L-<layout>

The analytic front end always reconstructs as if the layout were "cube"
(veml6031.VEML6031_NORMALS): its rows say what a wrong assumption costs. The
MLP front ends are trained on the layout that is flown, which is what a ground
sweep of the assembled unit would give them: their rows say whether the
network then needs the normals stated at all.

EXPERIMENT 1f-albedo. The tuned controller with Earth albedo on (0.3, 0.5):

    s6_fine20_calmag_lqr_tuned_3mode+alb<a>             cube; analytic, mlp_b, none
    s6_acq120_calmag_lqr_tuned_3mode+alb<a>             cube; analytic, mlp_b
    s6_fine20_calmag_lqr_tuned_3mode+alb<a>+veml6.25    VEML array; analytic, mlp_b

"none" is sun_front="none": the sun sensor is never used (magnetometer and gyro
only, the MEKF starts from the identity quaternion instead of TRIAD). Albedo
enters through the sun sensor only, so the two "none" rows are the same run.

WHERE THE NUMBERS COME FROM
    lux per count, 65535 ceiling, 133,300 lx at 1 AU, cosine response, dark offset,
    25 ms default            scalar/sensors/veml6031.py and the datasheet pages it cites
    noise, gains             config B's ideal cube, as veml6031.py takes them from its base:
                             2 % of face-on full scale per sample, unit gains
    6.25 ms                  computed below from veml6031.MAX_LUX_G05_PD4
    layouts                  LAYOUTS below; faces 5, 6 on -Z and 7 on +Z as veml6031.py states
    tilt of "cube_tilt2"     ASSUMED, see ASSUMED_TILT_DEG
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scalar.constants import SOLAR_CONSTANT_1AU  # noqa: E402
from scalar.harness.baseline import baseline_specs, tuned_controller_specs  # noqa: E402
from scalar.learning.sun_mlp import annual_irradiance_range  # noqa: E402
from scalar.sensors.veml6031 import LUX_1AU, MAX_LUX_G05_PD4, VEML6031_NORMALS, VEML6031Array  # noqa: E402

FLIGHT_MS = VEML6031Array().integration_ms      # 25: the module default


def non_clipping_integration_ms():
    """Longest integration time of Table 10 whose ceiling is above face-on sun all year."""
    face_on_max = LUX_1AU * annual_irradiance_range()[1] / SOLAR_CONSTANT_1AU
    return max(it for it, ceiling in MAX_LUX_G05_PD4.items() if ceiling > face_on_max)


NON_CLIPPING_MS = non_clipping_integration_ms()  # 6.25 (12.5 ms tops out at 114,083 lx, below face-on sun)

# ASSUMED. 1-sigma mounting tilt of each sensor normal [deg], used only by the "cube_tilt2"
# layout. No file in this repo gives a mounting tolerance; 2.0 is the default tilt_deg of
# sun_mlp.flight_like_sensor, and the draw below is that function's (normal + tilt, renormalised).
# Replaced by the plan's phase-5 rotary-stage sweep of the VEML6031s (measured normals).
ASSUMED_TILT_DEG = 2.0
ASSUMED_TILT_SEED = 1       # the unit that is drawn; flight_like_sensor's unit in the notes is seed 1 too

_PX, _NX, _PY, _NY, _PZ, _NZ = (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)


def _tilted(normals, tilt_deg, seed):
    rng = np.random.default_rng(seed)
    n = np.array(normals, float)
    n = n + np.deg2rad(tilt_deg) * rng.standard_normal(n.shape)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _as_tuple(normals):
    return tuple(tuple(float(x) for x in n) for n in normals)


# The candidate set of the sweep: physical outward normals of devicetree faces (0, 1, 2, 3, 5, 6, 7).
# What is not known is which side face is which (faces 0-3), and, given that the face numbering
# disagrees between three files (notes, section 9), whether the Z faces are the right way up.
LAYOUTS = {
    # the six cube faces, -Z carried twice: what veml6031.py and every reconstruction assume
    "cube": _as_tuple(VEML6031_NORMALS),
    # side faces as a body turned 90 deg about +Z
    "side_rot90": (_PY, _NY, _NX, _PX, _NZ, _NZ, _PZ),
    # side faces as a body turned 180 deg about +Z
    "side_rot180": (_NX, _PX, _NY, _PY, _NZ, _NZ, _PZ),
    # faces 0 and 1 exchanged, faces 2 and 3 as assumed
    "side_mirror_x": (_NX, _PX, _PY, _NY, _NZ, _NZ, _PZ),
    # faces numbered going round the body, so neither (0, 1) nor (2, 3) is an opposing pair
    "side_ring": (_PX, _PY, _NX, _NY, _NZ, _NZ, _PZ),
    # Z faces upside down: 5 and 6 on +Z, 7 on -Z
    "z_flip": (_PX, _NX, _PY, _NY, _PZ, _PZ, _NZ),
    # the assumed faces, each normal tilted (ASSUMED_TILT_DEG, ASSUMED_TILT_SEED)
    "cube_tilt2": _as_tuple(_tilted(VEML6031_NORMALS, ASSUMED_TILT_DEG, ASSUMED_TILT_SEED)),
}

MLP_FRONTS = ("mlp_a", "mlp_b")
ALBEDOS = (0.3, 0.5)


def veml_fields(integration_ms, layout="cube", clip_mode=None):
    """ScenarioSpec keyword arguments that put the VEML array in place of the cube."""
    kw = {"sun_array": "veml6031", "veml_integration_ms": integration_ms, "veml_clip_mode": clip_mode}
    if layout != "cube":
        kw["veml_normals"] = LAYOUTS[layout]
    return kw


def _tuned(name):
    (s,) = [s for s in tuned_controller_specs() if s.name == name]
    return s


def fine_base():
    return _tuned("s6_fine20_calmag_lqr_tuned_3mode")


def acq_base():
    return _tuned("s6_acq120_calmag_lqr_tuned_3mode")


def open_loop_base():
    (s,) = [s for s in baseline_specs() if s.name == "e2_alongtrack_0km"]
    return s


def realism_specs():
    """Experiment 1f: both bases, VEML at 25 ms and at the non-clipping time, every front end."""
    out = []
    for base in (fine_base(), open_loop_base()):
        for it in (FLIGHT_MS, NON_CLIPPING_MS):
            for front in ("analytic", *MLP_FRONTS):
                out.append(base.replace(name=f"{base.name}+veml{it:g}", experiment="1f", sun_front=front,
                                        **veml_fields(it)))
        out.append(base.replace(name=f"{base.name}+veml{FLIGHT_MS:g}_fill", experiment="1f",
                                **veml_fields(FLIGHT_MS, clip_mode="fill")))
    return out


def normals_specs():
    """Experiment 1f-normals: every layout flown as the truth at the non-clipping time."""
    out = []
    for base in (fine_base(), open_loop_base()):
        for layout in LAYOUTS:
            for front in ("analytic", *MLP_FRONTS):
                out.append(base.replace(name=f"{base.name}+veml{NON_CLIPPING_MS:g}_L-{layout}",
                                        experiment="1f-normals", sun_front=front,
                                        **veml_fields(NON_CLIPPING_MS, layout)))
    return out


def albedo_specs():
    """Experiment 1f-albedo: the tuned controller with Earth albedo on."""
    out = []
    for a in ALBEDOS:
        kw = dict(experiment="1f-albedo", albedo_coeff=a)
        fine, acq = fine_base(), acq_base()
        for front in ("analytic", "mlp_b", "none"):
            out.append(fine.replace(name=f"{fine.name}+alb{a:g}", sun_front=front, **kw))
        for front in ("analytic", "mlp_b"):
            out.append(acq.replace(name=f"{acq.name}+alb{a:g}", sun_front=front, **kw))
        for front in ("analytic", "mlp_b"):
            out.append(fine.replace(name=f"{fine.name}+alb{a:g}+veml{NON_CLIPPING_MS:g}", sun_front=front,
                                    **kw, **veml_fields(NON_CLIPPING_MS)))
    return out


def _specs_for_matrix(experiments=None):
    specs = realism_specs() + normals_specs() + albedo_specs()
    if experiments is None:
        return specs
    want = {str(e) for e in experiments}
    return [s for s in specs if s.experiment in want]


def main(argv=None):
    import matrix
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--gate" in argv:
        sys.exit("sun_realism.py: --gate does not apply (no doc numbers for these scenarios)")
    if not any(a == "--experiments" or a.startswith("--experiments=") for a in argv):
        argv = ["--experiments", "1f", *argv]
    matrix.select_specs = _specs_for_matrix
    return matrix.main(argv)


if __name__ == "__main__":
    sys.exit(main())
