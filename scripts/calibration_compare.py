#!/usr/bin/env python3
"""Compare sensor-calibration techniques and their effect on MEKF knowledge error.

    python3 scripts/calibration_compare.py [n_runs]

Runs a Monte-Carlo over randomized orbits with realistic magnetometer soft/hard
iron and sun-sensor gain/roll-off errors, and for each run re-estimates attitude
with the MEKF under five calibration regimes:

    none            raw, uncalibrated measurements
    on-ground sun   pre-flight sun-sensor response-curve calibration only
    on-orbit mag    in-flight attitude-independent ellipsoid fit only
    both            on-ground sun + on-orbit mag
    ST-referenced   star-tracker-referenced full soft-iron + bias recovery

The estimator uses the real IGRF field as its onboard reference (not the tilted
dipole), so the residual error isolates the *sensor* calibration effect rather
than the field-model mismatch. Prints a ranked table vs the 10-15 deg budget and
saves a grouped-bar figure + CSV to outputs/.

Determination only -- no control.
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.attitude import to_body                                   # noqa: E402
from scalar.calibration import (                                      # noqa: E402
    calibrate_mag_with_reference,
    calibrate_sun_response,
    fit_magnetometer_ellipsoid,
    sweep_currents,
)
from scalar.environment.orbit import CircularOrbit                    # noqa: E402
from scalar.estimator.mekf import MEKF, run_mekf                      # noqa: E402
from scalar.scenario import Scenario                                  # noqa: E402
from scalar.sensors.magnetometer import Magnetometer                  # noqa: E402
from scalar.sensors.star_tracker import StarTracker                   # noqa: E402
from scalar.sensors.sun_sensor import SunSensorArray                  # noqa: E402

DEG = np.pi / 180.0

# Representative, non-symmetric magnetometer distortion (scale + non-orthogonality
# + boresight misalignment) and hard-iron offset. Fixed across runs so the fitted
# calibrations are compared on the same physical error.
SOFT_IRON = np.array([[1.12, 0.05, -0.04],
                      [0.03, 0.90, 0.06],
                      [-0.05, 0.02, 1.07]])
HARD_IRON = np.array([2.2e-6, -1.5e-6, 3.0e-6])   # [T]

REGIMES = ["none", "on-ground sun", "on-orbit mag", "both", "ST-referenced"]


def _make_sensors(rng):
    """Magnetometer with fixed soft/hard iron; sun array with per-run gains."""
    mag = Magnetometer(bias=HARD_IRON.copy(), soft_iron=SOFT_IRON.copy())
    gains = np.ones(6) * (1.0 + 0.15 * rng.standard_normal(6))
    rolloff = float(np.clip(0.18 + 0.03 * rng.standard_normal(), 0.0, 0.4))
    ss = SunSensorArray(gains=gains, rolloff=rolloff)
    return mag, ss


def _corrected(res, ss, mag_cal=None, sun_gains=None):
    """Copy of ``res`` with calibrated magnetometer / sun measurements."""
    new = copy.copy(res)
    if mag_cal is not None:
        new.mag_body = mag_cal.apply(res.mag_body)
    if sun_gains is not None:
        cur = res.sun_currents / sun_gains[None, :]
        sb = np.full((len(res), 3), np.nan)
        sv = np.zeros(len(res), dtype=bool)
        for k in range(len(res)):
            s_hat, valid = ss.reconstruct(cur[k], res.irradiance[k])
            sv[k] = valid
            if valid:
                sb[k] = s_hat
        new.sun_body = sb
        new.sun_valid = sv
    return new


def _mekf_error(res):
    mk = run_mekf(res, MEKF())
    return mk.err_deg, res.lit


def run_one(seed, n_orbits=2.0, dt=2.0):
    """One randomized orbit; returns {regime: err_deg array} plus diagnostics."""
    rng = np.random.default_rng(seed)
    mag, ss = _make_sensors(rng)
    orb = CircularOrbit(altitude=550e3, inclination=51.6 * DEG, j2=True)
    # slow residual rate so the star tracker can lock (a de-tumbled platform)
    omega0 = np.array([0.3, -0.2, 0.4]) * DEG
    sc = Scenario(orbit=orb, field_truth="igrf", magnetometer=mag, sun_sensor=ss,
                  dt=dt, duration=n_orbits * orb.period, omega0=omega0, seed=seed)
    res = sc.run()
    # onboard reference = real IGRF (we model it), so residual = sensor error only
    res.B_eci_ref = res.B_eci.copy()

    # --- fit calibrations from THIS flight's data / a bench sweep ---
    # on-orbit magnetometer: attitude-independent ellipsoid fit vs |B_IGRF|
    Fref = np.linalg.norm(res.B_eci, axis=1)
    mag_ell = fit_magnetometer_ellipsoid(res.mag_body, Fref)

    # on-ground sun: response-curve sweep on a rate table under known irradiance
    angles = np.deg2rad(np.linspace(0.0, 60.0, 25))
    cur_bench = sweep_currents(ss, angles, 1361.0, np.random.default_rng(seed + 100),
                               noise_frac=0.006)
    sun_cal = calibrate_sun_response(cur_bench, angles, 1361.0, sensor=ss)

    # star-tracker-referenced: recover full D + bias where the tracker has a fix
    st = StarTracker()
    rng_st = np.random.default_rng(seed + 200)
    n = len(res)
    q_st = np.full((n, 4), np.nan)
    st_valid = np.zeros(n, dtype=bool)
    rate = np.linalg.norm(res.omega_true, axis=1)
    for k in range(n):
        r_hat = res.r_eci[k] / np.linalg.norm(res.r_eci[k])
        nadir_body = to_body(res.q_true[k], -r_hat)
        sun_body_true = to_body(res.q_true[k], res.sun_eci[k])
        qm, valid = st.measure(res.q_true[k], rng_st, sun_body_true, nadir_body,
                               rate[k])
        st_valid[k] = valid
        if valid:
            q_st[k] = qm
    idx = np.where(st_valid)[0]
    mag_ref = calibrate_mag_with_reference(res.mag_body[idx], q_st[idx],
                                           res.B_eci[idx])

    out = {}
    diag = dict(seed=seed, st_avail=float(st_valid.mean()),
                mag_ell_resid_nT=float(mag_ell.residual_rms * 1e9),
                mag_ref_D_err=float(np.linalg.norm(mag_ref.D - SOFT_IRON)),
                mag_ref_bias_err_nT=float(np.linalg.norm(mag_ref.b - HARD_IRON) * 1e9),
                sun_gain_err=float(np.linalg.norm(sun_cal.gains - ss.gains)))
    out["none"] = _mekf_error(_corrected(res, ss))
    out["on-ground sun"] = _mekf_error(_corrected(res, ss, sun_gains=sun_cal.gains))
    out["on-orbit mag"] = _mekf_error(_corrected(res, ss, mag_cal=mag_ell))
    out["both"] = _mekf_error(_corrected(res, ss, mag_cal=mag_ell,
                                         sun_gains=sun_cal.gains))
    out["ST-referenced"] = _mekf_error(_corrected(res, ss, mag_cal=mag_ref,
                                                  sun_gains=sun_cal.gains))
    return out, diag


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"Monte-Carlo over {n_runs} randomized IGRF orbits "
          f"(soft+hard iron, sun gain/roll-off; IGRF onboard reference)\n")

    # accumulate per-run medians and pooled samples per regime
    run_med = {r: [] for r in REGIMES}
    run_p95 = {r: [] for r in REGIMES}
    run_ecl = {r: [] for r in REGIMES}
    diags = []
    for s in range(n_runs):
        out, diag = run_one(seed=s)
        diags.append(diag)
        for r in REGIMES:
            e, lit = out[r]
            run_med[r].append(np.median(e))
            run_p95[r].append(np.percentile(e, 95))
            run_ecl[r].append(np.median(e[~lit]) if (~lit).any() else np.nan)
        print(f"  run {s:>2}: " + "  ".join(
            f"{r}={np.median(out[r][0]):.2f}°" for r in REGIMES))

    rows = []
    print(f"\n{'regime':<16}{'median':>9}{'p95':>9}{'eclipse':>9}"
          f"{'vs raw':>9}")
    print("-" * 52)
    raw_med = float(np.median(run_med["none"]))
    for r in REGIMES:
        med = float(np.median(run_med[r]))
        p95 = float(np.median(run_p95[r]))
        ecl = float(np.nanmedian(run_ecl[r]))
        rows.append(dict(regime=r, median_deg=med, p95_deg=p95, eclipse_deg=ecl,
                         improvement_x=raw_med / med if med > 0 else np.nan))
        print(f"{r:<16}{med:>8.2f}°{p95:>8.2f}°{ecl:>8.2f}°"
              f"{raw_med / med:>8.1f}x")

    davg = {k: float(np.mean([d[k] for d in diags]))
            for k in ("st_avail", "mag_ell_resid_nT", "mag_ref_D_err",
                      "mag_ref_bias_err_nT", "sun_gain_err")}
    print(f"\nCalibration quality (mean over runs):")
    print(f"  ellipsoid scalar residual   {davg['mag_ell_resid_nT']:.0f} nT")
    print(f"  ST-ref soft-iron D error    {davg['mag_ref_D_err']:.4f}")
    print(f"  ST-ref hard-iron bias error {davg['mag_ref_bias_err_nT']:.0f} nT")
    print(f"  sun-gain recovery error     {davg['sun_gain_err']:.4f}")
    print(f"  star-tracker availability   {100 * davg['st_avail']:.0f}%")

    os.makedirs("outputs", exist_ok=True)
    _write_csv(rows, "outputs/calibration_comparison.csv")
    print("\nTable saved to outputs/calibration_comparison.csv")
    _plot(run_med, raw_med, n_runs, "outputs/calibration_comparison.png")


def _write_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _plot(run_med, raw_med, n_runs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib unavailable; skipped plot)")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = REGIMES
    meds = [np.median(run_med[r]) for r in labels]
    # box of per-run medians to show spread
    ax.boxplot([run_med[r] for r in labels], positions=range(len(labels)),
               widths=0.5, showfliers=False,
               medianprops=dict(color="#c0392b", lw=1.3),
               boxprops=dict(color="#555"), whiskerprops=dict(color="#555"),
               capprops=dict(color="#555"))
    ax.axhspan(10, 15, color="#e8a33d", alpha=0.18, label="10-15° budget")
    for i, m in enumerate(meds):
        ax.annotate(f"{m:.1f}°", (i, m), textcoords="offset points",
                    xytext=(9, 0), va="center", fontsize=8, color="#c0392b")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("per-run median attitude error [deg]")
    ax.set_yscale("log")
    ax.set_title(f"Calibration technique vs MEKF knowledge error "
                 f"({n_runs} IGRF orbits)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Figure saved to {path}")


if __name__ == "__main__":
    main()
