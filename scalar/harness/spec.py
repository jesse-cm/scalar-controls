"""ScenarioSpec: one flat, frozen record of every switch a harness run depends on.

The harness wraps the existing simulators; it adds no physics. A field exists
here only if the wrapped code already has that switch today:

    kind="closed_loop"  -> scalar.control.tune.simulate_closed_loop_estimated
    kind="estimation"   -> scalar.scenario.Scenario + the MEKF
    kind="sysid"        -> scalar.sysid.magnetorquer_id.simulate_burst_campaign
    kind="sun_vector"   -> scalar.learning.sun_mlp.make_dataset (static geometry)

DEFAULTS. Sensor/estimator fields default to ``None`` = "whatever the wrapped
class or function defaults to today". Orbit, epoch and controller fields
default to the values the closed-loop scripts (experiments 4 and 5) use. The
open-loop experiments (1-3) run on other epochs/orbits; their baseline specs in
``baseline.py`` state every difference explicitly.

A field that the selected ``kind`` cannot honour must be left at its default;
``validate()`` raises instead of silently ignoring it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from dataclasses import replace as _dc_replace
from typing import Optional, Tuple

__all__ = ["ScenarioSpec", "KINDS"]

KINDS = ("closed_loop", "estimation", "sysid", "sun_vector")

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class ScenarioSpec:
    # ---- identity (ledger key = scenario name, estimator label, controller label) ----
    name: str
    experiment: str = ""
    kind: str = "closed_loop"

    # ---- orbit, epoch, environment ----
    epoch: Tuple[int, ...] = (2026, 9, 11, 12, 0, 0)
    altitude_m: float = 550e3
    inclination_deg: float = 98.61
    raan_deg: float = 0.0
    arg_lat0_deg: float = 0.0
    j2: bool = False
    albedo_coeff: float = 0.0
    field_truth: str = "dipole"             # estimation only (Scenario.field_truth)
    field_model_error_deg: float = 0.0      # estimation only (Scenario.field_model_error_deg)
    ref_along_track_error_km: float = 0.0   # estimation only: position error in the estimator's reference field

    # ---- spacecraft ----
    inertia: Optional[Tuple[Vec3, Vec3, Vec3]] = None   # None -> RigidBody() default
    inertia_scale: float = 1.0              # true plant only; the design keeps the nominal inertia
    coil_set: str = "s3_on_s6"              # sysid.magnetorquer_id.s3_coil_matrix()
    coil_matrix: Optional[Tuple[Vec3, Vec3, Vec3]] = None  # closed_loop only: dipole per amp [A m^2/A], column = channel; None -> coil_set's
    coil_i_max: Optional[Vec3] = None       # closed_loop only: full-scale current per channel [A]; None -> coil_set's
    coil_scale: float = 1.0                 # true plant only (rl_tune.Plant.scale_M)
    m_res: Vec3 = (0.0, 0.0, 0.0)           # residual spacecraft dipole [A m^2]

    # ---- initial condition (fixed per scenario; the seed drives sensor noise only) ----
    q0: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    omega0_deg_s: Vec3 = (0.0, 0.0, 0.0)

    # ---- sampling ----
    dt: float = 1.0                         # sensor = estimator = control step [s]
    n_orbits: float = 1.0
    plant_hold: str = "torque"              # closed_loop only: "torque" | "dipole" held over dt (dynamics.plant_step)
    plant_substeps: int = 1                 # closed_loop only: truth integration substeps per dt

    # ---- sensors (None -> the sensor class / wrapped function default) ----
    gyro_noise_density: Optional[float] = None      # [rad/s/sqrt(Hz)]
    gyro_bias_rw: Optional[float] = None            # [rad/s/sqrt(s)]
    gyro_initial_bias_deg_s: Optional[Vec3] = None
    # -- gyro realism (plan row 1h); None = off. Scale factor (1 %) and misalignment (0.5 deg) are the
    #    plan's sizes; the output step and the temperature coefficient have no source in this repo and
    #    any value given is ASSUMED (see sensors/gyro.py). The temperature drift is on when BOTH the
    #    coefficient and the amplitude are set; the swing is the board's, so with the magnetometer
    #    drift also on, amplitude and phase must equal mag_temp_amp_K / mag_temp_phase_deg.
    gyro_scale_factor: Optional[Vec3] = None            # scale-factor error per axis, fraction (0.01 = 1 %)
    gyro_misalign_deg: Optional[Vec3] = None            # rotation vector of the sensor-from-body DCM [deg]
    gyro_quant_deg_s: Optional[float] = None            # output LSB [deg/s]
    gyro_temp_coeff_deg_s_per_K: Optional[Vec3] = None  # zero-rate drift per axis [deg/s/K]
    gyro_temp_amp_K: Optional[float] = None             # half the peak-to-peak board temperature swing, once per orbit [K]
    gyro_temp_phase_deg: float = 0.0                    # phase of that swing at t = 0 (0 -> at its mean, warming)
    mag_noise_std: Optional[float] = None           # [T]
    mag_bias_T: Optional[Vec3] = None
    mag_soft_iron: Optional[Tuple[Vec3, Vec3, Vec3]] = None
    mag_calibrated: bool = False            # closed_loop, estimation: rl_tune.calibration_pass before the run
    # -- magnetometer realism (plan row 1c); None = off. The temperature drift is on when BOTH the
    #    coefficient and the amplitude are set. Neither has a source in this repo: any value given
    #    is ASSUMED until the phase-5 temperature ramp with the TMP112s (see sensors/magnetometer.py).
    mag_quant_T: Optional[float] = None             # output LSB [T]; the plan gives 0.15e-6
    mag_temp_coeff_T_per_K: Optional[Vec3] = None   # offset drift per axis [T/K]
    mag_temp_amp_K: Optional[float] = None          # half the peak-to-peak board temperature swing, once per orbit [K]
    mag_temp_phase_deg: float = 0.0                 # phase of that swing at t = 0 (0 -> at its mean, warming)
    # -- board temperature from a log (hardware.json board_temperature.log_csv), in place of the sinusoid,
    #    for both sensors: ((t [s], ...), (T - T_ref [K], ...)), replayed end to end and repeating
    #    (period = span + median step). Needs a temperature coefficient and no *_temp_amp_K / *_phase_deg.
    board_temp_log: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
    sun_unit: str = "ideal"                # "ideal" | "flight_like" (sun_mlp.flight_like_sensor)
    sun_unit_seed: int = 1                  # rng seed that draws the flight-like unit's systematics
    sun_gain_sigma: float = 0.05            # flight_like_sensor arguments
    sun_rolloff: float = 0.2
    sun_tilt_deg: float = 2.0
    sun_noise_frac: Optional[float] = None
    # -- sun-sensor array (plan row 1f). "cube" is the six-photodiode SunSensorArray every run has
    #    used; "veml6031" is sensors/veml6031.VEML6031Array: seven sensors, lux in 16-bit counts,
    #    clipping at 65535. The veml_* fields need sun_array="veml6031"; None = the class default.
    #    The analytic reconstruction always assumes veml6031.VEML6031_NORMALS, whatever veml_normals
    #    says; an MLP front end is trained on the array as specified here (ground sweep of the unit).
    sun_array: str = "cube"                         # "cube" | "veml6031"
    veml_integration_ms: Optional[float] = None     # a key of veml6031.MAX_LUX_G05_PD4 (class default 25)
    veml_normals: Optional[Tuple[Vec3, ...]] = None  # physical outward normals of the seven sensors
    veml_clip_mode: Optional[str] = None            # analytic front end only: None | "drop" | "fill"
    veml_gains: Optional[Tuple[float, ...]] = None  # per-sensor gain, seven values (faces 0,1,2,3,5,6,7)
    veml_dark_counts: Optional[float] = None        # dark offset [counts]

    # ---- estimator ----
    estimator: str = "mekf"                 # "mekf" | "truth" (closed_loop: controller fed the truth)
    sun_front: str = "analytic"             # "analytic" | "mlp_a" | "mlp_b" | "none" (sun sensor never used)
    mlp_train_seed: int = 0
    mlp_epochs: int = 15
    mekf_n_iter: int = 3
    mekf_sigma_sun: Optional[float] = None
    mekf_sigma_mag: Optional[float] = None  # estimation only
    mekf_sigma_v: Optional[float] = None    # estimation only
    mekf_sigma_u: Optional[float] = None    # estimation only

    # ---- controller (closed_loop only) ----
    controller: str = "none"                # none | projected_lqr | periodic_lqr | two_mode | three_mode
    proj_q_weight: Optional[float] = None   # projected_lqr: MagnetorquerLQR weights
    proj_w_weight: Optional[float] = None
    proj_r_weight: Optional[float] = None
    theta_max_deg: float = 10.0             # Bryson scales of the periodic schedule
    omega_max_deg_s: float = 0.2
    rho: float = 0.3
    schedule_bins: Optional[int] = 256      # None -> the full per-step schedule
    k_damp: float = 5e3                     # two_mode / three_mode supervisor
    enter_deg_s: float = 0.3
    exit_deg_s: float = 1.5
    fine_theta_max_deg: Optional[float] = None   # three_mode fine schedule (None -> acquisition weights)
    fine_omega_max_deg_s: Optional[float] = None
    fine_rho: Optional[float] = None
    start_mode: int = 1                     # two_mode / three_mode initial supervisor mode

    # ---- sysid (kind="sysid") ----
    pulse_duration_s: float = 0.32
    pulse_period_s: float = 10.0
    n_pulses: int = 30
    sample_hz: float = 10.0
    impulse_window_s: float = 1.0
    sysid_dt_sim: float = 0.01

    # ---- sun_vector (kind="sun_vector") ----
    dataset_n: int = 5000
    dataset_albedo_max: float = 0.0         # make_dataset(albedo_max=...): coefficient ~ U(0, albedo_max)

    # ---- scoring ----
    knowledge_window: str = "last_orbit"    # "last_orbit" | "after_first_fifth"
    actuation_window: str = "run"           # "run" | "last_orbit": sat, duty, power, max rate
    settle_thresh_deg: float = 10.0
    settle_hold: float = 0.9
    diverge_deg: float = 90.0

    # ------------------------------------------------------------------
    def replace(self, **kw) -> "ScenarioSpec":
        return _dc_replace(self, **kw)

    @property
    def estimator_label(self) -> str:
        if self.kind == "sysid":
            return "lsq_per_sample+impulse"
        if self.kind == "sun_vector":
            return f"sunvec:{self.sun_front}"
        if self.estimator == "truth":
            return "truth"
        return f"mekf+{self.sun_front}"

    @property
    def controller_label(self) -> str:
        c = self.controller
        if c == "none":
            return "none"
        if c == "projected_lqr":
            return (f"projected_lqr q={self.proj_q_weight:.4g} w={self.proj_w_weight:.4g} "
                    f"r={self.proj_r_weight:.4g}")
        sched = (f"th={self.theta_max_deg:g}deg w={self.omega_max_deg_s:g}dps rho={self.rho:g} "
                 f"bins={'full' if self.schedule_bins is None else self.schedule_bins}")
        if c == "periodic_lqr":
            return f"periodic_lqr {sched}"
        sup = f"k_damp={self.k_damp:g} enter={self.enter_deg_s:g} exit={self.exit_deg_s:g} start={self.start_mode}"
        if c == "two_mode":
            return f"two_mode {sched} {sup}"
        fth, fwm, frho = self.fine_weights
        return f"three_mode {sched} fine th={fth:g}deg w={fwm:g}dps rho={frho:g} {sup}"

    @property
    def fine_weights(self):
        """three_mode fine-schedule weights; unset ones fall back to the acquisition weights."""
        return (self.theta_max_deg if self.fine_theta_max_deg is None else self.fine_theta_max_deg,
                self.omega_max_deg_s if self.fine_omega_max_deg_s is None else self.fine_omega_max_deg_s,
                self.rho if self.fine_rho is None else self.fine_rho)

    @property
    def key(self) -> str:
        """Unique id of a ledger row's (scenario, estimator, controller)."""
        return f"{self.name} | {self.estimator_label} | {self.controller_label}"

    # ------------------------------------------------------------------
    def validate(self) -> "ScenarioSpec":
        """Raise ValueError if a non-default field cannot be honoured by ``kind``."""
        d = _DEFAULTS

        def need_default(*names, why):
            bad = [n for n in names if getattr(self, n) != d[n]]
            if bad:
                raise ValueError(f"{self.name}: {bad} not supported for kind={self.kind!r} ({why})")

        def one_of(name, allowed):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{self.name}: {name}={getattr(self, name)!r} not in {allowed}")

        one_of("kind", KINDS)
        one_of("sun_unit", ("ideal", "flight_like"))
        one_of("sun_front", ("analytic", "mlp_a", "mlp_b", "none"))
        # ---- sun-sensor array (plan row 1f) ----
        one_of("sun_array", ("cube", "veml6031"))
        one_of("veml_clip_mode", (None, "drop", "fill"))
        if self.sun_front == "none" and self.kind not in ("closed_loop", "estimation"):
            raise ValueError(f"{self.name}: sun_front='none' needs an MEKF run (closed_loop or estimation)")
        if self.sun_array == "cube":
            need_default("veml_integration_ms", "veml_normals", "veml_clip_mode", "veml_gains", "veml_dark_counts",
                         why="sun_array='cube' is the SunSensorArray; these belong to 'veml6031'")
        else:
            from ..sensors.veml6031 import MAX_LUX_G05_PD4
            if self.kind not in ("closed_loop", "estimation"):
                raise ValueError(f"{self.name}: sun_array='veml6031' is plumbed for closed_loop and estimation only")
            need_default("sun_unit", why="VEML6031Array has no roll-off or tilt, and the harness does not "
                                         "plumb per-sensor gains; give the physical normals in veml_normals")
            if self.veml_integration_ms is not None and self.veml_integration_ms not in MAX_LUX_G05_PD4:
                raise ValueError(f"{self.name}: veml_integration_ms={self.veml_integration_ms!r} not in "
                                 f"{sorted(MAX_LUX_G05_PD4)}")
            if self.veml_normals is not None and (len(self.veml_normals) != 7
                                                  or any(len(n) != 3 for n in self.veml_normals)):
                raise ValueError(f"{self.name}: veml_normals must be seven 3-vectors (devicetree faces 0,1,2,3,5,6,7)")
            if self.veml_clip_mode is not None and self.sun_front != "analytic":
                raise ValueError(f"{self.name}: veml_clip_mode acts in the analytic reconstruction only")
            if self.veml_gains is not None and len(self.veml_gains) != 7:
                raise ValueError(f"{self.name}: veml_gains must be seven numbers (devicetree faces 0,1,2,3,5,6,7)")
        one_of("estimator", ("mekf", "truth"))
        one_of("controller", ("none", "projected_lqr", "periodic_lqr", "two_mode", "three_mode"))
        one_of("knowledge_window", ("last_orbit", "after_first_fifth"))
        one_of("actuation_window", ("run", "last_orbit"))
        one_of("coil_set", ("s3_on_s6",))
        one_of("field_truth", ("dipole", "igrf"))
        one_of("plant_hold", ("torque", "dipole"))
        if not isinstance(self.plant_substeps, int) or self.plant_substeps < 1:
            raise ValueError(f"{self.name}: plant_substeps={self.plant_substeps!r} must be an integer >= 1")
        if self.kind != "closed_loop":
            need_default("coil_matrix", "coil_i_max", why="only the closed loop builds its coils from the spec")
        if self.coil_matrix is not None and (len(self.coil_matrix) != 3 or any(len(r) != 3 for r in self.coil_matrix)):
            raise ValueError(f"{self.name}: coil_matrix must be 3x3")
        if self.coil_i_max is not None and len(self.coil_i_max) != 3:
            raise ValueError(f"{self.name}: coil_i_max must be three numbers")

        ctrl_fields = ("proj_q_weight", "proj_w_weight", "proj_r_weight", "theta_max_deg",
                       "omega_max_deg_s", "rho", "schedule_bins", "k_damp", "enter_deg_s",
                       "exit_deg_s", "fine_theta_max_deg", "fine_omega_max_deg_s", "fine_rho",
                       "start_mode", "coil_scale", "inertia_scale")
        sysid_fields = ("pulse_duration_s", "pulse_period_s", "n_pulses", "sample_hz",
                        "impulse_window_s", "sysid_dt_sim")
        dataset_fields = ("dataset_n", "dataset_albedo_max")
        est_only = ("field_truth", "field_model_error_deg", "ref_along_track_error_km",
                    "mekf_sigma_mag", "mekf_sigma_v", "mekf_sigma_u")

        if self.kind == "closed_loop":
            need_default(*est_only, why="tune.py builds its own MEKF and truth field; no such argument")
            need_default("mekf_n_iter", why="tune.py line 108 hard-codes MEKF(n_iter=3)")
            need_default(*sysid_fields, *dataset_fields, why="not a closed-loop switch")
            if self.controller == "none":
                raise ValueError(f"{self.name}: closed_loop needs a controller")
            if self.controller == "projected_lqr":
                if None in (self.proj_q_weight, self.proj_w_weight, self.proj_r_weight):
                    raise ValueError(f"{self.name}: projected_lqr needs proj_q/w/r_weight")
                need_default("coil_scale", "inertia_scale", why="projected_lqr has no design/true split")
            if self.controller != "three_mode":
                need_default("fine_theta_max_deg", "fine_omega_max_deg_s", "fine_rho",
                             why="fine schedule exists only in three_mode")
            if self.controller == "three_mode":
                need_default("schedule_bins", why="rl_tune.build_controller fixes 256 bins")
            if self.controller in ("projected_lqr", "periodic_lqr"):
                need_default("start_mode", why="no supervisor")
        else:
            need_default(*ctrl_fields, why="no controller in this kind")
            need_default("plant_hold", "plant_substeps",
                         why="only tune.simulate_closed_loop_estimated takes the plant switches")
            if self.controller != "none" or self.estimator != "mekf":
                raise ValueError(f"{self.name}: kind={self.kind!r} has no controller and no truth feedback")
        if self.kind == "estimation":
            need_default(*sysid_fields, *dataset_fields, why="not an estimation switch")
            need_default("m_res", why="free tumble, no torque")
            one_of("knowledge_window", ("after_first_fifth",))
        if self.kind == "sysid":
            need_default(*est_only, *dataset_fields, "albedo_coeff", "sun_front", "sun_unit",
                         "gyro_bias_rw", "gyro_initial_bias_deg_s", "mag_bias_T", "mag_soft_iron",
                         "sun_noise_frac", "mekf_sigma_sun", "dt", "n_orbits",
                         why="simulate_burst_campaign has no such argument")
        if self.kind == "sun_vector":
            need_default(*est_only, *sysid_fields, "albedo_coeff", "m_res", "dt", "n_orbits",
                         "gyro_noise_density", "gyro_bias_rw", "gyro_initial_bias_deg_s",
                         "mag_noise_std", "mag_bias_T", "mag_soft_iron", "mekf_sigma_sun",
                         why="static sun-vector dataset; use dataset_albedo_max")
        # ---- board temperature from a log (hardware.json), in place of the sinusoids ----
        log = self.board_temp_log
        if log is not None:
            if self.kind not in ("closed_loop", "estimation"):
                raise ValueError(f"{self.name}: board_temp_log needs the harness Gyro or Magnetometer "
                                 "(closed_loop or estimation)")
            need_default("gyro_temp_amp_K", "gyro_temp_phase_deg", "mag_temp_amp_K", "mag_temp_phase_deg",
                         why="board_temp_log replaces the sinusoid")
            if self.gyro_temp_coeff_deg_s_per_K is None and self.mag_temp_coeff_T_per_K is None:
                raise ValueError(f"{self.name}: board_temp_log drives nothing: no temperature coefficient is set")
            if (len(log) != 2 or len(log[0]) != len(log[1]) or len(log[0]) < 2
                    or any(b <= a for a, b in zip(log[0], log[0][1:]))):
                raise ValueError(f"{self.name}: board_temp_log must be (times, temperatures), "
                                 "two or more of each, times increasing")
        # ---- gyro realism (plan row 1h) ----
        if self.kind not in ("closed_loop", "estimation"):
            need_default("gyro_scale_factor", "gyro_misalign_deg", "gyro_quant_deg_s",
                         "gyro_temp_coeff_deg_s_per_K", "gyro_temp_amp_K", "gyro_temp_phase_deg",
                         why="the harness Gyro is not used by this kind")
        if log is None and (self.gyro_temp_coeff_deg_s_per_K is None) != (self.gyro_temp_amp_K is None):
            raise ValueError(f"{self.name}: gyro_temp_coeff_deg_s_per_K and gyro_temp_amp_K go together")
        if self.gyro_temp_amp_K is None:
            need_default("gyro_temp_phase_deg", why="no temperature drift to phase")
        elif self.mag_temp_amp_K is not None and (
                (self.gyro_temp_amp_K, self.gyro_temp_phase_deg) != (self.mag_temp_amp_K, self.mag_temp_phase_deg)):
            raise ValueError(f"{self.name}: gyro and magnetometer sit on one board; "
                             "their temperature amplitude and phase must be equal")
        # ---- magnetometer realism (plan row 1c) ----
        if self.kind not in ("closed_loop", "estimation"):
            need_default("mag_calibrated", "mag_quant_T", "mag_temp_coeff_T_per_K", "mag_temp_amp_K",
                         "mag_temp_phase_deg", why="the harness Magnetometer is not used by this kind")
        if log is None and (self.mag_temp_coeff_T_per_K is None) != (self.mag_temp_amp_K is None):
            raise ValueError(f"{self.name}: mag_temp_coeff_T_per_K and mag_temp_amp_K go together")
        if self.mag_temp_amp_K is None:
            need_default("mag_temp_phase_deg", why="no temperature drift to phase")
        if self.kind == "estimation" and self.mag_calibrated:
            need_default("field_truth", why="rl_tune.calibration_pass flies the dipole field")
        return self


_DEFAULTS = {f.name: f.default for f in fields(ScenarioSpec) if f.name != "name"}
