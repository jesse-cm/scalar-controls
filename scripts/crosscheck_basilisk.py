#!/usr/bin/env python3
"""Phase 2 of docs/verification-loop-plan.md: independent cross-check against Basilisk.

    python3 scripts/crosscheck_basilisk.py

Runs the same orbit, inertia, six-face sun-sensor suite and S3 coil set in our
sim and in Basilisk 2.12 (SPICE sun + Earth orientation, Basilisk's centered
dipole, conical eclipse, CoarseSunSensor, MtbEffector), for one orbit of free
tumble and one orbit of rate-damping detumble, and compares truth attitude,
B_body, eclipse timing and sun-sensor currents at 1 Hz.

Gate (written before the first run): every direction error < 0.5 deg and every
eclipse edge within 5 s. Where the gate fails, the confirmation runs below
test one predicted cause each (rule 5: a cause counts only when it predicts a
result that a run then confirms).

Writes results/crosscheck_basilisk.json and prints the tables that
docs/crosscheck-basilisk.md quotes. Changes nothing else. About 2 min 20 s.

Our side is the repo code untouched: simulate_free_tumble / simulate_controlled
(dynamics.py), the TwoModeController mode-1 damping law (periodic_lqr.py),
actuators_from_matrix (tune.py), s3_coil_matrix, CircularOrbit, the dipole,
sun_direction_eci / eclipse_factor and SunSensorArray.currents. The detumble
path reproduces tune.simulate_closed_loop_estimated(use_truth=True) exactly.
scalar/control/lqr.py is imported by tune.py but nothing in it is called.

The damping law is an input, not a thing under test: the Basilisk side runs the
same law on Basilisk's own rate and field, so what is compared is the plant.

First use downloads Basilisk's own support data into
~/Library/Caches/bsk_support_data: de430.bsp, naif0012.tls, pck00010.tpc,
de-403-masses.tpc (naif.jpl.nasa.gov) and WMM2025.COF (Basilisk's GitHub repo).

Two configurations, because the repo uses two:
  control  550 km / 98.61 deg, epoch 2026-09-11 12:00, q0 = [.5,.5,.5,.5],
           omega0 = [1, -0.6, 0.8] deg/s, dt = 1 s
           (scripts/lqr_tune.py:46-49, scalar/control/rl_tune.py:72-73)
           -> free tumble AND detumble, with the S3 coil set.
  readme   550 km / 51.6 deg, epoch 2024-03-20 12:00, q0 = identity,
           omega0 = [4, -3, 5] deg/s, dt = 0.5 s
           (scalar/scenario.py:76-83, scalar/environment/orbit.py:51)
           -> free tumble only (the determination sim has no actuators).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scalar.attitude import dcm, to_body  # noqa: E402
from scalar.constants import DEG, MU_EARTH, R_EARTH_M, SOLAR_CONSTANT_1AU  # noqa: E402
from scalar.control.periodic_lqr import PeriodicLQR, TwoModeController  # noqa: E402
from scalar.control.tune import actuators_from_matrix  # noqa: E402
from scalar.dynamics import RigidBody, simulate_controlled, simulate_free_tumble  # noqa: E402
from scalar.environment.magfield import (B0_SURFACE_T, _POLE_COLAT, _POLE_EAST_LON,  # noqa: E402
                                         magnetic_field_eci)
from scalar.environment.orbit import CircularOrbit  # noqa: E402
from scalar.environment.sun import eclipse_factor, solar_irradiance, sun_direction_eci  # noqa: E402
from scalar.sensors.sun_sensor import CUBE_FACE_NORMALS, SunSensorArray  # noqa: E402
from scalar.sysid.magnetorquer_id import s3_coil_matrix  # noqa: E402
from scalar.timeutil import gmst_rad, julian_date  # noqa: E402

from Basilisk.architecture import messaging, sysModel  # noqa: E402
from Basilisk.simulation import (MtbEffector, coarseSunSensor, eclipse,  # noqa: E402
                                 magneticFieldCenteredDipole, spacecraft)
from Basilisk.utilities import RigidBodyKinematics as rbk  # noqa: E402
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody  # noqa: E402

# --- gate thresholds (plan lines 62-66), fixed before any run -----------------
GATE_DIR_DEG = 0.5
GATE_EDGE_S = 5.0
# A 0.5 deg direction error changes a face current by at most sin(0.5 deg) of
# full scale (face near grazing incidence). Derived here, not in the plan.
GATE_CURRENT_FS = float(np.sin(np.deg2rad(GATE_DIR_DEG)))

MASS_KG = 1.3                     # scalar/dynamics.py:28 default_inertia(mass=1.3)
K_DAMP = 5e3                      # scalar/control/periodic_lqr.py:131 TwoModeController.k_damp
BSK_DYN_DT = 0.1                  # Basilisk RK4 step, free tumble [s] (converged: halving moves it < 1e-7 deg)
BSK_DETUMBLE_DT = 0.02            # Basilisk RK4 step, detumble [s]: with coils on, its attitude is only
                                  # first-order in the step (0.038 deg between 0.04 s and 0.02 s)

CONFIGS = {
    "control": dict(
        source="scripts/lqr_tune.py:46-49, scalar/control/rl_tune.py:72-73",
        inclination_deg=98.61, epoch=dt.datetime(2026, 9, 11, 12),
        q0=[0.5, 0.5, 0.5, 0.5], omega0_deg_s=[1.0, -0.6, 0.8], dt=1.0,
        orbits=("free_tumble", "detumble")),
    "readme": dict(
        source="scalar/scenario.py:76-83, scalar/environment/orbit.py:51",
        inclination_deg=51.6, epoch=dt.datetime(2024, 3, 20, 12),
        q0=[0.0, 0.0, 0.0, 1.0], omega0_deg_s=[4.0, -3.0, 5.0], dt=0.5,
        orbits=("free_tumble",)),
}


# ---------------------------------------------------------------- small maths
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angle_deg(a, b):
    """Angle between rows of a and b [deg]."""
    c = np.sum(unit(a) * unit(b), axis=-1)
    s = np.linalg.norm(np.cross(unit(a), unit(b)), axis=-1)
    return np.degrees(np.arctan2(s, c))


def dcm_angle_deg(Ca, Cb):
    """Rotation angle of Ca Cb^T for stacks of DCMs [deg]."""
    R = np.einsum("nij,nkj->nik", Ca, Cb)
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    anti = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    return np.degrees(np.arctan2(np.linalg.norm(anti, axis=-1) / 2.0, tr))


def stats(x, t=None):
    x = np.asarray(x, float)
    out = {"max": float(np.max(x)), "median": float(np.median(x)), "p95": float(np.percentile(x, 95))}
    if t is not None:
        out["t_of_max_s"] = float(t[int(np.argmax(x))])
    return out


def precession_date_from_j2000(jd):
    """IAU 1976 precession matrix P with r_mean_of_date = P r_J2000."""
    T = (jd - 2451545.0) / 36525.0
    asec = np.deg2rad(1.0 / 3600.0)
    zeta = (2306.2181 * T + 0.30188 * T ** 2 + 0.017998 * T ** 3) * asec
    z = (2306.2181 * T + 1.09468 * T ** 2 + 0.018203 * T ** 3) * asec
    theta = (2004.3109 * T - 0.42665 * T ** 2 - 0.041833 * T ** 3) * asec

    def r3(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def r2(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])

    return r3(-z) @ r2(theta) @ r3(-zeta)


# ------------------------------------------------------------------- our sim
def make_orbit(cfg):
    return CircularOrbit(inclination=cfg["inclination_deg"] * DEG)


def make_actuators():
    M, i_max = s3_coil_matrix()
    return actuators_from_matrix(M, i_max)


def damping_controller(orbit, acts):
    """The repo's rate-damping law, verbatim: TwoModeController held in mode 1
    (enter threshold 0 deg/s is never met), i.e. m = sat(k (omega x B))."""
    lqr = PeriodicLQR(np.zeros((1, 3, 6)), orbit.period, acts)
    return TwoModeController(lqr, k_damp=K_DAMP, enter_deg_s=0.0)


def ours_truth(cfg, kind, n_sec, dt=None, hold="as-is"):
    """Truth attitude from our sim, returned at 1 Hz.

    hold="as-is"   our simulate_controlled exactly as tune.py uses it: torque
                   computed once per control step and held in the body frame.
    hold="torque"  same law at 1 Hz, torque held, but integrated at ``dt``.
    hold="dipole"  same law at 1 Hz, the DIPOLE held and tau = m x B_body(t)
                   re-evaluated every ``dt`` (what a real coil does).
    """
    dt = cfg["dt"] if dt is None else dt
    body = RigidBody()
    orbit = make_orbit(cfg)
    jd0 = julian_date(cfg["epoch"])
    q0 = np.asarray(cfg["q0"], float)
    w0 = np.deg2rad(cfg["omega0_deg_s"])
    n_steps = int(round(n_sec / dt))
    dip = None
    if kind == "free_tumble":
        tl = simulate_free_tumble(body, q0, w0, dt, n_steps)
    else:
        acts = make_actuators()
        ctrl = damping_controller(orbit, acts)
        held = {"m": np.zeros(3), "tau": np.zeros(3)}
        dip_log = []
        per_ctl = 1 if hold == "as-is" else int(round(1.0 / dt))   # plant steps per control step

        def torque_fn(t, q, w):
            B_body = to_body(q, magnetic_field_eci(orbit.position(t), jd0 + t / 86400.0))
            if int(round(t / dt)) % per_ctl == 0:                  # control instant: new command
                held["m"] = ctrl.dipole(t, q, w, B_body)
                held["tau"] = np.cross(held["m"], B_body)
                dip_log.append(held["m"])
            if hold == "dipole":
                return np.cross(held["m"], B_body)
            return held["tau"]

        tl, _ = simulate_controlled(body, q0, w0, dt, n_steps, torque_fn)
        dip = np.array(dip_log)
    stride = int(round(1.0 / dt))
    return {"t": tl.t[::stride], "q": tl.q[::stride], "omega": tl.omega[::stride], "dipole": dip}


def ours_environment(cfg, truth):
    """Our environment + sensor truth along our own attitude history (1 Hz)."""
    orbit = make_orbit(cfg)
    jd0 = julian_date(cfg["epoch"])
    sun = SunSensorArray()
    t = truth["t"]
    n = len(t)
    out = {k: np.empty((n, 3)) for k in ("r", "B_eci", "s_eci", "B_body", "s_body")}
    out.update(lit=np.empty(n, bool), E=np.empty(n), currents=np.empty((n, 6)), C=np.empty((n, 3, 3)))
    for k in range(n):
        jd = jd0 + t[k] / 86400.0
        r = orbit.position(t[k])
        B = magnetic_field_eci(r, jd)
        s = sun_direction_eci(jd)
        E = solar_irradiance(jd)
        lit = eclipse_factor(r, s) > 0
        q = truth["q"][k]
        out["r"][k], out["B_eci"][k], out["s_eci"][k] = r, B, s
        out["B_body"][k], out["s_body"][k] = to_body(q, B), to_body(q, s)
        out["lit"][k], out["E"][k] = lit, E
        out["currents"][k] = sun.currents(out["s_body"][k], E, lit)
        out["C"][k] = dcm(q)
    out["full_scale_A"] = sun.full_scale_current(SOLAR_CONSTANT_1AU)
    return out


def ours_eclipse_edges(cfg, n_sec, oblate=False):
    """Our eclipse edges [(t, 'entry'|'exit')], bisected to 1 ms on our own model.

    ``oblate`` evaluates the same cylinder test against the WGS-84 ellipsoid
    (z stretched by a/b maps the ellipsoid to our sphere) - a model NEITHER
    sim contains, used only to size what both leave out.
    """
    orbit = make_orbit(cfg)
    jd0 = julian_date(cfg["epoch"])
    k_obl = 6378137.0 / 6356752.314 if oblate else 1.0
    S = np.diag([1.0, 1.0, k_obl])

    def lit(t):
        s = S @ sun_direction_eci(jd0 + t / 86400.0)
        return eclipse_factor(S @ orbit.position(t), s / np.linalg.norm(s)) > 0

    edges = []
    prev = lit(0.0)
    for k in range(1, n_sec + 1):
        cur = lit(float(k))
        if cur != prev:
            a, b = k - 1.0, float(k)
            while b - a > 1e-3:
                m = 0.5 * (a + b)
                a, b = (m, b) if lit(m) == prev else (a, m)
            edges.append((0.5 * (a + b), "exit" if cur else "entry"))
        prev = cur
    return edges


# ------------------------------------------------------------------ Basilisk
class _OurEarthOrientation(sysModel.SysModel):
    """Confirmation-run aid: publishes OUR Earth orientation, Rz(GMST)^T, as the
    planet state Basilisk's dipole module rotates with, so both sims fly through
    the identical field and only the rigid-body/actuator side differs."""

    def __init__(self, jd0):
        super().__init__()
        self.jd0 = jd0
        self.out = messaging.SpicePlanetStateMsg()

    def Reset(self, t_ns):
        self.UpdateState(t_ns)

    def UpdateState(self, t_ns):
        g = gmst_rad(self.jd0 + t_ns * 1e-9 / 86400.0)
        c, s = float(np.cos(g)), float(np.sin(g))
        p = messaging.SpicePlanetStateMsgPayload()
        p.PositionVector = [0.0, 0.0, 0.0]
        p.VelocityVector = [0.0, 0.0, 0.0]
        p.J20002Pfix = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
        p.computeOrient = 1
        self.out.write(p, t_ns, self.moduleID)


class _DampingModule(sysModel.SysModel):
    """1 Hz flight-software stand-in on the Basilisk side: reads Basilisk's truth
    rate and Basilisk's field, applies the repo's damping law, commands the coils.

    It sits in the dynamics task after the spacecraft and the field model and
    acts only on whole seconds, so a command computed from the state at t_k is
    the one applied over (t_k, t_k + 1 s] - the same timing as our loop."""

    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.scIn = messaging.SCStatesMsgReader()
        self.magIn = messaging.MagneticFieldMsgReader()
        self.cmdOut = messaging.MTBCmdMsg()
        self.dipoles = []

    def Reset(self, t_ns):
        self.cmdOut.write(messaging.MTBCmdMsgPayload(), t_ns, self.moduleID)

    def UpdateState(self, t_ns):
        if t_ns % 1_000_000_000:
            return
        s = self.scIn()
        B_body = rbk.MRP2C(np.array(s.sigma_BN)) @ np.array(self.magIn().magField_N)
        m = self.ctrl.dipole(t_ns * 1e-9, None, np.array(s.omega_BN_B), B_body)
        p = messaging.MTBCmdMsgPayload()
        p.mtbDipoleCmds = [float(m[0]), float(m[1]), float(m[2])]
        self.cmdOut.write(p, t_ns, self.moduleID)
        self.dipoles.append(m)


def bsk_run(cfg, kind, n_sec, dyn_dt=None, ic_rotation=None, earth_orientation="spice"):
    """One Basilisk run. ``ic_rotation`` R re-expresses our initial state in
    Basilisk's inertial frame: r_N = R r, v_N = R v, C_BN = C0 R^T (default I:
    our 'ECI' numbers are handed to Basilisk's J2000 frame unchanged).
    ``earth_orientation="ours"`` feeds Basilisk's dipole our Rz(GMST) instead of
    SPICE IAU_EARTH (confirmation runs only)."""
    R = np.eye(3) if ic_rotation is None else np.asarray(ic_rotation, float)
    if dyn_dt is None:
        dyn_dt = BSK_DETUMBLE_DT if kind == "detumble" else BSK_DYN_DT
    orbit = make_orbit(cfg)
    r0, v0 = orbit.state(0.0)
    C0 = dcm(np.asarray(cfg["q0"], float))
    w0 = np.deg2rad(cfg["omega0_deg_s"])

    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("proc")
    proc.addTask(sim.CreateNewTask("dyn", macros.sec2nano(dyn_dt)))

    sc = spacecraft.Spacecraft()
    sc.ModelTag = "scalar"
    sc.hub.mHub = MASS_KG
    sc.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    sc.hub.IHubPntBc_B = RigidBody().inertia.tolist()
    sc.hub.r_CN_NInit = [[x] for x in R @ r0]
    sc.hub.v_CN_NInit = [[x] for x in R @ v0]
    sc.hub.sigma_BNInit = [[x] for x in rbk.C2MRP(C0 @ R.T)]
    sc.hub.omega_BN_BInit = [[x] for x in w0]

    gf = simIncludeGravBody.gravBodyFactory()
    earth = gf.createEarth()
    earth.isCentralBody = True
    earth.mu = MU_EARTH                     # same two-body constant as our orbit
    gf.addBodiesTo(sc)                      # Earth point mass only; the Sun is ephemeris, not gravity
    e = cfg["epoch"]
    utc = e.strftime("%Y %b %d %H:%M:%S.000 (UTC)").upper()
    spice = gf.createSpiceInterface(time=utc, spicePlanetNames=["earth", "sun"])
    spice.zeroBase = "earth"

    mag = magneticFieldCenteredDipole.MagneticFieldCenteredDipole()
    mag.ModelTag = "dipole"
    # Our tilted dipole (magfield.py:31-35) as first-degree Gauss coefficients.
    mag.g10 = -B0_SURFACE_T * np.cos(_POLE_COLAT)
    mag.g11 = -B0_SURFACE_T * np.sin(_POLE_COLAT) * np.cos(_POLE_EAST_LON)
    mag.h11 = -B0_SURFACE_T * np.sin(_POLE_COLAT) * np.sin(_POLE_EAST_LON)
    mag.planetRadius = R_EARTH_M
    mag.addSpacecraftToModel(sc.scStateOutMsg)
    keep = []
    if earth_orientation == "ours":
        eo = _OurEarthOrientation(julian_date(cfg["epoch"]))
        eo.ModelTag = "ourEarthOrientation"
        mag.planetPosInMsg.subscribeTo(eo.out)
        sim.AddModelToTask("dyn", eo, 95)
        keep.append(eo)
    else:
        mag.planetPosInMsg.subscribeTo(spice.planetStateOutMsgs[0])

    ecl = eclipse.Eclipse()
    ecl.ModelTag = "eclipse"
    ecl.addSpacecraftToModel(sc.scStateOutMsg)
    ecl.addPlanetToModel(spice.planetStateOutMsgs[0])
    ecl.sunInMsg.subscribeTo(spice.planetStateOutMsgs[1])

    ours_css = SunSensorArray()
    css = []
    for i, nrm in enumerate(CUBE_FACE_NORMALS):
        c = coarseSunSensor.CoarseSunSensor()
        c.ModelTag = f"css{i}"
        c.fov = 90.0 * DEG
        c.scaleFactor = ours_css.full_scale_current(SOLAR_CONSTANT_1AU)
        c.nHat_B = np.array(nrm, float)
        c.sunInMsg.subscribeTo(spice.planetStateOutMsgs[1])
        c.stateInMsg.subscribeTo(sc.scStateOutMsg)
        c.sunEclipseInMsg.subscribeTo(ecl.eclipseOutMsgs[0])
        css.append(c)

    sim.AddModelToTask("dyn", spice, 100)
    sim.AddModelToTask("dyn", sc, 90)
    ctl = None
    if kind == "detumble":
        acts = make_actuators()
        mtb = MtbEffector.MtbEffector()
        mtb.ModelTag = "mtb"
        pc = messaging.MTBArrayConfigMsgPayload()
        pc.numMTB = len(acts.torquers)
        pc.GtMatrix_B = list(np.column_stack([tq.axis for tq in acts.torquers]).ravel())
        pc.maxMtbDipoles = [float(x) for x in acts.m_max_vector]
        cfg_msg = messaging.MTBArrayConfigMsg().write(pc)
        mtb.mtbParamsInMsg.subscribeTo(cfg_msg)
        mtb.magInMsg.subscribeTo(mag.envOutMsgs[0])
        sc.addDynamicEffector(mtb)
        ctl = _DampingModule(damping_controller(orbit, acts))
        ctl.ModelTag = "damping"
        ctl.scIn.subscribeTo(sc.scStateOutMsg)
        ctl.magIn.subscribeTo(mag.envOutMsgs[0])
        mtb.mtbCmdInMsg.subscribeTo(ctl.cmdOut)
        # order at each instant: spacecraft -> field -> controller -> coil effector
        sim.AddModelToTask("dyn", ctl, 60)
        sim.AddModelToTask("dyn", mtb, 50)
        keep += [mtb, cfg_msg, pc]
    sim.AddModelToTask("dyn", mag, 80)
    sim.AddModelToTask("dyn", ecl, 79)
    for c in css:
        sim.AddModelToTask("dyn", c, 70)

    one = macros.sec2nano(1.0)
    rec = {"sc": sc.scStateOutMsg.recorder(one), "mag": mag.envOutMsgs[0].recorder(one),
           "earth": spice.planetStateOutMsgs[0].recorder(one),
           "sun": spice.planetStateOutMsgs[1].recorder(one),
           "ecl": ecl.eclipseOutMsgs[0].recorder(macros.sec2nano(dyn_dt))}
    for i, c in enumerate(css):
        rec[f"css{i}"] = c.cssDataOutMsg.recorder(one)
    for r in rec.values():
        sim.AddModelToTask("dyn", r, 1)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(float(n_sec)))
    sim.ExecuteSimulation()

    sigma = np.array(rec["sc"].sigma_BN)
    out = {
        "t": np.array(rec["sc"].times()) * 1e-9,
        "C": np.array([rbk.MRP2C(s) for s in sigma]),
        "omega": np.array(rec["sc"].omega_BN_B),
        "r": np.array(rec["sc"].r_BN_N),
        "B_N": np.array(rec["mag"].magField_N),
        "sun_pos": np.array(rec["sun"].PositionVector),
        "PN": np.array(rec["earth"].J20002Pfix),
        "t_ecl": np.array(rec["ecl"].times()) * 1e-9,
        "shadow": np.array(rec["ecl"].shadowFactor),
        "currents": np.column_stack([np.array(rec[f"css{i}"].OutputData) for i in range(6)]),
        "dipole": None if ctl is None else np.array(ctl.dipoles),
        "R": R,
    }
    out["s_N"] = unit(out["sun_pos"] - out["r"])
    out["B_body"] = np.einsum("nij,nj->ni", out["C"], out["B_N"])
    out["s_body"] = np.einsum("nij,nj->ni", out["C"], out["s_N"])
    out["shadow_1hz"] = np.interp(out["t"], out["t_ecl"], out["shadow"])
    gf.unloadSpiceKernels()
    return out


def bsk_eclipse_edges(b):
    """Basilisk eclipse edges: 50 % illumination crossing plus the penumbra
    limits (last fully lit / first fully dark sample) around it."""
    t, sf = b["t_ecl"], b["shadow"]
    edges = []
    above = sf >= 0.5
    for k in np.flatnonzero(above[1:] != above[:-1]):
        t50 = t[k] + (0.5 - sf[k]) * (t[k + 1] - t[k]) / (sf[k + 1] - sf[k])
        kind = "exit" if above[k + 1] else "entry"
        lo = k
        while lo > 0 and 0.0 < sf[lo] < 1.0:
            lo -= 1
        hi = k + 1
        while hi < len(sf) - 1 and 0.0 < sf[hi] < 1.0:
            hi += 1
        edges.append({"kind": kind, "t50": float(t50), "t_start": float(t[lo]), "t_end": float(t[hi])})
    return edges


# ---------------------------------------------------------------- comparison
def compare(o_truth, o_env, b, our_edges):
    """Gate metrics: our sim as it stands against one Basilisk run, at 1 Hz."""
    t = o_truth["t"]
    assert np.allclose(t, b["t"]), "1 Hz sample times do not line up"
    R = b["R"]                                    # our inertial coords -> Basilisk N coords
    fs = o_env["full_scale_A"]
    m = {}
    m["attitude_deg"] = dcm_angle_deg(o_env["C"] @ R.T, b["C"])
    m["rate_deg_s"] = np.degrees(np.linalg.norm(o_truth["omega"] - b["omega"], axis=1))
    m["position_m"] = np.linalg.norm(o_env["r"] @ R.T - b["r"], axis=1)
    m["B_inertial_dir_deg"] = angle_deg(o_env["B_eci"] @ R.T, b["B_N"])
    m["B_magnitude_pct"] = 100.0 * np.abs(np.linalg.norm(o_env["B_eci"], axis=1)
                                          / np.linalg.norm(b["B_N"], axis=1) - 1.0)
    m["B_body_dir_deg"] = angle_deg(o_env["B_body"], b["B_body"])
    m["sun_inertial_dir_deg"] = angle_deg(o_env["s_eci"] @ R.T, b["s_N"])
    m["sun_body_dir_deg"] = angle_deg(o_env["s_body"], b["s_body"])
    dI = np.max(np.abs(o_env["currents"] - b["currents"]), axis=1) / fs
    m["current_pctFS_all"] = 100.0 * dI
    # "steady" = both sims agree the sun is fully visible or fully hidden
    steady = ((b["shadow_1hz"] == 1.0) & o_env["lit"]) | ((b["shadow_1hz"] == 0.0) & ~o_env["lit"])
    both_lit = (b["shadow_1hz"] == 1.0) & o_env["lit"]

    def recon(c):
        return np.column_stack([c[:, 0] - c[:, 1], c[:, 2] - c[:, 3], c[:, 4] - c[:, 5]])

    summary = {k: stats(v, t) for k, v in m.items()}
    summary["current_pctFS_steady"] = stats(100.0 * dI[steady], t[steady])
    summary["current_dir_deg_lit"] = stats(angle_deg(recon(o_env["currents"][both_lit]),
                                                     recon(b["currents"][both_lit])), t[both_lit])
    summary["lit_state_mismatch_s"] = int(np.sum(o_env["lit"] != (b["shadow_1hz"] >= 0.5)))
    summary["unsteady_samples"] = int(np.sum(~steady))
    summary["irradiance_pct"] = stats(100.0 * np.abs(
        o_env["E"] / SOLAR_CONSTANT_1AU
        / (149597870693.0 / np.linalg.norm(b["sun_pos"] - b["r"], axis=1)) ** 2 - 1.0), t)

    b_edges = bsk_eclipse_edges(b)
    edges = []
    for te, kind in our_edges:
        cand = [x for x in b_edges if x["kind"] == kind]
        if not cand:
            edges.append({"kind": kind, "ours_s": te, "basilisk": None})
            continue
        x = min(cand, key=lambda c: abs(c["t50"] - te))
        edges.append({"kind": kind, "ours_s": round(te, 3), "bsk_50pct_s": round(x["t50"], 3),
                      "bsk_penumbra_start_s": x["t_start"], "bsk_penumbra_end_s": x["t_end"],
                      "ours_minus_bsk50_s": round(te - x["t50"], 3),
                      "ours_minus_bsk_start_s": round(te - x["t_start"], 3),
                      "ours_minus_bsk_end_s": round(te - x["t_end"], 3)})
    summary["eclipse_edges"] = edges
    summary["n_edges"] = {"ours": len(our_edges), "basilisk": len(b_edges)}
    series = {k: np.round(m[k], 4).tolist() for k in
              ("attitude_deg", "B_body_dir_deg", "sun_body_dir_deg", "current_pctFS_all")}
    return summary, series


def gate_rows(summary):
    """(quantity, max, median, threshold, unit, pass) rows for the gate table."""
    rows = []
    for key, name in (("attitude_deg", "truth attitude"), ("B_body_dir_deg", "B_body direction"),
                      ("sun_body_dir_deg", "sun direction in body"),
                      ("current_dir_deg_lit", "sun direction rebuilt from currents (lit)")):
        s = summary[key]
        rows.append((name, s["max"], s["median"], GATE_DIR_DEG, "deg", s["max"] < GATE_DIR_DEG))
    s = summary["current_pctFS_steady"]
    rows.append(("face current, outside eclipse edges", s["max"], s["median"],
                 100 * GATE_CURRENT_FS, "% FS", s["max"] < 100 * GATE_CURRENT_FS))
    d = [abs(e["ours_minus_bsk50_s"]) for e in summary["eclipse_edges"] if e.get("bsk_50pct_s") is not None]
    ok = bool(d) and max(d) < GATE_EDGE_S and summary["n_edges"]["ours"] == summary["n_edges"]["basilisk"]
    rows.append(("eclipse edge (ours vs 50 % illumination)", max(d) if d else float("nan"),
                 float(np.median(d)) if d else float("nan"), GATE_EDGE_S, "s", ok))
    return rows


def print_gate(title, summary):
    print(f"\n{title}")
    print("| quantity | max | median | threshold | result |")
    print("|---|---|---|---|---|")
    for name, mx, med, thr, u, ok in gate_rows(summary):
        print(f"| {name} | {mx:.4g} {u} | {med:.4g} {u} | {thr:.3g} {u} | {'PASS' if ok else 'FAIL'} |")


# ------------------------------------------------------------- confirmations
def confirm_frame(cfg, n_sec, o_env, b_asis):
    """Cause A: our 'ECI' is the mean equator/equinox OF DATE (sun.py:13-14 says
    so; timeutil.gmst_rad is referred to the same equinox), while Basilisk's N
    is J2000. Prediction from the precession matrix alone, before any run."""
    jd0 = julian_date(cfg["epoch"])
    P = precession_date_from_j2000(jd0)            # r_date = P r_J2000
    s0 = sun_direction_eci(jd0)
    orbit = make_orbit(cfg)
    r0, v0 = orbit.state(0.0)
    nrm = unit(np.cross(r0, v0))
    # handing of-date numbers to a J2000 frame mis-places the sun by angle(s, P s)
    pred_sun = float(angle_deg(s0[None], (P @ s0)[None])[0])
    # in-plane part of that shift, as orbit phase -> seconds of eclipse-edge shift
    sj = P.T @ s0                                   # true J2000 sun, vs s0 taken as J2000

    def phase(s):
        p = s - (s @ nrm) * nrm
        return np.arctan2(p @ np.cross(nrm, unit(r0)), p @ unit(r0))

    pred_edge = float((phase(s0) - phase(sj)) / orbit.mean_motion)
    return {"precession_angle_deg": float(np.degrees(np.arccos((np.trace(P) - 1) / 2))),
            "predicted_sun_offset_deg": pred_sun,
            "measured_sun_offset_deg": stats(angle_deg(o_env["s_eci"], b_asis["s_N"])),
            "predicted_edge_shift_s": pred_edge, "P_date_from_j2000": P.tolist()}


def confirm_dipole_formula(cfg, b):
    """Like with like for the field model itself: our dipole expression fed
    Basilisk's own position and Earth orientation, against Basilisk's output."""
    pole = np.array([np.sin(_POLE_COLAT) * np.cos(_POLE_EAST_LON),
                     np.sin(_POLE_COLAT) * np.sin(_POLE_EAST_LON), np.cos(_POLE_COLAT)])
    m_hat = np.einsum("nji,j->ni", b["PN"], -pole)                 # planet-fixed -> N
    r = np.linalg.norm(b["r"], axis=1, keepdims=True)
    r_hat = b["r"] / r
    B = B0_SURFACE_T * (R_EARTH_M / r) ** 3 * (3 * np.sum(m_hat * r_hat, 1, keepdims=True) * r_hat - m_hat)
    return {"dir_deg": stats(angle_deg(B, b["B_N"])),
            "magnitude_pct": stats(100 * np.abs(np.linalg.norm(B, axis=1) / np.linalg.norm(b["B_N"], axis=1) - 1))}


def confirm_earth_orientation(cfg, b):
    """Our ECEF<-ECI rotation Rz(GMST)^T against Basilisk's Earth orientation,
    as it stands and after the of-date -> J2000 precession; and what Basilisk's
    orientation is: the linear IAU_EARTH model of SPICE pck00010.tpc
    (pole RA = -0.641 T, Dec = 90 - 0.557 T, W = 190.147 + 360.9856235 d)."""
    jd0 = julian_date(cfg["epoch"])
    P = precession_date_from_j2000(jd0)

    def r3(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def r1(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])

    raw, fixed, iau = [], [], []
    for k in range(0, len(b["t"]), 60):
        ours = r3(gmst_rad(jd0 + b["t"][k] / 86400.0))                  # ECEF <- our ECI
        raw.append(ours)
        fixed.append(ours @ P)
        d = ((jd0 - 2451545.0) * 86400.0 + b["t"][k] + 69.184) / 86400.0   # TT days from J2000 (37 leap s)
        T = d / 36525.0
        iau.append(r3(np.deg2rad((190.147 + 360.9856235 * d) % 360.0))
                   @ r1(np.deg2rad(0.557 * T)) @ r3(np.deg2rad(90.0 - 0.641 * T)))
    PN = b["PN"][::60]
    return {"as_is_deg": stats(dcm_angle_deg(np.array(raw), PN)),
            "after_precession_deg": stats(dcm_angle_deg(np.array(fixed), PN)),
            "basilisk_vs_pck00010_linear_formula_deg": stats(dcm_angle_deg(np.array(iau), PN))}


def confirm_eclipse_model(cfg, b):
    """Cause B: cylinder (ours) vs cone with penumbra (Basilisk). Our
    eclipse_factor fed Basilisk's own geometry must flip at Basilisk's 50 %
    point; the penumbra half-width must match the two-line geometry estimate."""
    t = b["t_ecl"]
    r = np.column_stack([np.interp(t, b["t"], b["r"][:, i]) for i in range(3)])
    sp = np.column_stack([np.interp(t, b["t"], b["sun_pos"][:, i]) for i in range(3)])
    lit = np.array([eclipse_factor(r[k], unit(sp[k])) > 0 for k in range(len(t))])
    flips = np.flatnonzero(lit[1:] != lit[:-1])
    b_edges = bsk_eclipse_edges(b)
    rows = []
    for k in flips:
        te = 0.5 * (t[k] + t[k + 1])
        x = min(b_edges, key=lambda c: abs(c["t50"] - te))
        rows.append({"cylinder_on_bsk_geometry_s": float(te), "bsk_50pct_s": x["t50"],
                     "diff_s": float(te - x["t50"]),
                     "bsk_penumbra_duration_s": x["t_end"] - x["t_start"]})
    a = make_orbit(cfg).radius
    d = np.sqrt(a * a - R_EARTH_M ** 2)                       # distance behind the terminator plane
    au, r_sun = 149597870693.0, 695000e3
    half_km = d * (r_sun / au)                                # penumbra half-width at the satellite
    # speed toward the shadow axis for an orbit whose plane contains the sun line
    v_perp = a * make_orbit(cfg).mean_motion * (d / a)
    return {"edges": rows, "predicted_penumbra_duration_s_if_sun_in_plane": float(2 * half_km / v_perp)}


def confirm_integrator(cfg, n_sec, b_ref):
    """Cause C: dynamics.py advances q with the step-averaged rate - second
    order. Prediction: the attitude difference to Basilisk falls 4x per halving
    of our dt, and Basilisk does not move when its own step is halved."""
    out = {"ours_dt": {}}
    for dt_ in (1.0, 0.5, 0.25, 0.125):
        tr = ours_truth(cfg, "free_tumble", n_sec, dt=dt_)
        C = np.array([dcm(q) for q in tr["q"]])
        out["ours_dt"][str(dt_)] = stats(dcm_angle_deg(C, b_ref["C"]), tr["t"])
    b2 = bsk_run(cfg, "free_tumble", n_sec, dyn_dt=BSK_DYN_DT / 2)
    out["basilisk_step_halved_deg"] = stats(dcm_angle_deg(b2["C"], b_ref["C"]), b_ref["t"])
    return out


def confirm_hold(cfg, n_sec, tr_asis):
    """Cause D (detumble): our plant holds the TORQUE fixed in the body frame for
    the whole control step (dynamics.py:118-124, tune.py:153-155); a coil holds
    its DIPOLE and the torque follows B_body as the body turns and the orbit moves.

    Reference: Basilisk fed OUR Earth orientation, so the field is identical on
    both sides and only the rigid-body/actuator treatment differs. Prediction:
    our law with the dipole held (0.02 s plant step) lands on Basilisk; with the
    torque held (0.05 s plant step, so integrator order is out of the picture)
    it reproduces the as-is difference."""
    b = bsk_run(cfg, "detumble", n_sec, earth_orientation="ours")
    b2 = bsk_run(cfg, "detumble", n_sec, dyn_dt=2 * BSK_DETUMBLE_DT, earth_orientation="ours")
    env = ours_environment(cfg, tr_asis)
    out = {"reference": "Basilisk with our Rz(GMST) Earth orientation, dyn step %g s" % BSK_DETUMBLE_DT,
           "field_identity_check_deg": stats(angle_deg(env["B_eci"], b["B_N"])),
           "basilisk_step_doubled_deg": stats(dcm_angle_deg(b2["C"], b["C"]), b["t"])}

    def diff(tr):
        C = np.array([dcm(q) for q in tr["q"]])
        return {"attitude_deg": stats(dcm_angle_deg(C, b["C"]), tr["t"]),
                "rate_deg_s": stats(np.degrees(np.linalg.norm(tr["omega"] - b["omega"], axis=1)), tr["t"])}

    out["ours_as_is"] = diff(tr_asis)
    out["ours_torque_held_plant_dt0.05"] = diff(ours_truth(cfg, "detumble", n_sec, dt=0.05, hold="torque"))
    out["ours_dipole_held_plant_dt0.02"] = diff(ours_truth(cfg, "detumble", n_sec, dt=0.02, hold="dipole"))
    return out


def model_difference_igrf_wmm(cfg, n_sec):
    """MODEL DIFFERENCE, not pass/fail: our real-field path (igrf.py, ppigrf
    IGRF) against Basilisk's WMM2025 along the same orbit, with Basilisk given
    our Earth orientation so only the field model and igrf.py's own coordinate
    handling can differ."""
    from Basilisk.simulation import magneticFieldWMM
    from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path
    from scalar.environment.igrf import igrf_field_eci_batch
    orbit = make_orbit(cfg)
    r0, v0 = orbit.state(0.0)
    jd0 = julian_date(cfg["epoch"])
    sim = SimulationBaseClass.SimBaseClass()
    sim.CreateNewProcess("proc").addTask(sim.CreateNewTask("dyn", macros.sec2nano(1.0)))
    sc = spacecraft.Spacecraft()
    sc.hub.mHub = MASS_KG
    sc.hub.IHubPntBc_B = RigidBody().inertia.tolist()
    sc.hub.r_CN_NInit = [[x] for x in r0]
    sc.hub.v_CN_NInit = [[x] for x in v0]
    gf = simIncludeGravBody.gravBodyFactory()
    earth = gf.createEarth()
    earth.isCentralBody = True
    earth.mu = MU_EARTH
    gf.addBodiesTo(sc)
    eo = _OurEarthOrientation(jd0)
    wmm = magneticFieldWMM.MagneticFieldWMM()
    wmm.ModelTag = "wmm"
    wmm.wmmDataFullPath = str(get_path(DataFile.MagneticFieldData.WMM))
    e = cfg["epoch"]
    y0, y1 = dt.datetime(e.year, 1, 1), dt.datetime(e.year + 1, 1, 1)
    wmm.epochDateFractionalYear = e.year + (e - y0) / (y1 - y0)
    wmm.addSpacecraftToModel(sc.scStateOutMsg)
    wmm.planetPosInMsg.subscribeTo(eo.out)
    sim.AddModelToTask("dyn", eo, 95)
    sim.AddModelToTask("dyn", sc, 90)
    sim.AddModelToTask("dyn", wmm, 80)
    rb = wmm.envOutMsgs[0].recorder(macros.sec2nano(1.0))
    rs = sc.scStateOutMsg.recorder(macros.sec2nano(1.0))
    sim.AddModelToTask("dyn", rb, 1)
    sim.AddModelToTask("dyn", rs, 1)
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(float(n_sec)))
    sim.ExecuteSimulation()
    t = np.array(rb.times()) * 1e-9
    Bw, r = np.array(rb.magField_N), np.array(rs.r_BN_N)
    Bi = igrf_field_eci_batch(r, jd0 + t / 86400.0)
    Bd = np.array([magnetic_field_eci(r[k], jd0 + t[k] / 86400.0) for k in range(len(t))])
    return {"ours": "scalar/environment/igrf.py (ppigrf)", "basilisk": "MagneticFieldWMM, WMM2025.COF",
            "igrf_vs_wmm_dir_deg": stats(angle_deg(Bi, Bw), t),
            "igrf_vs_wmm_nT": stats(np.linalg.norm(Bi - Bw, axis=1) * 1e9, t),
            "for_scale_our_dipole_vs_wmm_dir_deg": stats(angle_deg(Bd, Bw), t)}


# ---------------------------------------------------------------------- main
def main():
    wall0 = time.time()
    result = {"gate": {"direction_deg": GATE_DIR_DEG, "eclipse_edge_s": GATE_EDGE_S,
                       "current_fraction_fs_derived": GATE_CURRENT_FS},
              "basilisk": {"dyn_dt_free_tumble_s": BSK_DYN_DT, "dyn_dt_detumble_s": BSK_DETUMBLE_DT,
                           "integrator": "RK4 (default)",
                           "sun_and_earth_orientation": "SPICE de430 + pck00010 IAU_EARTH, J2000",
                           "eclipse": "Basilisk Eclipse (cone, penumbra)",
                           "field": "MagneticFieldCenteredDipole with our B0/tilt/longitude/radius"},
              "runs": {}, "series_1hz": {}}
    result["model_difference_igrf_vs_wmm"] = model_difference_igrf_wmm(
        CONFIGS["control"], int(np.ceil(make_orbit(CONFIGS["control"]).period)))
    print("MODEL DIFFERENCE (not pass/fail) IGRF vs WMM2025, control orbit:",
          json.dumps(result["model_difference_igrf_vs_wmm"], indent=1))
    for name, cfg in CONFIGS.items():
        orbit = make_orbit(cfg)
        n_sec = int(np.ceil(orbit.period))
        jd0 = julian_date(cfg["epoch"])
        P = precession_date_from_j2000(jd0)
        our_edges = ours_eclipse_edges(cfg, n_sec)
        print(f"\n=== config '{name}': {cfg['inclination_deg']} deg, {cfg['epoch']:%Y-%m-%d %H:%M} UTC, "
              f"omega0 {cfg['omega0_deg_s']} deg/s, our dt {cfg['dt']} s, {n_sec} s  [{cfg['source']}]")
        for kind in cfg["orbits"]:
            key = f"{name}/{kind}"
            tr = ours_truth(cfg, kind, n_sec)
            env = ours_environment(cfg, tr)
            b = bsk_run(cfg, kind, n_sec)
            summary, series = compare(tr, env, b, our_edges)
            print_gate(f"[{key}] our sim as it stands vs Basilisk (same numbers in both inertial frames)", summary)
            entry = {"config": {k: (str(v) if k == "epoch" else v) for k, v in cfg.items()},
                     "n_samples": len(tr["t"]), "as_is": summary,
                     "gate_rows_as_is": [dict(quantity=r[0], max=r[1], median=r[2], threshold=r[3],
                                              unit=r[4], passed=bool(r[5])) for r in gate_rows(summary)]}
            result["series_1hz"][key] = series

            # confirmation run for cause A: same physical scenario, expressed in J2000
            b_al = bsk_run(cfg, kind, n_sec, ic_rotation=P.T)
            s_al, _ = compare(tr, env, b_al, our_edges)
            print_gate(f"[{key}] confirmation A: Basilisk started from our state rotated of-date -> J2000", s_al)
            entry["frame_aligned"] = s_al
            entry["gate_rows_frame_aligned"] = [dict(quantity=r[0], max=r[1], median=r[2], threshold=r[3],
                                                     unit=r[4], passed=bool(r[5])) for r in gate_rows(s_al)]
            if kind == "detumble":
                entry["dipole_command_diff_Am2"] = stats(
                    np.linalg.norm(tr["dipole"] - b["dipole"][:len(tr["dipole"])], axis=1))
                entry["final_rate_deg_s"] = {
                    "ours": float(np.degrees(np.linalg.norm(tr["omega"][-1]))),
                    "basilisk": float(np.degrees(np.linalg.norm(b["omega"][-1])))}
                entry["confirm_hold"] = confirm_hold(cfg, n_sec, tr)
                print("confirmation D (torque-hold vs dipole-hold):", json.dumps(entry["confirm_hold"], indent=1))
            else:
                entry["confirm_frame"] = confirm_frame(cfg, n_sec, env, b)
                entry["confirm_dipole_formula"] = confirm_dipole_formula(cfg, b)
                entry["confirm_earth_orientation"] = confirm_earth_orientation(cfg, b)
                entry["confirm_eclipse_model"] = confirm_eclipse_model(cfg, b)
                entry["confirm_integrator"] = confirm_integrator(cfg, n_sec, b)
                obl = ours_eclipse_edges(cfg, n_sec, oblate=True)
                entry["oblate_earth_edge_shift_s"] = [
                    {"kind": k1, "sphere_s": round(t1, 2), "wgs84_s": round(t2, 2), "shift_s": round(t2 - t1, 2)}
                    for (t1, k1), (t2, _) in zip(our_edges, obl)] if len(obl) == len(our_edges) else "edge count differs"
                for k in ("confirm_frame", "confirm_dipole_formula", "confirm_earth_orientation",
                          "confirm_eclipse_model", "confirm_integrator", "oblate_earth_edge_shift_s"):
                    v = dict(entry[k]) if isinstance(entry[k], dict) else entry[k]
                    if isinstance(v, dict):
                        v.pop("P_date_from_j2000", None)
                    print(f"{k}:", json.dumps(v, indent=1))
            print("eclipse edges (as-is):", json.dumps(summary["eclipse_edges"], indent=1))
            print("eclipse edges (frame-aligned):", json.dumps(s_al["eclipse_edges"], indent=1))
            result["runs"][key] = entry
    result["wall_time_s"] = round(time.time() - wall0, 1)
    out = os.path.join(REPO, "results", "crosscheck_basilisk.json")
    series = result.pop("series_1hz")
    result["series_1hz"] = "@@SERIES@@"                      # summary indented, 1 Hz arrays compact
    text = json.dumps(result, indent=1).replace('"@@SERIES@@"', json.dumps(series, separators=(",", ":")))
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\nwrote {out}  ({os.path.getsize(out) / 1e3:.0f} kB), wall {result['wall_time_s']} s")


if __name__ == "__main__":
    main()
