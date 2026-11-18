"""Frame + quaternion conventions for the SCALAR attitude sim.

This module is the **shared contract**: every other module expresses rotations
the way it is defined here. Read this once and the sign conventions elsewhere
follow.

FRAMES
------
ECI   Earth-Centered, non-rotating: the mean equator and equinox of the
      date (no nutation), NOT J2000/GCRF. It is the frame the sun series is
      reported in (environment/sun.py) and the one Rz(GMST) carries to ECEF
      (timeutil.gmst_rad); the frame in which we express the truth sun
      vector, orbit position, and B-field. These are consistent with each
      other, so nothing inside the sim depends on the choice. It differs from
      J2000 by precession since 2000 (0.37 deg in 2026,
      docs/crosscheck-basilisk.md, discrepancy A): an attitude or orbit taken
      from, or handed to, a J2000 source must be rotated first.
ECEF  Earth-Centered Earth-Fixed. Rotates with Earth about +z at OMEGA_EARTH;
      used only where geography matters (the dipole B-field tilt).
BODY  Spacecraft body frame, fixed to the CubeSat. Sensors live here.

ATTITUDE QUATERNION
-------------------
Attitude is carried as a unit quaternion ``q`` in **Hamilton, scalar-LAST**
layout ``[x, y, z, w]`` — identical to ``scipy.spatial.transform.Rotation``,
which we delegate to for all heavy lifting so there are no hand-rolled sign
bugs.

``q`` denotes the rotation **q_{B<-I}** that carries a vector's ECI coordinates
to its BODY coordinates:

    v_body = C(q) @ v_eci ,     C(q) = DCM of q  (the "body-from-inertial" DCM)

so ``to_body`` applies ``q`` and ``to_eci`` applies its inverse.

KINEMATICS
----------
With ``omega`` the body angular velocity of BODY w.r.t. ECI, expressed in BODY,
the body-from-inertial DCM obeys ``dC/dt = -[omega]x C``. ``propagate`` advances
``q`` by one step consistent with that relation (verified in tests by checking
that a body-fixed axis traces the physically correct sense in ECI).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "IDENTITY_QUAT",
    "normalize",
    "is_unit",
    "rand_quat",
    "quat_mul",
    "quat_conj",
    "quat_from_rotvec",
    "rotvec_from_quat",
    "quat_from_small_angle",
    "small_angle_from_quat",
    "dcm",
    "quat_from_dcm",
    "rotate",
    "to_body",
    "to_eci",
    "propagate",
    "angle_between",
    "attitude_error_angle",
    "attitude_error_rotvec",
]

# scalar-last identity quaternion [x, y, z, w]
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


# --- basic quaternion algebra ----------------------------------------------
def normalize(q):
    """Return ``q`` scaled to unit norm (raises on a zero quaternion)."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("cannot normalize a zero-norm quaternion")
    return q / n


def is_unit(q, tol=1e-9):
    """True if ``q`` has unit norm within ``tol``."""
    return abs(np.linalg.norm(q) - 1.0) <= tol


def rand_quat(rng):
    """A uniformly-random unit quaternion (for tests / Monte-Carlo init)."""
    return Rotation.random(random_state=rng).as_quat()


def quat_mul(a, b):
    """Hamilton product ``a ⊗ b`` (scalar-last), matching DCM composition
    ``C(a⊗b) = C(a) @ C(b)``."""
    return (Rotation.from_quat(a) * Rotation.from_quat(b)).as_quat()


def quat_conj(q):
    """Conjugate/inverse of a unit quaternion (scalar-last)."""
    q = np.asarray(q, dtype=float)
    return np.array([-q[0], -q[1], -q[2], q[3]])


# --- rotation-vector <-> quaternion ----------------------------------------
def quat_from_rotvec(rotvec):
    """Quaternion from a rotation vector (axis * angle_rad)."""
    return Rotation.from_rotvec(rotvec).as_quat()


def rotvec_from_quat(q):
    """Rotation vector (axis * angle_rad) from a quaternion."""
    return Rotation.from_quat(q).as_rotvec()


def quat_from_small_angle(dtheta):
    """First-order quaternion for a small rotation vector ``dtheta`` (rad).

    This is the MEKF error-state -> quaternion map: ``dq = [dtheta/2, 1]``,
    renormalized. Exact enough for the filter's small correction angles.
    """
    dtheta = np.asarray(dtheta, dtype=float)
    return normalize(np.array([dtheta[0] / 2, dtheta[1] / 2, dtheta[2] / 2, 1.0]))


def small_angle_from_quat(q):
    """Inverse of :func:`quat_from_small_angle`: the 3-vector ``2*vec(q)`` with
    the sign chosen so the scalar part is >= 0 (the MEKF residual)."""
    q = np.asarray(q, dtype=float)
    q = q if q[3] >= 0 else -q
    return 2.0 * q[:3]


# --- DCM <-> quaternion -----------------------------------------------------
def dcm(q):
    """Body-from-inertial DCM ``C_{B<-I}`` of attitude ``q`` (v_body = C @ v_eci)."""
    return Rotation.from_quat(q).as_matrix()


def quat_from_dcm(C):
    """Attitude quaternion from a body-from-inertial DCM."""
    return Rotation.from_matrix(C).as_quat()


# --- applying an attitude ---------------------------------------------------
def rotate(q, v):
    """Apply rotation ``q`` to vector(s) ``v`` (i.e. ``C(q) @ v``)."""
    return Rotation.from_quat(q).apply(v)


def to_body(q, v_eci):
    """Express an ECI vector in BODY coordinates: ``v_body = C_{B<-I} v_eci``."""
    return Rotation.from_quat(q).apply(v_eci)


def to_eci(q, v_body):
    """Express a BODY vector in ECI coordinates: ``v_eci = C_{B<-I}^T v_body``."""
    return Rotation.from_quat(q).apply(v_body, inverse=True)


# --- kinematics -------------------------------------------------------------
def propagate(q, omega_body, dt):
    """Advance attitude ``q = q_{B<-I}`` by ``dt`` under constant body rate.

    ``omega_body`` is the angular velocity of BODY w.r.t. ECI in BODY coords
    [rad/s]. Uses the exact rotation-vector exponential over the step
    (zeroth-order hold on omega), consistent with ``dC/dt = -[omega]x C``.
    """
    omega_body = np.asarray(omega_body, dtype=float)
    dq = Rotation.from_rotvec(-omega_body * dt)          # C_{B<-I} left-update
    return normalize((dq * Rotation.from_quat(q)).as_quat())


# --- error / comparison metrics --------------------------------------------
def angle_between(u, v):
    """Angle [rad] between two vectors, robust to numerical range."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def attitude_error_rotvec(q_est, q_true):
    """Rotation vector of the error rotation ``q_est ⊗ q_true^-1`` [rad]."""
    err = Rotation.from_quat(q_est) * Rotation.from_quat(q_true).inv()
    return err.as_rotvec()


def attitude_error_angle(q_est, q_true):
    """Total attitude error angle [rad] between two attitudes (the metric the
    10-15 deg knowledge-error budget is measured against)."""
    return float(np.linalg.norm(attitude_error_rotvec(q_est, q_true)))
