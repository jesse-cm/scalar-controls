"""TRIAD deterministic two-vector attitude — used to initialize the MEKF.

Given two non-parallel reference directions in ECI (sun, geomagnetic field) and
their measured counterparts in the body frame, TRIAD builds an orthonormal
triad from each pair and reads off the attitude that maps one to the other:

    C_{B<-I} = M_body @ M_eci^T

The first (more trusted) vector is honored exactly; error concentrates on the
second. We pass the sun vector first when available — the coarse sun sensor and
the dipole-model field are comparable here, but the ordering convention is fixed
so the caller controls it.
"""
from __future__ import annotations

import numpy as np

from ..attitude import quat_from_dcm

__all__ = ["triad"]


def _triad_matrix(v1, v2):
    t1 = v1 / np.linalg.norm(v1)
    t2 = np.cross(t1, v2)
    t2 = t2 / np.linalg.norm(t2)
    t3 = np.cross(t1, t2)
    return np.column_stack([t1, t2, t3])


def triad(r1_eci, b1_body, r2_eci, b2_body):
    """Attitude quaternion q_{B<-I} from two ECI/body vector pairs.

    ``r1``/``b1`` is the primary pair (fit exactly). Raises if the two reference
    (or body) directions are parallel, which makes the triad singular.
    """
    M_eci = _triad_matrix(np.asarray(r1_eci, float), np.asarray(r2_eci, float))
    M_body = _triad_matrix(np.asarray(b1_body, float), np.asarray(b2_body, float))
    C = M_body @ M_eci.T          # maps ECI -> body
    return quat_from_dcm(C)
