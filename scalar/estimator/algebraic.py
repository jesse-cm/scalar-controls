"""Algebraic TRIAD-only attitude — a gyro-free baseline for comparison.

Each sunlit sample gets an instantaneous deterministic two-vector (sun + mag)
TRIAD solution; in eclipse there is only one vector, so a stateless algebraic
solver has nothing to do and must *hold* its last estimate (no gyro to coast
on). Contrasting this with the MEKF is the clearest demonstration of what the
filter buys: gyro propagation through eclipse and noise averaging when lit.
"""
from __future__ import annotations

import numpy as np

from ..attitude import IDENTITY_QUAT, attitude_error_angle
from .triad import triad

__all__ = ["run_triad"]


def run_triad(res):
    """Run the TRIAD-only estimator over a SimResult -> (q_est, err_deg)."""
    n = len(res)
    q_est = np.empty((n, 4))
    err = np.empty(n)
    q_last = IDENTITY_QUAT.copy()
    for k in range(n):
        if res.sun_valid[k]:
            q_last = triad(res.sun_eci[k], res.sun_body[k],
                           res.B_eci[k] / np.linalg.norm(res.B_eci[k]),
                           res.mag_body[k] / np.linalg.norm(res.mag_body[k]))
        q_est[k] = q_last          # hold through eclipse (no gyro)
        err[k] = np.rad2deg(attitude_error_angle(q_last, res.q_true[k]))
    return q_est, err
