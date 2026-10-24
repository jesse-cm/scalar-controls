"""Attitude control with magnetorquers.

    lqr           time-invariant LQR + perpendicular-to-B projection (damps rate)
    periodic_lqr  periodic Riccati gain schedule, two- and three-mode supervisors
    lqi           periodic LQR with an integral state on attitude error
    tune          closed loop on the MEKF estimate, weight sweeps
    rl_tune       pass/fail criteria, reward, cross-entropy tuner
"""
from __future__ import annotations
