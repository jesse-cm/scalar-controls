"""SCALAR — CubeSat attitude determination and control simulation.

An ADCS sim for the WashU/PROVES 1U CubeSat. Determination: a tumbling rigid
body observed through a real sensor suite (LSM6DSO gyro, LIS2MDL
magnetometer, face-board sun sensors) whose 3D attitude is reconstructed by a
Multiplicative EKF (MEKF). Control: magnetorquer actuators and LQR-family
controllers closed on that estimate. Simulation only, not flight code.

Sub-packages:
    scalar.environment  truth world (sun, orbit, geomagnetic field)
    scalar.sensors      sensor models that observe the truth
    scalar.estimator    TRIAD, MEKF, USQUE, TRIAD-only baseline
    scalar.calibration  magnetometer and sun-sensor calibration
    scalar.actuators    magnetorquer models
    scalar.control      LQR, periodic LQR, LQI, closed-loop sim, tuner
    scalar.learning     learned sun-sensor front end
    scalar.sysid        magnetorquer and inertia identification
Modules:
    scalar.constants    physical constants + unit conventions
    scalar.attitude     frame + quaternion conventions (the shared contract)
    scalar.timeutil     time conversions
    scalar.dynamics     rigid-body dynamics, free and with applied torque
    scalar.scenario     one determination run: truth + environment + sensors
    scalar.montecarlo   randomized multi-run estimator comparison
"""
from __future__ import annotations

__version__ = "0.1.0"
