"""SCALAR — CubeSat attitude-determination simulation.

A determination-only ADCS sim for the WashU/PROVES 1U CubeSat: a freely
tumbling rigid body observed through a real sensor suite (LSM6DSO gyro,
LIS2MDL magnetometer, face-board CMOS photodiode sun sensors) whose 3D
attitude is reconstructed by a Multiplicative EKF (MEKF).

Sub-packages:
    scalar.environment  truth world (sun, orbit, geomagnetic field)
    scalar.sensors       sensor models that observe the truth
Modules:
    scalar.constants     physical constants + unit conventions
    scalar.attitude      frame + quaternion conventions (the shared contract)
"""
from __future__ import annotations

__version__ = "0.1.0"
