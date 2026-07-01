"""Magnetorquer models for the SCALAR actuator suite.

The flight actuator set is **one PCB-trace magnetorquer + two torque rods**:

  * **PCB-trace coil** — an air-core planar spiral etched into the board. Dipole
    is the pure coil law  m = N * I * A  (no core), so it is weak (~0.01-0.05
    A·m² on a 1U face) but has no mass/volume cost.
  * **Torque rod** — many turns wound on a ferromagnetic core (the 9 cm 1008-steel
    design from the sizing study). The core multiplies the dipole by an apparent-
    permeability gain from the demagnetizing factor, so a rod reaches ~0.1-0.3
    A·m² at the same drive. Two rods carry the two strong axes.

Each torquer produces a dipole along its body-fixed mounting axis, bounded by a
saturation dipole ``m_max`` (set by drive voltage / coil resistance / geometry).
The mechanical torque on the spacecraft is  τ = m × B  — which has no component
along B, so the instantaneous authority is only 2-DOF (the underactuation the
LQR must work around as B rotates through the orbit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

__all__ = ["Magnetorquer", "ActuatorSet", "default_actuator_set"]


@dataclass
class Magnetorquer:
    """A single magnetorquer producing a saturable dipole along ``axis``."""

    axis: np.ndarray            # body-frame unit vector
    m_max: float                # saturation dipole magnitude [A·m^2]
    name: str = ""

    def __post_init__(self):
        self.axis = np.asarray(self.axis, dtype=float)
        self.axis = self.axis / np.linalg.norm(self.axis)

    @classmethod
    def air_core(cls, axis, n_turns, area, current_max, name="pcb-coil"):
        """PCB-trace / air-core coil: m_max = N·I·A."""
        return cls(axis, n_turns * area * current_max, name)

    @classmethod
    def rod(cls, axis, n_turns, area, current_max, core_gain, name="rod"):
        """Ferromagnetic torque rod: m_max = core_gain·N·I·A."""
        return cls(axis, core_gain * n_turns * area * current_max, name)


@dataclass
class ActuatorSet:
    """A set of magnetorquers; maps a commanded body dipole to a realizable one."""

    torquers: List[Magnetorquer] = field(default_factory=list)

    @property
    def m_max_vector(self):
        """Per-torquer saturation magnitudes [A·m^2]."""
        return np.array([t.m_max for t in self.torquers])

    def command(self, m_desired_body):
        """Realizable body dipole for a desired dipole, saturating each torquer.

        Each torquer contributes its (clipped) projection of ``m_desired_body``
        onto its axis; for an orthogonal mounting this is per-axis saturation.
        """
        m_desired_body = np.asarray(m_desired_body, dtype=float)
        m = np.zeros(3)
        for t in self.torquers:
            c = np.clip(float(m_desired_body @ t.axis), -t.m_max, t.m_max)
            m += c * t.axis
        return m

    def torque(self, m_desired_body, B_body):
        """Return (torque [N·m], realized dipole [A·m^2]) for a desired dipole."""
        m = self.command(m_desired_body)
        return np.cross(m, np.asarray(B_body, dtype=float)), m


def default_actuator_set(m_rod=0.2, m_coil=0.02):
    """Two torque rods on X/Y + one PCB coil on Z (the flight configuration).

    Note the ~10x weaker coil axis — an intentional authority asymmetry the
    controller has to respect (and one reason full 3-axis pointing is hard).
    """
    return ActuatorSet([
        Magnetorquer([1.0, 0.0, 0.0], m_rod, "rod-x"),
        Magnetorquer([0.0, 1.0, 0.0], m_rod, "rod-y"),
        Magnetorquer([0.0, 0.0, 1.0], m_coil, "pcb-coil-z"),
    ])
