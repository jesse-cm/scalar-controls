"""Attitude-independent on-orbit magnetometer calibration.

The magnetometer measurement model is

    m = D (C_{B<-I} B_eci) + b + noise,

with ``D`` the soft-iron / scale / non-orthogonality matrix and ``b`` the
hard-iron bias.  The direction ``C_{B<-I} B_eci`` depends on the (unknown)
attitude, but its *magnitude* does not::

    |C_{B<-I} B_eci| = |B_eci| = F,

and ``F`` is known independently along the orbit from the IGRF model.  So the
tip of the *true* field traces a sphere of (time-varying) radius ``F``, and the
sensor distortion maps that sphere to an **ellipsoid**.  Recovering the
ellipsoid recovers the calibration — no attitude knowledge required.

Writing ``u = m`` for the raw measurement, we seek ``W`` (symmetric) and ``v``
such that ``|W (u - v)| = F``.  Expanding with ``G = W^T W = W^2``::

    (u - v)^T G (u - v) = F^2
    u^T G u - 2 (G v)^T u + (v^T G v) = F^2.

This is **linear** in the unknowns ``G`` (6 independent entries), ``d = G v``
(3) and ``e = v^T G v`` (1), so a single linear least-squares solve over all
samples yields the quadric.  Then ``v = G^{-1} d`` is the bias and
``W = sqrtm(G)`` the correction matrix.

Because only the magnitude constrains the fit, ``D`` is recovered only up to an
arbitrary rotation (``W D`` is orthogonal, not necessarily identity).  That is
exactly the right amount of information: the estimator already solves for
attitude, and the *calibrated direction* ``W(m - v)`` differs from the true
body-frame field only by that fixed rotation, which is absorbed into the
attitude solution.  For a symmetric ``D`` (pure scale + soft-iron, no rotation)
the recovery is exact, ``W D = I``.

References
----------
* J. L. Crassidis, K.-L. Lai, R. R. Harman, "Real-Time Attitude-Independent
  Three-Axis Magnetometer Calibration," J. Guidance, Control, and Dynamics,
  2005.
* R. Alonso, M. D. Shuster, "TWOSTEP: A Fast Robust Algorithm for
  Attitude-Independent Magnetometer-Bias Determination," J. Astronautical
  Sciences, 2002.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import sqrtm

__all__ = [
    "MagCalibration",
    "fit_magnetometer_ellipsoid",
    "apply_mag_calibration",
]


@dataclass
class MagCalibration:
    """Recovered magnetometer calibration ``|W (m - v)| = F``.

    Attributes
    ----------
    W : (3, 3) ndarray
        Symmetric correction matrix (inverse soft-iron, up to rotation).
    v : (3,) ndarray
        Hard-iron bias estimate, in tesla.
    G : (3, 3) ndarray
        The fitted quadric ``G = W^T W``.
    residual_rms : float
        RMS of ``|W (m - v)| - F`` over the fit samples, in tesla.
    n_samples : int
        Number of samples used in the fit.
    """

    W: np.ndarray
    v: np.ndarray
    G: np.ndarray
    residual_rms: float
    n_samples: int

    def apply(self, m: np.ndarray) -> np.ndarray:
        """Return the calibrated field(s) ``W (m - v)``."""
        return apply_mag_calibration(m, self)


def fit_magnetometer_ellipsoid(m: np.ndarray, F: np.ndarray) -> MagCalibration:
    """Fit the attitude-independent magnetometer calibration.

    Parameters
    ----------
    m : (N, 3) ndarray
        Raw magnetometer measurements in tesla (body frame, distorted).
    F : (N,) ndarray or float
        Known field magnitude at each sample (``|B_model|`` from IGRF), tesla.
        A scalar is broadcast to all samples.

    Returns
    -------
    MagCalibration
        The recovered ``W``, ``v`` and fit diagnostics.

    Raises
    ------
    ValueError
        If fewer than 9 samples are supplied (the quadric has 10 parameters,
        one of which is fixed by the overall scale), or the geometry is
        degenerate (the field directions do not span enough of the sphere).
    """
    m = np.asarray(m, dtype=float)
    if m.ndim != 2 or m.shape[1] != 3:
        raise ValueError(f"m must be (N, 3); got {m.shape}")
    N = m.shape[0]
    F = np.broadcast_to(np.asarray(F, dtype=float), (N,))
    if N < 10:
        raise ValueError(
            f"need at least 10 samples to fit the ellipsoid; got {N}")

    x, y, z = m[:, 0], m[:, 1], m[:, 2]
    # u^T G u - 2 d^T u + e = F^2, with G symmetric.
    A = np.column_stack([
        x * x, y * y, z * z,          # G diagonal
        2 * x * y, 2 * x * z, 2 * y * z,  # G off-diagonal
        -2 * x, -2 * y, -2 * z,       # -2 d
        np.ones_like(x),              # e
    ])
    rhs = F ** 2
    p, _res, rank, _sv = np.linalg.lstsq(A, rhs, rcond=None)
    if rank < 10:
        raise ValueError(
            "degenerate geometry: field directions do not span the sphere "
            f"(design-matrix rank {rank} < 10). Use a longer / more tumbled arc.")

    G = np.array([[p[0], p[3], p[4]],
                  [p[3], p[1], p[5]],
                  [p[4], p[5], p[2]]])
    d = p[6:9]
    # G must be positive definite for a real ellipsoid.
    evals = np.linalg.eigvalsh(G)
    if np.any(evals <= 0):
        raise ValueError(
            f"fitted quadric is not an ellipsoid (eigenvalues {evals}); "
            "the measurement geometry is degenerate or too noisy.")

    v = np.linalg.solve(G, d)
    W = np.real(sqrtm(G))

    cal = (W @ (m - v).T).T
    residual = np.linalg.norm(cal, axis=1) - F
    return MagCalibration(
        W=W, v=v, G=G,
        residual_rms=float(np.sqrt(np.mean(residual ** 2))),
        n_samples=int(N),
    )


def apply_mag_calibration(m: np.ndarray, cal: MagCalibration) -> np.ndarray:
    """Apply a :class:`MagCalibration` to raw measurement(s).

    Parameters
    ----------
    m : (3,) or (N, 3) ndarray
        Raw magnetometer measurement(s), tesla.
    cal : MagCalibration
        The calibration to apply.

    Returns
    -------
    ndarray
        Calibrated field(s) ``W (m - v)``, same shape as ``m``.
    """
    m = np.asarray(m, dtype=float)
    if m.ndim == 1:
        return cal.W @ (m - cal.v)
    return (cal.W @ (m - cal.v).T).T
