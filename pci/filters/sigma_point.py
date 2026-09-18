"""Sigma-point filters: unscented (UKF) and cubature (CKF).

Both filters share the same predict/update structure and differ only in the
choice of integration points and weights.

* UKF (Julier & Uhlmann): ``2n + 1`` points with a tunable central weight
  ``kappa / (n + kappa)``. ``kappa = 0`` reproduces the convention used in the
  paper scripts (``gamma_param = 0``).
* CKF (Arasaratnam & Haykin): ``2n`` spherical-radial cubature points with
  equal weights ``1 / (2n)``; numerically identical to the UKF with
  ``kappa = 0`` but without a central point.
"""

from __future__ import annotations

import numpy as np

from ..linalg import robust_cholesky, robust_inverse, symmetrize
from .base import Filter, FilterStep, register_filter


class _SigmaPointFilter(Filter):
    def points(self, x: np.ndarray, P: np.ndarray):
        """Return ``(X, W)`` with ``X`` of shape ``(n, m)`` and weights ``W`` of shape ``(m,)``."""
        raise NotImplementedError

    def step(self, x, P, u, y, f, h, Q, R, t) -> FilterStep:
        x = np.asarray(x, dtype=float).reshape(-1)
        n = x.size
        X, W = self.points(x, P)
        m = X.shape[1]

        # propagate through the transition
        Xp = np.empty((n, m))
        for i in range(m):
            Xp[:, i] = f(X[:, i], u, t)
        x_pred = Xp @ W
        dX = Xp - x_pred[:, None]
        P_pred = symmetrize((dX * W) @ dX.T + Q)

        if y is None:
            return FilterStep(x_pred, P_pred, x_pred, P_pred)

        y = np.asarray(y, dtype=float).reshape(-1)
        ny = y.size
        Yp = np.empty((ny, m))
        for i in range(m):
            Yp[:, i] = np.asarray(h(Xp[:, i], u, t), dtype=float).reshape(-1)
        y_pred = Yp @ W
        dY = Yp - y_pred[:, None]
        Pyy = symmetrize((dY * W) @ dY.T + R)
        Pxy = (dX * W) @ dY.T

        K = Pxy @ robust_inverse(Pyy)
        x_new = x_pred + K @ (y - y_pred)
        P_new = symmetrize(P_pred - K @ Pyy @ K.T)
        return FilterStep(x_new, P_new, x_pred, P_pred, y_pred, Pyy)


@register_filter("ukf", "unscented", "unscented_kalman")
class UnscentedKalmanFilter(_SigmaPointFilter):
    """Unscented Kalman filter with the ``kappa`` (a.k.a. ``gamma``) scaling parameter."""

    name = "ukf"

    def __init__(self, kappa: float = 0.0):
        self.kappa = float(kappa)

    def points(self, x, P):
        n = x.size
        c = n + self.kappa
        if c <= 0:
            raise ValueError("UKF requires n + kappa > 0.")
        L = robust_cholesky(c * P)
        X = np.empty((n, 2 * n + 1))
        X[:, 0] = x
        X[:, 1 : n + 1] = x[:, None] + L
        X[:, n + 1 :] = x[:, None] - L
        W = np.full(2 * n + 1, 1.0 / (2.0 * c))
        W[0] = self.kappa / c
        return X, W


@register_filter("ckf", "cubature", "cubature_kalman")
class CubatureKalmanFilter(_SigmaPointFilter):
    """Third-degree spherical-radial cubature Kalman filter."""

    name = "ckf"

    def points(self, x, P):
        n = x.size
        L = robust_cholesky(P) * np.sqrt(n)
        X = np.empty((n, 2 * n))
        X[:, :n] = x[:, None] + L
        X[:, n:] = x[:, None] - L
        W = np.full(2 * n, 1.0 / (2.0 * n))
        return X, W
