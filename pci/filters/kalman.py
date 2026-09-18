"""Linear and extended Kalman filters.

Both filters linearize the (possibly nonlinear) transition and measurement
functions with central finite differences. The *linear* filter linearizes
once, about the initial belief, and then keeps the resulting ``A`` and ``H``
matrices fixed, which is exact for linear models with known parameters. The
*extended* filter re-linearizes at every step and therefore handles unknown
parameters carried in the augmented state.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..linalg import numerical_jacobian, robust_inverse, symmetrize
from .base import Filter, FilterStep, register_filter


def _joseph_update(x_pred, P_pred, y, y_pred, H, R):
    S = symmetrize(H @ P_pred @ H.T + R)
    K = P_pred @ H.T @ robust_inverse(S)
    x = x_pred + K @ (y - y_pred)
    I = np.eye(x_pred.size)
    IKH = I - K @ H
    P = IKH @ P_pred @ IKH.T + K @ R @ K.T
    return x, symmetrize(P), S


@register_filter("ekf", "extended", "extended_kalman")
class ExtendedKalmanFilter(Filter):
    """Extended Kalman filter with numerical Jacobians and Joseph-form update."""

    name = "ekf"

    def __init__(self, jac_eps: float = 1e-6):
        self.jac_eps = jac_eps

    def _linearize(self, x, P, u, f, h, t):
        A = numerical_jacobian(lambda z: f(z, u, t), x, eps=self.jac_eps)
        x_pred = np.asarray(f(x, u, t), dtype=float)
        H = numerical_jacobian(lambda z: h(z, u, t), x_pred, eps=self.jac_eps)
        return A, H, x_pred

    def step(self, x, P, u, y, f, h, Q, R, t) -> FilterStep:
        x = np.asarray(x, dtype=float)
        A, H, x_pred = self._linearize(x, P, u, f, h, t)
        P_pred = symmetrize(A @ P @ A.T + Q)
        if y is None:
            return FilterStep(x_pred, P_pred, x_pred, P_pred)
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = np.asarray(h(x_pred, u, t), dtype=float).reshape(-1)
        x_new, P_new, S = _joseph_update(x_pred, P_pred, y, y_pred, H, R)
        return FilterStep(x_new, P_new, x_pred, P_pred, y_pred, S)


@register_filter("kf", "linear", "linear_kalman", "lkf")
class LinearKalmanFilter(ExtendedKalmanFilter):
    """Linear Kalman filter.

    ``A`` and ``H`` are obtained by linearizing ``f`` and ``h`` once, at the
    first call, and are then held fixed. Because the mean is still propagated
    through ``f`` itself, the filter is exact for linear models and behaves
    like a "frozen-Jacobian" EKF otherwise. Use ``ekf``/``ukf``/``ckf`` when the
    subsystem carries unknown parameters.
    """

    name = "kf"

    def __init__(self, jac_eps: float = 1e-6, A: Optional[np.ndarray] = None, H: Optional[np.ndarray] = None):
        super().__init__(jac_eps=jac_eps)
        self._A = None if A is None else np.asarray(A, dtype=float)
        self._H = None if H is None else np.asarray(H, dtype=float)

    def _linearize(self, x, P, u, f, h, t):
        x_pred = np.asarray(f(x, u, t), dtype=float)
        if self._A is None:
            self._A = numerical_jacobian(lambda z: f(z, u, t), x, eps=self.jac_eps)
        if self._H is None:
            self._H = numerical_jacobian(lambda z: h(z, u, t), x_pred, eps=self.jac_eps)
        return self._A, self._H, x_pred
