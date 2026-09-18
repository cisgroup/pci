"""Explicit time integrators for continuous-time subsystem dynamics.

Every filter propagates the subsystem mean by integrating the user-supplied
continuous-time dynamics ``f(x, u, p, t)`` over one step with one of these
schemes (the input ``u`` is held constant over the step). The KF/EKF obtain
their transition matrix as the numerical Jacobian of the *discretized* step,
so the covariance propagation is consistent with the chosen integrator; the
UKF/CKF push their sigma points through the same discretized step.

Available: ``"euler"`` (1st order), ``"heun"`` (explicit trapezoidal / RK2,
2nd order, default), ``"rk4"`` (classical Runge-Kutta, 4th order). Select per
subsystem with ``Subsystem(..., integrator="rk4")``, in
``MassSpringChain.decompose(integrator=...)`` (one name for all, a list per
subsystem or a dict by subsystem name) or in a YAML config. A custom
``step(rhs, x, t, dt) -> x_next`` callable is accepted anywhere a name is.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

RHS = Callable[[np.ndarray, float], np.ndarray]


def euler(rhs: RHS, x: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Forward Euler step."""
    return x + dt * np.asarray(rhs(x, t), dtype=float)


def heun(rhs: RHS, x: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Heun (explicit trapezoidal, RK2) predictor-corrector step."""
    k1 = np.asarray(rhs(x, t), dtype=float)
    x_pred = x + dt * k1
    k2 = np.asarray(rhs(x_pred, t + dt), dtype=float)
    return x + 0.5 * dt * (k1 + k2)


def rk4(rhs: RHS, x: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Classical fourth-order Runge-Kutta step."""
    k1 = np.asarray(rhs(x, t), dtype=float)
    k2 = np.asarray(rhs(x + 0.5 * dt * k1, t + 0.5 * dt), dtype=float)
    k3 = np.asarray(rhs(x + 0.5 * dt * k2, t + 0.5 * dt), dtype=float)
    k4 = np.asarray(rhs(x + dt * k3, t + dt), dtype=float)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


INTEGRATORS = {"euler": euler, "heun": heun, "rk4": rk4}


def available_integrators():
    """Names of the built-in integrators."""
    return tuple(INTEGRATORS)


def get_integrator(name_or_fn):
    """Resolve an integrator from its name (``'euler'``, ``'heun'``, ``'rk4'``) or pass a callable through."""
    if callable(name_or_fn):
        return name_or_fn
    key = str(name_or_fn).lower()
    if key not in INTEGRATORS:
        raise ValueError(f"Unknown integrator '{name_or_fn}'. Choose from {sorted(INTEGRATORS)}.")
    return INTEGRATORS[key]
