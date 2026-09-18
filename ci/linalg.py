"""Small numerical helpers shared by the filters."""

from __future__ import annotations

import numpy as np
import scipy.linalg


def symmetrize(P: np.ndarray) -> np.ndarray:
    """Return the symmetric part of ``P``."""
    return 0.5 * (P + P.T)


def robust_cholesky(P: np.ndarray, base_jitter: float = 1e-10, max_tries: int = 12) -> np.ndarray:
    """Lower-triangular Cholesky factor with diagonal jitter escalation.

    Falls back to an eigenvalue square root when the matrix cannot be made
    positive definite by jitter alone.
    """
    P_sym = symmetrize(np.asarray(P, dtype=float))
    n = P_sym.shape[0]
    I = np.eye(n)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            return scipy.linalg.cholesky(P_sym + jitter * I, lower=True)
        except scipy.linalg.LinAlgError:
            jitter = base_jitter if jitter == 0.0 else 10.0 * jitter
    w, V = np.linalg.eigh(P_sym)
    w = np.clip(w, 1e-12, None)
    return V @ np.diag(np.sqrt(w))


def robust_inverse(A: np.ndarray, base_jitter: float = 1e-10, max_tries: int = 12) -> np.ndarray:
    """Matrix inverse with diagonal jitter escalation for near-singular matrices."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    I = np.eye(n)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            return np.linalg.inv(A + jitter * I)
        except np.linalg.LinAlgError:
            jitter = base_jitter if jitter == 0.0 else 10.0 * jitter
    raise RuntimeError("Matrix inversion failed even after jitter escalation.")


def numerical_jacobian(fun, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference Jacobian of a vector-valued function."""
    x = np.asarray(x, dtype=float).copy()
    f0 = np.asarray(fun(x), dtype=float)
    J = np.zeros((f0.size, x.size))
    for j in range(x.size):
        step = eps * max(1.0, abs(x[j]))
        dx = np.zeros(x.size)
        dx[j] = step
        fp = np.asarray(fun(x + dx), dtype=float)
        fm = np.asarray(fun(x - dx), dtype=float)
        J[:, j] = (fp - fm) / (2.0 * step)
    return J
