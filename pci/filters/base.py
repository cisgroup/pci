"""Common interface for local estimators.

A filter advances one subsystem's Gaussian belief ``(x, P)`` from time ``t``
to ``t + dt`` given the discrete transition ``f(x, u, t) -> x_next``, the
measurement model ``h(x, u, t) -> y`` and the measurement ``y``.

Filters are stateless apart from their hyper-parameters, so the same filter
object can serve several subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

Transition = Callable[[np.ndarray, np.ndarray, float], np.ndarray]
Measurement = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


@dataclass
class FilterStep:
    """Result of one filter step."""

    x: np.ndarray
    P: np.ndarray
    x_pred: np.ndarray
    P_pred: np.ndarray
    y_pred: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None  # innovation covariance


class Filter:
    """Base class for local Gaussian filters."""

    name: str = "base"

    def step(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        y: Optional[np.ndarray],
        f: Transition,
        h: Measurement,
        Q: np.ndarray,
        R: np.ndarray,
        t: float,
    ) -> FilterStep:
        """Predict with ``f`` and, if ``y`` is not ``None``, update with ``h``."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}()"


_REGISTRY: Dict[str, type] = {}


def register_filter(*names: str):
    """Class decorator registering a filter under one or more string names."""

    def deco(cls):
        for n in names:
            _REGISTRY[n.lower()] = cls
        return cls

    return deco


def get_filter(spec, **kwargs) -> Filter:
    """Resolve a filter from a string name, a class, or an instance.

    Examples
    --------
    >>> get_filter("ukf")
    >>> get_filter("ukf", kappa=0.0)
    >>> get_filter(UnscentedKalmanFilter(kappa=1.0))
    """
    if isinstance(spec, Filter):
        return spec
    if isinstance(spec, type) and issubclass(spec, Filter):
        return spec(**kwargs)
    key = str(spec).lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown filter '{spec}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def available_filters() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
