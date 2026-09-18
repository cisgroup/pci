"""Local estimators available to subsystems.

Use string names in configurations: ``"kf"``, ``"ekf"``, ``"ukf"``, ``"ckf"``.
New filters (particle filters, PINN-based estimators, ...) are added by
subclassing :class:`Filter` and decorating with :func:`register_filter`.
"""

from .base import Filter, FilterStep, available_filters, get_filter, register_filter
from .kalman import ExtendedKalmanFilter, LinearKalmanFilter
from .sigma_point import CubatureKalmanFilter, UnscentedKalmanFilter

__all__ = [
    "Filter",
    "FilterStep",
    "available_filters",
    "get_filter",
    "register_filter",
    "LinearKalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "CubatureKalmanFilter",
]
