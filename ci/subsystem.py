"""Subsystem: a local model with its own states, unknown parameters and estimator.

A subsystem is described by

* ``states``: names of its dynamic states (e.g. ``["x1", "x2", "v1", "v2"]``);
* ``inputs``: names of its input ports; external loads *and* incoming
  messages from neighbouring subsystems are summed on these ports;
* ``dynamics(x, u, p, t) -> dx/dt``: continuous-time right-hand side;
* ``measurement(x, u, p, t) -> y``: observation model;
* ``parameters``: known constants, available inside ``dynamics`` through ``p``;
* ``unknowns``: parameters to be estimated jointly with the states. They are
  appended to the state vector as random walks (``theta_{k+1} = theta_k + w``).

``p`` is a plain ``dict`` mapping parameter names to their current values, so
the same physics function serves both the forward model (all parameters known)
and the inverse problem (some parameters replaced by their running estimates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from .filters import Filter, get_filter
from .integrators import get_integrator

Dynamics = Callable[[np.ndarray, np.ndarray, Dict[str, float], float], np.ndarray]


@dataclass
class Unknown:
    """Specification of an unknown parameter estimated as a random walk.

    Parameters
    ----------
    initial : float
        Initial guess.
    std : float, optional
        Initial standard deviation (prior uncertainty). Defaults to ``|initial|``.
    process_std : float, optional
        Random-walk standard deviation per time step. ``0`` freezes the
        parameter once the filter has converged; a small value lets it drift
        (useful for slowly varying parameters).
    lower, upper : float, optional
        Clamp applied to the value handed to ``dynamics``. The filter state
        itself is never clipped; the clamp only protects the physics
        (e.g. ``k_eff = max(k, 1.0)``).
    """

    initial: float
    std: Optional[float] = None
    process_std: float = 0.0
    lower: Optional[float] = None
    upper: Optional[float] = None

    def __post_init__(self):
        if self.std is None:
            self.std = abs(float(self.initial)) if self.initial != 0 else 1.0

    @classmethod
    def coerce(cls, spec) -> "Unknown":
        if isinstance(spec, Unknown):
            return spec
        if isinstance(spec, Mapping):
            return cls(**spec)
        if isinstance(spec, (tuple, list)):
            return cls(*spec)
        return cls(float(spec))


def _as_cov(value, n: int, what: str) -> np.ndarray:
    """Coerce a scalar, a vector of variances, or a full matrix into an ``(n, n)`` covariance."""
    if value is None:
        raise ValueError(f"{what} must be provided.")
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.eye(n) * float(arr)
    if arr.ndim == 1:
        if arr.size != n:
            raise ValueError(f"{what} has {arr.size} entries, expected {n}.")
        return np.diag(arr)
    if arr.shape != (n, n):
        raise ValueError(f"{what} has shape {arr.shape}, expected ({n}, {n}).")
    return arr.copy()


class Subsystem:
    """A node of the interaction graph."""

    def __init__(
        self,
        name: str,
        states: Sequence[str],
        inputs: Sequence[str],
        dynamics: Dynamics,
        measurement: Optional[Dynamics] = None,
        *,
        parameters: Optional[Mapping[str, float]] = None,
        unknowns: Optional[Mapping[str, Union[Unknown, Mapping, float, tuple]]] = None,
        filter: Union[str, Filter, type] = "ukf",
        filter_options: Optional[Mapping] = None,
        integrator: Union[str, Callable] = "heun",
        x0: Optional[Union[Sequence[float], Mapping[str, float]]] = None,
        P0: Union[float, Sequence[float], np.ndarray] = 1e-4,
        Q: Union[float, Sequence[float], np.ndarray] = 1e-12,
        R: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        measured: Optional[Sequence[str]] = None,
    ):
        self.name = str(name)
        self.states: List[str] = list(states)
        self.inputs: List[str] = list(inputs)
        self.dynamics = dynamics
        self.measurement = measurement
        self.parameters: Dict[str, float] = dict(parameters or {})
        self.unknowns: Dict[str, Unknown] = {k: Unknown.coerce(v) for k, v in (unknowns or {}).items()}
        self.measured: Optional[List[str]] = list(measured) if measured is not None else None

        if len(set(self.states)) != len(self.states):
            raise ValueError(f"Subsystem '{name}': duplicate state names.")
        clash = set(self.states) & set(self.unknowns)
        if clash:
            raise ValueError(f"Subsystem '{name}': names used as both state and unknown: {sorted(clash)}")
        clash = set(self.parameters) & set(self.unknowns)
        if clash:
            raise ValueError(f"Subsystem '{name}': names used as both known parameter and unknown: {sorted(clash)}")

        self.filter: Filter = get_filter(filter, **(filter_options or {}))
        self.integrator = get_integrator(integrator)

        nx = self.nx
        if x0 is None:
            x0_arr = np.zeros(nx)
        elif isinstance(x0, Mapping):
            x0_arr = np.array([float(x0.get(s, 0.0)) for s in self.states])
        else:
            x0_arr = np.asarray(x0, dtype=float).reshape(-1)
            if x0_arr.size != nx:
                raise ValueError(f"Subsystem '{name}': x0 has {x0_arr.size} entries, expected {nx}.")
        self.x0 = x0_arr
        self._P0_states = _as_cov(P0, nx, f"Subsystem '{name}': P0")
        self._Q_states = _as_cov(Q, nx, f"Subsystem '{name}': Q")
        self._R = R  # coerced lazily, once ny is known

    # ------------------------------------------------------------------ sizes
    @property
    def nx(self) -> int:
        return len(self.states)

    @property
    def nu(self) -> int:
        return len(self.inputs)

    @property
    def n_unknowns(self) -> int:
        return len(self.unknowns)

    @property
    def nz(self) -> int:
        """Augmented state dimension (states + unknown parameters)."""
        return self.nx + self.n_unknowns

    @property
    def variables(self) -> List[str]:
        """Names of the augmented state: states followed by unknowns."""
        return self.states + list(self.unknowns)

    @property
    def ny(self) -> int:
        if self.measured is not None:
            return len(self.measured)
        if self.measurement is None:
            return 0
        y = self.measurement(self.x0, np.zeros(self.nu), self.params(self.z0), 0.0)
        return int(np.asarray(y).size)

    # --------------------------------------------------------------- beliefs
    @property
    def z0(self) -> np.ndarray:
        theta0 = np.array([u.initial for u in self.unknowns.values()], dtype=float)
        return np.concatenate([self.x0, theta0])

    @property
    def P0(self) -> np.ndarray:
        P = np.zeros((self.nz, self.nz))
        P[: self.nx, : self.nx] = self._P0_states
        for i, u in enumerate(self.unknowns.values()):
            P[self.nx + i, self.nx + i] = float(u.std) ** 2
        return P

    @property
    def Q(self) -> np.ndarray:
        Q = np.zeros((self.nz, self.nz))
        Q[: self.nx, : self.nx] = self._Q_states
        for i, u in enumerate(self.unknowns.values()):
            Q[self.nx + i, self.nx + i] = float(u.process_std) ** 2
        return Q

    @property
    def R(self) -> np.ndarray:
        ny = self.ny
        if ny == 0:
            return np.zeros((0, 0))
        if self._R is None:
            raise ValueError(f"Subsystem '{self.name}': R (measurement noise covariance) must be provided.")
        return _as_cov(self._R, ny, f"Subsystem '{self.name}': R")

    # ---------------------------------------------------------------- models
    def params(self, z: np.ndarray) -> Dict[str, float]:
        """Known parameters merged with the current unknown estimates carried in ``z``."""
        p = dict(self.parameters)
        for i, (name, spec) in enumerate(self.unknowns.items()):
            val = float(z[self.nx + i])
            if spec.lower is not None:
                val = max(val, spec.lower)
            if spec.upper is not None:
                val = min(val, spec.upper)
            p[name] = val
        return p

    def project(self, z: np.ndarray) -> np.ndarray:
        """Clip the unknown-parameter entries of the augmented mean to their ``[lower, upper]`` bounds.

        Applied after every filter update so that an estimate pushed outside the
        physical range by a transient does not get stuck where the clamped
        physics has zero sensitivity to it.
        """
        if not self.unknowns:
            return z
        z = np.array(z, dtype=float, copy=True)
        for i, spec in enumerate(self.unknowns.values()):
            j = self.nx + i
            if spec.lower is not None and z[j] < spec.lower:
                z[j] = spec.lower
            if spec.upper is not None and z[j] > spec.upper:
                z[j] = spec.upper
        return z

    def split(self, z: np.ndarray):
        z = np.asarray(z, dtype=float)
        return z[: self.nx], z[self.nx :]

    def transition(self, z: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Discrete transition of the augmented state over one step ``dt``."""
        x, theta = self.split(z)
        p = self.params(z)
        rhs = lambda x_, t_: self.dynamics(x_, u, p, t_)
        x_next = self.integrator(rhs, x, t, dt)
        return np.concatenate([x_next, theta])

    def measure(self, z: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        if self.measurement is None:
            return np.zeros(0)
        x, _ = self.split(z)
        return np.asarray(self.measurement(x, u, self.params(z), t), dtype=float).reshape(-1)

    def derivative(self, z: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        x, _ = self.split(z)
        return np.asarray(self.dynamics(x, u, self.params(z), t), dtype=float)

    # -------------------------------------------------------------- indexing
    def index(self, name: str) -> int:
        try:
            return self.variables.index(name)
        except ValueError:
            raise KeyError(f"Subsystem '{self.name}' has no variable '{name}'. Known: {self.variables}") from None

    def indices(self, names: Iterable[str]) -> np.ndarray:
        return np.array([self.index(n) for n in names], dtype=int)

    def port(self, name: str) -> int:
        try:
            return self.inputs.index(name)
        except ValueError:
            raise KeyError(f"Subsystem '{self.name}' has no input port '{name}'. Known: {self.inputs}") from None

    def __repr__(self) -> str:
        return (
            f"Subsystem({self.name!r}, states={self.states}, inputs={self.inputs}, "
            f"unknowns={list(self.unknowns)}, filter={self.filter.name!r})"
        )
