"""Message-passing schedules (how and when subsystems exchange messages).

* :class:`Jacobi` -- all subsystems build their incoming messages from the
  neighbours' values of the previous iteration and then update in parallel.
* :class:`GaussSeidel` -- subsystems update one after another in a prescribed
  order; each uses the most recent values of already-updated neighbours.
* :class:`AdamsBashforth2` (AB2) -- parallel update like Jacobi, but the
  neighbours' interface variables are extrapolated to the middle of the step
  with the two-step Adams-Bashforth rule ``1.5 s_n - 0.5 s_{n-1}`` (script 04
  and Algorithm 3 of the paper). The first step falls back to Jacobi.

Both schedules can run ``iterations`` inner sweeps per time step. Each sweep
restarts from the prior belief at the beginning of the time step and only the
messages change, so the measurement is never assimilated twice. The first
sweep uses the lagged interface variables (explicit coupling, as in the paper).
Later sweeps use the *trapezoidal* message, i.e. the average of the lagged
message and the message rebuilt from the neighbours' latest end-of-step
estimates, which makes the coupling second-order consistent with the Heun
integrator instead of a backward-Euler treatment.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

Belief = Tuple[np.ndarray, np.ndarray]  # (mean, covariance)
MessagesInto = Callable[[str, Mapping[str, np.ndarray]], Tuple[np.ndarray, Dict[str, float]]]
StepFn = Callable[[str, Belief, np.ndarray], Belief]


class Schedule:
    name = "base"

    def __init__(self, iterations: int = 1):
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.iterations = int(iterations)

    def order(self, names: Sequence[str]) -> Sequence[str]:
        return list(names)

    def reset(self) -> None:
        """Forget any history kept between time steps (called at the start of a run)."""

    def sweep(
        self,
        names: Sequence[str],
        prior: Mapping[str, Belief],
        messages_into: MessagesInto,
        step: StepFn,
    ) -> Tuple[Dict[str, Belief], Dict[str, float]]:
        """Advance all subsystems by one time step.

        ``messages_into(name, means)`` returns the message contribution to the
        input ports of ``name`` computed from the mapping ``means`` (subsystem
        name -> augmented mean), together with the named message values.
        ``step(name, belief, u_msg)`` advances one subsystem from ``belief``.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(iterations={self.iterations})"


class Jacobi(Schedule):
    """Parallel update from lagged interface variables."""

    name = "jacobi"

    def sweep(self, names, prior, messages_into, step):
        prior_means = {n: prior[n][0] for n in names}
        lagged = {}
        logged: Dict[str, float] = {}
        for n in names:
            lagged[n], vals = messages_into(n, prior_means)
            logged.update(vals)
        posterior = {n: step(n, prior[n], lagged[n]) for n in names}
        for _ in range(self.iterations - 1):
            means = {n: posterior[n][0] for n in names}
            u_msgs = {}
            for n in names:
                latest, vals = messages_into(n, means)
                u_msgs[n] = 0.5 * (lagged[n] + latest)
                logged.update({k: 0.5 * (logged[k] + v) for k, v in vals.items()})
            posterior = {n: step(n, prior[n], u_msgs[n]) for n in names}
        return posterior, logged


class GaussSeidel(Schedule):
    """Sequential update; later subsystems see the already-updated neighbours."""

    name = "gauss_seidel"

    def __init__(self, iterations: int = 1, order: Optional[Sequence[str]] = None):
        super().__init__(iterations)
        self._order = list(order) if order is not None else None

    def order(self, names):
        if self._order is None:
            return list(names)
        missing = set(names) - set(self._order)
        extra = set(self._order) - set(names)
        if missing or extra:
            raise ValueError(f"GaussSeidel order mismatch. Missing: {sorted(missing)}, unknown: {sorted(extra)}")
        return list(self._order)

    def sweep(self, names, prior, messages_into, step):
        order = self.order(names)
        prior_means = {n: prior[n][0] for n in names}
        current: Dict[str, Belief] = dict(prior)
        lagged: Dict[str, np.ndarray] = {}
        logged: Dict[str, float] = {}
        # first sweep: sequential, using the most recent neighbour information (paper Algorithm 2)
        for n in order:
            means = {m: current[m][0] for m in names}
            lagged[n], vals = messages_into(n, prior_means)
            u_msg, vals = messages_into(n, means)
            logged.update(vals)
            current[n] = step(n, prior[n], u_msg)
        # refinement sweeps: trapezoidal message between lagged and latest interface variables
        for _ in range(self.iterations - 1):
            for n in order:
                means = {m: current[m][0] for m in names}
                latest, vals = messages_into(n, means)
                logged.update({k: 0.5 * (logged[k] + v) for k, v in vals.items()})
                current[n] = step(n, prior[n], 0.5 * (lagged[n] + latest))
        return current, logged


class AdamsBashforth2(Schedule):
    """Parallel update from AB2-extrapolated interface variables."""

    name = "ab2"

    def __init__(self, iterations: int = 1):
        super().__init__(iterations)
        self._prev: Optional[Dict[str, np.ndarray]] = None

    def reset(self):
        self._prev = None

    def sweep(self, names, prior, messages_into, step):
        prior_means = {n: prior[n][0] for n in names}
        if self._prev is None:
            means = prior_means
        else:
            means = {n: 1.5 * prior_means[n] - 0.5 * self._prev[n] for n in names}
        logged: Dict[str, float] = {}
        u_msgs = {}
        for n in names:
            u_msgs[n], vals = messages_into(n, means)
            logged.update(vals)
        posterior = {n: step(n, prior[n], u_msgs[n]) for n in names}
        for _ in range(self.iterations - 1):
            new_means = {n: posterior[n][0] for n in names}
            for n in names:
                latest, vals = messages_into(n, new_means)
                u_msgs[n] = 0.5 * (u_msgs[n] + latest)
                logged.update({k: 0.5 * (logged[k] + v) for k, v in vals.items()})
            posterior = {n: step(n, prior[n], u_msgs[n]) for n in names}
        self._prev = prior_means
        return posterior, logged


SCHEDULES = {
    "jacobi": Jacobi,
    "parallel": Jacobi,
    "gauss_seidel": GaussSeidel,
    "gauss-seidel": GaussSeidel,
    "gs": GaussSeidel,
    "sequential": GaussSeidel,
    "ab2": AdamsBashforth2,
    "adams_bashforth": AdamsBashforth2,
    "adams-bashforth": AdamsBashforth2,
}


def get_schedule(spec, **kwargs) -> Schedule:
    """Resolve a schedule from a name, a dict (``{"type": "jacobi", "iterations": 2}``), a class or an instance."""
    if isinstance(spec, Schedule):
        return spec
    if isinstance(spec, type) and issubclass(spec, Schedule):
        return spec(**kwargs)
    if isinstance(spec, Mapping):
        spec = dict(spec)
        kind = spec.pop("type", spec.pop("name", "jacobi"))
        spec.update(kwargs)
        return get_schedule(kind, **spec)
    key = str(spec).lower()
    if key not in SCHEDULES:
        raise ValueError(f"Unknown schedule '{spec}'. Available: {sorted(SCHEDULES)}")
    return SCHEDULES[key](**kwargs)
