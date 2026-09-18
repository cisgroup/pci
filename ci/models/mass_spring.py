"""Damped mass-spring systems: monolithic truth model and automatic decomposition.

A :class:`MassSpringChain` is a set of masses connected by spring-damper
elements (to each other or to the ground). It provides

* the monolithic forward model (``matrices``, ``simulate``) used to generate
  synthetic ground truth and measurements, and
* ``decompose(partition, ...)`` which turns the same physics into a
  :class:`ci.System` of subsystems coupled by spring-damper interfaces, with
  selected parameters declared unknown.

Naming conventions (all 1-based):
``m{d}`` mass of DOF ``d``; ``k{i}``/``c{i}`` stiffness/damping of spring ``i``
(or a custom name given in :class:`Spring`); ``x{d}``, ``v{d}``, ``a{d}``
displacement, velocity and acceleration of DOF ``d``; ``f{d}`` the force
input port of DOF ``d``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..integrators import get_integrator
from ..interface import spring_damper
from ..loads import as_load
from ..subsystem import Subsystem, Unknown
from ..system import System

GROUND = "ground"


@dataclass
class Spring:
    """Spring-damper element between DOF ``a`` and DOF ``b`` (``a`` may be ``"ground"``)."""

    a: Union[int, str]
    b: int
    k: float
    c: float = 0.0
    name: str = ""  # parameter suffix: stiffness "k<name>", damping "c<name>"

    @property
    def grounded(self) -> bool:
        return isinstance(self.a, str)

    @property
    def k_name(self) -> str:
        return f"k{self.name}"

    @property
    def c_name(self) -> str:
        return f"c{self.name}"

    def dofs(self) -> List[int]:
        return [self.b] if self.grounded else [int(self.a), int(self.b)]


@dataclass
class Truth:
    """Monolithic simulation output."""

    t: np.ndarray
    x: np.ndarray  # (N+1, n) displacements
    v: np.ndarray  # (N+1, n) velocities
    a: np.ndarray  # (N+1, n) accelerations
    f: np.ndarray  # (N, n) applied forces
    params: Dict[str, float]

    @property
    def n_dof(self) -> int:
        return self.x.shape[1]

    def signal(self, name: str) -> np.ndarray:
        kind, d = name[0], int(name[1:])
        arr = {"x": self.x, "v": self.v, "a": self.a}[kind]
        return arr[:, d - 1]

    def as_dict(self) -> Dict[str, np.ndarray]:
        """All trajectories and parameters keyed by name, for ``Results.truth``."""
        out: Dict[str, np.ndarray] = {}
        for d in range(1, self.n_dof + 1):
            out[f"x{d}"] = self.x[:, d - 1]
            out[f"v{d}"] = self.v[:, d - 1]
            out[f"a{d}"] = self.a[:, d - 1]
        for k, val in self.params.items():
            out[k] = np.asarray(float(val))
        return out


class MassSpringChain:
    """Damped mass-spring system with an arbitrary spring topology.

    Parameters
    ----------
    masses : sequence of floats, one per DOF (DOFs are numbered 1..n).
    springs : sequence of :class:`Spring`. If omitted, a serial chain is built
        from ``k`` and ``c``: spring ``i`` connects DOF ``i-1`` (DOF 0 = ground)
        to DOF ``i``.
    """

    def __init__(self, masses: Sequence[float], springs: Optional[Sequence[Spring]] = None, *,
                 k: Optional[Sequence[float]] = None, c: Optional[Sequence[float]] = None):
        self.masses = [float(m) for m in masses]
        n = self.n_dof
        if springs is None:
            if k is None:
                raise ValueError("Provide either `springs` or the chain stiffness list `k`.")
            k = list(k)
            c = list(c) if c is not None else [0.0] * len(k)
            if len(k) != n or len(c) != n:
                raise ValueError(f"Chain needs {n} stiffness and damping values (spring i connects DOF i-1 to i).")
            springs = [Spring(GROUND if i == 1 else i - 1, i, k[i - 1], c[i - 1]) for i in range(1, n + 1)]
        self.springs: List[Spring] = []
        for i, s in enumerate(springs, start=1):
            s = Spring(s.a if isinstance(s.a, str) else int(s.a), int(s.b), float(s.k), float(s.c), s.name or str(i))
            if s.b < 1 or s.b > n or (not s.grounded and (s.a < 1 or s.a > n)):
                raise ValueError(f"Spring {s} references a DOF outside 1..{n}.")
            self.springs.append(s)
        names = [s.k_name for s in self.springs]
        if len(set(names)) != len(names):
            raise ValueError("Spring names must be unique.")

    @classmethod
    def uniform(cls, n_dof: int, mass: float, k: float, c: float = 0.0) -> "MassSpringChain":
        """Serial chain of ``n_dof`` identical masses and spring-dampers."""
        return cls([mass] * n_dof, k=[k] * n_dof, c=[c] * n_dof)

    # ---------------------------------------------------------------- basics
    @property
    def n_dof(self) -> int:
        return len(self.masses)

    @property
    def parameters(self) -> Dict[str, float]:
        p = {f"m{d}": m for d, m in enumerate(self.masses, start=1)}
        for s in self.springs:
            p[s.k_name] = s.k
            p[s.c_name] = s.c
        return p

    def with_parameters(self, **changes) -> "MassSpringChain":
        """Copy of the chain with some parameters changed (e.g. to define an initial-guess model)."""
        p = self.parameters
        unknown = set(changes) - set(p)
        if unknown:
            raise KeyError(f"Unknown parameters {sorted(unknown)}. Known: {sorted(p)}")
        p.update(changes)
        masses = [p[f"m{d}"] for d in range(1, self.n_dof + 1)]
        springs = [Spring(s.a, s.b, p[s.k_name], p[s.c_name], s.name) for s in self.springs]
        return MassSpringChain(masses, springs)

    def matrices(self, params: Optional[Mapping[str, float]] = None):
        """Mass, stiffness and damping matrices ``(M, K, C)``."""
        p = dict(self.parameters)
        if params:
            p.update(params)
        n = self.n_dof
        M = np.diag([p[f"m{d}"] for d in range(1, n + 1)])
        K = np.zeros((n, n))
        C = np.zeros((n, n))
        for s in self.springs:
            k, c = p[s.k_name], p[s.c_name]
            if s.grounded:
                j = s.b - 1
                K[j, j] += k
                C[j, j] += c
            else:
                i, j = s.a - 1, s.b - 1
                K[i, i] += k; K[j, j] += k; K[i, j] -= k; K[j, i] -= k
                C[i, i] += c; C[j, j] += c; C[i, j] -= c; C[j, i] -= c
        return M, K, C

    def acceleration(self, x: np.ndarray, v: np.ndarray, f: np.ndarray, params=None) -> np.ndarray:
        M, K, C = self.matrices(params)
        return np.linalg.solve(M, f - C @ v - K @ x)

    # ------------------------------------------------------------- forward
    def _sample_forces(self, loads, t_steps: np.ndarray) -> np.ndarray:
        F = np.zeros((t_steps.size, self.n_dof))
        for key, spec in (loads or {}).items():
            d = int(str(key).lstrip("f"))
            if d < 1 or d > self.n_dof:
                raise KeyError(f"Load on DOF {d}: outside 1..{self.n_dof}.")
            F[:, d - 1] += as_load(spec).sample(t_steps)
        return F

    def simulate(self, loads: Optional[Mapping] = None, dt: float = 1e-3, T: float = 1.0, *,
                 x0: Optional[Mapping[str, float]] = None, integrator: str = "heun",
                 params: Optional[Mapping[str, float]] = None, t0: float = 0.0) -> Truth:
        """Monolithic forward simulation (the ground truth for synthetic experiments).

        ``loads`` maps DOF numbers (or ``"f3"``) to load specifications, see :mod:`ci.loads`.
        ``x0`` maps ``x{d}``/``v{d}`` names to initial values (default: at rest).
        """
        n = self.n_dof
        N = int(round(T / dt))
        t = t0 + dt * np.arange(N + 1)
        F = self._sample_forces(loads, t[:-1])
        M, K, C = self.matrices(params)
        Minv = np.linalg.inv(M)
        p = dict(self.parameters)
        if params:
            p.update(params)
        step = get_integrator(integrator)

        z = np.zeros(2 * n)
        for name, val in (x0 or {}).items():
            kind, d = name[0], int(name[1:])
            z[(0 if kind == "x" else n) + d - 1] = float(val)

        def accel(q, v, f):
            return Minv @ (f - C @ v - K @ q)

        X = np.zeros((N + 1, n)); V = np.zeros((N + 1, n)); A = np.zeros((N + 1, n))
        X[0], V[0] = z[:n], z[n:]
        A[0] = accel(X[0], V[0], F[0])
        for k in range(N):
            f = F[k]
            rhs = lambda zz, tt: np.concatenate([zz[n:], accel(zz[:n], zz[n:], f)])
            z = step(rhs, z, t[k], dt)
            X[k + 1], V[k + 1] = z[:n], z[n:]
            A[k + 1] = accel(X[k + 1], V[k + 1], f)
        return Truth(t=t, x=X, v=V, a=A, f=F, params=p)

    @staticmethod
    def measure(truth: Truth, sensors: Sequence[str], noise_std: Union[float, Mapping[str, float]] = 0.0,
                seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Noisy synthetic measurements ``{"a1": array, ...}`` from a truth simulation."""
        rng = np.random.default_rng(seed)
        out = {}
        for s in sensors:
            std = float(noise_std[s] if isinstance(noise_std, Mapping) else noise_std)
            clean = truth.signal(s)
            out[s] = clean + rng.normal(0.0, std, size=clean.shape)
        return out

    # -------------------------------------------------------- decomposition
    def _crossing_springs(self, partition: Sequence[Sequence[int]]):
        group_of = {}
        for gi, g in enumerate(partition):
            for d in g:
                if d in group_of:
                    raise ValueError(f"DOF {d} appears in more than one subsystem.")
                group_of[int(d)] = gi
        missing = set(range(1, self.n_dof + 1)) - set(group_of)
        if missing:
            raise ValueError(f"DOFs {sorted(missing)} are not assigned to any subsystem.")
        crossing = []
        for s in self.springs:
            if not s.grounded and group_of[s.a] != group_of[s.b]:
                crossing.append(s)
        return group_of, crossing

    def interface_forces(self, truth: Truth, partition: Sequence[Sequence[int]]) -> Dict[str, np.ndarray]:
        """True interface forces ``F = k (x_a - x_b) + c (v_a - v_b)`` for every crossing spring."""
        _, crossing = self._crossing_springs(partition)
        out = {}
        for s in crossing:
            i, j = s.a - 1, s.b - 1
            F = s.k * (truth.x[:, i] - truth.x[:, j]) + s.c * (truth.v[:, i] - truth.v[:, j])
            out[f"F_{s.k_name}"] = F[:-1]
        return out

    def decompose(
        self,
        partition: Sequence[Sequence[int]],
        *,
        unknowns: Optional[Mapping[str, Union[Unknown, Mapping, float]]] = None,
        sensors: Sequence[str] = (),
        noise_std: Union[float, Mapping[str, float]] = 1e-3,
        filters: Union[str, Sequence[str], Mapping[str, str]] = "ukf",
        filter_options: Optional[Mapping] = None,
        integrator: Union[str, Sequence[str], Mapping[str, str]] = "heun",
        schedule: Union[str, Mapping] = "jacobi",
        message_type: str = "mean",
        x0: Optional[Mapping[str, float]] = None,
        state_var: float = 1e-4,
        process_var: float = 1e-12,
        R: Optional[Mapping[str, np.ndarray]] = None,
        names: Optional[Sequence[str]] = None,
        interface_params: Optional[Mapping[str, float]] = None,
        positive_floor: float = 1.0,
        r_inflation: float = 10.0,
    ) -> System:
        """Build a :class:`ci.System` from a partition of the DOFs.

        Parameters
        ----------
        partition : e.g. ``[[1, 2], [3, 4]]`` (DOF numbers, 1-based).
        unknowns : ``{"k4": {"initial": 30000, "std": 50000}, "m3": 400, ...}``.
            Each unknown must be internal to one subsystem (a mass, or a spring
            whose two ends lie in the same subsystem or on the ground).
        sensors : e.g. ``["a1", "a4"]`` (``a`` acceleration, ``x`` displacement, ``v`` velocity).
        noise_std : scalar or per-sensor dict; the sensor noise standard deviation.
        r_inflation : factor multiplying ``noise_std`` when building the measurement
            covariance ``R = (r_inflation * noise_std)**2``. With mean-only messages
            the incoming interface force carries an error the local filter does not
            know about; sensors at interface DOFs see that error directly and, with
            ``R`` equal to the true sensor noise, the coupled filters over-trust them
            and the message loop can diverge within a few steps. The paper scripts
            inflate ``R`` by 10-30x; the default of 10 does the same. Set ``1.0`` to
            use the raw sensor noise (uncertainty-carrying messages will remove the
            need for this heuristic).
        filters : one name for all subsystems, a list per subsystem, or a dict by subsystem name.
        integrator : ``"euler"``, ``"heun"`` or ``"rk4"`` (or a custom step function); one value
            for all subsystems, a list per subsystem, or a dict by subsystem name.
        x0 : initial state guesses by name, default zeros.
        state_var, process_var : diagonal prior/process variance for the dynamic states.
        interface_params : override the interface law constants, e.g. ``{"k3": 4.8e4}``,
            when the coupling used for message passing differs from the truth.
        positive_floor : lower clamp applied to unknown masses/stiffnesses/dampings
            inside the dynamics (``k_eff = max(k, positive_floor)``), unless an
            :class:`Unknown` sets its own ``lower``. Keeps sigma points with
            negative stiffness from destabilising the local model; the posterior
            mean is projected back onto the bound after each update.
        """
        partition = [[int(d) for d in g] for g in partition]
        group_of, crossing = self._crossing_springs(partition)
        n_sub = len(partition)
        names = list(names) if names is not None else [f"S{i + 1}" for i in range(n_sub)]
        if len(names) != n_sub:
            raise ValueError("`names` must have one entry per subsystem.")

        unknowns = {k: Unknown.coerce(v) for k, v in (unknowns or {}).items()}
        for spec in unknowns.values():
            if spec.lower is None:  # masses, stiffnesses and dampings are positive: protect the physics
                spec.lower = positive_floor
        all_params = self.parameters
        bad = set(unknowns) - set(all_params)
        if bad:
            raise KeyError(f"Unknown parameter names {sorted(bad)}. Known: {sorted(all_params)}")
        crossing_names = {s.k_name for s in crossing} | {s.c_name for s in crossing}
        bad = set(unknowns) & crossing_names
        if bad:
            raise NotImplementedError(
                f"{sorted(bad)} belong to interface springs. Unknown interface parameters (learned interface "
                f"laws) are not supported yet; move the partition boundary or fix these parameters."
            )

        def per_subsystem(spec, default, what):
            if isinstance(spec, (str, bytes)) or callable(spec):
                return {n: spec for n in names}
            if isinstance(spec, Mapping):
                return {n: spec.get(n, default) for n in names}
            spec = list(spec)
            if len(spec) != n_sub:
                raise ValueError(f"`{what}` list has {len(spec)} entries for {n_sub} subsystems.")
            return dict(zip(names, spec))

        filt = per_subsystem(filters, "ukf", "filters")
        integ = per_subsystem(integrator, "heun", "integrator")

        # which sensors / unknowns belong to which subsystem
        sensors = list(sensors)
        sensor_group = {}
        for s in sensors:
            d = int(s[1:])
            if s[0] not in "axv" or d not in group_of:
                raise ValueError(f"Sensor '{s}' must be a{{d}}, x{{d}} or v{{d}} with d in 1..{self.n_dof}.")
            sensor_group.setdefault(group_of[d], []).append(s)

        param_group = {}
        for s in self.springs:
            if s in crossing:
                continue
            g = group_of[s.b]
            param_group[s.k_name] = g
            param_group[s.c_name] = g
        for d in range(1, self.n_dof + 1):
            param_group[f"m{d}"] = group_of[d]

        subsystems = []
        for gi, g in enumerate(partition):
            pos = {d: i for i, d in enumerate(g)}
            ng = len(g)
            internal = [s for s in self.springs if s not in crossing and all(d in pos for d in s.dofs())]
            local_params = {f"m{d}": all_params[f"m{d}"] for d in g}
            for s in internal:
                local_params[s.k_name] = all_params[s.k_name]
                local_params[s.c_name] = all_params[s.c_name]
            local_unknowns = {k: v for k, v in unknowns.items() if param_group[k] == gi}
            for k in local_unknowns:
                local_params.pop(k, None)

            m_names = [f"m{d}" for d in g]
            spring_specs = [(s.grounded, None if s.grounded else pos[s.a], pos[s.b], s.k_name, s.c_name) for s in internal]

            def make_dynamics(ng=ng, m_names=m_names, spring_specs=spring_specs):
                def accel(q, v, u, p):
                    a = np.array(u, dtype=float)
                    for grounded, ia, ib, kn, cn in spring_specs:
                        if grounded:
                            a[ib] -= p[kn] * q[ib] + p[cn] * v[ib]
                        else:
                            F = p[kn] * (q[ia] - q[ib]) + p[cn] * (v[ia] - v[ib])
                            a[ia] -= F
                            a[ib] += F
                    for i, mn in enumerate(m_names):
                        a[i] /= p[mn]
                    return a

                def dynamics(x, u, p, t):
                    q, v = x[:ng], x[ng:]
                    return np.concatenate([v, accel(q, v, u, p)])

                return dynamics, accel

            dynamics, accel = make_dynamics()
            local_sensors = sensor_group.get(gi, [])

            def make_measurement(ng=ng, pos=pos, sensors=tuple(local_sensors), accel=accel):
                specs = [(s[0], pos[int(s[1:])]) for s in sensors]
                need_a = any(k == "a" for k, _ in specs)

                def measurement(x, u, p, t):
                    q, v = x[:ng], x[ng:]
                    a = accel(q, v, u, p) if need_a else None
                    return np.array([{"x": q, "v": v, "a": a}[k][i] for k, i in specs])

                return measurement

            measurement = make_measurement() if local_sensors else None
            if R is not None and names[gi] in R:
                R_local = R[names[gi]]
            else:
                R_local = [(r_inflation * float(noise_std[s] if isinstance(noise_std, Mapping) else noise_std)) ** 2
                           for s in local_sensors]
                R_local = np.diag(R_local) if R_local else None

            states = [f"x{d}" for d in g] + [f"v{d}" for d in g]
            x0_local = {k: v for k, v in (x0 or {}).items() if k in states}
            subsystems.append(
                Subsystem(
                    names[gi], states, [f"f{d}" for d in g], dynamics, measurement,
                    parameters=local_params, unknowns=local_unknowns,
                    filter=filt[names[gi]], filter_options=filter_options, integrator=integ[names[gi]],
                    x0=x0_local, P0=state_var, Q=process_var, R=R_local,
                    measured=local_sensors or None,
                )
            )

        interfaces = []
        ip = dict(interface_params or {})
        for s in crossing:
            ga, gb = group_of[s.a], group_of[s.b]
            interfaces += spring_damper(
                names[ga], [f"x{s.a}", f"v{s.a}"], f"f{s.a}",
                names[gb], [f"x{s.b}", f"v{s.b}"], f"f{s.b}",
                k=ip.get(s.k_name, s.k), c=ip.get(s.c_name, s.c), name=f"F_{s.k_name}",
            )
        return System(subsystems, interfaces, schedule=schedule, message_type=message_type, name="mass_spring")
