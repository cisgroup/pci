"""System: the interaction graph of subsystems and interfaces, plus the solvers.

``System.estimate`` runs the distributed inverse problem (every subsystem
runs its own filter and exchanges messages according to the schedule);
``System.simulate`` runs the coupled forward problem with the same message
passing but without measurements.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .interface import Interface
from .loads import as_load
from .results import Results, SubsystemHistory
from .schedules import Schedule, get_schedule
from .subsystem import Subsystem

MESSAGE_TYPES = ("mean", "mean_variance")


def _flatten(items) -> List[Interface]:
    out: List[Interface] = []
    for it in items:
        if isinstance(it, Interface):
            out.append(it)
        else:
            out.extend(_flatten(it))
    return out


class System:
    """A system of coupled subsystems.

    Parameters
    ----------
    subsystems : sequence of Subsystem
    interfaces : sequence of Interface (nested lists are flattened, so the
        output of :func:`pci.spring_damper` can be passed directly)
    schedule : "jacobi" | "gauss_seidel" | dict | Schedule
    message_type : "mean" (currently the only implemented option)
    """

    def __init__(
        self,
        subsystems: Sequence[Subsystem],
        interfaces: Iterable = (),
        schedule: Union[str, Mapping, Schedule] = "jacobi",
        message_type: str = "mean",
        name: str = "system",
    ):
        self.name = name
        self.subsystems: Dict[str, Subsystem] = {}
        for s in subsystems:
            self.add_subsystem(s)
        self.interfaces: List[Interface] = []
        for itf in _flatten(interfaces):
            self.add_interface(itf)
        self.schedule: Schedule = get_schedule(schedule)
        if message_type not in MESSAGE_TYPES:
            raise ValueError(f"message_type must be one of {MESSAGE_TYPES}")
        if message_type != "mean":
            raise NotImplementedError(
                "Only mean-valued messages are implemented in this version. "
                "Uncertainty-carrying messages ('mean_variance') are the next planned step."
            )
        self.message_type = message_type

    # ------------------------------------------------------------ building
    def add_subsystem(self, sub: Subsystem) -> "System":
        if sub.name in self.subsystems:
            raise ValueError(f"Duplicate subsystem name '{sub.name}'.")
        self.subsystems[sub.name] = sub
        return self

    def add_interface(self, itf: Interface) -> "System":
        for role, nm in (("sender", itf.sender), ("receiver", itf.receiver)):
            if nm not in self.subsystems:
                raise ValueError(f"Interface '{itf.name}': {role} '{nm}' is not a subsystem of this system.")
        self.subsystems[itf.sender].indices(itf.sender_vars)  # validates names
        self.subsystems[itf.receiver].indices(itf.receiver_vars)
        self.subsystems[itf.receiver].port(itf.target)
        self.interfaces.append(itf)
        return self

    @property
    def names(self) -> List[str]:
        return list(self.subsystems)

    def neighbours(self, name: str) -> List[str]:
        return sorted({i.sender for i in self.interfaces if i.receiver == name})

    def describe(self) -> str:
        lines = [f"System '{self.name}': {len(self.subsystems)} subsystems, {len(self.interfaces)} directed interfaces, "
                 f"schedule={self.schedule!r}, messages='{self.message_type}'"]
        for s in self.subsystems.values():
            lines.append(f"  - {s.name}: states={s.states}, unknowns={list(s.unknowns)}, inputs={s.inputs}, "
                         f"measured={s.measured}, filter={s.filter.name}, integrator={s.integrator.__name__}")
        for i in self.interfaces:
            lines.append(f"  - {i.name}: {i.sender}{i.sender_vars} -> {i.receiver}.{i.target} (sign {i.sign:+.0f})")
        return "\n".join(lines)

    # ----------------------------------------------------------- utilities
    def _resolve_port(self, key) -> Tuple[str, str]:
        if isinstance(key, tuple):
            sub, port = key
            return str(sub), str(port)
        key = str(key)
        if "." in key:
            sub, port = key.split(".", 1)
            return sub, port
        owners = [s.name for s in self.subsystems.values() if key in s.inputs]
        if len(owners) == 1:
            return owners[0], key
        if not owners:
            raise KeyError(f"No subsystem has an input port named '{key}'.")
        raise KeyError(f"Port '{key}' exists in several subsystems {owners}; use 'Subsystem.port'.")

    def _sample_loads(self, loads, t_steps: np.ndarray) -> Dict[str, np.ndarray]:
        u_ext = {s.name: np.zeros((t_steps.size, s.nu)) for s in self.subsystems.values()}
        for key, spec in (loads or {}).items():
            sub, port = self._resolve_port(key)
            if sub not in self.subsystems:
                raise KeyError(f"Load key '{key}': unknown subsystem '{sub}'.")
            j = self.subsystems[sub].port(port)
            u_ext[sub][:, j] += as_load(spec).sample(t_steps)
        return u_ext

    def _prepare_interfaces(self):
        prepared = []
        for itf in self.interfaces:
            s_idx = self.subsystems[itf.sender].indices(itf.sender_vars)
            r_idx = self.subsystems[itf.receiver].indices(itf.receiver_vars)
            port = self.subsystems[itf.receiver].port(itf.target)
            prepared.append((itf, s_idx, r_idx, port))
        return prepared

    def _resolve_measurements(self, measurements, n_steps: int) -> Dict[str, Optional[np.ndarray]]:
        """Return per-subsystem arrays of shape (n_steps, ny) holding y_{k+1} at row k."""
        out: Dict[str, Optional[np.ndarray]] = {}
        for s in self.subsystems.values():
            if s.ny == 0 or measurements is None:
                out[s.name] = None
                continue
            if s.name in measurements:
                Y = np.asarray(measurements[s.name], dtype=float)
                if Y.ndim == 1:
                    Y = Y[:, None]
            elif s.measured is not None:
                missing = [m for m in s.measured if m not in measurements]
                if missing:
                    raise KeyError(f"Subsystem '{s.name}': measurements missing for {missing}.")
                Y = np.column_stack([np.asarray(measurements[m], dtype=float) for m in s.measured])
            else:
                raise KeyError(
                    f"Subsystem '{s.name}' measures {s.ny} quantities but no `measured` names were given; "
                    f"provide measurements keyed by the subsystem name."
                )
            if Y.shape[1] != s.ny:
                raise ValueError(f"Subsystem '{s.name}': measurement has {Y.shape[1]} columns, expected {s.ny}.")
            if Y.shape[0] == n_steps + 1:
                Y = Y[1:]
            elif Y.shape[0] < n_steps:
                raise ValueError(f"Subsystem '{s.name}': {Y.shape[0]} measurement rows for {n_steps} steps.")
            out[s.name] = Y[:n_steps]
        return out

    # -------------------------------------------------------------- solvers
    def estimate(
        self,
        measurements: Optional[Mapping[str, np.ndarray]],
        loads: Optional[Mapping] = None,
        dt: float = 1e-3,
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        t0: float = 0.0,
        truth: Optional[Mapping[str, np.ndarray]] = None,
        progress: bool = False,
    ) -> Results:
        """Distributed state/parameter estimation.

        Parameters
        ----------
        measurements : dict
            Either ``{measurement_name: array}`` (names as listed in each
            subsystem's ``measured``) or ``{subsystem_name: 2-D array}``.
            Arrays of length ``n_steps + 1`` are indexed so that row ``k+1`` is
            assimilated when stepping from ``k`` to ``k+1``; arrays of length
            ``n_steps`` are used as-is.
        loads : dict
            ``{"S1.f1": load_spec, ...}`` external loads on input ports.
        dt, T, n_steps : time step, horizon or number of steps.
        truth : optional dict of reference trajectories attached to the result.
        """
        return self._run(measurements, loads, dt, T, n_steps, t0, truth, progress, mode="estimate")

    def simulate(
        self,
        loads: Optional[Mapping] = None,
        dt: float = 1e-3,
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        t0: float = 0.0,
        truth: Optional[Mapping[str, np.ndarray]] = None,
        progress: bool = False,
    ) -> Results:
        """Coupled forward simulation (direct problem) using the message-passing schedule."""
        return self._run(None, loads, dt, T, n_steps, t0, truth, progress, mode="simulate")

    def _run(self, measurements, loads, dt, T, n_steps, t0, truth, progress, mode) -> Results:
        if n_steps is None:
            if T is not None:
                n_steps = int(round(T / dt))
            elif measurements:
                lengths = {np.asarray(v).shape[0] for v in measurements.values()}
                n_steps = min(lengths) - 1
            else:
                raise ValueError("Provide `T` or `n_steps`.")
        n_steps = int(n_steps)
        t = t0 + dt * np.arange(n_steps + 1)
        names = self.names
        subs = self.subsystems

        u_ext = self._sample_loads(loads, t[:-1])
        Y = self._resolve_measurements(measurements if mode == "estimate" else None, n_steps)
        prepared = self._prepare_interfaces()
        Qs = {n: subs[n].Q for n in names}
        Rs = {n: subs[n].R for n in names}

        hist = {
            n: SubsystemHistory(
                name=n,
                variables=list(subs[n].variables),
                inputs=list(subs[n].inputs),
                mean=np.zeros((n_steps + 1, subs[n].nz)),
                var=np.zeros((n_steps + 1, subs[n].nz)),
                u=np.zeros((n_steps, subs[n].nu)),
                n_params=subs[n].n_unknowns,
            )
            for n in names
        }
        msg_hist: Dict[str, np.ndarray] = {itf.name: np.zeros(n_steps) for itf in self.interfaces}

        belief = {n: (subs[n].z0.copy(), subs[n].P0.copy()) for n in names}
        for n in names:
            hist[n].mean[0] = belief[n][0]
            hist[n].var[0] = np.diag(belief[n][1])

        state = {"k": 0}

        def messages_into(name, means):
            u_msg = np.zeros(subs[name].nu)
            logged = {}
            tk = t[state["k"]]
            for itf, s_idx, r_idx, port in prepared:
                if itf.receiver != name:
                    continue
                val = itf.value(means[itf.sender][s_idx], means[name][r_idx], tk)
                u_msg[port] += itf.sign * val
                logged[itf.name] = val
            return u_msg, logged

        def step(name, prior, u_msg):
            k = state["k"]
            sub = subs[name]
            u = u_ext[name][k] + u_msg
            hist[name].u[k] = u
            z, P = prior
            tk = t[k]
            if mode == "simulate":
                return sub.transition(z, u, tk, dt), P
            y = None if Y[name] is None else Y[name][k]
            res = sub.filter.step(
                z, P, u, y,
                f=lambda zz, uu, tt: sub.transition(zz, uu, tt, dt),
                h=sub.measure,
                Q=Qs[name], R=Rs[name], t=tk,
            )
            return sub.project(res.x), res.P

        self.schedule.reset()
        tic = time.perf_counter()
        report_every = max(1, n_steps // 10)
        for k in range(n_steps):
            state["k"] = k
            belief, logged = self.schedule.sweep(names, belief, messages_into, step)
            for nm, val in logged.items():
                msg_hist[nm][k] = val
            for n in names:
                hist[n].mean[k + 1] = belief[n][0]
                hist[n].var[k + 1] = np.diag(belief[n][1])
                m_abs = np.abs(belief[n][0])
                if not np.all(np.isfinite(m_abs)) or np.any(m_abs > 1e12) or np.any(hist[n].var[k + 1] > 1e20):
                    raise RuntimeError(
                        f"Subsystem '{n}' diverged at step {k + 1} (t = {t[k + 1]:.4g} s). Typical causes: the local "
                        f"states are not observable from the chosen sensors (e.g. a free-floating subsystem measured "
                        f"by a single accelerometer), an over-wide prior on an unknown parameter, or a time step too "
                        f"large for the integrator. Check `System.describe()` and the sensor placement."
                    )
            if progress and (k % report_every == 0 or k == n_steps - 1):
                print(f"\r[{self.name}] {mode}: step {k + 1}/{n_steps}", end="", flush=True)
        if progress:
            print()
        runtime = time.perf_counter() - tic

        return Results(
            t=t,
            subsystems=hist,
            messages=msg_hist,
            truth=dict(truth) if truth else None,
            runtime=runtime,
            mode=mode,
            schedule=repr(self.schedule),
            final_covariances={n: belief[n][1].copy() for n in names},
        )
