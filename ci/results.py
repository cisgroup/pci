"""Result containers with convenience accessors, error metrics and plots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass
class SubsystemHistory:
    name: str
    variables: List[str]
    inputs: List[str]
    mean: np.ndarray  # (N+1, nz)
    var: np.ndarray  # (N+1, nz) diagonal of the covariance
    u: np.ndarray  # (N, nu) total input actually applied (loads + messages)
    n_params: int = 0  # trailing variables that are unknown parameters

    def __getitem__(self, var: str) -> np.ndarray:
        return self.mean[:, self.variables.index(var)]

    def std(self, var: str) -> np.ndarray:
        return np.sqrt(np.maximum(self.var[:, self.variables.index(var)], 0.0))


@dataclass
class Results:
    t: np.ndarray
    subsystems: Dict[str, SubsystemHistory]
    messages: Dict[str, np.ndarray]
    truth: Optional[Dict[str, np.ndarray]] = None
    runtime: float = 0.0
    mode: str = "estimate"
    schedule: str = ""
    final_covariances: Dict[str, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------ lookups
    def _locate(self, name: str):
        if "." in name:
            sub, var = name.split(".", 1)
            if sub not in self.subsystems:
                raise KeyError(f"Unknown subsystem '{sub}'.")
            return self.subsystems[sub], var
        owners = [h for h in self.subsystems.values() if name in h.variables]
        if len(owners) == 1:
            return owners[0], name
        if not owners:
            raise KeyError(f"No subsystem has a variable named '{name}'. Known: {self.variables}")
        raise KeyError(f"Variable '{name}' exists in several subsystems; use 'Subsystem.variable'.")

    def __getitem__(self, name: str) -> np.ndarray:
        return self.state(name)

    def state(self, name: str) -> np.ndarray:
        """Posterior-mean trajectory of a state or parameter, by name (``"x2"`` or ``"S1.x2"``)."""
        h, var = self._locate(name)
        return h[var]

    parameter = state

    def std(self, name: str) -> np.ndarray:
        h, var = self._locate(name)
        return h.std(var)

    def final(self, name: str) -> float:
        return float(self.state(name)[-1])

    @property
    def variables(self) -> List[str]:
        out: List[str] = []
        for h in self.subsystems.values():
            out.extend(h.variables)
        return out

    def qualified_variables(self) -> List[str]:
        return [f"{h.name}.{v}" for h in self.subsystems.values() for v in h.variables]

    def estimates(self) -> Dict[str, np.ndarray]:
        """All trajectories keyed by variable name (qualified when a name is shared)."""
        counts: Dict[str, int] = {}
        for v in self.variables:
            counts[v] = counts.get(v, 0) + 1
        out = {}
        for h in self.subsystems.values():
            for v in h.variables:
                key = v if counts[v] == 1 else f"{h.name}.{v}"
                out[key] = h[v]
        return out

    def input(self, subsystem: str, port: Optional[str] = None) -> np.ndarray:
        h = self.subsystems[subsystem]
        if port is None:
            return h.u
        return h.u[:, h.inputs.index(port)]

    # ------------------------------------------------------------ metrics
    def _truth_for(self, name: str, truth: Optional[Mapping[str, np.ndarray]]):
        truth = truth if truth is not None else self.truth
        if truth is None:
            return None
        if name in truth:
            return truth[name]
        if name in self.messages:
            return None
        _, var = self._locate(name)
        return truth.get(var)

    def message(self, name: str) -> np.ndarray:
        """Mean message trajectory (length N) of the interface ``name``."""
        return self.messages[name]

    def series(self, name: str) -> np.ndarray:
        """State/parameter trajectory, or message trajectory when ``name`` is an interface."""
        if name in self.messages and name not in self.variables:
            return self.messages[name]
        return self.state(name)

    def errors(self, name: str, truth=None) -> np.ndarray:
        ref = self._truth_for(name, truth)
        if ref is None:
            raise KeyError(f"No truth available for '{name}'.")
        est = self.series(name)
        ref = np.asarray(ref, dtype=float)
        if ref.ndim == 0 or ref.size == 1:
            ref = np.full_like(est, float(np.ravel(ref)[0]))
        n = min(est.size, ref.size)
        return est[:n] - ref[:n]

    def rmse(self, names: Optional[Sequence[str]] = None, truth=None, start: float = 0.0) -> Dict[str, float]:
        """RMSE against the truth, optionally starting from time ``start`` (seconds)."""
        names = list(names) if names is not None else [v for v in self.estimates() if self._truth_for(v, truth) is not None]
        mask = self.t >= start
        out = {}
        for n in names:
            e = self.errors(n, truth)
            m = mask[: e.size]
            out[n] = float(np.sqrt(np.mean(e[m] ** 2)))
        return out

    def nrmse(self, names: Optional[Sequence[str]] = None, truth=None, start: float = 0.0) -> Dict[str, float]:
        """RMSE normalised by the truth range (or by the true value for constant parameters)."""
        out = {}
        for n, r in self.rmse(names, truth, start).items():
            ref = np.asarray(self._truth_for(n, truth), dtype=float).ravel()
            scale = np.ptp(ref) if ref.size > 1 and np.ptp(ref) > 0 else abs(float(ref[0]))
            out[n] = float(r / scale) if scale > 0 else float("nan")
        return out

    def summary(self, truth=None, start: float = 0.0) -> str:
        lines = [f"mode={self.mode}  schedule={self.schedule}  steps={self.t.size - 1}  runtime={self.runtime:.2f}s"]
        for h in self.subsystems.values():
            lines.append(f"[{h.name}]")
            for v in h.variables:
                q = f"{h.name}.{v}"
                s = f"  {v:>10s}: final={h[v][-1]: .6g}  std={h.std(v)[-1]: .3g}"
                ref = self._truth_for(q, truth)
                if ref is not None:
                    r = self.rmse([q], truth, start)[q]
                    refv = np.asarray(ref, float).ravel()
                    if refv.size == 1:
                        s += f"  true={refv[0]: .6g}"
                    s += f"  rmse={r: .4g}"
                lines.append(s)
        if self.messages:
            lines.append("[messages] " + ", ".join(self.messages))
        return "\n".join(lines)

    # ------------------------------------------------------------- export
    def to_dataframe(self):
        """Long-format table with one column per (qualified) variable, its std and the truth when known."""
        import pandas as pd

        data = {"t": self.t}
        for h in self.subsystems.values():
            for v in h.variables:
                data[f"{h.name}.{v}"] = h[v]
                data[f"{h.name}.{v}.std"] = h.std(v)
        for nm, arr in self.messages.items():
            data[f"msg:{nm}"] = np.concatenate([arr, [np.nan]])
        if self.truth:
            for nm, arr in self.truth.items():
                arr = np.asarray(arr, float).ravel()
                if arr.size == self.t.size:
                    data[f"true.{nm}"] = arr
        return pd.DataFrame(data)

    # -------------------------------------------------------------- plots
    def plot_states(self, names=None, truth=None, bands=True, **kw):
        from .plotting import plot_trajectories

        if names is None:
            names = [f"{h.name}.{v}" for h in self.subsystems.values() for v in h.variables]
        return plot_trajectories(self, list(names), truth=truth, bands=bands, **kw)

    def plot_parameters(self, names=None, truth=None, bands=True, **kw):
        from .plotting import plot_trajectories

        if names is None:
            names = []
            for h in self.subsystems.values():
                nx = h.mean.shape[1] - self._n_params(h)
                names += [f"{h.name}.{v}" for v in h.variables[nx:]]
        if not names:
            raise ValueError("No unknown parameters in this result.")
        return plot_trajectories(self, list(names), truth=truth, bands=bands, **kw)

    def plot_messages(self, **kw):
        from .plotting import plot_messages

        return plot_messages(self, **kw)

    @staticmethod
    def _n_params(h: SubsystemHistory) -> int:
        return int(h.n_params)

    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        """Trajectories of all unknown parameters, keyed by name."""
        out = {}
        for h in self.subsystems.values():
            for v in h.variables[len(h.variables) - h.n_params:]:
                out[v] = h[v]
        return out
