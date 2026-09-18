"""External loads (excitations) applied to subsystem input ports.

A load is anything with ``sample(t) -> array`` where ``t`` is the array of
step start times. Loads can be combined with ``+`` and scaled with ``*``.
Specify loads in configurations as dictionaries, e.g.
``{"type": "random", "std": 400, "seed": 1}`` or ``{"type": "harmonic",
"amplitude": 100, "frequency": 2.0}``.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence, Union

import numpy as np


class Load:
    def sample(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __add__(self, other):
        return SumLoad([self, as_load(other)])

    __radd__ = __add__

    def __mul__(self, factor: float):
        return ScaledLoad(self, float(factor))

    __rmul__ = __mul__


class ZeroLoad(Load):
    def sample(self, t):
        return np.zeros_like(np.asarray(t, dtype=float))


class ConstantLoad(Load):
    def __init__(self, value: float):
        self.value = float(value)

    def sample(self, t):
        return np.full(np.asarray(t).shape, self.value)


class HarmonicLoad(Load):
    """``amplitude * sin(2 pi frequency t + phase)`` with ``frequency`` in Hz."""

    def __init__(self, amplitude: float, frequency: float, phase: float = 0.0, offset: float = 0.0):
        self.amplitude, self.frequency, self.phase, self.offset = map(float, (amplitude, frequency, phase, offset))

    def sample(self, t):
        t = np.asarray(t, dtype=float)
        return self.offset + self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)


class RandomLoad(Load):
    """Zero-mean (or ``mean``) Gaussian white-noise force with standard deviation ``std``."""

    def __init__(self, std: float, seed: Optional[int] = None, mean: float = 0.0):
        self.std, self.mean, self.seed = float(std), float(mean), seed

    def sample(self, t):
        rng = np.random.default_rng(self.seed)
        return rng.normal(self.mean, self.std, size=np.asarray(t).shape)


class StepLoad(Load):
    """``amplitude`` for ``t_on <= t < t_off`` (``t_off=None`` keeps it on)."""

    def __init__(self, amplitude: float, t_on: float = 0.0, t_off: Optional[float] = None):
        self.amplitude, self.t_on, self.t_off = float(amplitude), float(t_on), t_off

    def sample(self, t):
        t = np.asarray(t, dtype=float)
        on = t >= self.t_on
        if self.t_off is not None:
            on &= t < self.t_off
        return np.where(on, self.amplitude, 0.0)


class ImpulseLoad(StepLoad):
    """Rectangular pulse of ``amplitude`` starting at ``t0`` lasting ``duration`` seconds."""

    def __init__(self, amplitude: float, t0: float = 0.0, duration: float = 0.01):
        super().__init__(amplitude, t0, t0 + duration)


class RecordedLoad(Load):
    """A recorded time history, resampled onto the simulation grid by linear interpolation.

    Pass either ``time`` (array, same length as ``values``) or ``dt`` of the
    record. ``scale`` multiplies the record (e.g. to convert ground
    acceleration to force, ``scale = -mass``).
    """

    def __init__(self, values: Sequence[float], dt: Optional[float] = None, time: Optional[Sequence[float]] = None,
                 scale: float = 1.0, loop: bool = False, stretch: bool = False):
        self.values = np.asarray(values, dtype=float).reshape(-1)
        if time is None:
            if dt is None:
                raise ValueError("RecordedLoad needs either `time` or `dt`.")
            time = np.arange(self.values.size) * float(dt)
        self.time = np.asarray(time, dtype=float)
        self.scale, self.loop, self.stretch = float(scale), loop, stretch

    def sample(self, t):
        t = np.asarray(t, dtype=float)
        if self.stretch:  # map the whole record onto the simulation horizon (as scipy.signal.resample would)
            t = self.time[0] + (t - t[0]) / max(t[-1] - t[0], 1e-300) * (self.time[-1] - self.time[0])
        if self.loop:
            span = self.time[-1] - self.time[0]
            t = self.time[0] + np.mod(t - self.time[0], span)
        return self.scale * np.interp(t, self.time, self.values, left=0.0, right=0.0)


class ElCentroLoad(RecordedLoad):
    """The 1940 El Centro record shipped with the paper repository (``elcentro.mat``).

    ``path`` defaults to the ``PCI_ELCENTRO`` environment variable. The
    ``.mat`` file holds a 2 x n array ``e`` with time in row 0 and ground
    acceleration in row 1. Use ``scale`` to turn ground acceleration into a
    force (the paper's 6-DOF example uses ``scale = 3 * 500``); ``stretch``
    maps the 50 s record onto the simulation horizon.
    """

    def __init__(self, path: Optional[str] = None, scale: float = 1.0, loop: bool = False, stretch: bool = False):
        path = os.path.expanduser(path or os.environ.get("PCI_ELCENTRO") or os.environ.get("CI_ELCENTRO") or "")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                "El Centro record not found. Pass `path=` or set the PCI_ELCENTRO environment variable."
            )
        import scipy.io

        mat = scipy.io.loadmat(path)
        e = np.asarray(mat["e"], dtype=float)
        super().__init__(values=e[1], time=e[0], scale=scale, loop=loop, stretch=stretch)


class ArrayLoad(Load):
    """A precomputed array aligned with the simulation grid (length must match ``len(t)``)."""

    def __init__(self, values: Sequence[float]):
        self.values = np.asarray(values, dtype=float).reshape(-1)

    def sample(self, t):
        t = np.asarray(t)
        if self.values.size < t.size:
            raise ValueError(f"ArrayLoad has {self.values.size} samples but {t.size} steps are requested.")
        return self.values[: t.size].copy()


class SumLoad(Load):
    def __init__(self, loads: Sequence[Load]):
        self.loads = list(loads)

    def sample(self, t):
        return sum(l.sample(t) for l in self.loads)


class ScaledLoad(Load):
    def __init__(self, load: Load, factor: float):
        self.load, self.factor = load, factor

    def sample(self, t):
        return self.factor * self.load.sample(t)


LOAD_TYPES = {
    "zero": ZeroLoad,
    "none": ZeroLoad,
    "constant": ConstantLoad,
    "harmonic": HarmonicLoad,
    "sine": HarmonicLoad,
    "random": RandomLoad,
    "white_noise": RandomLoad,
    "step": StepLoad,
    "impulse": ImpulseLoad,
    "recorded": RecordedLoad,
    "elcentro": ElCentroLoad,
    "el_centro": ElCentroLoad,
    "array": ArrayLoad,
}


def as_load(spec: Union[Load, Mapping, float, Sequence[float], None]) -> Load:
    """Coerce a load specification to a :class:`Load`.

    Accepts a ``Load``, ``None`` (zero), a number (constant), an array
    (sample-aligned values) or a dict ``{"type": ..., **kwargs}``.
    """
    if spec is None:
        return ZeroLoad()
    if isinstance(spec, Load):
        return spec
    if isinstance(spec, Mapping):
        spec = dict(spec)
        kind = str(spec.pop("type", "constant")).lower()
        if kind not in LOAD_TYPES:
            raise ValueError(f"Unknown load type '{kind}'. Available: {sorted(LOAD_TYPES)}")
        return LOAD_TYPES[kind](**spec)
    if np.ndim(spec) == 0:
        return ConstantLoad(float(spec))
    return ArrayLoad(np.asarray(spec, dtype=float))
