"""Declarative problem definition (dict / YAML / JSON).

Two model families are supported.

``model: mass_spring_chain`` (built-in physics)::

    model: mass_spring_chain
    mode: estimate                             # estimate (default) | simulate (forward problem only)
    masses: [500, 500, 500, 500]
    stiffness: [50000, 50000, 50000, 50000]   # spring i connects DOF i-1 (0 = ground) to DOF i
    damping: [300, 300, 300, 300]
    # or an explicit topology:  springs: [{a: ground, b: 1, k: 5.0e4, c: 300}, {a: 1, b: 2, ...}]
    subsystems: [[1, 2], [3, 4]]
    names: [S1, S2]                            # optional
    filters: ukf                               # or [ukf, ckf] or {S1: ukf, S2: ekf}
    integrator: heun                           # euler | heun | rk4; or {S1: rk4, S2: heun} per subsystem
    schedule: {type: jacobi, iterations: 1}    # jacobi | gauss_seidel | ab2
    messages: mean
    unknowns:
      k4: {initial: 30000, std: 50000, process_std: 0}
    sensors: [a1, a4]
    noise_std: 1.0e-3                          # or {a1: 1.0e-3, a4: 5.0e-3}
    loads:
      1: {type: random, std: 400, seed: 1}
      4: {type: harmonic, amplitude: 200, frequency: 1.5}
    time: {dt: 1.0e-3, T: 5.0}
    initial_state: {x1: 0.01, v1: 0.01}
    truth: {seed: 123, initial_state: {x1: 0.01, v1: 0.01}, integrator: heun}
    #   truth.noise_std: sensor noise used to generate the synthetic data (default: noise_std above)
    prior: {state_var: 1.0e-4, process_var: 1.0e-12, r_inflation: 10}

``model: custom`` (equations written as strings, needs ``sympy``)::

    model: custom
    subsystems:
      A:
        states: [q, v]
        inputs: [f]
        parameters: {m: 1.0, c: 0.3}
        unknowns: {k: {initial: 5.0, std: 5.0}}
        equations: {q: v, v: (f - k*q - c*v)/m}          # d/dt of each state
        measurements: {a_A: (f - k*q - c*v)/m}          # sensor name -> expression
        noise_std: {a_A: 1.0e-2}
        filter: ekf
        integrator: heun
        x0: {q: 0.3}
        P0: 1.0e-4
        Q: 1.0e-10
      B: {...}
    interfaces:
      - {type: spring_damper, a: A, a_vars: [q, v], a_port: f, b: B, b_vars: [q, v], b_port: f,
         k: 6.0, c: 0.2, name: F_AB}
      - {type: custom, sender: A, receiver: B, sender_vars: [q, v], receiver_vars: [q, v], target: f,
         law: "k*(s_q - r_q) + c*(s_v - r_v)", constants: {k: 6.0, c: 0.2}, sign: 1, name: F_AB}
    schedule: gauss_seidel
    loads: {A.f: {type: harmonic, amplitude: 2.0, frequency: 0.7}}
    time: {dt: 1.0e-3, T: 20.0}
    truth:                                     # synthetic data from the same coupled model
      parameters: {A.k: 12.0}                  # true values of the unknowns (and any other override)
      x0: {A.q: 0.3}
      schedule: ab2
      seed: 0
    prior: {r_inflation: 1}

Then ``results = pci.solve("problem.yaml")``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .interface import Interface, spring_damper
from .models.mass_spring import MassSpringChain, Spring
from .results import Results
from .subsystem import Subsystem
from .system import System


def load_config(path: str) -> Dict[str, Any]:
    """Read a YAML or JSON configuration file."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as fh:
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ImportError("YAML configs require pyyaml: pip install pci[config]") from exc
            return yaml.safe_load(fh)
        return json.load(fh)


def _as_config(config) -> Dict[str, Any]:
    if isinstance(config, (str, os.PathLike)):
        return load_config(str(config))
    return dict(config)


@dataclass
class Problem:
    """Everything ``pci.build`` instantiates from a configuration."""

    system: System
    loads: Dict[str, Any]  # keyed by input port ("S1.f1" or "f1")
    dt: float
    T: float
    mode: str = "estimate"
    truth: Optional[Dict[str, np.ndarray]] = None
    measurements: Optional[Dict[str, np.ndarray]] = None
    chain: Optional[MassSpringChain] = None
    truth_system: Optional[System] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def solve(self, progress: bool = False) -> Results:
        if self.mode == "simulate":
            return self.system.simulate(loads=self.loads, dt=self.dt, T=self.T, truth=self.truth, progress=progress)
        return self.system.estimate(self.measurements, loads=self.loads, dt=self.dt, T=self.T,
                                    truth=self.truth, progress=progress)


# ----------------------------------------------------------------------------- mass-spring
def build_chain(cfg: Mapping[str, Any]) -> MassSpringChain:
    masses = cfg["masses"]
    if "springs" in cfg:
        springs = [Spring(s.get("a", "ground"), s["b"], s["k"], s.get("c", 0.0), str(s.get("name", ""))) for s in cfg["springs"]]
        return MassSpringChain(masses, springs)
    return MassSpringChain(masses, k=cfg["stiffness"], c=cfg.get("damping"))


def _build_mass_spring(cfg, measurements) -> Problem:
    chain = build_chain(cfg)
    time_cfg = cfg.get("time", {})
    dt, T = float(time_cfg.get("dt", 1e-3)), float(time_cfg.get("T", 1.0))
    mode = str(cfg.get("mode", "estimate")).lower()
    loads = {int(str(k).lstrip("f")): v for k, v in (cfg.get("loads") or {}).items()}
    partition = cfg["subsystems"]
    sensors = list(cfg.get("sensors", []))
    noise_std = cfg.get("noise_std", 1e-3)
    prior = cfg.get("prior", {}) or {}

    system = chain.decompose(
        partition,
        unknowns=cfg.get("unknowns"),
        sensors=sensors,
        noise_std=noise_std,
        filters=cfg.get("filters", "ukf"),
        filter_options=cfg.get("filter_options"),
        integrator=cfg.get("integrator", "heun"),
        schedule=cfg.get("schedule", "jacobi"),
        message_type=cfg.get("messages", "mean"),
        x0=cfg.get("initial_state"),
        state_var=float(prior.get("state_var", 1e-4)),
        process_var=float(prior.get("process_var", 1e-12)),
        r_inflation=float(prior.get("r_inflation", 10.0)),
        names=cfg.get("names"),
        interface_params=cfg.get("interface_params"),
    )

    truth_dict = None
    truth_cfg = cfg.get("truth", {}) or {}
    if measurements is None:
        integ = cfg.get("integrator", "heun")
        truth = chain.simulate(
            loads, dt=dt, T=T,
            x0=truth_cfg.get("initial_state", cfg.get("initial_state") if mode == "simulate" else None),
            integrator=truth_cfg.get("integrator", integ if isinstance(integ, str) else "heun"),
        )
        truth_dict = truth.as_dict()
        truth_dict.update(chain.interface_forces(truth, partition))
        if mode == "estimate":
            data_noise = truth_cfg.get("noise_std", noise_std)  # sensor noise of the synthetic data (default: noise_std)
            measurements = MassSpringChain.measure(truth, sensors, data_noise, seed=truth_cfg.get("seed"))
    port_loads = {f"f{d}": spec for d, spec in loads.items()}
    return Problem(system, port_loads, dt, T, mode, truth_dict, None if measurements is None else dict(measurements),
                   chain=chain, config=dict(cfg))


# ----------------------------------------------------------------------------- custom
def _make_subsystem(name: str, spec: Mapping[str, Any], r_inflation: float, override_params=None,
                    drop_unknowns=False) -> Subsystem:
    from .symbolic import compile_model

    states = list(spec["states"])
    inputs = list(spec.get("inputs", []))
    params = dict(spec.get("parameters", {}))
    unknowns = dict(spec.get("unknowns", {}) or {})
    if drop_unknowns:
        params.update({k: (v["initial"] if isinstance(v, Mapping) else float(v)) for k, v in unknowns.items()})
        unknowns = {}
    if override_params:
        params.update(override_params)
    all_param_names = list(params) + list(unknowns)

    eqs = spec["equations"]
    if isinstance(eqs, Mapping):
        missing = [s for s in states if s not in eqs]
        if missing:
            raise KeyError(f"Subsystem '{name}': no equation for state(s) {missing}.")
        exprs = [eqs[s] for s in states]
    else:
        exprs = list(eqs)
    dynamics = compile_model(exprs, states, inputs, all_param_names)

    meas = spec.get("measurements") or {}
    measured = list(meas) if isinstance(meas, Mapping) else None
    measurement = compile_model(list(meas.values()) if isinstance(meas, Mapping) else list(meas),
                                states, inputs, all_param_names) if meas else None
    noise = spec.get("noise_std", 1e-3)
    R = None
    if measured:
        R = np.diag([(r_inflation * float(noise[m] if isinstance(noise, Mapping) else noise)) ** 2 for m in measured])
    if spec.get("R") is not None:
        R = spec["R"]
    return Subsystem(
        name, states, inputs, dynamics, measurement, parameters=params, unknowns=unknowns,
        filter=spec.get("filter", "ukf"), filter_options=spec.get("filter_options"),
        integrator=spec.get("integrator", "heun"), x0=spec.get("x0"), P0=spec.get("P0", 1e-4),
        Q=spec.get("Q", 1e-12), R=R, measured=measured,
    )


def _make_interfaces(specs) -> list:
    from .symbolic import compile_interface_law

    out = []
    for i, s in enumerate(specs or []):
        kind = str(s.get("type", "spring_damper")).lower()
        if kind == "spring_damper":
            out += spring_damper(s["a"], s["a_vars"], s["a_port"], s["b"], s["b_vars"], s["b_port"],
                                 k=float(s["k"]), c=float(s.get("c", 0.0)), name=s.get("name"))
        elif kind == "custom":
            law = compile_interface_law(s["law"], s["sender_vars"], s["receiver_vars"], s.get("constants"))
            out.append(Interface(s["sender"], s["receiver"], s["sender_vars"], s["receiver_vars"], s["target"],
                                 law, float(s.get("sign", 1.0)), s.get("name", f"interface_{i}")))
        else:
            raise ValueError(f"Unknown interface type '{kind}' (spring_damper | custom).")
    return out


def _split_qualified(key: str, default_sub: Optional[str] = None):
    if "." in key:
        sub, var = key.split(".", 1)
        return sub, var
    return default_sub, key


def _build_custom(cfg, measurements) -> Problem:
    time_cfg = cfg.get("time", {})
    dt, T = float(time_cfg.get("dt", 1e-3)), float(time_cfg.get("T", 1.0))
    mode = str(cfg.get("mode", "estimate")).lower()
    prior = cfg.get("prior", {}) or {}
    r_inflation = float(prior.get("r_inflation", 10.0))
    sub_specs = cfg["subsystems"]
    if not isinstance(sub_specs, Mapping):
        raise ValueError("For model: custom, `subsystems` must be a mapping name -> specification.")

    subsystems = [_make_subsystem(n, s, r_inflation) for n, s in sub_specs.items()]
    system = System(subsystems, _make_interfaces(cfg.get("interfaces")), schedule=cfg.get("schedule", "jacobi"),
                    message_type=cfg.get("messages", "mean"), name=cfg.get("name", "custom"))
    loads = dict(cfg.get("loads") or {})

    truth_dict, truth_system = None, None
    truth_cfg = cfg.get("truth", {}) or {}
    if measurements is None and (mode == "estimate" or truth_cfg):
        # truth = same coupled model with the unknowns replaced by their true values
        overrides: Dict[str, Dict[str, float]] = {n: {} for n in sub_specs}
        for key, val in (truth_cfg.get("parameters") or {}).items():
            sub, var = _split_qualified(key)
            if sub is None:
                owners = [n for n, s in sub_specs.items() if var in (s.get("unknowns") or {}) or var in (s.get("parameters") or {})]
                if len(owners) != 1:
                    raise KeyError(f"truth.parameters '{key}': qualify it as '<subsystem>.{var}'.")
                sub = owners[0]
            overrides[sub][var] = float(val)
        x0_true: Dict[str, Dict[str, float]] = {n: dict(s.get("x0") or {}) for n, s in sub_specs.items()}
        for key, val in (truth_cfg.get("x0") or {}).items():
            sub, var = _split_qualified(key)
            if sub is None:
                raise KeyError(f"truth.x0 '{key}': qualify it as '<subsystem>.{var}'.")
            x0_true[sub][var] = float(val)
        true_subs = []
        for n, s in sub_specs.items():
            s2 = dict(s)
            s2["x0"] = x0_true[n]
            if truth_cfg.get("integrator"):
                s2["integrator"] = truth_cfg["integrator"]
            true_subs.append(_make_subsystem(n, s2, r_inflation, override_params=overrides[n], drop_unknowns=True))
        truth_system = System(true_subs, _make_interfaces(cfg.get("interfaces")),
                              schedule=truth_cfg.get("schedule", "ab2"), name="truth")
        sim = truth_system.simulate(loads=loads, dt=dt, T=T)
        truth_dict = {}
        for n, h in sim.subsystems.items():
            for v in h.variables:
                truth_dict[f"{n}.{v}"] = h[v]
        for n, ov in overrides.items():
            for var, val in ov.items():
                truth_dict[f"{n}.{var}"] = np.asarray(val)
        for nm, arr in sim.messages.items():
            truth_dict[nm] = arr
        if mode == "estimate":
            rng = np.random.default_rng(truth_cfg.get("seed"))
            measurements = {}
            for sub, tsub in zip(subsystems, true_subs):
                if not sub.measured:
                    continue
                h = sim.subsystems[sub.name]
                Z, U = h.mean, h.u
                Y = np.zeros((Z.shape[0], len(sub.measured)))
                Y[0] = tsub.measure(Z[0], U[0], sim.t[0])
                for k in range(U.shape[0]):
                    Y[k + 1] = tsub.measure(Z[k + 1], U[k], sim.t[k])
                noise = sub_specs[sub.name].get("noise_std", 0.0)
                if "noise_std" in truth_cfg:
                    noise = truth_cfg["noise_std"]
                for j, m in enumerate(sub.measured):
                    std = float(noise[m] if isinstance(noise, Mapping) else noise)
                    measurements[m] = Y[:, j] + rng.normal(0.0, std, size=Y.shape[0])
    return Problem(system, loads, dt, T, mode, truth_dict, None if measurements is None else dict(measurements),
                   truth_system=truth_system, config=dict(cfg))


# ----------------------------------------------------------------------------- public API
def build(config, measurements: Optional[Mapping[str, np.ndarray]] = None) -> Problem:
    """Instantiate everything described by ``config`` (dict or path) without solving."""
    cfg = _as_config(config)
    model = str(cfg.get("model", "mass_spring_chain")).lower()
    if model in ("mass_spring_chain", "mass_spring", "chain"):
        return _build_mass_spring(cfg, measurements)
    if model in ("custom", "equations", "symbolic"):
        return _build_custom(cfg, measurements)
    raise ValueError(f"Unknown model '{model}'. Available: mass_spring_chain, custom")


def solve(config, measurements: Optional[Mapping[str, np.ndarray]] = None, progress: bool = False) -> Results:
    """Build and solve the problem described by ``config`` (dict or path).

    If ``measurements`` is not given, synthetic ground truth and noisy sensor
    data are generated from the model (``truth`` block: seed, initial state,
    true parameter values). ``mode: simulate`` runs the coupled forward
    problem instead of the estimation.
    """
    return build(config, measurements).solve(progress=progress)
