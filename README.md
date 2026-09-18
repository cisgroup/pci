# pci: Probabilistic Compositional Inference

[![CI](https://github.com/cisgroup/pci/actions/workflows/ci.yml/badge.svg)](https://github.com/cisgroup/pci/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-mkdocs--material-303f9f)](https://cisgroup.github.io/pci/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/cisgroup/pci/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2605.27544-b31b1b.svg)](https://arxiv.org/abs/2605.27544)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Distributed state and parameter estimation for coupled engineered systems, one
subsystem at a time.**

<p align="center">
  <a href="https://cisgroup.github.io/pci/">
    <img src="docs/images/hero.png" width="760" alt="Two subsystems identify an unknown stiffness while exchanging the interface force" />
  </a>
</p>

<p align="center"><em>The paper's 4-DOF testbed: subsystem S2 identifies the stiffness k4 from one accelerometer while S1 and S2 exchange the interface force as a message.</em></p>

`pci` (`import pci`) implements *probabilistic compositional inference*. A complex
system is decomposed into **subsystems**. Each keeps its own physics, its own unknown
parameters and its own estimator (a Kalman-type filter). Subsystems talk to their
neighbours through **interfaces** that carry **messages** (for now: mean values of the
interface forces), and a **schedule** decides how messages are exchanged (Jacobi in
parallel, Gauss-Seidel in sequence, AB2 extrapolated). No global augmented state or
covariance is ever assembled.

The library implements the framework of

> Ghorbani, E. and Hackl, J. (2026). *Subsystem Structure as an Inferential Resource
> for Coupled Engineered Systems.* [arXiv:2605.27544](https://arxiv.org/abs/2605.27544).

## Install

The import name is `pci`. The distribution name is `pci-inference` (the name `pci` on
PyPI is taken by an unrelated package).

```bash
pip install "pci-inference[all]"                                             # from PyPI (once released)
pip install "pci-inference[all] @ git+https://github.com/cisgroup/pci.git"   # latest main
```

Extras: `plot` (matplotlib), `config` (pyyaml), `data` (pandas), `symbolic` (sympy),
`all` (everything), `docs` (the documentation toolchain), `dev` (all of it plus pytest,
ruff, jupytext, nbmake).

## Quick start

```python
import pci

# 1) a physical system and synthetic data (in practice: your measurements)
chain = pci.MassSpringChain.uniform(4, mass=500.0, k=50_000.0, c=300.0)
loads = {d: {"type": "random", "std": 400.0, "seed": d} for d in range(1, 5)}
truth = chain.simulate(loads, dt=1e-3, T=5.0)
data = chain.measure(truth, ["a1", "a4"], noise_std=1e-3, seed=123)

# 2) subsystems, interfaces, local estimators, schedule
system = chain.decompose([[1, 2], [3, 4]], unknowns={"k4": {"initial": 30_000.0, "std": 50_000.0}},
                         sensors=["a1", "a4"], filters="ukf", schedule="jacobi", r_inflation=100.0)

# 3) distributed estimation
results = system.estimate(data, loads={f"f{d}": v for d, v in loads.items()}, dt=1e-3, T=5.0,
                          truth=truth.as_dict())
print(results.summary(start=1.0))
results.plot_parameters(); results.plot_messages()
```

Or the same problem as YAML and `pci.solve("problem.yaml")`. Your own equations are
subsystems too: any `f(x, u, p, t) -> dx/dt` with a measurement `h(x, u, p, t)`, or
equation strings in YAML.

> New to `pci`? Work through the runnable
> **[tutorials](https://cisgroup.github.io/pci/tutorials/)**: the 4-DOF testbed →
> problems in YAML → the forward problem → custom subsystems → six DOF with three
> filters → the paper's 6-DOF system → Van der Pol meets Bouc-Wen.

## Documentation

The full documentation is at **<https://cisgroup.github.io/pci/>** (or `mkdocs serve`
after `pip install -e ".[dev]"`):

- [Get started](https://cisgroup.github.io/pci/getting-started/): install, first run, the two ways of defining a problem.
- [Tutorials](https://cisgroup.github.io/pci/tutorials/): seven notebooks in [`examples/`](examples/), each with a YAML twin.
- [Case studies](https://cisgroup.github.io/pci/case-studies/): chains of 9 to 40 DOF in [`case_studies/`](case_studies/), a different filter, sensor set and task per subsystem.
- [Concepts](https://cisgroup.github.io/pci/concepts/): nodes, edges, messages, schedules; why mean-only messages need an inflated `R`; the roadmap.
- [Configuration](https://cisgroup.github.io/pci/configuration/): every YAML key.
- [API reference](https://cisgroup.github.io/pci/reference/): generated from the docstrings.

## Layout

```
pci/
  subsystem.py    Subsystem, Unknown                (node of the graph)
  interface.py    Interface, spring_damper           (edge / interface law)
  schedules.py    Jacobi, GaussSeidel, AB2           (message passing)
  system.py       System.estimate / System.simulate  (the solver loop)
  filters/        kf, ekf, ukf, ckf + registry
  integrators.py  euler, heun, rk4
  loads.py        load library
  models/         MassSpringChain, Spring, paper case-study definitions
  config.py       pci.solve(yaml | dict)
  results.py      Results, metrics, plots
examples/         tutorials: jupytext-paired notebooks (.py + .ipynb) and their YAML twins
case_studies/     9-, 16-, 20- and 40-DOF case studies, same format
docs/             MkDocs Material site (tutorials and case studies are symlinked in)
tests/            pytest suite
```

## Roadmap

1. Uncertainty-carrying messages (`mean_variance`): interface-force variance injected
   as process noise in the receiver, incremental update rule.
2. Learned interface laws with SINDy as drop-in `Interface.law`.
3. More local estimators: particle filter, PINN / neural surrogates, WLS/WNLS.
4. Optimisation-based (batch) estimators and further built-in models (Kuramoto power
   networks, turbine modules) from the paper.

## Contributing and citing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and the quality gate,
[RELEASING.md](RELEASING.md) for how a version is cut, and
[CITATION.cff](CITATION.cff) (or the [Citing pci](https://cisgroup.github.io/pci/citation/)
page) for how to cite the paper and the software. MIT-licensed; developed at Princeton
University, Complex Infrastructure Systems Group.
