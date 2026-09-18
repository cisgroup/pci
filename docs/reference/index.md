# API reference

The complete public API, generated from the docstrings. If you are new, start with
[Get started](../getting-started.md) and [Concepts](../concepts.md); this section is the
exhaustive symbol-level reference.

## Where things live

| You want to... | See |
|---|---|
| Solve a problem written in YAML or a dict | [Top level: `pci.solve`](api.md) |
| Assemble subsystems and interfaces, run `estimate` / `simulate` | [System](system.md) |
| Write your own physics and choose its estimator | [Subsystems](subsystems.md) |
| Couple subsystems with a spring-damper or a custom law | [Interfaces](interfaces.md) |
| Choose how messages are exchanged | [Schedules](schedules.md) |
| Pick or register a local estimator | [Filters](filters.md) |
| Pick the time integrator | [Integrators](integrators.md) |
| Define external forces | [Loads](loads.md) |
| Use the built-in mass-spring chain and the paper's systems | [Models](models.md) |
| Read estimates, metrics and plots | [Results](results.md) |
| Equation strings, linear algebra helpers, plotting | [Internals](internals.md) |

## The public surface

Everything importable from the top-level `pci` package:

```python
import pci

pci.solve, pci.build, pci.load_config, pci.Problem          # declarative problems
pci.System                                                  # the graph + the solvers
pci.Subsystem, pci.Unknown                                  # nodes
pci.Interface, pci.spring_damper, pci.custom_interface      # edges
pci.Jacobi, pci.GaussSeidel, pci.AdamsBashforth2            # schedules
pci.KF, pci.EKF, pci.UKF, pci.CKF, pci.register_filter      # local estimators
pci.euler, pci.heun, pci.rk4                                # integrators
pci.RandomLoad, pci.HarmonicLoad, pci.ElCentroLoad, ...     # loads
pci.MassSpringChain, pci.Spring, pci.Truth                  # built-in model
pci.Results                                                 # what estimate/simulate return
```

## Module layout

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
```
