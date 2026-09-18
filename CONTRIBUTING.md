# Contributing to compositional-inference

Thanks for your interest. This guide covers the development setup, the checks
to run before opening a pull request, and the conventions used in the code.

## Setup

```bash
git clone https://github.com/cisgroup/compositional-inference
cd compositional-inference
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # numpy, scipy, matplotlib, pyyaml, pandas, sympy, pytest
```

## Before you open a pull request

```bash
pytest                       # unit tests in tests/
python examples/01_four_dof_jacobi_ukf.py --no-save   # smoke-test the canonical example
```

New features need tests in `tests/`, and every example script must keep a
YAML twin that reproduces it (see `examples/02_solve_from_yaml.py`).

## Conventions

- **Docstrings** follow the NumPy style (`Parameters`, `Returns`, `Examples`
  sections). Every module carries a docstring that says what it does and
  where it fits in the graph (node, edge, schedule, solver).
- **Names**: subsystems are nodes, interfaces are directed edges, messages are
  what edges carry, schedules decide when messages are exchanged. Keep the
  code and the paper using the same words.
- **New filters** subclass `ci.filters.Filter` and register with
  `@register_filter("name")`. **New models** live in `ci/models/` and return a
  `System` from a `decompose`-style method.
- Keep the physics functions (`dynamics`, `measurement`) pure: they receive a
  plain `dict` of parameters so the same function serves the forward model
  and the inverse problem.

## Reporting problems

Open an issue with the YAML (or a minimal script) that reproduces the
behaviour, the `Results.summary()` output, and the `ci` version.
