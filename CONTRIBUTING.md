# Contributing to pci

Thanks for your interest. This guide covers the development setup, the quality gate
that CI enforces, and the conventions used in the code and the notebooks.

## Setup

```bash
git clone https://github.com/cisgroup/pci
cd pci
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # all extras + pytest, ruff, jupytext, nbmake, mkdocs
pre-commit install           # ruff, whitespace and notebook-sync hooks on commit
```

## The quality gate

CI (`.github/workflows/ci.yml`) runs these on every push and pull request; all must
pass. Run them locally before opening a PR:

```bash
ruff check .                                        # lint (E, F, W, B, I)
pytest -q                                           # unit tests in tests/
PCI_T=5 pytest --nbmake examples/*.ipynb            # tutorials execute from scratch
PCI_T=5 pytest --nbmake case_studies/*.ipynb        # case studies (short horizon)
mkdocs build --strict                               # docs build without warnings
```

`pre-commit run --all-files` runs the fast subset (ruff, whitespace, notebook sync) on
the whole tree.

## Tests

- Tests live in `tests/` and use the shared `four_dof` fixture in `conftest.py`.
- New features need tests. Keep them fast: short horizons (`T` of a second or two) are
  enough to exercise a filter, a schedule or a configuration key.
- Every tutorial and case study must keep a YAML twin that reproduces it (the notebooks
  check this at the end).

## Notebooks (tutorials and case studies)

Every notebook under `examples/` and `case_studies/` is a
[jupytext](https://jupytext.readthedocs.io/) pair: the `.py` in percent format is the
reviewable source of truth, the `.ipynb` carries the executed outputs shown in the docs.

- Edit the `.py`. Then re-execute to refresh the outputs and the pairing:
  ```bash
  cd examples && jupytext --to ipynb --execute 01_four_dof_jacobi_ukf.py
  ```
  (`pre-commit` runs `jupytext --sync` so the pair never drifts, but only execution
  refreshes the outputs.)
- Run scripts from their own directory so that the YAML twins are found.
- Keep notebooks self-contained and deterministic (seeds everywhere); no `plt.show()`,
  no progress bars.
- After changing a notebook's first figure, regenerate the gallery thumbnails and the
  hero image: `python scripts/make_doc_images.py`.

## Conventions

- **Docstrings** follow the NumPy style (`Parameters`, `Returns`, `Examples`
  sections); the API reference is generated from them by mkdocstrings. Every module
  carries a docstring that says what it does and where it fits in the graph (node,
  edge, schedule, solver).
- **Names**: subsystems are nodes, interfaces are directed edges, messages are what
  edges carry, schedules decide when messages are exchanged. Keep the code and the
  paper using the same words.
- **New filters** subclass `pci.filters.Filter` and register with
  `@register_filter("name")`. **New models** live in `pci/models/` and return a
  `System` from a `decompose`-style method.
- Keep the physics functions (`dynamics`, `measurement`) pure: they receive a plain
  `dict` of parameters so the same function serves the forward model and the inverse
  problem.
- **Lint**: `ruff check` with the rules in `pyproject.toml` (line length 120). The
  formatter is not enforced.

## Docs

```bash
mkdocs serve                 # live preview at http://127.0.0.1:8000
```

Narrative pages live in `docs/`; `docs/tutorials` and `docs/case-studies` are symlinks
to `examples/` and `case_studies/`, so the notebooks are rendered with their committed
outputs (`mkdocs-jupyter`, no execution at build time). When you add a public symbol,
add it to the relevant `docs/reference/*.md` page.

## Reporting problems

Open an issue with the YAML (or a minimal script) that reproduces the behaviour, the
`Results.summary()` output, and the `pci` version.
