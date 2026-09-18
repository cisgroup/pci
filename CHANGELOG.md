# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Documentation site (MkDocs Material): landing page, getting started, concepts,
  YAML configuration reference, API reference generated from the docstrings
  (mkdocstrings), tutorials and case studies rendered from their notebooks
  (mkdocs-jupyter). Build with `mkdocs serve`.
- Tutorials 01-07 and case studies 01-04 as jupytext-paired notebooks
  (`.py` percent scripts + executed `.ipynb`), with gallery thumbnails generated
  by `scripts/make_doc_images.py`.
- Tutorial 07 (`07_vanderpol_boucwen.py`) for the Van der Pol / Bouc-Wen YAML
  problem, which previously existed as a YAML file only.
- GitHub Actions: `ci.yml` (ruff, pytest on Python 3.10-3.13, notebook execution
  with nbmake, strict docs build), `docs.yml` (GitHub Pages deployment),
  `examples.yml` (weekly full-horizon notebook run), `release.yml` (PyPI trusted
  publishing on `v*` tags). `.pre-commit-config.yaml` with ruff, whitespace and
  notebook-sync hooks.
- `ruff` lint configuration (`E, F, W, B, I`), `RELEASING.md`, `docs` and `dev`
  extras in `pyproject.toml`.
- The El Centro record ships with the tutorials (`examples/data/elcentro.mat`);
  `ElCentroLoad` reads the `PCI_ELCENTRO` environment variable (`CI_ELCENTRO`
  still works).
- `load_config` resolves relative `path` entries of recorded loads against the
  directory of the configuration file.

### Changed
- Case studies moved from `examples/07-10_*` to `case_studies/01-04_*`; the
  shared helpers are `case_studies/_support.py`. Their horizon is set with
  `PCI_T` instead of `CI_T`.
- README trimmed to a landing page; the detailed material moved to the docs.

### Planned
- Uncertainty-carrying messages (`message_type="mean_variance"`) with the
  incremental injection rule of the paper.
- Learned interface laws (SINDy) as drop-in `Interface.law`.

## [0.1.0] - 2026-09-18

### Added
- `Subsystem`, `Interface`, `System` with `estimate` and `simulate`.
- Local estimators: KF, EKF, UKF, CKF, with a registry for custom filters.
- Message-passing schedules: Jacobi, Gauss-Seidel, Adams-Bashforth 2, with
  optional inner iterations.
- Integrators: Euler, Heun, RK4.
- Built-in `MassSpringChain` model with `decompose`, the paper's 4-DOF and
  6-DOF configurations, and a load library (random, harmonic, El Centro).
- Declarative problem definition through YAML or dict (`pci.solve`).
- `Results` with metrics (RMSE, NRMSE), plotting helpers and pandas export.
- Examples 01-11 and case studies with 9, 16, 20 and 40 DOF.
