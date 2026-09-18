# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Uncertainty-carrying messages (`message_type="mean_variance"`) with the
  incremental injection rule of the paper.
- Documentation site (MkDocs Material) with tutorials, case studies and an
  API reference generated from docstrings.
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
