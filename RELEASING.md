# Releasing pci

The library is versioned in one place, `pyproject.toml` (`project.version`), and
published to PyPI as **`pci-inference`** by `.github/workflows/release.yml` when a
`v*` tag is pushed. The documentation is deployed to GitHub Pages by
`.github/workflows/docs.yml` on every push to `main`.

## Cut a release

1. Move the `## [Unreleased]` items of `CHANGELOG.md` into a new `## [X.Y.Z] - YYYY-MM-DD`
   section.
2. Bump `pyproject.toml` (`project.version = "X.Y.Z"`) and `pci/__init__.py`
   (`__version__`).
3. Update `CITATION.cff` (`version`, `date-released`).
4. Re-execute the notebooks if the estimates changed (`jupytext --to ipynb --execute`
   in `examples/` and `case_studies/`), then `python scripts/make_doc_images.py`.
5. Commit, open a PR, merge to `main` with CI green. Optionally run the
   **Examples (full horizon)** workflow by hand as a last check.
6. Tag and push:
   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
   `release.yml` fails if the tag does not match the `pyproject` version, runs the gate
   (ruff, pytest, `mkdocs build --strict`), builds the sdist and wheel, and publishes
   to PyPI through **Trusted Publishing (OIDC)**; no token is stored.

## One-time setup (before the first release)

- Confirm the distribution name `pci-inference` is free on PyPI.
- On PyPI, add a **trusted publisher**: repository `cisgroup/pci`, workflow
  `release.yml`, environment `pypi`.
- On GitHub, create an **environment** named `pypi` (Settings → Environments).
- Turn on **GitHub Pages** for the repository (source: the `gh-pages` branch that
  `mkdocs gh-deploy` creates). On a private repository Pages needs a plan that
  supports it; the site can be previewed locally with `mkdocs serve` in the meantime.
- Rehearse once against TestPyPI if in doubt.

## The paper's code archive

The scripts and manuscript of the paper live in the separate public repository
[cisgroup/compositional-inference](https://github.com/cisgroup/compositional-inference),
cited by arXiv:2605.27544. Releases of the library never touch it.
