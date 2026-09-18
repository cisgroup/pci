"""MkDocs build hooks for the pci documentation.

Injects the ``pci`` version into the site at build time, so the docs always show the
version they were built from without hardcoding it anywhere. The version is taken from
the installed distribution (``pci-inference``) and falls back to ``pyproject.toml`` when
the docs are built from a checkout without installing the package.
"""

from __future__ import annotations

import pathlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def _version() -> str:
    try:
        return version("pci-inference")
    except PackageNotFoundError:
        pass
    try:  # Python >= 3.11
        import tomllib

        pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
        return tomllib.loads(pyproject.read_text())["project"]["version"]
    except Exception:  # pragma: no cover - last resort
        return "dev"


PCI_VERSION = _version()


def on_config(config: Any, **kwargs: Any) -> Any:
    """Show the built pci version in the site footer."""
    config.copyright = (
        f"pci v{PCI_VERSION} · MIT-licensed · Princeton University, Complex Infrastructure Systems Group"
    )
    return config


def on_page_markdown(markdown: str, **kwargs: Any) -> str:
    """Replace the ``{{ pci_version }}`` token with the built version."""
    return markdown.replace("{{ pci_version }}", PCI_VERSION)
