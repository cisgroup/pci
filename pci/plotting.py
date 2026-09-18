"""Matplotlib helpers (optional dependency)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib: pip install pci[plot]") from exc
    return plt


def plot_trajectories(results, names: Sequence[str], truth=None, bands: bool = True, ncols: int = 2,
                      figsize=None, sharex: bool = True, band_sigma: float = 2.0, show: bool = False):
    plt = _plt()
    n = len(names)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (6 * ncols, 2.6 * nrows), sharex=sharex, squeeze=False)
    t = results.t
    for ax, name in zip(axes.ravel(), names):
        est = results.state(name)
        ref = results._truth_for(name, truth)
        if ref is not None:
            ref = np.asarray(ref, float).ravel()
            if ref.size == 1:
                ax.axhline(ref[0], color="k", lw=1.2, label="true")
            else:
                ax.plot(t[: ref.size], ref[: t.size], color="k", lw=1.2, label="true")
        ax.plot(t, est, color="C0", lw=1.0, ls="--", label="estimate")
        if bands and results.mode == "estimate":
            s = results.std(name)
            ax.fill_between(t, est - band_sigma * s, est + band_sigma * s, color="C0", alpha=0.2,
                            label=f"±{band_sigma:g}σ")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_messages(results, truth: Optional[dict] = None, figsize=None, show: bool = False):
    plt = _plt()
    names = list(results.messages)
    fig, axes = plt.subplots(len(names), 1, figsize=figsize or (8, 2.4 * len(names)), sharex=True, squeeze=False)
    tm = results.t[:-1]
    for ax, nm in zip(axes.ravel(), names):
        ref = (truth or results.truth or {}).get(nm)
        if ref is not None:
            ax.plot(tm, np.asarray(ref)[: tm.size], "k", lw=1.2, label="true")
        ax.plot(tm, results.messages[nm], "C1--", lw=1.0, label="message (mean)")
        ax.set_title(nm)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    axes.ravel()[-1].set_xlabel("time [s]")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
