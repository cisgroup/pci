"""Shared helpers for the case-study scripts 07-10 (chain definition, report, figures)."""

from __future__ import annotations

import os

import numpy as np

import pci

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def chain_properties(n_dof: int):
    """Deterministic, non-uniform masses / stiffnesses / dampings (rounded so the YAML twins are exact)."""
    i = np.arange(1, n_dof + 1)
    masses = np.round(500.0 * (1.0 + 0.10 * np.cos(0.7 * i)), 1)
    k = np.round(50_000.0 * (1.0 + 0.25 * np.sin(1.3 * i)), 1)
    c = np.round(300.0 * (1.0 + 0.30 * np.cos(0.9 * i)), 1)
    return masses.tolist(), k.tolist(), c.tolist()


def print_design(title, names, partition, design, dt, T, schedule, r_inflation):
    print("=" * 100)
    print(f"{title}   schedule = {schedule}   dt = {dt}   T = {T} s   steps = {int(round(T / dt))}   R inflation = {r_inflation:g}x")
    print("-" * 100)
    print(f"{'subsystem':<10s}{'DOFs':<18s}{'filter':<8s}{'integrator':<12s}{'task':<26s}{'sensors (noise std)':<40s}")
    for name, g, spec in zip(names, partition, design):
        task = "state estimation" if not spec.get("unknowns") else "joint: " + ", ".join(spec["unknowns"])
        sens = ", ".join(f"{s} ({std:g})" for s, std in spec["sensors"].items())
        print(f"{name:<10s}{str(g):<18s}{spec['filter']:<8s}{spec.get('integrator', 'heun'):<12s}{task:<26s}{sens:<40s}")


def report(res, true_params, unknowns, names, partition, T):
    print(f"\nDistributed run time: {res.runtime:.1f} s (sequential; ~{res.runtime / len(partition):.1f} s with one core per subsystem)")
    print(f"\n{'parameter':<10s}{'true':>10s}{'initial':>10s}{'final':>12s}{'error %':>9s}{'post. std':>11s}")
    for p, spec in unknowns.items():
        true, fin, sd = true_params[p], res.final(p), res.std(p)[-1]
        print(f"{p:<10s}{true:>10.1f}{spec['initial']:>10.1f}{fin:>12.1f}{100 * (fin - true) / true:>+9.2f}{sd:>11.3g}")
    print(f"\n{'subsystem':<10s}{'disp. NRMSE (2nd half)':>24s}{'vel. NRMSE (2nd half)':>24s}")
    for name, g in zip(names, partition):
        nx = np.mean(list(res.nrmse([f"x{d}" for d in g], start=T / 2).values()))
        nv = np.mean(list(res.nrmse([f"v{d}" for d in g], start=T / 2).values()))
        print(f"{name:<10s}{nx:>24.3%}{nv:>24.3%}")
    msg = res.nrmse(list(res.messages), start=T / 2)
    print("interface-force NRMSE (2nd half): " + ", ".join(f"{k} {v:.2%}" for k, v in msg.items()))


def save_figures(res, tag, partition, unknowns):
    os.makedirs(OUT, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    if unknowns:
        res.plot_parameters().savefig(os.path.join(OUT, f"{tag}_parameters.png"), dpi=130)
    show = [f"S{i + 1}.x{g[len(g) // 2]}" for i, g in enumerate(partition)][:6]
    res.plot_states(show).savefig(os.path.join(OUT, f"{tag}_states.png"), dpi=130)
    res.plot_messages().savefig(os.path.join(OUT, f"{tag}_messages.png"), dpi=130)
    try:
        res.to_dataframe().to_csv(os.path.join(OUT, f"{tag}_trajectories.csv"), index=False)
    except ImportError:
        pass
    print(f"figures saved to {OUT}/{tag}_*.png")


def compare_with_yaml(res, yaml_path, unknowns):
    """Solve the YAML twin and check that it reproduces the Python-API result."""
    yres = pci.solve(yaml_path)
    worst = max(abs(yres.final(p) - res.final(p)) / max(abs(res.final(p)), 1e-12) for p in unknowns) if unknowns else 0.0
    print(f"\nYAML twin ({os.path.basename(yaml_path)}): "
          + ", ".join(f"{p}={yres.final(p):.1f}" for p in unknowns)
          + f"   max relative difference to Python API: {worst:.2e}")
    return yres
