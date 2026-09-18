"""Shared helpers for the case studies (chain definition, design table, report, YAML check).

Not a tutorial: imported by ``01_case_09dof.py`` ... ``04_case_40dof.py``.
"""

from __future__ import annotations

import os

import numpy as np

import pci

HERE = os.path.dirname(os.path.abspath(__file__))
# The 1940 El Centro record shipped with the tutorials (from the paper repository).
ELCENTRO = os.path.join(HERE, "..", "examples", "data", "elcentro.mat")


def chain_properties(n_dof: int):
    """Deterministic, non-uniform masses / stiffnesses / dampings (rounded so the YAML twins are exact)."""
    i = np.arange(1, n_dof + 1)
    masses = np.round(500.0 * (1.0 + 0.10 * np.cos(0.7 * i)), 1)
    k = np.round(50_000.0 * (1.0 + 0.25 * np.sin(1.3 * i)), 1)
    c = np.round(300.0 * (1.0 + 0.30 * np.cos(0.9 * i)), 1)
    return masses.tolist(), k.tolist(), c.tolist()


def build(n_dof, partition, names, design, loads, dt, T, schedule, r_inflation):
    """Physical chain, synthetic data and the decomposed system for one case study.

    Returns ``(chain, truth_dict, data, system, unknowns, sensors, noise)``.
    """
    masses, k, c = chain_properties(n_dof)
    chain = pci.MassSpringChain(masses, k=k, c=c)
    truth = chain.simulate(loads, dt=dt, T=T)
    truth_dict = {**truth.as_dict(), **chain.interface_forces(truth, partition)}

    sensors, noise, filters, integrators, unknowns = [], {}, {}, {}, {}
    for name, spec in zip(names, design):
        filters[name] = spec["filter"]
        integrators[name] = spec["integrator"]
        for s, std in spec["sensors"].items():
            sensors.append(s)
            noise[s] = std
        for p, frac in spec["unknowns"].items():
            unknowns[p] = {"initial": round(frac * chain.parameters[p], 1), "std": round(0.5 * chain.parameters[p], 1)}
    data = chain.measure(truth, sensors, noise_std=noise, seed=1)

    system = chain.decompose(partition, unknowns=unknowns, sensors=sensors, noise_std=noise, filters=filters,
                             integrator=integrators, schedule=schedule, state_var=1e-4, process_var=1e-10,
                             names=names, r_inflation=r_inflation)
    return chain, truth_dict, data, system, unknowns, sensors, noise


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
    print(f"Distributed run time: {res.runtime:.1f} s (sequential; ~{res.runtime / len(partition):.1f} s with one core per subsystem)")
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


def centralized(chain, n_dof, unknowns, sensors, noise, data, loads, dt, T, truth_dict, r_inflation, res):
    """Run a single UKF on the full augmented state and print it next to the distributed result."""
    mono = chain.decompose([list(range(1, n_dof + 1))], unknowns=unknowns, sensors=sensors, noise_std=noise,
                           filters="ukf", state_var=1e-4, process_var=1e-10, names=["central"], r_inflation=r_inflation)
    cres = mono.estimate(data, loads={f"f{d}": v for d, v in loads.items()}, dt=dt, T=T, truth=truth_dict)
    print(f"Centralized UKF ({2 * n_dof + len(unknowns)} augmented states): {cres.runtime:.1f} s")
    for p in unknowns:
        tp = chain.parameters[p]
        print(f"  {p:<6s} central {cres.final(p):>10.1f} ({100 * (cres.final(p) - tp) / tp:+.2f}%)"
              f"   distributed {res.final(p):>10.1f} ({100 * (res.final(p) - tp) / tp:+.2f}%)")
    return cres


def compare_with_yaml(res, yaml_path, unknowns):
    """Solve the YAML twin and check that it reproduces the Python-API result."""
    yres = pci.solve(yaml_path)
    worst = max(abs(yres.final(p) - res.final(p)) / max(abs(res.final(p)), 1e-12) for p in unknowns) if unknowns else 0.0
    print(f"YAML twin ({os.path.basename(yaml_path)}): "
          + ", ".join(f"{p}={yres.final(p):.1f}" for p in unknowns)
          + f"   max relative difference to Python API: {worst:.2e}")
    return yres
