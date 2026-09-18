"""Case study: 40 DOF = 10 x 4-DOF.

Serial mass-spring chain with non-uniform properties, split into 10 subsystems.
Every subsystem has its own sensors, noise level, task (state estimation only or
joint state-and-parameter estimation), filter and integrator; messages are mean
interface forces exchanged with the jacobi schedule.

Two equivalent routes are shown: the Python API and the YAML twin 10_case_40dof.yaml.
Usage: python 10_case_40dof.py [--central] [--no-save]      (CI_T overrides the 20 s horizon)
"""

import argparse
import os

import numpy as np

import ci
from casestudy_report import chain_properties, compare_with_yaml, print_design, report, save_figures

HERE = os.path.dirname(os.path.abspath(__file__))
N_DOF = 40
PARTITION = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36], [37, 38, 39, 40]]
NAMES = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
SCHEDULE = "jacobi"
DT, T = 1e-3, float(os.environ.get("CI_T", 20.0))
R_INFLATION = 10.0     # R = (10 x sensor noise)^2, needed with mean-only messages (see README)

# one entry per subsystem: filter, integrator, sensors {name: noise std}, unknowns {param: initial guess / true}
DESIGN = [
    dict(filter='kf', integrator='heun', sensors={'a2': 0.005, 'a4': 0.005}, unknowns={}),  # grounded, accelerations only,
    dict(filter='ukf', integrator='heun', sensors={'x6': 0.001, 'a8': 0.01}, unknowns={'k7': 0.7}),
    dict(filter='ckf', integrator='heun', sensors={'x11': 0.001, 'a12': 0.005}, unknowns={'k10': 0.6, 'c10': 1.5}),
    dict(filter='kf', integrator='heun', sensors={'x14': 0.0001, 'x16': 0.0001}, unknowns={}),
    dict(filter='ukf', integrator='rk4', sensors={'x19': 0.001}, unknowns={'k18': 1.3}),
    dict(filter='ckf', integrator='heun', sensors={'x22': 0.001, 'a24': 0.01}, unknowns={'k23': 0.7}),
    dict(filter='ekf', integrator='heun', sensors={'x26': 0.0001, 'a28': 0.005}, unknowns={'m27': 0.8}),
    dict(filter='kf', integrator='euler', sensors={'a30': 0.005, 'x32': 0.0001}, unknowns={}),
    dict(filter='ukf', integrator='heun', sensors={'x35': 0.001, 'a36': 0.01}, unknowns={'k34': 0.6, 'c34': 0.6}),
    dict(filter='ckf', integrator='heun', sensors={'x40': 0.001}, unknowns={'k39': 1.4}),
]
LOADS = {d: {"type": "random", "std": 400.0, "seed": 40 + d} for d in range(1, N_DOF + 1)}

# ---- physical system and synthetic data --------------------------------------------------------
masses, k, c = chain_properties(N_DOF)
chain = ci.MassSpringChain(masses, k=k, c=c)
truth = chain.simulate(LOADS, dt=DT, T=T)
truth_dict = {**truth.as_dict(), **chain.interface_forces(truth, PARTITION)}

sensors, noise, filters, integrators, unknowns = [], {}, {}, {}, {}
for name, spec in zip(NAMES, DESIGN):
    filters[name] = spec["filter"]
    integrators[name] = spec["integrator"]
    for s, std in spec["sensors"].items():
        sensors.append(s)
        noise[s] = std
    for p, frac in spec["unknowns"].items():
        unknowns[p] = {"initial": round(frac * chain.parameters[p], 1), "std": round(0.5 * chain.parameters[p], 1)}
data = chain.measure(truth, sensors, noise_std=noise, seed=1)

# ---- decomposition, local estimators, message passing ------------------------------------------
system = chain.decompose(PARTITION, unknowns=unknowns, sensors=sensors, noise_std=noise, filters=filters,
                         integrator=integrators, schedule=SCHEDULE, state_var=1e-4, process_var=1e-10,
                         names=NAMES, r_inflation=R_INFLATION)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--central", action="store_true", help="also run a centralized UKF on the full augmented state")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    print_design("40 DOF = 10 x 4-DOF", NAMES, PARTITION, DESIGN, DT, T, SCHEDULE, R_INFLATION)
    res = system.estimate(data, loads={f"f{d}": v for d, v in LOADS.items()}, dt=DT, T=T, truth=truth_dict, progress=True)
    report(res, chain.parameters, unknowns, NAMES, PARTITION, T)

    if args.central:
        mono = chain.decompose([list(range(1, N_DOF + 1))], unknowns=unknowns, sensors=sensors, noise_std=noise,
                               filters="ukf", state_var=1e-4, process_var=1e-10, names=["central"], r_inflation=R_INFLATION)
        cres = mono.estimate(data, loads={f"f{d}": v for d, v in LOADS.items()}, dt=DT, T=T, truth=truth_dict, progress=True)
        print(f"\nCentralized UKF ({2 * N_DOF + len(unknowns)} augmented states): {cres.runtime:.1f} s")
        for p in unknowns:
            tp = chain.parameters[p]
            print(f"  {p:<6s} central {cres.final(p):>10.1f} ({100 * (cres.final(p) - tp) / tp:+.2f}%)"
                  f"   distributed {res.final(p):>10.1f} ({100 * (res.final(p) - tp) / tp:+.2f}%)")

    if not args.no_save:
        save_figures(res, "case40_" + SCHEDULE, PARTITION, unknowns)

    # ---- YAML twin: identical problem, no Python beyond ci.solve ---------------------------------
    if T == 20.0:
        compare_with_yaml(res, os.path.join(HERE, "10_case_40dof.yaml"), unknowns)
