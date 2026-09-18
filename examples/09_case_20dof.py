"""Case study: 20 DOF = 5 x 4-DOF.

Serial mass-spring chain with non-uniform properties, split into 5 subsystems.
Every subsystem has its own sensors, noise level, task (state estimation only or
joint state-and-parameter estimation), filter and integrator; messages are mean
interface forces exchanged with the gauss_seidel schedule.

Two equivalent routes are shown: the Python API and the YAML twin 09_case_20dof.yaml.
Usage: python 09_case_20dof.py [--central] [--no-save]      (CI_T overrides the 20 s horizon)
"""

import argparse
import os

import numpy as np

import pci
from casestudy_report import chain_properties, compare_with_yaml, print_design, report, save_figures

HERE = os.path.dirname(os.path.abspath(__file__))
N_DOF = 20
PARTITION = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
NAMES = ['S1', 'S2', 'S3', 'S4', 'S5']
SCHEDULE = "gauss_seidel"
DT, T = 1e-3, float(os.environ.get("CI_T", 20.0))
R_INFLATION = 10.0     # R = (10 x sensor noise)^2, needed with mean-only messages (see README)

# one entry per subsystem: filter, integrator, sensors {name: noise std}, unknowns {param: initial guess / true}
DESIGN = [
    dict(filter='kf', integrator='heun', sensors={'a2': 0.005, 'a4': 0.005}, unknowns={}),  # grounded: accelerations only, states only,
    dict(filter='ukf', integrator='heun', sensors={'x6': 0.001, 'a8': 0.01}, unknowns={'k7': 0.7}),  # displacement + acceleration,
    dict(filter='ckf', integrator='heun', sensors={'x11': 0.001}, unknowns={'k10': 0.6, 'c10': 1.5}),  # a single displacement sensor, two unknowns,
    dict(filter='ekf', integrator='rk4', sensors={'x14': 0.0001, 'a16': 0.005}, unknowns={'m15': 0.8}),  # unknown mass, EKF, RK4,
    dict(filter='ukf', integrator='heun', sensors={'a18': 0.02, 'x20': 0.001}, unknowns={'k19': 1.3}),  # noisy accelerometer + displacement,
]
LOADS = {d: {"type": "random", "std": 300.0, "seed": 30 + d} for d in range(1, N_DOF)}
LOADS[N_DOF] = {"type": "elcentro", "path": os.path.expanduser("~/Documents/Princeton/Projects/Active-projects/ntwks/parl_KFs/"
                "Selected_code_for_sub/github-compositional-inference/data/elcentro.mat"), "scale": 1500.0, "stretch": True}

# ---- physical system and synthetic data --------------------------------------------------------
masses, k, c = chain_properties(N_DOF)
chain = pci.MassSpringChain(masses, k=k, c=c)
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

    print_design("20 DOF = 5 x 4-DOF", NAMES, PARTITION, DESIGN, DT, T, SCHEDULE, R_INFLATION)
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
        save_figures(res, "case20_" + SCHEDULE, PARTITION, unknowns)

    # ---- YAML twin: identical problem, no Python beyond pci.solve ---------------------------------
    if T == 20.0:
        compare_with_yaml(res, os.path.join(HERE, "09_case_20dof.yaml"), unknowns)
