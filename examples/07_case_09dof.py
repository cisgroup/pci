"""Case study: 9 DOF = 3 x 3-DOF.

Serial mass-spring chain with non-uniform properties, split into 3 subsystems.
Every subsystem has its own sensors, noise level, task (state estimation only or
joint state-and-parameter estimation), filter and integrator; messages are mean
interface forces exchanged with the jacobi schedule.

Two equivalent routes are shown: the Python API and the YAML twin 07_case_09dof.yaml.
Usage: python 07_case_09dof.py [--central] [--no-save]      (CI_T overrides the 20 s horizon)
"""

import argparse
import os

import numpy as np

import ci
from casestudy_report import chain_properties, compare_with_yaml, print_design, report, save_figures

HERE = os.path.dirname(os.path.abspath(__file__))
N_DOF = 9
PARTITION = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
NAMES = ['S1', 'S2', 'S3']
SCHEDULE = "jacobi"
DT, T = 1e-3, float(os.environ.get("CI_T", 20.0))
R_INFLATION = 10.0     # R = (10 x sensor noise)^2, needed with mean-only messages (see README)

# one entry per subsystem: filter, integrator, sensors {name: noise std}, unknowns {param: initial guess / true}
DESIGN = [
    dict(filter='kf', integrator='heun', sensors={'a1': 0.005, 'a3': 0.005}, unknowns={}),  # grounded: accelerations only, states only (linear KF),
    dict(filter='ukf', integrator='heun', sensors={'a5': 0.01, 'a6': 0.01}, unknowns={'k5': 0.6}),  # free-floating, two accelerometers, joint estimation,
    dict(filter='ckf', integrator='rk4', sensors={'x8': 0.001, 'a9': 0.005}, unknowns={'k9': 0.7, 'c9': 1.5}),  # displacement + acceleration, two unknowns, RK4,
]
LOADS = {1: {"type": "harmonic", "amplitude": 300.0, "frequency": 1.2}}
LOADS.update({d: {"type": "random", "std": 400.0, "seed": 10 + d} for d in range(1, N_DOF + 1)})

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

    print_design("9 DOF = 3 x 3-DOF", NAMES, PARTITION, DESIGN, DT, T, SCHEDULE, R_INFLATION)
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
        save_figures(res, "case09_" + SCHEDULE, PARTITION, unknowns)

    # ---- YAML twin: identical problem, no Python beyond ci.solve ---------------------------------
    if T == 20.0:
        compare_with_yaml(res, os.path.join(HERE, "07_case_09dof.yaml"), unknowns)
