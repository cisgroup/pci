# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Case study 1: 9 DOF = 3 x 3-DOF
#
# Three 3-DOF subsystems with three different filters and tasks: the grounded subsystem only tracks its states with a linear KF, the free-floating middle one identifies a stiffness from two accelerometers with a UKF, and the last one identifies a stiffness and a damping from a displacement and an acceleration sensor with a CKF and RK4.
#
# The chain has non-uniform masses, stiffnesses and dampings, random forces act on
# every mass, and messages are mean interface forces exchanged with the
# **jacobi** schedule. The measurement covariance is inflated
# 10x to account for the message error (see [Concepts](../concepts.md)).
#
# The paper-length horizon is 20 s at 1 ms; set `PCI_T` to shorten it. Set
# `RUN_CENTRAL = True` to also run a single UKF on the full augmented state for
# comparison.

# %%
import os

from _support import build, centralized, compare_with_yaml, print_design, report

N_DOF = 9
PARTITION = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
NAMES = ['S1', 'S2', 'S3']
SCHEDULE = "jacobi"
DT, T = 1e-3, float(os.environ.get("PCI_T", 20.0))
R_INFLATION = 10.0     # R = (10 x sensor noise)^2, needed with mean-only messages
RUN_CENTRAL = False

# %% [markdown]
# ## Design: one line per subsystem
#
# Filter, integrator, sensors with their noise standard deviation, and the unknowns as
# a fraction of the true value used for the initial guess.

# %%
DESIGN = [
    dict(filter='kf', integrator='heun', sensors={'a1': 0.005, 'a3': 0.005}, unknowns={}),  # grounded: accelerations only, states only (linear KF)
    dict(filter='ukf', integrator='heun', sensors={'a5': 0.01, 'a6': 0.01}, unknowns={'k5': 0.6}),  # free-floating, two accelerometers, joint estimation
    dict(filter='ckf', integrator='rk4', sensors={'x8': 0.001, 'a9': 0.005}, unknowns={'k9': 0.7, 'c9': 1.5}),  # displacement + acceleration, two unknowns, RK4
]
LOADS = {1: {"type": "harmonic", "amplitude": 300.0, "frequency": 1.2}}
LOADS.update({d: {"type": "random", "std": 400.0, "seed": 10 + d} for d in range(1, N_DOF + 1)})

# %% [markdown]
# ## Physical system, synthetic data, decomposition

# %%
chain, truth_dict, data, system, unknowns, sensors, noise = build(N_DOF, PARTITION, NAMES, DESIGN, LOADS, DT, T, SCHEDULE, R_INFLATION)
print_design("9 DOF = 3 x 3-DOF", NAMES, PARTITION, DESIGN, DT, T, SCHEDULE, R_INFLATION)

# %% [markdown]
# ## Distributed estimation

# %%
res = system.estimate(data, loads={f"f{d}": v for d, v in LOADS.items()}, dt=DT, T=T, truth=truth_dict)
report(res, chain.parameters, unknowns, NAMES, PARTITION, T)

# %%
res.plot_parameters();

# %%
res.plot_states([f"{name}.x{g[len(g) // 2]}" for name, g in zip(NAMES, PARTITION)][:6]);

# %%
res.plot_messages();

# %% [markdown]
# ## Centralized comparison (optional)

# %%
if RUN_CENTRAL:
    centralized(chain, N_DOF, unknowns, sensors, noise, data, LOADS, DT, T, truth_dict, R_INFLATION, res)

# %% [markdown]
# ## The YAML twin
#
# `01_case_09dof.yaml` describes the identical problem; with the full 20 s horizon it
# reproduces the Python-API result to machine precision.

# %%
if T == 20.0:
    compare_with_yaml(res, "01_case_09dof.yaml", unknowns)

# %% [markdown]
# The three schedules give the same parameters on this case; Jacobi and AB2 give the smallest state errors. The centralized UKF on the full augmented state agrees with the distributed estimates (k5 about -0.4 % in both) at 2-3x the sequential cost and without the one-core-per-subsystem speed-up.
