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
# # 5. Six DOF, three subsystems, three filters
#
# A 6-DOF chain with springs to ground at several masses and a parallel spring
# between DOFs 3 and 4, split into three 2-DOF subsystems. Seven internal parameters
# are unknown and each subsystem runs a different local estimator (UKF, CKF, EKF).
# Loosely follows the 6-DOF example of the paper.

# %%
import pci
from pci import Spring

# %% [markdown]
# ## An explicit spring topology
#
# Beyond the serial `uniform` chain, `MassSpringChain` takes any list of `Spring`
# objects: grounded or between two DOFs, with a stiffness, a damping and an optional
# name. Parameter names follow the spring index (`k3`, `c3`) or the given name (`k34`).

# %%
springs = [
    Spring("ground", 1, 50_000.0, 300.0),          # k1
    Spring("ground", 2, 40_000.0, 350.0),          # k2
    Spring(1, 2, 45_000.0, 320.0),                 # k3  (unknown, S1)
    Spring(2, 3, 30_000.0, 400.0),                 # k4  interface S1-S2
    Spring(3, 4, 25_000.0, 480.0),                 # k5
    Spring("ground", 4, 55_000.0, 700.0),          # k6  (unknown, S2)
    Spring(4, 5, 30_000.0, 350.0),                 # k7  interface S2-S3
    Spring("ground", 5, 45_000.0, 460.0),          # k8
    Spring(5, 6, 40_000.0, 480.0),                 # k9  (unknown, S3)
    Spring(3, 4, 40_000.0, 0.0, name="34"),        # k34 parallel spring (unknown, S2)
]
chain = pci.MassSpringChain([500, 500, 600, 500, 500, 500], springs)

dt, T = 2e-3, 20.0
loads = {4: {"type": "random", "std": 1500.0, "seed": 123}}
truth = chain.simulate(loads, dt=dt, T=T)
sensors = ["a2", "a3", "a4", "a5"]
data = chain.measure(truth, sensors, noise_std=1e-3, seed=1)

# %% [markdown]
# ## Decompose with heterogeneous estimators
#
# Unknowns must be internal to a subsystem: a mass, or a spring whose two ends lie in
# the same subsystem or on the ground. Interface springs (`k4`, `k7`) stay known;
# learning them is a roadmap item. Here the measurement noise given to the filters is
# already the inflated value of the paper script, so `r_inflation=1`.

# %%
partition = [[1, 2], [3, 4], [5, 6]]
guess = 0.7
system = chain.decompose(
    partition,
    unknowns={
        "k3": {"initial": guess * 45_000, "std": 1e4},
        "c3": {"initial": guess * 320, "std": 3e2},
        "k6": {"initial": guess * 55_000, "std": 1e4},
        "c6": {"initial": guess * 700, "std": 3e1},
        "k34": {"initial": guess * 40_000, "std": 1e4},
        "k9": {"initial": guess * 40_000, "std": 1e4},
        "c9": {"initial": guess * 480, "std": 3e2},
    },
    sensors=sensors,
    noise_std={"a2": 3e-2, "a3": 1e-1, "a4": 1e-1, "a5": 3e-2},
    filters={"S1": "ukf", "S2": "ckf", "S3": "ekf"},
    integrator="heun",
    schedule="jacobi",
    process_var=1e-11,
    r_inflation=1.0,
)
print(system.describe())

# %%
results = system.estimate(data, loads={"f4": loads[4]}, dt=dt, T=T,
                          truth={**truth.as_dict(), **chain.interface_forces(truth, partition)})
print(results.summary(start=10.0))

# %%
results.plot_parameters();

# %%
results.plot_messages();

# %% [markdown]
# Stiffnesses converge within a few seconds; dampings identified from accelerations
# alone converge more slowly, as in the paper's 6-DOF study.
#
# ## The YAML twin

# %%
yaml_results = pci.solve("05_six_dof_three_subsystems.yaml")
print("YAML twin:", {p: round(yaml_results.final(p), 1) for p in ["k3", "c3", "k6", "c6", "k34", "k9", "c9"]})
