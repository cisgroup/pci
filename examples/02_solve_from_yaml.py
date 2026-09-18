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
# # 2. Problems in YAML
#
# A problem for the built-in mass-spring model is a dozen lines of YAML: the chain, the
# partition, the local estimators, the unknowns, the sensors, the loads and the time
# grid. `pci.solve` builds the system, generates synthetic ground truth and data,
# runs the distributed estimation and returns a `Results` object. The
# [Configuration](../configuration.md) page lists every key.

# %%
import copy

import pci

# %% [markdown]
# ## The file

# %%
print(open("01_four_dof_jacobi_ukf.yaml").read())

# %% [markdown]
# ## Solve it

# %%
results = pci.solve("01_four_dof_jacobi_ukf.yaml")
print(results.summary(start=1.0))

# %%
results.plot_parameters();

# %%
results.plot_states(["S1.x2", "S2.x3"]);

# %% [markdown]
# ## Override a configuration
#
# `load_config` returns a plain dictionary, so a study over schedules, filters or
# partitions is a loop that changes a key and calls `pci.solve` on the copy.

# %%
base = pci.load_config("01_four_dof_jacobi_ukf.yaml")
variants = {
    "baseline (jacobi, ukf + ckf)": {},
    "gauss_seidel": {"schedule": "gauss_seidel"},
    "ab2": {"schedule": "ab2"},
    "all ckf": {"filters": "ckf"},
    "jacobi, 2 inner iterations": {"schedule": {"type": "jacobi", "iterations": 2}},
}
print(f"{'variant':<32s}{'k4 final':>10s}{'error %':>9s}{'x3 rmse':>10s}{'time [s]':>9s}")
for label, overrides in variants.items():
    cfg = copy.deepcopy(base)
    cfg.update(overrides)
    res = pci.solve(cfg)
    k4 = res.final("k4")
    print(f"{label:<32s}{k4:>10.1f}{100 * (k4 - 50_000) / 50_000:>+9.2f}{res.rmse(['x3'], start=1.0)['x3']:>10.2e}{res.runtime:>9.1f}")

# %% [markdown]
# ## Bring your own measurements
#
# Without a `truth` block nothing is simulated; pass the recorded sensor signals as
# `measurements={name: array}` instead. The arrays must be sampled on the `time` grid
# of the configuration (`n_steps + 1` samples).

# %%
chain = pci.MassSpringChain.uniform(4, mass=500.0, k=50_000.0, c=300.0)
loads = {d: {"type": "random", "std": 400.0, "seed": 10 + d} for d in range(1, 5)}
truth = chain.simulate(loads, dt=1e-3, T=5.0)
data = chain.measure(truth, ["a1", "a4"], noise_std=1e-3, seed=3)

cfg = copy.deepcopy(base)
cfg.pop("truth")                                   # no synthetic data: use the arrays below
cfg["loads"] = loads                               # the loads that acted on the real structure
res = pci.solve(cfg, measurements=data)
print(f"k4 from recorded data: {res.final('k4'):.1f}")
