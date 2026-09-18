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
# # 6. The paper's 6-DOF system under many configurations
#
# `06_paper_six_dof.yaml` holds the baseline of the paper's second script: three 2-DOF
# subsystems, all UKF, Jacobi messages, the El Centro record scaled as a force on
# mass 4, eight unknowns. Each variant below copies that configuration and overrides a
# few keys: the schedule, the filters, the number of inner iterations, the integrator,
# or the decomposition itself.
#
# The paper runs 50 s. Set `PCI_T` to change the horizon (this notebook uses 20 s so
# that it executes in a couple of minutes).

# %%
import copy
import os

import numpy as np

import pci

BASE = pci.load_config("06_paper_six_dof.yaml")
BASE["time"]["T"] = float(os.environ.get("PCI_T", 20.0))
TRUE = {"k3": 45_000.0, "c3": 320.0, "m3": 600.0, "k6": 55_000.0, "c6": 700.0, "k34": 40_000.0, "k9": 40_000.0, "c9": 480.0}
print(open("06_paper_six_dof.yaml").read())

# %% [markdown]
# ## A helper that runs one variant

# %%
def run(label, **overrides):
    cfg = copy.deepcopy(BASE)
    for key, val in overrides.items():
        cfg[key] = val
    try:
        res = pci.solve(cfg)
    except (NotImplementedError, RuntimeError) as exc:
        print(f"{label:<44s} FAILED: {str(exc)[:90]}")
        return None
    T = cfg["time"]["T"]
    errs = {p: 100.0 * (res.final(p) - TRUE[p]) / TRUE[p] for p in cfg["unknowns"]}
    x_rmse = np.mean([res.rmse([f"x{d}"], start=T / 2)[f"x{d}"] for d in range(1, 7)])
    worst = max(errs, key=lambda p: abs(errs[p]))
    print(f"{label:<44s} {res.runtime:6.1f}s  mean x-RMSE {x_rmse:.2e}  max |param err| {abs(errs[worst]):5.2f}% ({worst})  "
          + "  ".join(f"{p}={errs[p]:+.1f}%" for p in errs))
    return res


# %% [markdown]
# ## Schedules, filters, integrators

# %%
print(f"6-DOF paper system, T = {BASE['time']['T']} s, dt = {BASE['time']['dt']}\n")
print(f"{'configuration':<44s} {'time':>7s}")
baseline = run("paper baseline: 3x2-DOF, UKF, Jacobi")
run("3x2-DOF, UKF, Gauss-Seidel", schedule="gauss_seidel")
run("3x2-DOF, UKF, AB2", schedule="ab2")
run("3x2-DOF, UKF, Jacobi x2 inner iterations", schedule={"type": "jacobi", "iterations": 2})
run("3x2-DOF, CKF, Jacobi", filters="ckf")
run("3x2-DOF, EKF, Jacobi", filters="ekf")
run("3x2-DOF, mixed UKF/CKF/EKF, Jacobi", filters={"S1": "ukf", "S2": "ckf", "S3": "ekf"})
run("3x2-DOF, UKF, Jacobi, Euler integrator", integrator="euler")
run("3x2-DOF, UKF, Jacobi, RK4 integrator", integrator="rk4");

# %% [markdown]
# ## Other decompositions of the same physical system
#
# Unknowns must stay internal to a subsystem, so each partition drops the unknowns
# that would sit on an interface spring.

# %%
unk = BASE["unknowns"]
run("2 subsystems [1,2,3],[4,5,6], UKF", subsystems=[[1, 2, 3], [4, 5, 6]],
    unknowns={k: v for k, v in unk.items() if k not in ("k34", "k6", "c6")})
run("uneven [1],[2,3],[4,5,6], UKF, Gauss-Seidel", subsystems=[[1], [2, 3], [4, 5, 6]], schedule="gauss_seidel",
    sensors=["a1", "a2", "a3", "a4", "a5", "a6"],
    noise_std={"a1": 3e-2, "a2": 3e-2, "a3": 1e-1, "a4": 1e-1, "a5": 3e-2, "a6": 3e-2},
    unknowns={k: v for k, v in unk.items() if k not in ("k3", "c3", "k34", "m3")})
run("monolithic (1 subsystem), UKF", subsystems=[[1, 2, 3, 4, 5, 6]]);

# %% [markdown]
# ## The baseline's parameter trajectories

# %%
baseline.plot_parameters();
