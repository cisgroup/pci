"""Direct (forward) problem: decomposed simulation vs the monolithic model.

Compares Jacobi, Gauss-Seidel and AB2 message passing on the 4-DOF chain,
first through the Python API and then from the YAML twin (03_forward.yaml)
with the `schedule` key overridden.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import ci

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEDULES = ["jacobi", "gauss_seidel", "ab2"]

# ---- Python API ----------------------------------------------------------------------------
chain = ci.MassSpringChain.uniform(4, mass=500.0, k=50_000.0, c=300.0)
dt, T = 1e-3, 5.0
loads = {d: {"type": "random", "std": 50.0, "seed": 123 + d} for d in range(1, 5)}
truth = chain.simulate(loads, dt=dt, T=T, x0={"x1": 0.01, "v1": 0.01}, integrator="heun")
port_loads = {f"f{d}": v for d, v in loads.items()}

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for schedule, style in zip(SCHEDULES, ["--", "-.", ":"]):
    system = chain.decompose([[1, 2], [3, 4]], integrator="heun", schedule=schedule, x0={"x1": 0.01, "v1": 0.01})
    res = system.simulate(loads=port_loads, dt=dt, T=T, truth=truth.as_dict())
    err = res.rmse(["x1", "x2", "x3", "x4"])
    print(f"{schedule:>13s}: RMSE vs monolithic", {k: f"{v:.2e}" for k, v in err.items()})
    axes[0].plot(res.t, res["x2"], style, label=schedule)
    axes[1].plot(res.t, np.abs(res.errors("x2")), style, label=schedule)
axes[0].plot(truth.t, truth.signal("x2"), "k", lw=1, label="monolithic")
axes[0].set_ylabel("x2 [m]"); axes[0].legend(frameon=False)
axes[1].set_ylabel("|error x2|"); axes[1].set_yscale("log"); axes[1].set_xlabel("time [s]")
fig.tight_layout()

# ---- YAML twin -----------------------------------------------------------------------------
cfg = ci.load_config(os.path.join(HERE, "03_forward.yaml"))
for schedule in SCHEDULES:
    cfg["schedule"] = schedule
    res = ci.solve(cfg)
    print(f"YAML {schedule:>13s}: mean x-RMSE vs monolithic {np.mean(list(res.rmse(['x1', 'x2', 'x3', 'x4']).values())):.2e}")
plt.show()
