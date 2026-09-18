"""Canonical testbed of the paper: 4-DOF chain split into two 2-DOF subsystems.

Both subsystems run a UKF, exchange the interface force by Jacobi message
passing, and subsystem S2 estimates the unknown stiffness k4 from a single
boundary acceleration sensor.
"""

import matplotlib.pyplot as plt

import ci

# 1) physical system (truth)
chain = ci.MassSpringChain.uniform(4, mass=500.0, k=50_000.0, c=300.0)
dt, T = 1e-3, 5.0
loads = {d: {"type": "random", "std": 400.0, "seed": d} for d in range(1, 5)}  # white-noise force on every mass
truth = chain.simulate(loads, dt=dt, T=T, x0={"x1": 0.01, "v1": 0.01})
data = chain.measure(truth, ["a1", "a4"], noise_std=1e-3, seed=123)

# 2) decomposition + local estimators
partition = [[1, 2], [3, 4]]
system = chain.decompose(
    partition,
    unknowns={"k4": {"initial": 30_000.0, "std": 50_000.0}},
    sensors=["a1", "a4"],
    noise_std=1e-3,
    filters="ukf",            # or "ckf", "ekf", "kf", or a list/dict per subsystem
    integrator="heun",
    schedule="jacobi",        # or "gauss_seidel"
    x0={"x1": 0.01, "v1": 0.01},
    state_var=1e-4,
    process_var=1e-18,
    r_inflation=100.0,        # R = (0.1)^2 = 1e-2 as in the paper script, although the sensor noise std is 1e-3
)
print(system.describe())

# 3) solve the inverse problem
truth_dict = {**truth.as_dict(), **chain.interface_forces(truth, partition)}
results = system.estimate(data, loads={f"f{d}": v for d, v in loads.items()}, dt=dt, T=T,
                          truth=truth_dict, progress=True)
print(results.summary(start=1.0))

results.plot_states(["S1.x2", "S2.x3", "S1.v2", "S2.v3"])
results.plot_parameters()
results.plot_messages()
plt.show()

# ---- the same problem from its YAML twin (01_four_dof_jacobi_ukf.yaml) ----------------------
import os
yaml_results = ci.solve(os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_four_dof_jacobi_ukf.yaml"))
print(f"\nYAML twin: k4 final = {yaml_results.final('k4'):.1f}")
