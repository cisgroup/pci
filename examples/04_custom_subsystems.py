"""Writing your own subsystems with the generic API.

Two single-degree-of-freedom oscillators, one of them with a cubic (Duffing)
spring, coupled by a linear spring-damper. Subsystem A estimates its unknown
linear stiffness with an EKF, subsystem B tracks its states with a UKF.
The interface force is exchanged by Gauss-Seidel message passing.
"""

import matplotlib.pyplot as plt
import numpy as np

import ci


def duffing(x, u, p, t):
    """x = [q, v]; p holds m, k, k3, c; u[0] is the total force on the mass."""
    q, v = x
    return np.array([v, (u[0] - p["k"] * q - p["k3"] * q**3 - p["c"] * v) / p["m"]])


def linear_sdof(x, u, p, t):
    q, v = x
    return np.array([v, (u[0] - p["k"] * q - p["c"] * v) / p["m"]])


def measure_accel(x, u, p, t):
    return duffing(x, u, p, t)[1:]  # acceleration of A


def measure_disp(x, u, p, t):
    return x[:1]


A = ci.Subsystem("A", states=["q", "v"], inputs=["f"], dynamics=duffing, measurement=measure_accel,
                 parameters={"m": 1.0, "k3": 50.0, "c": 0.3},
                 unknowns={"k": ci.Unknown(initial=5.0, std=5.0)},
                 filter="ekf", x0={"q": 0.3}, P0=1e-4, Q=1e-10, R=1e-4, measured=["a_A"])
B = ci.Subsystem("B", states=["q", "v"], inputs=["f"], dynamics=linear_sdof, measurement=measure_disp,
                 parameters={"m": 2.0, "k": 20.0, "c": 0.4},
                 filter="ukf", P0=1e-4, Q=1e-10, R=1e-6, measured=["q_B"])
system = ci.System([A, B],
                   ci.spring_damper("A", ["q", "v"], "f", "B", ["q", "v"], "f", k=6.0, c=0.2, name="F_AB"),
                   schedule="gauss_seidel")
print(system.describe())

# --- monolithic truth with the true k = 12 ------------------------------------------
dt, T = 1e-3, 20.0
N = int(T / dt)
t = dt * np.arange(N + 1)
p_true = {"m": 1.0, "k": 12.0, "k3": 50.0, "c": 0.3}
p_B = {"m": 2.0, "k": 20.0, "c": 0.4}
force = ci.HarmonicLoad(amplitude=2.0, frequency=0.7).sample(t[:-1])


def rhs(z, k):
    Fb = 6.0 * (z[0] - z[2]) + 0.2 * (z[1] - z[3])
    dA = duffing(z[:2], [force[k] - Fb], p_true, 0.0)
    dB = linear_sdof(z[2:], [Fb], p_B, 0.0)
    return np.concatenate([dA, dB])


Z = np.zeros((N + 1, 4)); Z[0] = [0.3, 0.0, 0.0, 0.0]
acc_A = np.zeros(N + 1)
for k in range(N):
    Z[k + 1] = ci.heun(lambda z, tt: rhs(z, k), Z[k], t[k], dt)
    acc_A[k + 1] = rhs(Z[k + 1], k)[1]
rng = np.random.default_rng(0)
data = {"a_A": acc_A + rng.normal(0, 1e-2, N + 1), "q_B": Z[:, 2] + rng.normal(0, 1e-3, N + 1)}

results = system.estimate(data, loads={"A.f": force}, dt=dt, n_steps=N,
                          truth={"A.q": Z[:, 0], "B.q": Z[:, 2], "k": 12.0}, progress=True)
print(results.summary(start=5.0))
results.plot_states(["A.q", "B.q", "k"])
plt.show()

# ---- the same problem from its YAML twin (04_custom_subsystems.yaml, equations as strings) ---
import os
yaml_results = ci.solve(os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_custom_subsystems.yaml"))
print(f"\nYAML twin: k final = {yaml_results.final('k'):.3f} (true 12.0)")
