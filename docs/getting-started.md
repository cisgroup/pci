# Get started

Install `pci`, run the paper's 4-DOF testbed end to end, and learn the two ways of
defining a problem. Prefer runnable notebooks? The **[tutorials](tutorials/README.md)**
cover all of this in depth.

## Install

=== "pip"

    ```bash
    pip install "pci-inference[all]"      # numpy, scipy + matplotlib, pyyaml, pandas, sympy
    ```

=== "from GitHub"

    ```bash
    pip install "pci-inference[all] @ git+https://github.com/cisgroup/pci.git"
    ```

=== "development"

    ```bash
    git clone https://github.com/cisgroup/pci.git && cd pci
    pip install -e ".[dev]"               # all extras + pytest, ruff, jupytext, mkdocs
    pytest
    ```

The core needs only `numpy` and `scipy`. Extras: `plot` (matplotlib), `config` (pyyaml,
for YAML problems), `data` (pandas, for `Results.to_dataframe`), `symbolic` (sympy, for
equation strings), `all` (everything).

## Your first distributed estimation

A serial chain of four masses is split into two 2-DOF subsystems. Each runs an unscented
Kalman filter, they exchange the interface force by Jacobi message passing, and the
second subsystem estimates the stiffness `k4` from one accelerometer.

```python
import pci

# 1) physical system and synthetic data (in practice: your measurements)
chain = pci.MassSpringChain.uniform(4, mass=500.0, k=50_000.0, c=300.0)
loads = {d: {"type": "random", "std": 400.0, "seed": d} for d in range(1, 5)}
truth = chain.simulate(loads, dt=1e-3, T=5.0)
data = chain.measure(truth, ["a1", "a4"], noise_std=1e-3, seed=123)

# 2) decomposition: subsystems, interfaces, local estimators, schedule
system = chain.decompose([[1, 2], [3, 4]],
                         unknowns={"k4": {"initial": 30_000.0, "std": 50_000.0}},
                         sensors=["a1", "a4"], filters="ukf", schedule="jacobi", r_inflation=100.0)
print(system.describe())

# 3) solve
results = system.estimate(data, loads={f"f{d}": v for d, v in loads.items()}, dt=1e-3, T=5.0,
                          truth=truth.as_dict())
print(results.summary(start=1.0))
results.plot_parameters()
```

```text
k4: final= 49871.3  std= 312.4  true= 50000  rmse= ...
```

`Results` holds the posterior mean and standard deviation of every state and parameter
of every subsystem, the message time histories and, if a truth was attached, RMSE and
NRMSE metrics:

```python
results.state("k4"), results.std("k4"), results.final("k4")
results.messages["F_k3"]                   # interface force exchanged at every step
results.rmse(["x2", "x3", "k4"], start=1.0)
results.to_dataframe()                     # needs pandas
```

## Two ways to define a problem

=== "Declarative (YAML)"

    For the built-in mass-spring model, and for custom subsystems written as equation
    strings. Every key is documented on the [Configuration](configuration.md) page.

    ```yaml
    # problem.yaml
    model: mass_spring_chain
    masses: [500, 500, 500, 500]
    stiffness: [50000, 50000, 50000, 50000]   # spring i connects DOF i-1 (0 = ground) to DOF i
    damping: [300, 300, 300, 300]
    subsystems: [[1, 2], [3, 4]]
    filters: {S1: ukf, S2: ckf}
    schedule: {type: jacobi, iterations: 1}
    unknowns:
      k4: {initial: 30000, std: 50000}
    sensors: [a1, a4]
    noise_std: 1.0e-3
    loads:
      1: {type: random, std: 400, seed: 1}
    time: {dt: 1.0e-3, T: 5.0}
    truth: {seed: 123}                        # synthetic ground truth + noisy sensors
    prior: {state_var: 1.0e-4, process_var: 1.0e-18, r_inflation: 100}
    ```

    ```python
    results = pci.solve("problem.yaml")
    ```

=== "Programmatic (your equations)"

    Any continuous-time model `f(x, u, p, t) -> dx/dt` with a measurement `h(x, u, p, t)`
    is a subsystem. `p` is a plain dict of parameters, unknown ones included.

    ```python
    import numpy as np, pci

    def sdof(x, u, p, t):
        return np.array([x[1], (u[0] - p["k"] * x[0] - p["c"] * x[1]) / p["m"]])

    def h(x, u, p, t):
        return np.array([x[0]])

    A = pci.Subsystem("A", ["x", "v"], ["f"], sdof, h, parameters={"m": 1.0, "c": 0.2},
                      unknowns={"k": pci.Unknown(initial=6.0, std=3.0)}, filter="ukf",
                      P0=1e-4, Q=1e-10, R=1e-6, measured=["xa"])
    B = pci.Subsystem("B", ["x", "v"], ["f"], sdof, h, parameters={"m": 2.0, "k": 15.0, "c": 0.3},
                      filter="kf", P0=1e-4, Q=1e-10, R=1e-6, measured=["xb"])

    system = pci.System([A, B],
                        pci.spring_damper("A", ["x", "v"], "f", "B", ["x", "v"], "f", k=4.0, c=0.1),
                        schedule="gauss_seidel")
    results = system.estimate({"xa": xa_data, "xb": xb_data}, dt=1e-3, n_steps=4000)
    ```

The built-in `MassSpringChain.decompose` produces exactly these objects from a partition
of the DOFs, so the two routes can be mixed.

!!! tip "Mean-only messages need an inflated measurement covariance"
    The incoming interface force carries an error the local filter does not know about.
    With `R` equal to the raw sensor noise the coupled filters over-trust sensors at the
    interface and the message loop can diverge. `r_inflation` (default 10) inflates `R`;
    see [Concepts](concepts.md#mean-only-messages-and-r-inflation) for tuning guidance.

## Next steps

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } __Tutorials__

    ---

    From the 4-DOF testbed to Van der Pol and Bouc-Wen subsystems, as runnable notebooks.

    [:octicons-arrow-right-24: Start the tutorials](tutorials/README.md)

-   :material-chart-timeline-variant:{ .lg .middle } __Case studies__

    ---

    Chains of 9 to 40 DOF with a different filter, sensor set and task per subsystem.

    [:octicons-arrow-right-24: Browse case studies](case-studies/README.md)

-   :material-lightbulb-on:{ .lg .middle } __Concepts__

    ---

    Nodes, edges, messages, schedules; why mean-only messages need care; what is next.

    [:octicons-arrow-right-24: Read Concepts](concepts.md)

-   :material-api:{ .lg .middle } __API reference__

    ---

    Every public class and function, generated from the docstrings.

    [:octicons-arrow-right-24: Browse the API](reference/index.md)

</div>
