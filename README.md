# pci: Probabilistic Compositional Inference

`pci` (`import pci`) is a Python library for probabilistic compositional
inference: distributed state and parameter estimation for coupled engineered
systems.

A complex system is decomposed into **subsystems**. Each subsystem keeps its
own physics, its own unknown parameters and its own estimator (Kalman-type
filter). Subsystems talk to their neighbours through **interfaces** that carry
**messages** (for now: mean values of interface forces). A **schedule**
decides how messages are exchanged (Jacobi = parallel, Gauss-Seidel =
sequential). No global augmented state or covariance is ever assembled.

The library implements the framework of

> Ghorbani, E. and Hackl, J. (2026). *Subsystem Structure as an Inferential
> Resource for Coupled Engineered Systems.* arXiv:2605.27544.

## Install

The import name is `pci`. The distribution name on PyPI is `pci-inference`
(the name `pci` is taken by an unrelated package).

```bash
pip install "pci-inference[all]"                              # from PyPI (once released)
pip install "pci-inference[all] @ git+https://github.com/cisgroup/pci.git"   # latest main
```

Extras: `plot` (matplotlib), `config` (pyyaml), `data` (pandas), `symbolic`
(sympy), `all` (everything). For development, clone the repo and install it
in editable mode so that changes to the source take effect without reinstalling:

```bash
git clone https://github.com/cisgroup/pci.git && cd pci
pip install -e ".[dev]"        # all extras + pytest
pytest                         # run the test suite
```

## Two ways to define a problem

### 1. Declarative (YAML / dict) for the built-in mass-spring model

```yaml
# problem.yaml
model: mass_spring_chain
masses: [500, 500, 500, 500]
stiffness: [50000, 50000, 50000, 50000]   # spring i connects DOF i-1 (0 = ground) to DOF i
damping: [300, 300, 300, 300]
subsystems: [[1, 2], [3, 4]]              # any grouping of DOFs: [[1], [2, 3], [4]] works too
filters: {S1: ukf, S2: ckf}               # kf | ekf | ukf | ckf, per subsystem
integrator: heun                          # euler | heun | rk4, or per subsystem {S1: rk4, S2: heun}
schedule: {type: jacobi, iterations: 1}   # jacobi | gauss_seidel
messages: mean
unknowns:
  k4: {initial: 30000, std: 50000, process_std: 0}
sensors: [a1, a4]                         # a = acceleration, x = displacement, v = velocity
noise_std: 1.0e-3
loads:
  1: {type: random, std: 400, seed: 1}
  4: {type: harmonic, amplitude: 200, frequency: 1.5}
time: {dt: 1.0e-3, T: 5.0}
initial_state: {x1: 0.01, v1: 0.01}
truth: {seed: 123}                        # synthetic ground truth + noisy sensors are generated
prior: {state_var: 1.0e-4, process_var: 1.0e-18, r_inflation: 10}
```

```python
import pci
results = pci.solve("problem.yaml")
print(results.summary())
results.plot_parameters(); results.plot_states(["S1.x2", "S2.x3"]); results.plot_messages()
```

### 2. Programmatic, with your own equations

```python
import numpy as np, pci

def sdof(x, u, p, t):                      # x = states, u = inputs, p = parameters (dict), t = time
    return np.array([x[1], (u[0] - p["k"] * x[0] - p["c"] * x[1]) / p["m"]])

def h(x, u, p, t):                         # what the sensor sees
    return np.array([x[0]])

A = pci.Subsystem("A", ["x", "v"], ["f"], sdof, h, parameters={"m": 1.0, "c": 0.2},
                 unknowns={"k": pci.Unknown(initial=6.0, std=3.0)}, filter="ukf",
                 x0={"x": 0.5}, P0=1e-4, Q=1e-10, R=1e-6, measured=["xa"])
B = pci.Subsystem("B", ["x", "v"], ["f"], sdof, h, parameters={"m": 2.0, "k": 15.0, "c": 0.3},
                 filter="kf", P0=1e-4, Q=1e-10, R=1e-6, measured=["xb"])

system = pci.System([A, B], pci.spring_damper("A", ["x", "v"], "f", "B", ["x", "v"], "f", k=4.0, c=0.1),
                   schedule="gauss_seidel")
results = system.estimate({"xa": xa_data, "xb": xb_data}, dt=1e-3, n_steps=4000)
```

The built-in `MassSpringChain` produces exactly these objects from a
partition of the DOFs (`chain.decompose([[1, 2], [3, 4]], ...)`), so the two
routes can be mixed.

## What you have to specify

| Level | Input | Options / notes |
|---|---|---|
| **System** | number of subsystems and which DOFs (or states) belong to each | any sizes, e.g. 1-, 2- and 3-DOF subsystems side by side |
| | interfaces between subsystems | `spring_damper(...)` or any `Interface` with a custom law `m = law(s_sender, s_receiver, t)` |
| | message-passing schedule | `jacobi` (parallel), `gauss_seidel` (sequential, optional `order`), `ab2` (extrapolated); `iterations` per step |
| | message type | `mean` (mean-value messages). `mean_variance` is reserved for the next step |
| **Subsystem** | state names, input-port names | e.g. `["x1","x2","v1","v2"]`, `["f1","f2"]` |
| | dynamics `f(x, u, p, t) -> dx/dt` and measurement `h(x, u, p, t) -> y` | continuous time; the library discretises them |
| | known parameters `p` | dict |
| | unknowns | `Unknown(initial, std, process_std, lower, upper)`; estimated as random walks |
| | filter | `kf`, `ekf`, `ukf` (`kappa`), `ckf`; register your own with `@pci.register_filter` |
| | integrator | `euler` (1st order), `heun` (RK2, default), `rk4` (4th order), or a custom step function; one per subsystem |
| | initial state `x0`, prior `P0`, process noise `Q`, measurement noise `R` | scalars, variance vectors or full matrices |
| | `measured` names | which columns of the data belong to this subsystem |
| **Loads** | external force per input port | `random`, `harmonic`, `step`, `impulse`, `constant`, `recorded`, `elcentro`, arrays; combinable with `+` and `*` |
| **Time** | `dt`, `T` or `n_steps` | |
| **Data** | measurements `{name: array}` | or generate synthetic data with `chain.simulate` + `chain.measure` |

## What you get back

`Results` holds the posterior mean and standard deviation of every state and
parameter of every subsystem, the total input applied to each port, the
message time histories and, if a truth was attached, RMSE/NRMSE metrics.

```python
results.state("k4"), results.std("k4"), results.final("k4")
results.messages["F_k3"]                   # interface force exchanged at every step
results.rmse(["x2", "x3", "k4"], start=1.0)
results.summary(); results.to_dataframe()
```

`System.simulate(...)` solves the coupled *forward* (direct) problem with the
same message passing, for comparing Jacobi and Gauss-Seidel against a
monolithic solution.

## Examples

Every example exists twice: as a Python script and as a YAML twin solved
with `pci.solve("<name>.yaml")`. Each script ends by running its twin and
printing the same estimates.

| script + YAML | shows |
|---|---|
| `01_four_dof_jacobi_ukf` | the paper's canonical 4-DOF testbed (unknown k4, sensors a1/a4); `02_solve_from_yaml.py` solves the YAML alone |
| `03_forward` | direct problem, `mode: simulate`, Jacobi vs Gauss-Seidel vs AB2 against the monolithic model |
| `04_custom_subsystems` | user-written nonlinear (Duffing) subsystem, EKF/UKF mix; `model: custom` with equations as strings |
| `05_six_dof_three_subsystems` | the paper's 6-DOF system, 7 unknowns, three different filters, explicit spring topology |
| `06_paper_six_dof` | the 6-DOF system under other schedules, filters, integrators and partitions, all as YAML overrides |
| `07_case_09dof`, `08_case_16dof`, `09_case_20dof`, `10_case_40dof` | case studies: 3x3, 4x4, 5x4 and 10x4 DOF chains with mixed sensors, noise, tasks, filters and integrators |

Custom equations (`model: custom`, needs `sympy`): states, inputs and
parameters are plain symbols, `t` is time; interface laws use `s_<var>` and
`r_<var>` for the sender's and receiver's interface variables. The ground truth
is the same coupled model simulated with the true parameter values
(`truth.parameters`), so no monolithic model has to be written.

## Practical notes

* **Mean-only messages need an inflated measurement covariance.** The incoming
  interface force carries an error the local filter does not know about.
  Sensors at interface DOFs see that error directly; with `R` equal to the raw
  sensor noise the coupled filters over-trust them and the message loop can
  diverge within a few steps (the run stops with a clear `diverged` error).
  The paper scripts inflate `R` by 10-30x; `MassSpringChain.decompose` does
  the same through `r_inflation` (default 10). Uncertainty-carrying messages
  (`mean_variance`, next step) will replace this heuristic.
* Unknown masses, stiffnesses and dampings in `MassSpringChain.decompose` are
  clamped to a small positive value inside the physics (`positive_floor`), like
  `k_eff = max(k, 1)` in the paper scripts; the filter state itself is not clipped.
* Unknown *interface* parameters (a spring that crosses a partition boundary)
  are not supported yet: that is where learned interface laws (SINDy) come in.
* Schedules: `jacobi` (parallel, lagged messages), `gauss_seidel` (sequential),
  `ab2` (parallel, interface variables extrapolated with Adams-Bashforth 2;
  the most accurate forward coupling with Heun). `iterations > 1` refines the
  coupling with a trapezoidal message (average of the lagged and the updated
  interface force).

## Case studies (`examples/07_case_09dof.py` ... `10_case_40dof.py`)

Serial chains with non-uniform masses, stiffnesses and dampings, random
forces on every mass (El Centro on the last mass of the 20-DOF case), mixed
Heun/RK4/Euler integrators, Jacobi messages (Gauss-Seidel for 20 DOF), `R`
inflated 10x, 20 s horizon at 1 ms. Each case is one script plus one YAML
twin that reproduces it to machine precision. Every subsystem has its own sensors (displacement `x`
and/or acceleration `a`, one or two per subsystem, noise std from 1e-4 m to
2e-2 m/s²), its own task and its own filter.

| case | subsystems | filters | unknowns | stiffness / mass error | damping error | state NRMSE | run time (sequential) |
|---|---|---|---|---|---|---|---|
| 9 DOF | 3 x 3-DOF | KF, UKF, CKF | k5, k9, c9 | < 0.4 % | 2.7 % | < 0.1 % | 11 s |
| 16 DOF | 4 x 4-DOF | UKF, KF, CKF, UKF | k2, k11, c11, k15 | < 2.7 % (k15 10.6 %, one sensor) | 1.4 % | < 0.4 % | 20 s |
| 20 DOF | 5 x 4-DOF | KF, UKF, CKF, EKF, UKF | k7, k10, c10, m15, k19 | < 2 % | 38 % (c10, one sensor) | < 0.3 % | 26 s |
| 40 DOF | 10 x 4-DOF | KF, UKF, CKF, EKF (cycled) | 9 parameters | < 6 % (mass 0.4 %) | 13-78 % (single-sensor subsystems) | < 0.3 % | 42 s |

The centralized UKF on the full augmented state agrees with the distributed
estimates (9 DOF: k5 -0.44 % vs -0.47 %) at 2-3x the sequential cost and
without the one-core-per-subsystem speed-up. Dampings identified from a single
displacement sensor converge slowly, as in the paper's 6-DOF study (50 s).
Jacobi, Gauss-Seidel and AB2 give the same parameters on the 9-DOF case;
Jacobi and AB2 give the smallest state errors.

### Tuning guidance for mean-only messages

* `r_inflation` (default 10): inflate the measurement covariance to account for
  the message error. 10-30 was robust in all multi-subsystem case studies.
  The paper's canonical 4-DOF testbed (free-floating second subsystem, one
  accelerometer, 40 % biased stiffness guess) is the delicate one: the
  paper's own value (100, `R = 1e-2`) and the raw sensor noise (1) both work,
  intermediate values can push the estimate to its lower bound during the
  first second and the collapsed covariance never recovers.
* `process_std` on an unknown keeps its covariance from collapsing so the
  filter can re-learn after a bad transient; use it for slowly varying or
  poorly excited parameters.
* `process_var` on the states: 1e-12 to 1e-10 worked for the chains here;
  the paper's scripts range from 1e-18 to 1e-8.

## Roadmap

1. Uncertainty-carrying messages (`mean_variance`): interface-force variance
   injected as process noise in the receiver, incremental update rule.
2. Learned interface laws with SINDy (pysindy) as drop-in `Interface.law`.
3. More local estimators: particle filter, PINN / neural surrogates, WLS/WNLS.
4. Optimisation-based (batch) estimators and further built-in models
   (Kuramoto power networks, turbine modules) from the paper.

## Layout

```
pci/
  subsystem.py    Subsystem, Unknown                (node of the graph)
  interface.py    Interface, spring_damper           (edge / interface law)
  schedules.py    Jacobi, GaussSeidel                (message passing)
  system.py       System.estimate / System.simulate  (the solver loop)
  filters/        kf, ekf, ukf, ckf + registry
  integrators.py  euler, heun, rk4
  loads.py        load library
  models/         MassSpringChain, Spring, paper case-study definitions
  config.py       pci.solve(yaml|dict)
  results.py      Results, metrics, plots
```
