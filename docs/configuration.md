# Configuration (YAML)

`pci.solve(config)` accepts a path to a YAML or JSON file, or a dictionary. Two model
families exist: the built-in mass-spring chain and custom subsystems written as equation
strings. `pci.load_config(path)` returns the dictionary so that a study can override
keys before solving; `pci.build(config)` instantiates everything without solving.

Relative `path` entries of recorded loads are resolved against the directory of the
configuration file.

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
| **Data** | measurements `{name: array}` | or generate synthetic data with a `truth` block |

## `model: mass_spring_chain`

```yaml
model: mass_spring_chain
mode: estimate                             # estimate (default) | simulate (forward problem only)
masses: [500, 500, 500, 500]
stiffness: [50000, 50000, 50000, 50000]   # spring i connects DOF i-1 (0 = ground) to DOF i
damping: [300, 300, 300, 300]
# or an explicit topology instead of stiffness/damping:
# springs: [{a: ground, b: 1, k: 5.0e4, c: 300}, {a: 1, b: 2, k: 4.5e4, c: 320}, {a: 3, b: 4, k: 4.0e4, c: 0, name: "34"}]
subsystems: [[1, 2], [3, 4]]               # DOF groups -> S1, S2 (any sizes)
names: [S1, S2]                            # optional
filters: ukf                               # or [ukf, ckf] or {S1: ukf, S2: ekf}
integrator: heun                           # euler | heun | rk4; or {S1: rk4, S2: heun}
schedule: {type: jacobi, iterations: 1}    # jacobi | gauss_seidel | ab2; or just the name
messages: mean
unknowns:
  k4: {initial: 30000, std: 50000, process_std: 0}
sensors: [a1, a4]                          # a = acceleration, x = displacement, v = velocity
noise_std: 1.0e-3                          # or {a1: 1.0e-3, a4: 5.0e-3}
loads:                                     # external forces per DOF
  1: {type: random, std: 400, seed: 1}
  4: {type: harmonic, amplitude: 200, frequency: 1.5}
time: {dt: 1.0e-3, T: 5.0}
initial_state: {x1: 0.01, v1: 0.01}        # filter initial guess
truth: {seed: 123, initial_state: {x1: 0.01, v1: 0.01}, integrator: heun}
#   truth.noise_std: sensor noise used to generate the synthetic data (default: noise_std above)
prior: {state_var: 1.0e-4, process_var: 1.0e-12, r_inflation: 10}
```

Parameter names follow the spring index (`k3`, `c3`), the mass index (`m3`) or the
spring's `name` (`k34`). Unknowns must be internal to one subsystem: a mass, or a
spring whose two ends lie in the same subsystem or on the ground.

Without a `truth` block nothing is simulated; pass the recorded sensor signals as
`pci.solve(cfg, measurements={name: array})` with `n_steps + 1` samples each.

## `model: custom`

Needs `sympy`. States, inputs and parameters are plain symbols, `t` is time. The ground
truth is the same coupled model simulated with the true parameter values under
`truth.parameters`, so no monolithic model has to be written.

```yaml
model: custom
subsystems:
  A:
    states: [q, v]
    inputs: [f]                            # total force on the mass (external load + interface message)
    parameters: {m: 1.0, k3: 50.0, c: 0.3}
    unknowns: {k: {initial: 5.0, std: 5.0}}
    equations:                             # d/dt of each state
      q: v
      v: (f - k*q - k3*q**3 - c*v) / m
    measurements:
      a_A: (f - k*q - k3*q**3 - c*v) / m   # sensor name -> expression
    noise_std: {a_A: 1.0e-2}
    filter: ekf
    integrator: heun
    x0: {q: 0.3}
    P0: 1.0e-4
    Q: 1.0e-10
  B:
    states: [q, v]
    inputs: [f]
    parameters: {m: 2.0, k: 20.0, c: 0.4}
    equations: {q: v, v: (f - k*q - c*v) / m}
    measurements: {q_B: q}
    noise_std: {q_B: 1.0e-3}
    filter: ukf
interfaces:
  - {type: spring_damper, a: A, a_vars: [q, v], a_port: f, b: B, b_vars: [q, v], b_port: f, k: 6.0, c: 0.2, name: F_AB}
  # equivalent explicit form (one direction shown); s_<var> and r_<var> are the sender's and receiver's variables
  # - {type: custom, sender: B, receiver: A, sender_vars: [q, v], receiver_vars: [q, v], target: f,
  #    law: "k*(r_q - s_q) + c*(r_v - s_v)", constants: {k: 6.0, c: 0.2}, sign: -1, name: F_AB}
schedule: gauss_seidel
loads: {A.f: {type: harmonic, amplitude: 2.0, frequency: 0.7}}
time: {dt: 1.0e-3, T: 20.0}
truth:
  parameters: {A.k: 12.0}                  # true values of the unknowns (and any other override)
  x0: {A.q: 0.3}
  schedule: ab2                            # coupled forward simulation used as ground truth
  seed: 0
prior: {r_inflation: 1}
```

## Loads

| `type` | keys | notes |
|---|---|---|
| `random` | `std`, `seed` | white noise, one sample per step |
| `harmonic` | `amplitude`, `frequency`, `phase` | sine |
| `step` | `amplitude`, `t_on`, `t_off` | |
| `impulse` | `amplitude`, `t0`, `duration` | |
| `constant` | `value` | |
| `recorded` | `path`, `scale`, `loop`, `stretch` | time / value columns |
| `elcentro` | `path`, `scale`, `loop`, `stretch` | the 1940 El Centro record shipped with the tutorials (`examples/data/elcentro.mat`); `path` defaults to `$PCI_ELCENTRO` |

In Python any `Load` can be combined with `+` and `*`, and a plain array of length
`n_steps` is accepted wherever a load is expected.
