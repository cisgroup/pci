import numpy as np
import pytest

import pci


def test_loads_from_specs():
    t = np.arange(0, 1, 0.1)
    assert np.allclose(pci.as_load(5.0).sample(t), 5.0)
    assert np.allclose(pci.as_load({"type": "step", "amplitude": 2.0, "t_on": 0.5}).sample(t), np.where(t >= 0.5, 2.0, 0.0))
    h = pci.as_load({"type": "harmonic", "amplitude": 1.0, "frequency": 1.0}).sample(t)
    assert np.allclose(h, np.sin(2 * np.pi * t))
    r1 = pci.RandomLoad(1.0, seed=3).sample(t)
    r2 = pci.RandomLoad(1.0, seed=3).sample(t)
    assert np.allclose(r1, r2)
    combo = (pci.ConstantLoad(1.0) + pci.ConstantLoad(2.0)) * 2.0
    assert np.allclose(combo.sample(t), 6.0)
    rec = pci.RecordedLoad([0.0, 1.0], dt=0.5).sample(np.array([0.25]))
    assert np.allclose(rec, 0.5)


def test_schedule_resolution():
    assert isinstance(pci.get_schedule("jacobi"), pci.Jacobi)
    gs = pci.get_schedule({"type": "gauss_seidel", "iterations": 3, "order": ["B", "A"]})
    assert isinstance(gs, pci.GaussSeidel) and gs.iterations == 3
    assert gs.order(["A", "B"]) == ["B", "A"]
    assert isinstance(pci.get_schedule("ab2"), pci.AdamsBashforth2)
    with pytest.raises(ValueError):
        pci.get_schedule("runge")


def test_custom_subsystems_with_user_equations():
    """Two user-defined SDOF oscillators coupled by a spring, written with the generic API."""

    def sdof(x, u, p, t):
        return np.array([x[1], (u[0] - p["k"] * x[0] - p["c"] * x[1]) / p["m"]])

    def h_disp(x, u, p, t):
        return np.array([x[0]])

    A = pci.Subsystem("A", ["x", "v"], ["f"], sdof, h_disp, parameters={"m": 1.0, "c": 0.2},
                     unknowns={"k": pci.Unknown(initial=6.0, std=3.0)}, filter="ukf",
                     x0={"x": 0.5}, P0=1e-4, Q=1e-10, R=1e-6, measured=["xa"])
    B = pci.Subsystem("B", ["x", "v"], ["f"], sdof, h_disp, parameters={"m": 2.0, "k": 15.0, "c": 0.3},
                     filter="kf", x0={"x": 0.0}, P0=1e-4, Q=1e-10, R=1e-6, measured=["xb"])
    system = pci.System([A, B], pci.spring_damper("A", ["x", "v"], "f", "B", ["x", "v"], "f", k=4.0, c=0.1),
                       schedule="gauss_seidel")

    # ground truth from a monolithic integration with the true k = 10
    dt, N = 1e-3, 4000
    z = np.array([0.5, 0.0, 0.0, 0.0])
    X = np.zeros((N + 1, 4)); X[0] = z
    for k in range(N):
        Fb = 4.0 * (z[0] - z[2]) + 0.1 * (z[1] - z[3])
        a1 = (-10.0 * z[0] - 0.2 * z[1] - Fb) / 1.0
        a2 = (-15.0 * z[2] - 0.3 * z[3] + Fb) / 2.0
        k1 = np.array([z[1], a1, z[3], a2])
        zp = z + dt * k1
        Fb = 4.0 * (zp[0] - zp[2]) + 0.1 * (zp[1] - zp[3])
        k2 = np.array([zp[1], (-10.0 * zp[0] - 0.2 * zp[1] - Fb), zp[3], (-15.0 * zp[2] - 0.3 * zp[3] + Fb) / 2.0])
        z = z + 0.5 * dt * (k1 + k2)
        X[k + 1] = z
    rng = np.random.default_rng(0)
    data = {"xa": X[:, 0] + rng.normal(0, 1e-3, N + 1), "xb": X[:, 2] + rng.normal(0, 1e-3, N + 1)}
    res = system.estimate(data, dt=dt, n_steps=N, truth={"A.x": X[:, 0], "B.x": X[:, 2], "k": 10.0})
    assert abs(res.final("k") - 10.0) < 0.3
    assert res.rmse(["A.x", "B.x"], start=1.0)["B.x"] < 5e-3
    df = res.to_dataframe()
    assert "A.k" in df.columns and "msg:F[A.x-B.x]" in df.columns


def test_config_solve_roundtrip(tmp_path):
    cfg = {
        "model": "mass_spring_chain",
        "masses": [500, 500, 500, 500],
        "stiffness": [50000, 50000, 50000, 50000],
        "damping": [300, 300, 300, 300],
        "subsystems": [[1, 2], [3, 4]],
        "filters": ["ukf", "ckf"],
        "integrator": "euler",
        "schedule": {"type": "jacobi", "iterations": 1},
        "unknowns": {"k4": {"initial": 30000, "std": 50000}},
        "sensors": ["a1", "a4"],
        "noise_std": 1e-3,
        "loads": {d: {"type": "random", "std": 400, "seed": d} for d in range(1, 5)},
        "time": {"dt": 1e-3, "T": 1.5},
        "initial_state": {"x1": 0.01, "v1": 0.01},
        "truth": {"seed": 123, "initial_state": {"x1": 0.01, "v1": 0.01}},
        "prior": {"state_var": 1e-4, "process_var": 1e-18, "r_inflation": 1.0},
    }
    res = pci.solve(cfg)
    assert abs(res.final("k4") - 50000) / 50000 < 0.02
    pytest.importorskip("yaml")
    import yaml

    path = tmp_path / "problem.yaml"
    path.write_text(yaml.safe_dump(cfg))
    res2 = pci.solve(str(path))
    assert np.allclose(res2.state("k4"), res.state("k4"))


def test_integrators_orders_and_selection():
    # exact solution of x' = -x over one step
    rhs = lambda x, t: -x
    x0, dt = np.array([1.0]), 0.1
    exact = np.exp(-dt)
    errs = {n: abs(pci.get_integrator(n)(rhs, x0, 0.0, dt)[0] - exact) for n in pci.available_integrators()}
    assert errs["rk4"] < errs["heun"] < errs["euler"]
    assert errs["rk4"] < 1e-7
    assert pci.get_integrator(pci.rk4) is pci.rk4
    with pytest.raises(ValueError):
        pci.get_integrator("ab3")

    chain = pci.MassSpringChain.uniform(6, 500.0, 50e3, 300.0)
    system = chain.decompose([[1, 2], [3, 4], [5, 6]], integrator=["euler", "heun", "rk4"])
    assert [s.integrator.__name__ for s in system.subsystems.values()] == ["euler", "heun", "rk4"]
    system = chain.decompose([[1, 2], [3, 4], [5, 6]], integrator={"S3": "rk4"})
    assert [s.integrator.__name__ for s in system.subsystems.values()] == ["heun", "heun", "rk4"]
    with pytest.raises(ValueError):
        chain.decompose([[1, 2], [3, 4], [5, 6]], integrator=["euler", "heun"])


def test_custom_model_from_yaml_dict():
    pytest.importorskip("sympy")
    cfg = {
        "model": "custom",
        "subsystems": {
            "A": {"states": ["q", "v"], "inputs": ["f"], "parameters": {"m": 1.0, "c": 0.3},
                  "unknowns": {"k": {"initial": 6.0, "std": 4.0}},
                  "equations": {"q": "v", "v": "(f - k*q - c*v)/m"},
                  "measurements": {"qa": "q"}, "noise_std": {"qa": 1e-3},
                  "filter": "ukf", "x0": {"q": 0.5}, "P0": 1e-4, "Q": 1e-10},
            "B": {"states": ["q", "v"], "inputs": ["f"], "parameters": {"m": 2.0, "k": 15.0, "c": 0.4},
                  "equations": ["v", "(f - k*q - c*v)/m"], "measurements": {"qb": "q"}, "noise_std": {"qb": 1e-3},
                  "filter": "kf", "P0": 1e-4, "Q": 1e-10},
        },
        "interfaces": [{"type": "custom", "sender": "A", "receiver": "B", "sender_vars": ["q", "v"],
                        "receiver_vars": ["q", "v"], "target": "f", "law": "k*(s_q - r_q) + c*(s_v - r_v)",
                        "constants": {"k": 4.0, "c": 0.1}, "sign": 1, "name": "F_AB"},
                       {"type": "custom", "sender": "B", "receiver": "A", "sender_vars": ["q", "v"],
                        "receiver_vars": ["q", "v"], "target": "f", "law": "k*(r_q - s_q) + c*(r_v - s_v)",
                        "constants": {"k": 4.0, "c": 0.1}, "sign": -1, "name": "F_AB"}],
        "schedule": "gauss_seidel",
        "time": {"dt": 1e-3, "T": 4.0},
        "truth": {"parameters": {"A.k": 10.0}, "schedule": "ab2", "seed": 0},
        "prior": {"r_inflation": 1.0},
    }
    res = pci.solve(cfg)
    assert abs(res.final("k") - 10.0) < 0.3
    assert res.rmse(["B.q"], start=1.0)["B.q"] < 5e-3
    assert "F_AB" in res.messages and "F_AB" in res.truth
    prob = pci.build(cfg)
    assert prob.truth_system is not None and prob.mode == "estimate"


def test_simulate_mode_from_config():
    cfg = {"model": "mass_spring_chain", "mode": "simulate", "masses": [500] * 4, "stiffness": [5e4] * 4,
           "damping": [300] * 4, "subsystems": [[1, 2], [3, 4]], "schedule": "ab2", "integrator": "heun",
           "loads": {1: {"type": "random", "std": 50, "seed": 3}}, "time": {"dt": 1e-3, "T": 2.0},
           "initial_state": {"x1": 0.01}}
    res = pci.solve(cfg)
    assert res.mode == "simulate"
    assert np.mean(list(res.rmse(["x1", "x2", "x3", "x4"]).values())) < 1e-6


def test_symbolic_errors():
    pytest.importorskip("sympy")
    from pci.symbolic import compile_model

    with pytest.raises(NameError):
        compile_model(["v", "(f - kk*q)/m"], ["q", "v"], ["f"], ["m", "k"])
    with pytest.raises(ValueError):
        compile_model(["v"], ["q", "v"], ["q"], ["m"])
