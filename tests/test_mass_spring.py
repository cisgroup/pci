import numpy as np
import pytest

import ci


def test_matrices_match_hand_built_chain():
    chain = ci.MassSpringChain([500.0, 500.0, 500.0, 500.0], k=[50e3] * 4, c=[300.0] * 4)
    M, K, C = chain.matrices()
    K_ref = 50e3 * np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]], float)
    assert np.allclose(K, K_ref)
    assert np.allclose(C, K_ref * 300.0 / 50e3)
    assert np.allclose(M, 500.0 * np.eye(4))


def test_forward_jacobi_matches_monolithic_euler(four_dof):
    d = four_dof
    system = d["chain"].decompose([[1, 2], [3, 4]], integrator="euler", schedule="jacobi",
                                  x0={"x1": 0.01, "v1": 0.01})
    res = system.simulate(loads=d["port_loads"], dt=d["dt"], T=d["T"], truth=d["truth_dict"])
    for name in ["x1", "x2", "x3", "x4", "v1", "v4"]:
        assert res.rmse([name])[name] < 1e-12


def test_forward_gauss_seidel_close_to_monolithic(four_dof):
    d = four_dof
    system = d["chain"].decompose([[1, 2], [3, 4]], integrator="euler", schedule="gauss_seidel",
                                  x0={"x1": 0.01, "v1": 0.01})
    res = system.simulate(loads=d["port_loads"], dt=d["dt"], T=d["T"], truth=d["truth_dict"])
    for name in ["x1", "x2", "x3", "x4"]:
        assert res.rmse([name])[name] < 1e-2 * np.ptp(d["truth"].signal(name))
    # the message value is the interface force F = k3 (x2 - x3) + c3 (v2 - v3)
    assert res.rmse(["F_k3"])["F_k3"] < 0.05 * np.std(d["truth_dict"]["F_k3"])


def test_forward_ab2_beats_jacobi_with_heun(four_dof):
    d = four_dof
    truth = d["chain"].simulate(d["loads"], dt=d["dt"], T=d["T"], x0={"x1": 0.01, "v1": 0.01}, integrator="heun")
    errs = {}
    for sched in ["jacobi", "ab2"]:
        system = d["chain"].decompose([[1, 2], [3, 4]], integrator="heun", schedule=sched, x0={"x1": 0.01, "v1": 0.01})
        res = system.simulate(loads=d["port_loads"], dt=d["dt"], T=d["T"], truth=truth.as_dict())
        errs[sched] = np.mean(list(res.rmse(["x1", "x2", "x3", "x4"]).values()))
    assert errs["ab2"] < 0.1 * errs["jacobi"]
    assert errs["jacobi"] < 1e-2 * np.ptp(truth.signal("x2"))


@pytest.mark.parametrize("filt", ["ukf", "ckf", "ekf"])
@pytest.mark.parametrize("schedule", ["jacobi", "gauss_seidel", "ab2"])
def test_parameter_estimation_recovers_k4(four_dof, filt, schedule):
    d = four_dof
    system = d["chain"].decompose(
        [[1, 2], [3, 4]], unknowns={"k4": {"initial": 30_000.0, "std": 50_000.0}},
        sensors=["a1", "a4"], noise_std=1e-3, filters=filt, integrator="euler", schedule=schedule,
        x0={"x1": 0.01, "v1": 0.01}, state_var=1e-4, process_var=1e-18, r_inflation=1.0,
    )
    res = system.estimate(d["data"], loads=d["port_loads"], dt=d["dt"], T=d["T"], truth=d["truth_dict"])
    assert abs(res.final("k4") - 50_000.0) / 50_000.0 < 0.01
    assert res.rmse(["x2"], start=0.5)["x2"] < 5e-4
    assert res.std("k4")[-1] < res.std("k4")[0]


def test_paper_six_dof_heterogeneous_filters_and_uneven_partition():
    """Paper 6-DOF system (script 02) with mixed filters, then re-partitioned into 1-, 2- and 3-DOF subsystems."""
    from ci.models import paper

    chain = paper.six_dof_chain()
    dt, T = 2e-3, 6.0
    loads = {4: {"type": "random", "std": 1500.0, "seed": 123}}
    truth = chain.simulate(loads, dt=dt, T=T)
    port_loads = {"f4": loads[4]}

    sensors = ["a2", "a3", "a4", "a5"]
    data = chain.measure(truth, sensors, noise_std=1e-3, seed=1)
    system = chain.decompose([[1, 2], [3, 4], [5, 6]],
                             unknowns={"k3": {"initial": 0.7 * 45_000, "std": 1e4}, "k34": {"initial": 0.7 * 40_000, "std": 1e4}},
                             sensors=sensors, noise_std={"a2": 3e-2, "a3": 1e-1, "a4": 1e-1, "a5": 3e-2},
                             filters={"S1": "ukf", "S2": "ckf", "S3": "ekf"}, integrator="heun",
                             schedule={"type": "jacobi", "iterations": 2}, process_var=1e-11, r_inflation=1.0)
    assert [s.filter.name for s in system.subsystems.values()] == ["ukf", "ckf", "ekf"]
    assert system.neighbours("S2") == ["S1", "S3"]
    res = system.estimate(data, loads=port_loads, dt=dt, T=T, truth=truth.as_dict())
    assert abs(res.final("k3") - 45_000.0) / 45_000.0 < 0.05
    assert abs(res.final("k34") - 40_000.0) / 40_000.0 < 0.05
    assert set(res.messages) == {"F_k4", "F_k7"}

    # same physical system, different decomposition: subsystems of size 1, 2 and 3
    sensors = ["a1", "a2", "a3", "a4", "a6"]
    data = chain.measure(truth, sensors, noise_std=1e-3, seed=1)
    system = chain.decompose([[1], [2, 3], [4, 5, 6]], unknowns={"k9": {"initial": 0.7 * 40_000, "std": 1e4}},
                             sensors=sensors, noise_std=3e-2, filters="ukf", integrator="heun",
                             schedule="gauss_seidel", process_var=1e-11, r_inflation=1.0)
    assert [s.nx for s in system.subsystems.values()] == [2, 4, 6]
    res = system.estimate(data, loads=port_loads, dt=dt, T=T, truth=truth.as_dict())
    assert set(res.messages) == {"F_k3", "F_k5", "F_k34"}
    assert abs(res.final("k9") - 40_000.0) / 40_000.0 < 0.05


def test_divergence_raises_clear_error():
    """With R equal to the raw sensor noise the mean-only message loop can diverge; it must fail loudly."""
    chain = ci.MassSpringChain.uniform(6, mass=500.0, k=50_000.0, c=300.0)
    dt, T = 1e-3, 1.5
    loads = {6: {"type": "random", "std": 300.0, "seed": 3}}
    truth = chain.simulate(loads, dt=dt, T=T)
    data = chain.measure(truth, ["a2", "a4", "a6"], noise_std=1e-3, seed=1)
    system = chain.decompose([[1, 2], [3, 4], [5, 6]], sensors=["a2", "a4", "a6"], filters="ukf", r_inflation=1.0)
    with pytest.raises(RuntimeError, match="diverged"):
        system.estimate(data, loads={"f6": loads[6]}, dt=dt, T=T)


def test_unknown_interface_parameter_is_rejected():
    chain = ci.MassSpringChain.uniform(4, 500.0, 50e3, 300.0)
    with pytest.raises(NotImplementedError):
        chain.decompose([[1, 2], [3, 4]], unknowns={"k3": 40e3})


def test_partition_validation():
    chain = ci.MassSpringChain.uniform(4, 500.0, 50e3, 300.0)
    with pytest.raises(ValueError):
        chain.decompose([[1, 2], [3]])
    with pytest.raises(ValueError):
        chain.decompose([[1, 2], [2, 3, 4]])
