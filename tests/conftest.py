import pytest

import pci


@pytest.fixture(scope="session")
def four_dof():
    """Canonical 4-DOF testbed: two 2-DOF subsystems, unknown k4, sensors a1 and a4."""
    chain = pci.MassSpringChain.uniform(4, mass=500.0, k=50_000.0, c=300.0)
    dt, T = 1e-3, 2.0
    loads = {d: {"type": "random", "std": 400.0, "seed": 100 + d} for d in range(1, 5)}
    truth = chain.simulate(loads, dt=dt, T=T, x0={"x1": 0.01, "v1": 0.01}, integrator="euler")
    data = chain.measure(truth, ["a1", "a4"], noise_std=1e-3, seed=7)
    port_loads = {f"f{d}": v for d, v in loads.items()}
    truth_dict = {**truth.as_dict(), **chain.interface_forces(truth, [[1, 2], [3, 4]])}
    return dict(chain=chain, dt=dt, T=T, loads=loads, port_loads=port_loads, truth=truth, data=data, truth_dict=truth_dict)
