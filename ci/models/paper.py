"""The case-study systems of the paper, ready to instantiate.

These reproduce the mass-spring configurations of the paper scripts so that
results can be compared against the published figures and so that other
decompositions, filters and schedules can be tried on the same systems.
"""

from __future__ import annotations

from typing import Dict

from .mass_spring import GROUND, MassSpringChain, Spring


def four_dof_chain() -> MassSpringChain:
    """Canonical testbed (scripts 01/04/05): four equal masses in a serial chain."""
    return MassSpringChain.uniform(4, mass=500.0, k=50_000.0, c=300.0)


FOUR_DOF = dict(
    partition=[[1, 2], [3, 4]],
    sensors=["a1", "a4"],
    unknowns={"k4": {"initial": 30_000.0, "std": 50_000.0}},
    x0={"x1": 0.01, "v1": 0.01},
    dt=1e-3,
    T=10.0,
    load_std=400.0,
    noise_std=1e-3,
    state_var=1e-4,
    process_var=1e-18,
    R_std=1e-1,  # the scripts inflate R to 1e-2 (variance) although the noise std is 1e-3
)


def six_dof_chain(m_star: float = 100.0) -> MassSpringChain:
    """6-DOF system of scripts 02/03: three 2-DOF subsystems S1 -- S3 -- S2 (here S1, S2, S3 in DOF order)."""
    springs = [
        Spring(GROUND, 1, 50_000.0, 300.0),          # k1, c1
        Spring(GROUND, 2, 40_000.0, 350.0),          # k2, c2
        Spring(1, 2, 45_000.0, 320.0),               # k3, c3   unknown (S1)
        Spring(2, 3, 30_000.0, 400.0),               # k4, c4   interface S1-S2
        Spring(3, 4, 25_000.0, 480.0),               # k5, c5
        Spring(GROUND, 4, 55_000.0, 700.0),          # k6, c6   unknown (S2)
        Spring(4, 5, 30_000.0, 350.0),               # k7, c7   interface S2-S3
        Spring(GROUND, 5, 45_000.0, 460.0),          # k8, c8
        Spring(5, 6, 40_000.0, 480.0),               # k9, c9   unknown (S3)
        Spring(3, 4, 40_000.0, 0.0, name="34"),      # k34      parallel spring, unknown (S2)
    ]
    masses = [500.0, 500.0, 500.0 + m_star, 500.0, 500.0, 500.0]
    return MassSpringChain(masses, springs)


def six_dof_unknowns(guess: float = 0.7) -> Dict[str, dict]:
    """Biased initial guesses (``guess`` x true) and the prior variances used in script 02."""
    true = dict(k3=45_000.0, c3=320.0, m3=600.0, k6=55_000.0, c6=700.0, k34=40_000.0, k9=40_000.0, c9=480.0)
    prior_var = dict(k3=1e8, c3=1e5, m3=1e4, k6=1e8, c6=1e3, k34=1e8, k9=1e8, c9=1e5)
    out = {}
    for name, val in true.items():
        # the script estimates the added mass m* = m3 - 500 with guess 0.7 m*; here the full m3 is the unknown
        init = 500.0 + guess * (val - 500.0) if name == "m3" else guess * val
        out[name] = {"initial": init, "std": prior_var[name] ** 0.5}
    return out


SIX_DOF = dict(
    partition=[[1, 2], [3, 4], [5, 6]],
    sensors=["a2", "a3", "a4", "a5"],
    # script 02 inflates the measurement covariances (R = 1e-3 for a2/a5, 1e-2 for a3/a4)
    # although the synthetic sensor noise std is 1e-3
    R_std={"a2": 1e-3 ** 0.5, "a3": 1e-2 ** 0.5, "a4": 1e-2 ** 0.5, "a5": 1e-3 ** 0.5},
    measurement_noise=1e-3,
    dt=2e-3,
    T=50.0,
    load_dof=4,
    elcentro_scale=3.0 * 500.0,
    state_var=1e-4,
    process_var={"S1": 1e-11, "S2": 1e-12, "S3": 1e-11},
)
