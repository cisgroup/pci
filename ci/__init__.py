"""compositional-inference (``import ci``)

Probabilistic compositional inference for systems of coupled subsystems:
each subsystem keeps its own model and estimator, and coupling is handled by
message passing across interfaces.

Quick start::

    import ci

    chain = ci.MassSpringChain.uniform(4, mass=500.0, k=50_000.0, c=300.0)
    truth = chain.simulate({1: {"type": "random", "std": 400, "seed": 1}}, dt=1e-3, T=5.0)
    data = chain.measure(truth, ["a1", "a4"], noise_std=1e-3, seed=2)

    system = chain.decompose([[1, 2], [3, 4]], unknowns={"k4": 30_000.0},
                             sensors=["a1", "a4"], filters="ukf", schedule="jacobi")
    results = system.estimate(data, loads={"f1": {"type": "random", "std": 400, "seed": 1}},
                              dt=1e-3, T=5.0, truth=truth.as_dict())
    print(results.summary())
"""

from .config import Problem, build, load_config, solve
from .filters import (
    CubatureKalmanFilter,
    ExtendedKalmanFilter,
    Filter,
    LinearKalmanFilter,
    UnscentedKalmanFilter,
    available_filters,
    get_filter,
    register_filter,
)
from .integrators import available_integrators, euler, get_integrator, heun, rk4
from .interface import Interface, custom_interface, spring_damper, spring_damper_law
from .loads import (
    ArrayLoad,
    ConstantLoad,
    ElCentroLoad,
    HarmonicLoad,
    ImpulseLoad,
    Load,
    RandomLoad,
    RecordedLoad,
    StepLoad,
    ZeroLoad,
    as_load,
)
from .models import MassSpringChain, Spring, Truth
from .results import Results
from .schedules import AdamsBashforth2, GaussSeidel, Jacobi, Schedule, get_schedule
from .subsystem import Subsystem, Unknown
from .system import System

KF = LinearKalmanFilter
EKF = ExtendedKalmanFilter
UKF = UnscentedKalmanFilter
CKF = CubatureKalmanFilter

__version__ = "0.1.0"

__all__ = [
    "Subsystem", "Unknown", "System", "Results",
    "Interface", "spring_damper", "spring_damper_law", "custom_interface",
    "Schedule", "Jacobi", "GaussSeidel", "AdamsBashforth2", "get_schedule",
    "Filter", "KF", "EKF", "UKF", "CKF",
    "LinearKalmanFilter", "ExtendedKalmanFilter", "UnscentedKalmanFilter", "CubatureKalmanFilter",
    "get_filter", "register_filter", "available_filters",
    "euler", "heun", "rk4", "get_integrator", "available_integrators",
    "Load", "ZeroLoad", "ConstantLoad", "HarmonicLoad", "RandomLoad", "StepLoad", "ImpulseLoad",
    "RecordedLoad", "ElCentroLoad", "ArrayLoad", "as_load",
    "MassSpringChain", "Spring", "Truth",
    "solve", "build", "load_config", "Problem",
]
