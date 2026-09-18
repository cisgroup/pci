import numpy as np
import pytest

import ci
from ci.filters import get_filter


def linear_oscillator(dt=1e-2, wn=2.0 * np.pi, zeta=0.05):
    A = np.array([[0.0, 1.0], [-wn**2, -2 * zeta * wn]])
    Ad = np.eye(2) + dt * A  # Euler discretization used by both truth and filter

    def f(x, u, t):
        return Ad @ x + dt * np.array([0.0, u[0]])

    def h(x, u, t):
        return x[:1]  # measure displacement

    return f, h


@pytest.mark.parametrize("name", ["kf", "ekf", "ukf", "ckf"])
def test_filters_track_linear_oscillator(name):
    rng = np.random.default_rng(0)
    f, h = linear_oscillator()
    filt = get_filter(name)
    Q = 1e-8 * np.eye(2)
    R = np.array([[1e-4]])
    x_true = np.array([1.0, 0.0])
    x, P = np.zeros(2), np.eye(2)
    errs = []
    for k in range(400):
        u = np.array([rng.normal(0, 0.5)])
        x_true = f(x_true, u, 0.0)
        y = h(x_true, u, 0.0) + rng.normal(0, 1e-2, size=1)
        res = filt.step(x, P, u, y, f, h, Q, R, 0.0)
        x, P = res.x, res.P
        if k > 200:
            errs.append(np.abs(x - x_true))
    errs = np.array(errs)
    assert errs[:, 0].mean() < 2e-2
    assert errs[:, 1].mean() < 2e-1
    assert np.allclose(P, P.T)
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_ukf_kappa_zero_equals_ckf():
    f, h = linear_oscillator()
    x, P = np.array([0.3, -0.2]), np.array([[2.0, 0.3], [0.3, 1.0]])
    u, y = np.array([0.1]), np.array([0.35])
    Q, R = 1e-6 * np.eye(2), np.array([[1e-3]])
    a = ci.UKF(kappa=0.0).step(x, P, u, y, f, h, Q, R, 0.0)
    b = ci.CKF().step(x, P, u, y, f, h, Q, R, 0.0)
    assert np.allclose(a.x, b.x)
    assert np.allclose(a.P, b.P)


def test_predict_only_when_no_measurement():
    f, h = linear_oscillator()
    x, P = np.array([1.0, 0.0]), 0.1 * np.eye(2)
    res = ci.UKF().step(x, P, np.zeros(1), None, f, h, 1e-6 * np.eye(2), np.eye(1), 0.0)
    assert np.allclose(res.x, f(x, np.zeros(1), 0.0), atol=1e-8)
    assert res.y_pred is None


def test_registry_and_aliases():
    assert isinstance(get_filter("unscented"), ci.UnscentedKalmanFilter)
    assert isinstance(get_filter(ci.CKF), ci.CubatureKalmanFilter)
    inst = ci.EKF()
    assert get_filter(inst) is inst
    with pytest.raises(ValueError):
        get_filter("particle")
