import numpy as np
import pytest

from navsim.kf_altitude import AltitudeKF, van_loan
from navsim.runner import dead_reckoning, simulate_and_run
from navsim.sensors import AccelConfig, BaroConfig, simulate_accel
from navsim.trajectory import fixed_wing_vertical_profile


@pytest.fixture(scope="module")
def profile():
    return fixed_wing_vertical_profile(seed=0)


def test_truth_is_self_consistent(profile):
    """Integrating the truth acceleration must give back v and h."""
    dt = profile.dt
    v = profile.v[0] + np.concatenate([[0], np.cumsum(0.5 * (profile.a[1:] + profile.a[:-1]) * dt)])
    h = profile.h[0] + np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
    assert np.max(np.abs(v - profile.v)) < 1e-3
    assert np.max(np.abs(h - profile.h)) < 1e-2


def test_perfect_accel_dead_reckoning(profile):
    """A noise-free, bias-free accelerometer integrated alone stays on the truth."""
    cfg = AccelConfig(noise_density=0.0, bias_sigma0=0.0, bias_rw=0.0)
    acc = simulate_accel(profile, cfg, np.random.default_rng(0))
    h, v = dead_reckoning(profile, acc.a)
    assert np.max(np.abs(h - profile.h)) < 1e-2
    assert np.max(np.abs(v - profile.v)) < 1e-6


def test_van_loan_double_integrator():
    """Closed form for a white-noise acceleration model."""
    q, dt = 0.3, 0.1
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    L = np.array([[0.0], [1.0]])
    Phi, Qd = van_loan(A, L, np.array([[q]]), dt)
    np.testing.assert_allclose(Phi, [[1, dt], [0, 1]], atol=1e-12)
    np.testing.assert_allclose(Qd, q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]), rtol=1e-9)


@pytest.mark.parametrize("drift", [0.0, 1.5])
def test_covariance_stays_symmetric_positive(profile, drift):
    accel, baro = AccelConfig(), BaroConfig(drift_sigma=drift)
    r = simulate_and_run(profile, accel, baro, accel, baro, np.random.default_rng(3))
    P = r.P[::500]
    np.testing.assert_allclose(P, np.transpose(P, (0, 2, 1)), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_bias_is_estimated(profile):
    accel, baro = AccelConfig(), BaroConfig(drift_sigma=0.0)
    r = simulate_and_run(profile, accel, baro, accel, baro, np.random.default_rng(4))
    err = r.x[-1, 2] - r.accel.bias[-1]
    assert abs(err) < 3 * np.sqrt(r.P[-1, 2, 2])
    assert np.sqrt(r.P[-1, 2, 2]) < 0.2 * accel.bias_sigma0


def _mean_nees(profile, baro_truth, baro_model, runs=12):
    accel = AccelConfig()
    rng = np.random.default_rng(10)
    return np.mean([
        simulate_and_run(profile, accel, baro_truth, accel, baro_model, rng).nees_hv.mean()
        for _ in range(runs)
    ])


def test_filter_is_consistent_when_model_matches(profile):
    """NEES on [h, v] should average to 2 (its dimension)."""
    assert 1.6 < _mean_nees(profile, BaroConfig(drift_sigma=0.0), BaroConfig(drift_sigma=0.0)) < 2.5
    assert 1.6 < _mean_nees(profile, BaroConfig(), BaroConfig()) < 2.5


def test_unmodelled_drift_makes_filter_overconfident(profile):
    assert _mean_nees(profile, BaroConfig(), BaroConfig(drift_sigma=0.0)) > 50


def test_filter_rejects_mismatched_rate(profile):
    with pytest.raises(ValueError):
        simulate_accel(profile, AccelConfig(rate_hz=200.0), np.random.default_rng(0))


def test_kf_dimension_follows_model():
    assert AltitudeKF(AccelConfig(), BaroConfig(drift_sigma=0.0), 0.01).n == 3
    assert AltitudeKF(AccelConfig(), BaroConfig(), 0.01).n == 4
