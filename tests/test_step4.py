import dataclasses

import numpy as np
import pytest

from navsim import fusion, gnss, imu, trajectory3d
from navsim.eskf import ErrorStateEKF


def test_correlated_error_statistics():
    cfg = gnss.GNSSConfig(corr_sigma_h=1.4, corr_sigma_v=2.8, corr_tau=60.0)
    ce = gnss.CorrelatedError(cfg, np.random.default_rng(0), runs=4000)
    first = ce.next().copy()
    second = ce.next().copy()
    for _ in range(200):
        last = ce.next()
    # stationary from the first draw on
    np.testing.assert_allclose(first.std(axis=0), cfg.corr_sigma, rtol=0.05)
    np.testing.assert_allclose(last.std(axis=0), cfg.corr_sigma, rtol=0.05)
    # one-fix correlation exp(-0.2 / 60)
    rho = [np.corrcoef(first[:, i], second[:, i])[0, 1] for i in range(3)]
    np.testing.assert_allclose(rho, np.exp(-0.2 / 60.0), atol=0.002)


def test_realistic_config_keeps_the_total_error():
    np.testing.assert_allclose(np.sqrt(np.diag(gnss.REALISTIC.R_total)[:3]),
                               np.sqrt(np.diag(gnss.GNSSConfig().R)[:3]), rtol=0.02)


def test_initial_covariance_correlates_position_and_gnss_bias():
    P = fusion.initial_covariance(imu.IMUConfig(), gnss.REALISTIC, fusion.InitConfig(), True)
    assert P.shape == (18, 18)
    assert P[0, 15] == pytest.approx(-gnss.REALISTIC.corr_sigma_h**2)
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_gnss_bias_states_follow_gauss_markov():
    ekf = ErrorStateEKF(imu.IMUConfig(), gnss.REALISTIC, 0.005, runs=1, model_gnss_bias=True)
    assert ekf.n == 18
    assert ekf.gnss_phi == pytest.approx(np.exp(-0.005 / 60.0))
    # H adds the bias to the position measurement
    np.testing.assert_array_equal(ekf.H[0:3, 15:18], np.eye(3))


@pytest.fixture(scope="module")
def short_turning():
    plan = trajectory3d.FlightPlan(duration=90.0, bank_deg=((20, 24, 30), (40, 44, -30),
                                                            (60, 64, -30), (75, 79, 30)))
    return trajectory3d.generate(plan)


def test_delayed_mode_without_latency_equals_plain_filter(short_turning):
    cfg = gnss.GNSSConfig()
    a = fusion.run(short_turning, imu.IMUConfig(), cfg, fusion.InitConfig(), 2,
                   np.random.default_rng(5), latency_mode="ignore")
    b = fusion.run(short_turning, imu.IMUConfig(), cfg, fusion.InitConfig(), 2,
                   np.random.default_rng(5), latency_mode="delayed")
    np.testing.assert_allclose(a.err, b.err, atol=1e-12)


def test_delayed_horizon_handles_latency(short_turning):
    """With 150 ms of latency, fusing the fix as if it were current corrupts the
    attitude in turns; the delayed horizon keeps the filter consistent."""
    cfg = dataclasses.replace(gnss.GNSSConfig(), latency_s=0.15)
    kwargs = dict(init=fusion.InitConfig(yaw_sigma_deg=1.0), runs=10)
    ign = fusion.run(short_turning, imu.IMUConfig(), cfg, rng=np.random.default_rng(6),
                     latency_mode="ignore", **kwargs)
    dly = fusion.run(short_turning, imu.IMUConfig(), cfg, rng=np.random.default_rng(6),
                     latency_mode="delayed", **kwargs)
    m = ign.t > 30
    assert ign.nees["attitude"][:, m].mean() / 3 > 5
    assert 0.6 < dly.nees["attitude"][:, dly.t > 30].mean() / 3 < 1.5
    # the delivered (real-time) position is better with the delayed horizon
    rms = lambda lg: np.sqrt((lg.out_err[:, lg.t > 30, 0:2] ** 2).mean())  # noqa: E731
    assert rms(dly) < 0.5 * rms(ign)
