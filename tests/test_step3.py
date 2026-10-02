import numpy as np
import pytest

from navsim import fusion, gnss, imu, strapdown, trajectory3d
from navsim.eskf import ErrorStateEKF
from navsim.rotations import (dcm_from_quat, quat_conj, quat_from_euler, quat_from_rotvec,
                              quat_mul, quat_rotate, rotvec_from_quat, skew)


@pytest.fixture(scope="module")
def mission():
    return trajectory3d.generate(trajectory3d.MISSION)


def test_dcm_and_skew_helpers():
    rng = np.random.default_rng(0)
    q = quat_from_euler(*rng.uniform(-1, 1, (3, 10)))
    v = rng.normal(size=(10, 3))
    np.testing.assert_allclose(np.einsum("rij,rj->ri", dcm_from_quat(q), v), quat_rotate(q, v), atol=1e-12)
    a, b = rng.normal(size=(2, 10, 3))
    np.testing.assert_allclose(np.einsum("rij,rj->ri", skew(a), b), np.cross(a, b), atol=1e-12)


def _error(truth, nom):
    """truth minus nominal, attitude as rotvec of q_t * conj(q_n)."""
    (pt, vt, qt, bgt, bat), (pn, vn, qn, bgn, ban) = truth, nom
    return np.concatenate([pt - pn, vt - vn, rotvec_from_quat(quat_mul(qt, quat_conj(qn))),
                           bgt - bgn, bat - ban])


@pytest.mark.parametrize("start_s", [30.0, 130.0])  # straight climb-out, then mid-loiter
def test_transition_matrix_matches_nonlinear_propagation(mission, start_s):
    """Perturb the true state by a small error, propagate truth and nominal
    through the nonlinear strapdown, and compare the resulting error with the
    product of the filter's transition matrices. This is what checks the signs
    and the blocks of F."""
    k0 = int(start_s / mission.dt)
    n = 40  # 0.2 s
    rng = np.random.default_rng(1)
    dx0 = rng.normal(size=15) * np.array([0.5] * 3 + [0.05] * 3 + [2e-3] * 3 + [2e-4] * 3 + [2e-2] * 3)

    # truth: state on the reference trajectory, true biases bg_t, ba_t
    bg_t, ba_t = np.array([3e-4, -2e-4, 1e-4]), np.array([0.03, -0.02, 0.04])
    pt, vt, qt = mission.p_n[k0], mission.v_n[k0], mission.q_nb[k0]
    # nominal = truth minus the error
    pn, vn = pt - dx0[0:3], vt - dx0[3:6]
    qn = quat_mul(quat_from_rotvec(-dx0[6:9]), qt)
    bgn, ban = bg_t - dx0[9:12], ba_t - dx0[12:15]

    ekf = ErrorStateEKF(imu.IMUConfig(), gnss.GNSSConfig(), mission.dt, runs=1)
    ekf.initialise(pn[None], vn[None], qn[None], np.eye(15), bgn[None], ban[None])
    Phi_total = np.eye(15)
    for k in range(k0, k0 + n):
        dth_m = mission.dtheta[k] + bg_t * mission.dt
        dv_m = mission.dvel[k] + ba_t * mission.dt
        prev = (mission.dtheta[k - 1], mission.dvel[k - 1])
        pt, vt, qt = strapdown.step(pt, vt, qt, mission.dtheta[k], mission.dvel[k], *prev,
                                    mission.dt, "full", first=(k == k0))
        ekf.predict(dth_m[None], dv_m[None])
        Phi_total = ekf.Phi[0] @ Phi_total
    actual = _error((pt, vt, qt, bg_t, ba_t), (ekf.p[0], ekf.v[0], ekf.q[0], ekf.bg[0], ekf.ba[0]))
    predicted = Phi_total @ dx0
    scale = np.abs(predicted) + np.array([1e-3] * 6 + [1e-6] * 3 + [1e-9] * 6)
    assert np.max(np.abs(actual - predicted) / scale) < 0.05


def test_filter_tracks_truth_with_perfect_imu(mission):
    """Perfect IMU, perfect initial attitude: the filter stays at GNSS-averaged
    accuracy and does not invent biases."""
    short = trajectory3d.generate(trajectory3d.FlightPlan(duration=60.0))
    # "perfect" up to tiny values: an exactly zero bias variance makes the
    # bias covariance block singular, and the NEES needs its inverse.
    tiny = imu.IMUConfig(gyro_noise=1e-9, gyro_bias0=1e-9, gyro_bias_rw=1e-12,
                         accel_noise=1e-9, accel_bias0=1e-9, accel_bias_rw=1e-12)
    log = fusion.run(short, tiny, gnss.GNSSConfig(),
                     fusion.InitConfig(tilt_sigma_deg=1e-3, yaw_sigma_deg=1e-3), 2,
                     np.random.default_rng(2))
    assert np.abs(log.err[:, -1, 0:3]).max() < 1.5
    assert np.abs(np.degrees(log.err[:, -1, 6:9])).max() < 0.05


def test_filter_is_consistent_on_a_short_flight():
    """Mean NEES per degree of freedom close to 1 and mean NIS close to 6, with a
    small initial heading error (large ones break the linearisation before the
    first turn, see README)."""
    plan = trajectory3d.FlightPlan(duration=100.0, bank_deg=((40, 44, 30), (70, 74, -30)))
    traj = trajectory3d.generate(plan)
    log = fusion.run(traj, imu.IMUConfig(), gnss.GNSSConfig(),
                     fusion.InitConfig(yaw_sigma_deg=1.0), 12, np.random.default_rng(3))
    total = log.nees["total"][:, 1:].mean() / 15
    assert 0.8 < total < 1.25
    assert 5.5 < np.nanmean(log.nis) < 6.5


def test_heading_converges_only_after_a_turn(mission):
    log = fusion.run(trajectory3d.generate(trajectory3d.FlightPlan(
        duration=110.0, bank_deg=((70, 74, 25), (80, 84, -25)))), imu.IMUConfig(), gnss.GNSSConfig(),
        fusion.InitConfig(), 4, np.random.default_rng(4))
    yaw_sigma = np.degrees(log.sigma[:, :, 8].mean(axis=0))
    before = yaw_sigma[np.searchsorted(log.t, 65.0)]
    after = yaw_sigma[np.searchsorted(log.t, 95.0)]
    assert before > 3.0 and after < 1.0
