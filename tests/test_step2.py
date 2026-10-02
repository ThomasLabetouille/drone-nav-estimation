import numpy as np
import pytest

from navsim import imu, strapdown, trajectory3d
from navsim.rotations import (attitude_error, euler_from_quat, quat_conj, quat_from_euler,
                              quat_from_rotvec, quat_mul, quat_rotate, rotvec_from_quat)

G = trajectory3d.G


@pytest.fixture(scope="module")
def mission():
    return trajectory3d.generate(trajectory3d.MISSION)


@pytest.fixture(scope="module")
def straight():
    return trajectory3d.generate(trajectory3d.STRAIGHT_LEVEL)


# --- rotations --------------------------------------------------------------

def test_quaternion_roundtrips():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1.0, (50, 3))
    np.testing.assert_allclose(rotvec_from_quat(quat_from_rotvec(r)), r, atol=1e-12)
    phi, theta, psi = rng.uniform(-1.4, 1.4, (3, 50))
    out = euler_from_quat(quat_from_euler(phi, theta, psi))
    np.testing.assert_allclose(np.stack(out), np.stack([phi, theta, psi]), atol=1e-12)


def test_quat_rotate_matches_euler_convention():
    """Yaw of +90 deg turns the body x axis (forward) to the east."""
    q = quat_from_euler(0.0, 0.0, np.pi / 2)
    np.testing.assert_allclose(quat_rotate(q, np.array([1.0, 0, 0])), [0, 1, 0], atol=1e-12)
    # Pitch up: forward axis points up, i.e. negative down component.
    q = quat_from_euler(0.0, 0.3, 0.0)
    assert quat_rotate(q, np.array([1.0, 0, 0]))[2] < 0


def test_small_rotvec_is_stable():
    q = quat_from_rotvec(np.array([1e-12, 0.0, 0.0]))
    assert np.all(np.isfinite(q)) and abs(np.linalg.norm(q) - 1) < 1e-15


# --- trajectory -------------------------------------------------------------

def test_specific_force_matches_acceleration_and_attitude(mission):
    """f_b computed analytically must equal C_bn (a_n - g)."""
    f_check = quat_rotate(quat_conj(mission.q_nb), mission.a_n - np.array([0, 0, G]))
    np.testing.assert_allclose(f_check, mission.f_b, atol=1e-10)


def test_specific_force_in_steady_turn(mission):
    """Level turn at constant speed: the horizontal acceleration is g tan(phi).
    With a nonzero pitch (trim angle), the body y axis is not exactly
    perpendicular to it, so a small side force g sin(phi)(1 - cos(theta))
    remains: about 7 mm/s^2 at 30 deg of bank and 3 deg of pitch."""
    k = np.searchsorted(mission.t, 130.0)  # middle of the loiter
    phi, theta = mission.euler[k, 0], mission.euler[k, 1]
    assert np.degrees(phi) == pytest.approx(30.0)
    assert mission.f_b[k, 1] == pytest.approx(G * np.sin(phi) * (1 - np.cos(theta)), rel=1e-9)
    assert mission.f_b[k, 2] == pytest.approx(
        -G * (np.sin(phi) ** 2 / np.cos(phi) + np.cos(phi) * np.cos(theta)), rel=1e-9)
    assert abs(mission.f_b[k, 1]) < 0.01


def test_velocity_is_derivative_of_position(mission):
    dt = mission.dt
    v_mid = np.diff(mission.p_n, axis=0) / dt
    v_avg = 0.5 * (mission.v_n[1:] + mission.v_n[:-1])
    np.testing.assert_allclose(v_mid, v_avg, atol=1e-4)


def test_straight_level_imu(straight):
    """Level, unaccelerated: no rotation, specific force is -g along the
    lift direction (body z tilted by the trim angle)."""
    assert np.abs(straight.omega_b).max() < 1e-12
    alpha = np.radians(straight.plan.alpha_trim_deg)
    np.testing.assert_allclose(straight.f_b[0], [G * np.sin(alpha), 0, -G * np.cos(alpha)], atol=1e-12)


# --- strapdown --------------------------------------------------------------

def test_perfect_imu_reproduces_mission(mission):
    sol = strapdown.integrate(mission.p_n[0], mission.v_n[0], mission.q_nb[0],
                              mission.dtheta, mission.dvel, mission.dt, "full")
    assert np.linalg.norm(sol.p_n[-1] - mission.p_n[-1]) < 0.05
    assert np.degrees(np.linalg.norm(attitude_error(mission.q_nb, sol.q_nb), axis=1)).max() < 1e-6


def test_each_correction_reduces_the_error(mission):
    """Guards the signs of the coning / sculling terms: a wrong sign would
    make the 'full' variant worse than 'rotcomp'."""
    errs = []
    for m in strapdown.METHODS:
        sol = strapdown.integrate(mission.p_n[0], mission.v_n[0], mission.q_nb[0],
                                  mission.dtheta, mission.dvel, mission.dt, m)
        errs.append(np.linalg.norm(sol.p_n[-1] - mission.p_n[-1]))
    assert errs[0] > errs[1] > errs[2]


def test_batch_integration_matches_single_runs(straight):
    rng = np.random.default_rng(3)
    meas = imu.simulate(straight.dtheta[:2000], straight.dvel[:2000], imu.IMUConfig(), rng, runs=3)
    batch = strapdown.integrate(straight.p_n[0], straight.v_n[0], straight.q_nb[0],
                                meas.dtheta, meas.dvel, straight.dt)
    for r in range(3):
        single = strapdown.integrate(straight.p_n[0], straight.v_n[0], straight.q_nb[0],
                                     meas.dtheta[r], meas.dvel[r], straight.dt)
        np.testing.assert_allclose(batch.p_n[r], single.p_n, atol=1e-9)


def test_imu_noise_statistics(straight):
    """Increment noise must have std N*sqrt(dt) (density convention)."""
    cfg = imu.IMUConfig(gyro_bias0=0.0, gyro_bias_rw=0.0, accel_bias0=0.0, accel_bias_rw=0.0)
    meas = imu.simulate(straight.dtheta, straight.dvel, cfg, np.random.default_rng(1))
    dt = straight.dt
    assert np.std(meas.dtheta - straight.dtheta) == pytest.approx(cfg.gyro_noise * np.sqrt(dt), rel=0.02)
    assert np.std(meas.dvel - straight.dvel) == pytest.approx(cfg.accel_noise * np.sqrt(dt), rel=0.02)


def test_dead_reckoning_matches_error_budget(straight):
    """Monte-Carlo RMS horizontal error vs analytic budget at 60 s."""
    runs = 150
    k = int(round(60.0 / straight.dt))
    cfg = imu.IMUConfig()
    meas = imu.simulate(straight.dtheta[:k], straight.dvel[:k], cfg, np.random.default_rng(11), runs=runs)
    nav = strapdown.integrate(straight.p_n[0], straight.v_n[0], straight.q_nb[0],
                              meas.dtheta, meas.dvel, straight.dt)
    err = nav.p_n[:, -1, :2] - straight.p_n[k, :2]
    rms_axis = np.sqrt(np.mean(err**2) )
    budget = strapdown.error_budget(60.0, cfg)["total"]
    half = 2.5 / np.sqrt(2 * 2 * runs)  # two axes pooled, ~99 % interval
    assert abs(rms_axis / budget - 1) < half


def test_heading_error_alone_gives_no_position_error_in_straight_flight(straight):
    """A pure yaw-gyro bias rotates the heading estimate, but in unaccelerated
    flight the specific force is vertical, so the position is unaffected:
    heading is not observable from the IMU alone in this phase."""
    n = len(straight.dtheta)
    bias = np.zeros((n, 3))
    # yaw rate about the NED down axis, expressed in body (tilted by the trim angle)
    alpha = np.radians(straight.plan.alpha_trim_deg)
    bias[:] = np.radians(0.05) * np.array([-np.sin(alpha), 0.0, np.cos(alpha)])
    sol = strapdown.integrate(straight.p_n[0], straight.v_n[0], straight.q_nb[0],
                              straight.dtheta + bias * straight.dt, straight.dvel, straight.dt)
    heading_err = np.degrees(attitude_error(straight.q_nb[-1], sol.q_nb[-1])[2])
    assert heading_err == pytest.approx(0.05 * straight.t[-1], rel=1e-3)
    assert np.linalg.norm(sol.p_n[-1] - straight.p_n[-1]) < 0.5
