"""Independent checks: each test compares the project's code with something
it does not share code with (another library, another integration method,
another formulation of the same estimator, or a closed-form result).

A test written next to the code it checks tends to share its mistakes: same
convention, same sign, same misunderstanding. These tests are built to avoid
that, by going through a different path to the same answer.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from scipy.stats import chi2, kstest

from navsim import fusion, gnss, imu, strapdown, trajectory3d
from navsim.eskf import ErrorStateEKF
from navsim.kf_altitude import AltitudeKF, input_matrix, van_loan
from navsim.rotations import (attitude_error, dcm_from_quat, euler_from_quat, quat_conj,
                              quat_from_euler, quat_from_rotvec, quat_mul, quat_rotate,
                              rotvec_from_quat)
from navsim.sensors import AccelConfig, BaroConfig
from navsim.trajectory3d import G, body_kinematics

RNG = np.random.default_rng(12345)


def same_quat(q1, q2, atol=1e-12):
    """Equal up to the sign ambiguity of quaternions."""
    s = np.sign(np.sum(q1 * q2, axis=-1, keepdims=True))
    np.testing.assert_allclose(q1 * s, q2, atol=atol)


# ---------------------------------------------------------------------------
# Rotations against scipy.spatial.transform
# ---------------------------------------------------------------------------

class TestRotationsAgainstScipy:
    angles = RNG.uniform([-np.pi, -1.5, -np.pi], [np.pi, 1.5, np.pi], (200, 3))  # roll, pitch, yaw

    def scipy_rot(self):
        # ZYX intrinsic = yaw, then pitch, then roll (scipy uses upper case for intrinsic)
        return Rotation.from_euler("ZYX", self.angles[:, ::-1])

    def test_euler_to_quaternion(self):
        q = quat_from_euler(*self.angles.T)
        xyzw = self.scipy_rot().as_quat()
        same_quat(q, np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=1))

    def test_quaternion_to_euler(self):
        q = quat_from_euler(*self.angles.T)
        np.testing.assert_allclose(np.stack(euler_from_quat(q), axis=1), self.angles, atol=1e-10)

    def test_rotate_vector_is_body_to_ned(self):
        v = RNG.normal(size=(200, 3))
        np.testing.assert_allclose(quat_rotate(quat_from_euler(*self.angles.T), v),
                                   self.scipy_rot().apply(v), atol=1e-12)

    def test_dcm(self):
        np.testing.assert_allclose(dcm_from_quat(quat_from_euler(*self.angles.T)),
                                   self.scipy_rot().as_matrix(), atol=1e-12)

    def test_product_is_composition(self):
        r1, r2 = Rotation.random(100, random_state=1), Rotation.random(100, random_state=2)
        to_wxyz = lambda r: np.roll(r.as_quat(), 1, axis=1)  # noqa: E731
        same_quat(quat_mul(to_wxyz(r1), to_wxyz(r2)), to_wxyz(r1 * r2))

    def test_rotation_vector_maps(self):
        rv = RNG.normal(size=(200, 3))
        rv *= (RNG.uniform(0, 3.1, 200) / np.linalg.norm(rv, axis=1))[:, None]
        same_quat(quat_from_rotvec(rv), np.roll(Rotation.from_rotvec(rv).as_quat(), 1, axis=1))
        np.testing.assert_allclose(rotvec_from_quat(quat_from_rotvec(rv)),
                                   Rotation.from_rotvec(rv).as_rotvec(), atol=1e-10)


# ---------------------------------------------------------------------------
# Step 1: discretisation and Kalman filter
# ---------------------------------------------------------------------------

def continuous_altitude_model(accel: AccelConfig, baro: BaroConfig):
    """The model of kf_altitude, typed again from its docstring, not imported."""
    A = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0], [0, 0, 0, -1 / baro.drift_tau]], float)
    B = np.array([[0], [1], [0], [0]], float)
    L = np.array([[0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    Qc = np.diag([accel.noise_density**2, accel.bias_rw**2, 2 * baro.drift_sigma**2 / baro.drift_tau])
    return A, B, L, Qc


def test_van_loan_matches_covariance_ode():
    """Qd from Van Loan vs direct integration of dP/dt = A P + P A^T + L Qc L^T."""
    accel, baro, dt = AccelConfig(), BaroConfig(), 0.01
    A, B, L, Qc = continuous_altitude_model(accel, baro)
    n = 4

    def rhs(_, y):
        Phi = y[: n * n].reshape(n, n)
        P = y[n * n: 2 * n * n].reshape(n, n)
        dPhi = A @ Phi
        dP = A @ P + P @ A.T + L @ Qc @ L.T
        dGam = Phi @ B
        return np.concatenate([dPhi.ravel(), dP.ravel(), dGam.ravel()])

    y0 = np.concatenate([np.eye(n).ravel(), np.zeros(n * n), np.zeros(n)])
    sol = solve_ivp(rhs, (0, dt), y0, rtol=1e-12, atol=1e-18)
    Phi_ode = sol.y[: n * n, -1].reshape(n, n)
    Q_ode = sol.y[n * n: 2 * n * n, -1].reshape(n, n)
    Gam_ode = sol.y[2 * n * n:, -1]

    kf = AltitudeKF(accel, baro, dt)  # the filter's own matrices
    np.testing.assert_allclose(kf.Phi, Phi_ode, atol=1e-12)
    np.testing.assert_allclose(kf.Q, Q_ode, rtol=1e-6, atol=1e-16)
    np.testing.assert_allclose(kf.Gamma, Gam_ode, atol=1e-12)
    # and the generic helpers on the same model
    Phi_vl, Q_vl = van_loan(A, L, Qc, dt)
    np.testing.assert_allclose(Q_vl, Q_ode, rtol=1e-6, atol=1e-16)
    np.testing.assert_allclose(input_matrix(A, B, dt)[:, 0], Gam_ode, atol=1e-12)


def test_kalman_filter_equals_batch_least_squares():
    """Without process noise, the Kalman filter is exactly recursive least
    squares: its final estimate and covariance must equal those of a batch
    weighted least-squares fit of the initial state, propagated to the end."""
    accel = AccelConfig(noise_density=0.0, bias_rw=0.0)
    baro = BaroConfig(drift_sigma=0.0, noise_sigma=0.4)
    dt = 0.01
    kf = AltitudeKF(accel, baro, dt)
    x0 = np.array([100.0, 2.0, 0.0])
    P0 = np.diag([4.0, 0.5, 0.01])
    kf.x, kf.P = x0.copy(), P0.copy()

    rng = np.random.default_rng(0)
    u = rng.normal(0, 1.0, 2000)
    rows, rhs_off, zs = [], [], []
    M = np.eye(3)          # Phi^k
    c = np.zeros(3)        # state reached from x0 = 0 with the inputs
    for k in range(2000):
        kf.predict(u[k])
        M = kf.Phi @ M
        c = kf.Phi @ c + kf.Gamma * u[k]
        if (k + 1) % 4 == 0:
            z = rng.normal(100.0, 5.0)
            kf.update_baro(z)
            rows.append(kf.H @ M)
            rhs_off.append(kf.H @ c)
            zs.append(z)
    Hb = np.array(rows)
    Rinv = 1.0 / kf.R
    info = np.linalg.inv(P0) + Hb.T @ Hb * Rinv
    x0_hat = np.linalg.solve(info, np.linalg.inv(P0) @ x0 + Hb.T @ (np.array(zs) - np.array(rhs_off)) * Rinv)
    x_end = M @ x0_hat + c
    P_end = M @ np.linalg.inv(info) @ M.T
    np.testing.assert_allclose(kf.x, x_end, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(kf.P, P_end, rtol=1e-6, atol=1e-12)


def test_altitude_nees_follows_chi_square():
    """Correctly specified linear filter: across runs, the NEES on [h, v] at a
    given time must follow a chi-square law with 2 degrees of freedom, not just
    average 2. Kolmogorov-Smirnov test."""
    from navsim.runner import simulate_and_run
    from navsim.trajectory import fixed_wing_vertical_profile

    profile = fixed_wing_vertical_profile(duration=60.0, seed=0)
    accel, baro = AccelConfig(), BaroConfig(drift_sigma=0.0)
    rng = np.random.default_rng(7)
    k = np.searchsorted(profile.t, 45.0)
    nees = [simulate_and_run(profile, accel, baro, accel, baro, rng).nees_hv[k] for _ in range(300)]
    assert kstest(nees, chi2(2).cdf).pvalue > 1e-3


# ---------------------------------------------------------------------------
# Step 2: trajectory and strapdown against an ODE solver
# ---------------------------------------------------------------------------

def mission_ode(plan, t0, t1, y0):
    """Integrate the continuous kinematics p' = v, v' = C(q) f_b + g,
    q' = q * (0, w_b) / 2 with scipy's adaptive Runge-Kutta, using the
    analytic body rates and specific force at any instant."""
    alpha = np.radians(plan.alpha_trim_deg)

    def channels(t):
        tt = np.array([t])
        phi, dphi = trajectory3d._channel(tt, 0.0, plan.bank_deg, np.pi / 180)
        gam, dgam = trajectory3d._channel(tt, 0.0, plan.gamma_deg, np.pi / 180)
        V, dV = trajectory3d._channel(tt, plan.airspeed0, plan.airspeed)
        return body_kinematics(phi, dphi, gam, dgam, V, dV, alpha)

    def rhs(t, y):
        omega, f, *_ = channels(t)
        q = y[6:10] / np.linalg.norm(y[6:10])
        dq = 0.5 * quat_mul(q, np.concatenate([[0.0], omega[0]]))
        dv = quat_rotate(q, f[0]) + np.array([0, 0, G])
        return np.concatenate([y[3:6], dv, dq])

    return solve_ivp(rhs, (t0, t1), y0, rtol=1e-11, atol=1e-11, method="DOP853")


@pytest.fixture(scope="module")
def mission():
    return trajectory3d.generate(trajectory3d.MISSION)


def test_trajectory_matches_ode_integration(mission):
    """The truth generator (analytic rates + Simpson on heading and position)
    against an adaptive ODE solver on 60 s with a roll-in, two circles and a
    roll-out."""
    k0, k1 = np.searchsorted(mission.t, [100.0, 160.0])
    y0 = np.concatenate([mission.p_n[k0], mission.v_n[k0], mission.q_nb[k0]])
    sol = mission_ode(mission.plan, mission.t[k0], mission.t[k1], y0)
    p, v, q = sol.y[0:3, -1], sol.y[3:6, -1], sol.y[6:10, -1] / np.linalg.norm(sol.y[6:10, -1])
    np.testing.assert_allclose(p, mission.p_n[k1], atol=1e-5)
    np.testing.assert_allclose(v, mission.v_n[k1], atol=1e-7)
    same_quat(q, mission.q_nb[k1], atol=1e-8)


def test_strapdown_matches_ode_integration(mission):
    k0, k1 = np.searchsorted(mission.t, [100.0, 160.0])
    sol = strapdown.integrate(mission.p_n[k0], mission.v_n[k0], mission.q_nb[k0],
                              mission.dtheta[k0:k1], mission.dvel[k0:k1], mission.dt, "full")
    y0 = np.concatenate([mission.p_n[k0], mission.v_n[k0], mission.q_nb[k0]])
    ode = mission_ode(mission.plan, mission.t[k0], mission.t[k1], y0)
    assert np.linalg.norm(sol.p_n[-1] - ode.y[0:3, -1]) < 2e-3


def test_body_rates_match_finite_difference_of_attitude(mission):
    """omega_b from the Euler-rate formula vs the rotation between two
    consecutive attitudes: an independent way to get the body rates."""
    dq = quat_mul(quat_conj(mission.q_nb[:-1]), mission.q_nb[1:])
    omega_fd = rotvec_from_quat(dq) / mission.dt
    omega_mid = 0.5 * (mission.omega_b[:-1] + mission.omega_b[1:])
    np.testing.assert_allclose(omega_fd, omega_mid, atol=2e-5)


def test_acceleration_matches_finite_difference_of_velocity(mission):
    a_fd = np.diff(mission.v_n, axis=0) / mission.dt
    a_mid = 0.5 * (mission.a_n[:-1] + mission.a_n[1:])
    np.testing.assert_allclose(a_fd, a_mid, atol=2e-4)


@pytest.fixture(scope="module")
def straight():
    return trajectory3d.generate(trajectory3d.FlightPlan(duration=60.0, altitude0=100.0))


def test_error_budget_constant_accel_bias(straight):
    """Deterministic, single error source: a constant forward accelerometer
    bias b gives a north error b cos(alpha) t^2 / 2 (closed form)."""
    b = 0.05
    dv = straight.dvel + np.array([b, 0, 0]) * straight.dt
    sol = strapdown.integrate(straight.p_n[0], straight.v_n[0], straight.q_nb[0],
                              straight.dtheta, dv, straight.dt)
    t = straight.t[-1]
    alpha = np.radians(straight.plan.alpha_trim_deg)
    assert sol.p_n[-1, 0] - straight.p_n[-1, 0] == pytest.approx(b * np.cos(alpha) * t**2 / 2, rel=1e-3)
    assert strapdown.error_budget(t, dataclasses.replace(imu.PERFECT, accel_bias0=b))["total"] == \
        pytest.approx(b * t**2 / 2)


def test_error_budget_constant_gyro_bias(straight):
    """A constant roll-gyro bias b tilts the estimate about north by b t; the
    accelerometer then reads g b t towards east: east error g b t^3 / 6."""
    b = np.radians(0.02)
    alpha = np.radians(straight.plan.alpha_trim_deg)
    # bias about the body x axis; the body is pitched up by alpha
    dth = straight.dtheta + np.array([b, 0, 0]) * straight.dt
    sol = strapdown.integrate(straight.p_n[0], straight.v_n[0], straight.q_nb[0], dth,
                              straight.dvel, straight.dt)
    t = straight.t[-1]
    east = sol.p_n[-1, 1] - straight.p_n[-1, 1]
    assert east == pytest.approx(G * b * np.cos(alpha) * t**3 / 6, rel=0.02)


# ---------------------------------------------------------------------------
# Step 3: error-state EKF
# ---------------------------------------------------------------------------

def nonlinear_error_after(mission, k0, n, dx0, bg_t, ba_t):
    """Propagate truth and a nominal state offset by dx0 through the nonlinear
    strapdown; return truth minus nominal after n steps."""
    pt, vt, qt = mission.p_n[k0], mission.v_n[k0], mission.q_nb[k0]
    pn, vn = pt - dx0[0:3], vt - dx0[3:6]
    qn = quat_mul(quat_from_rotvec(-dx0[6:9]), qt)
    bgn, ban = bg_t - dx0[9:12], ba_t - dx0[12:15]
    prev_t = prev_n = None
    dt = mission.dt
    for k in range(k0, k0 + n):
        dth_m = mission.dtheta[k] + bg_t * dt
        dv_m = mission.dvel[k] + ba_t * dt
        a, b = dth_m - bgn * dt, dv_m - ban * dt
        first = k == k0
        pt, vt, qt = strapdown.step(pt, vt, qt, mission.dtheta[k], mission.dvel[k],
                                    *(prev_t or (mission.dtheta[k], mission.dvel[k])), dt, "full", first)
        pn, vn, qn = strapdown.step(pn, vn, qn, a, b, *(prev_n or (a, b)), dt, "full", first)
        prev_t, prev_n = (mission.dtheta[k], mission.dvel[k]), (a, b)
    return np.concatenate([pt - pn, vt - vn, rotvec_from_quat(quat_mul(qt, quat_conj(qn))),
                           bg_t - bgn, ba_t - ban])


@pytest.mark.parametrize("start_s", [30.0, 130.0])
def test_transition_matrix_column_by_column(mission, start_s):
    """Numerical Jacobian of the nonlinear error propagation (central
    differences, one column per state) against the product of the filter's
    transition matrices. A wrong block shows up in its own column."""
    k0, n = int(start_s / mission.dt), 40
    bg_t, ba_t = np.array([3e-4, -2e-4, 1e-4]), np.array([0.03, -0.02, 0.04])
    ekf = ErrorStateEKF(imu.IMUConfig(), gnss.GNSSConfig(), mission.dt, runs=1)
    ekf.initialise(mission.p_n[k0][None], mission.v_n[k0][None], mission.q_nb[k0][None],
                   np.eye(15), bg_t[None], ba_t[None])
    Phi = np.eye(15)
    for k in range(k0, k0 + n):
        ekf.predict((mission.dtheta[k] + bg_t * mission.dt)[None],
                    (mission.dvel[k] + ba_t * mission.dt)[None])
        Phi = ekf.Phi[0] @ Phi
    eps = np.array([1e-3] * 3 + [1e-4] * 3 + [1e-6] * 3 + [1e-7] * 3 + [1e-5] * 3)
    J = np.empty((15, 15))
    for i in range(15):
        d = np.zeros(15)
        d[i] = eps[i]
        J[:, i] = (nonlinear_error_after(mission, k0, n, d, bg_t, ba_t)
                   - nonlinear_error_after(mission, k0, n, -d, bg_t, ba_t)) / (2 * eps[i])
    for i in range(15):
        scale = np.linalg.norm(J[:, i])
        assert np.linalg.norm(J[:, i] - Phi[:, i]) < 2e-3 * scale + 1e-9, f"colonne {i}"


def test_gnss_update_equals_information_form():
    """Joseph-form update vs the information form P+ = (P^-1 + H^T R^-1 H)^-1,
    on a random covariance. The measurement equals the prediction, so the
    injected correction is zero and the reset leaves P unchanged."""
    cfg = gnss.GNSSConfig()
    ekf = ErrorStateEKF(imu.IMUConfig(), cfg, 0.005, runs=1)
    A = RNG.normal(size=(15, 15))
    P = A @ A.T + np.eye(15) * 0.1
    ekf.initialise(np.zeros((1, 3)), np.array([[17.0, 0, 0]]), np.array([[1.0, 0, 0, 0]]), P)
    z = np.concatenate([ekf.p, ekf.v], axis=1)
    ekf.update_gnss(z)
    H = ekf.H
    expected = np.linalg.inv(np.linalg.inv(P) + H.T @ np.linalg.inv(cfg.R) @ H)
    np.testing.assert_allclose(ekf.P[0], expected, rtol=1e-8, atol=1e-10)


NOISE_ONLY = dataclasses.replace(imu.IMUConfig(), gyro_bias0=0.0, gyro_bias_rw=0.0,
                                 accel_bias0=0.0, accel_bias_rw=0.0)


@pytest.mark.parametrize("case", ["full", "noise_only"])
def test_propagated_covariance_matches_monte_carlo(mission, case):
    """Without measurements, the covariance the filter propagates must be the
    actual covariance of its errors. 600 open-loop runs over 20 s through a
    roll-in, real IMU errors.

    "full": initial errors drawn from P0, biases and noise.
    "noise_only": perfect start, no bias, white noise only. Added after the
    mutation tests showed that a gyro noise mis-scaled by sqrt(dt) went
    unnoticed in "full", where it is hidden by the larger terms."""
    runs, k0, n = 600, int(100.0 / mission.dt), int(20.0 / mission.dt)
    cfg = imu.IMUConfig() if case == "full" else NOISE_ONLY
    rng = np.random.default_rng(21)
    stream = imu.IMUStream(cfg, rng, runs, mission.dt)
    if case == "full":
        sig0 = np.concatenate([[1.0] * 3, [0.1] * 3, np.radians([1.0, 1.0, 3.0]),
                               [cfg.gyro_bias0] * 3, [cfg.accel_bias0] * 3])
    else:
        sig0 = np.zeros(15)
    dx0 = rng.normal(size=(runs, 15)) * sig0
    dx0[:, 9:12], dx0[:, 12:15] = stream.gyro_bias, stream.accel_bias   # nominal biases = 0
    ekf = ErrorStateEKF(cfg, gnss.GNSSConfig(), mission.dt, runs)
    q0 = quat_mul(quat_from_rotvec(-dx0[:, 6:9]), np.broadcast_to(mission.q_nb[k0], (runs, 4)))
    ekf.initialise(mission.p_n[k0] - dx0[:, 0:3], mission.v_n[k0] - dx0[:, 3:6], q0, np.diag(sig0**2))
    for k in range(k0, k0 + n):
        ekf.predict(*stream.sample(mission.dtheta[k], mission.dvel[k]))
    kend = k0 + n
    err = np.concatenate([
        mission.p_n[kend] - ekf.p, mission.v_n[kend] - ekf.v,
        -attitude_error(np.broadcast_to(mission.q_nb[kend], (runs, 4)), ekf.q),
        stream.gyro_bias - ekf.bg, stream.accel_bias - ekf.ba], axis=1)
    keep = slice(0, 15) if case == "full" else slice(0, 9)  # no bias states to compare in noise_only
    emp = np.cov(err[:, keep].T)
    P = ekf.P.mean(axis=0)[keep, keep]
    ratio = np.diag(emp) / np.diag(P)
    assert np.all((ratio > 0.8) & (ratio < 1.25)), ratio.round(2)
    # correlations, where the filter predicts a strong one
    d = np.sqrt(np.diag(P))
    corr_P = P / np.outer(d, d)
    corr_emp = emp / np.sqrt(np.outer(np.diag(emp), np.diag(emp)))
    strong = np.abs(corr_P) > 0.5
    assert np.max(np.abs(corr_P - corr_emp)[strong]) < 0.1


def test_noise_densities_are_rate_independent():
    """Process noise is set from physical densities: propagating the same
    flight at 100 Hz or 200 Hz must give the same covariance."""
    plan = trajectory3d.FlightPlan(duration=20.0, bank_deg=((5, 9, 25),))
    P_end = []
    for rate in (100.0, 200.0):
        tr = trajectory3d.generate(plan, rate_hz=rate)
        cfg = dataclasses.replace(imu.IMUConfig(), rate_hz=rate)
        ekf = ErrorStateEKF(cfg, gnss.GNSSConfig(), tr.dt, runs=1)
        ekf.initialise(tr.p_n[:1], tr.v_n[:1], tr.q_nb[:1],
                       fusion.initial_covariance(cfg, gnss.GNSSConfig(), fusion.InitConfig()))
        for k in range(len(tr.dtheta)):
            ekf.predict(tr.dtheta[k][None], tr.dvel[k][None])
        P_end.append(np.diag(ekf.P[0]))
    np.testing.assert_allclose(P_end[0], P_end[1], rtol=0.02)


def test_imu_stream_matches_its_specification():
    """The streaming IMU (used by every Monte-Carlo of steps 3 and 4) checked
    against the error model it claims: per-sample noise N sqrt(dt), initial
    bias sigma0, bias random walk K sqrt(t). Added after the mutation tests:
    the streaming generator had no direct test."""
    cfg, dt, runs = imu.IMUConfig(), 0.005, 20000
    stream = imu.IMUStream(cfg, np.random.default_rng(4), runs, dt)
    bg0, ba0 = stream.gyro_bias.copy(), stream.accel_bias.copy()
    np.testing.assert_allclose(bg0.std(axis=0), cfg.gyro_bias0, rtol=0.03)
    np.testing.assert_allclose(ba0.std(axis=0), cfg.accel_bias0, rtol=0.03)
    dth, dv = stream.sample(np.zeros(3), np.zeros(3))
    np.testing.assert_allclose((dth - bg0 * dt).std(axis=0), cfg.gyro_noise * np.sqrt(dt), rtol=0.03)
    np.testing.assert_allclose((dv - ba0 * dt).std(axis=0), cfg.accel_noise * np.sqrt(dt), rtol=0.03)
    for _ in range(199):  # 1 s in total
        stream.sample(np.zeros(3), np.zeros(3))
    np.testing.assert_allclose((stream.gyro_bias - bg0).std(axis=0), cfg.gyro_bias_rw, rtol=0.03)
    np.testing.assert_allclose((stream.accel_bias - ba0).std(axis=0), cfg.accel_bias_rw, rtol=0.03)


# ---------------------------------------------------------------------------
# Step 4: GNSS latency, deterministic geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,expected", [("ignore", 17.0 * 0.15), ("compensate", 0.0), ("delayed", 0.0)])
def test_latency_geometry(mode, expected):
    """Near-perfect sensors, straight flight at 17 m/s, 150 ms of latency.
    Fusing the late fix as current must leave the estimate V*latency behind
    (2.55 m); compensating or using the delayed horizon removes it."""
    tr = trajectory3d.generate(trajectory3d.FlightPlan(duration=30.0, altitude0=100.0))
    tiny = imu.IMUConfig(gyro_noise=1e-9, gyro_bias0=1e-9, gyro_bias_rw=1e-12,
                         accel_noise=1e-9, accel_bias0=1e-9, accel_bias_rw=1e-12)
    g = gnss.GNSSConfig(pos_sigma_h=0.01, pos_sigma_v=0.01, vel_sigma_h=0.001, vel_sigma_v=0.001,
                        latency_s=0.15)
    log = fusion.run(tr, tiny, g, fusion.InitConfig(tilt_sigma_deg=1e-4, yaw_sigma_deg=1e-4), 1,
                     np.random.default_rng(0), latency_mode=mode)
    north_err = log.out_err[0, -1, 0]  # truth minus estimate, at real time
    assert north_err == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reset_jacobian_matches_rotation_composition(seed):
    """After injecting dtheta_hat, the new attitude error is
    log(exp(dtheta) exp(-dtheta_hat)) (global error, scipy composition).
    Its numerical derivative at dtheta = dtheta_hat must equal G, up to
    second-order terms in dtheta_hat."""
    from navsim.eskf import reset_jacobian

    dth_hat = np.random.default_rng(seed).normal(0.0, 0.03, 3)

    def new_error(dth):
        return (Rotation.from_rotvec(dth) * Rotation.from_rotvec(dth_hat).inv()).as_rotvec()

    eps = 1e-7
    J = np.column_stack([(new_error(dth_hat + e) - new_error(dth_hat - e)) / (2 * eps)
                         for e in np.eye(3) * eps])
    G = reset_jacobian(dth_hat)
    first_order = np.abs(G - np.eye(3)).max()
    assert np.abs(J - G).max() < 0.05 * first_order   # the wrong sign gives ~2x first_order
