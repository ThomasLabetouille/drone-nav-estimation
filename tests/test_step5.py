"""Step 5: aiding sensors, wind, and the measurement models of the filter.

The measurement Jacobians are checked against measurement functions retyped
here from their physical definition with scipy rotations, and differentiated
numerically: the filter code is not used to build the reference.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from navsim import aiding, fusion, gnss, imu, trajectory3d
from navsim.eskf import ErrorStateEKF
from navsim.rotations import quat_from_euler

DEG = np.pi / 180


# --- configuration ----------------------------------------------------------

def test_aiding_config_rejects_inconsistent_combinations():
    with pytest.raises(ValueError):
        aiding.AidingConfig(pitot=aiding.PitotConfig())          # no wind model
    with pytest.raises(ValueError):
        aiding.AidingConfig(sideslip_sigma_deg=2.0)              # no Pitot


def test_earth_field_has_the_requested_intensity_and_angles():
    m = aiding.earth_field(0.47, 61.0, 1.5)
    assert np.linalg.norm(m) == pytest.approx(0.47)
    assert np.degrees(np.arctan2(m[2], np.hypot(m[0], m[1]))) == pytest.approx(61.0)
    assert np.degrees(np.arctan2(m[1], m[0])) == pytest.approx(1.5)


# --- truth with wind ----------------------------------------------------------

def test_constant_wind_only_shifts_the_ground_track():
    """Galilean invariance: in a constant wind the aircraft flies the same
    air-relative path. Attitude, body rates and specific force are the same
    as in still air; the position drifts by w t."""
    plan = dataclasses.replace(trajectory3d.MISSION, duration=180.0)
    still = trajectory3d.generate(plan)
    windy = trajectory3d.generate(dataclasses.replace(plan, wind_n0=-3.0, wind_e0=4.0))
    np.testing.assert_allclose(windy.q_nb, still.q_nb, atol=1e-12)
    np.testing.assert_allclose(windy.f_b, still.f_b, atol=1e-10)
    np.testing.assert_allclose(windy.omega_b, still.omega_b, atol=1e-12)
    shift = np.outer(windy.t - windy.t[0], [-3.0, 4.0, 0.0])
    np.testing.assert_allclose(windy.p_n - still.p_n, shift, atol=1e-6)


@pytest.fixture(scope="module")
def windy():
    return trajectory3d.generate(trajectory3d.MISSION_WIND)


def test_changing_wind_is_in_the_specific_force(windy):
    """f_b = C^T (a_n - g): the wind acceleration must be inside, otherwise
    the IMU would not feel the gust the GNSS sees."""
    a_fd = np.diff(windy.v_n, axis=0) / windy.dt
    a_mid = 0.5 * (windy.a_n[:-1] + windy.a_n[1:])
    np.testing.assert_allclose(a_fd, a_mid, atol=2e-4)
    g = np.array([0.0, 0.0, trajectory3d.G])
    f_n = np.einsum("kij,kj->ki", Rotation.from_quat(windy.q_nb[:, [1, 2, 3, 0]]).as_matrix(), windy.f_b)
    np.testing.assert_allclose(f_n, windy.a_n - g, atol=1e-9)


def test_airspeed_and_sideslip_of_the_truth(windy):
    v_air = windy.v_n - windy.wind_n
    np.testing.assert_allclose(windy.airspeed, np.linalg.norm(v_air, axis=1), atol=1e-9)
    v_b = Rotation.from_quat(windy.q_nb[:, [1, 2, 3, 0]]).inv().apply(v_air)
    beta = np.degrees(np.arcsin(v_b[:, 1] / windy.airspeed))
    level = np.abs(windy.euler[:, 0]) < 1e-9
    assert np.abs(beta[level]).max() < 1e-6       # wings level: exactly zero
    assert np.abs(beta).max() < 2.0               # what the 6 deg sigma has to cover


# --- sensor models ------------------------------------------------------------

def test_magnetometer_heading_recovers_the_true_heading():
    """Rebuild the body field with scipy from random attitudes, then the
    tilt-compensated heading must give back the yaw."""
    rng = np.random.default_rng(1)
    roll, pitch, yaw = rng.uniform(-0.6, 0.6, 200), rng.uniform(-0.4, 0.4, 200), rng.uniform(-np.pi, np.pi, 200)
    m_n = aiding.earth_field()
    m_b = Rotation.from_euler("ZYX", np.c_[yaw, pitch, roll]).inv().apply(m_n)
    psi = aiding.heading_from_magnetometer(m_b, roll, pitch, np.arctan2(m_n[1], m_n[0]))
    np.testing.assert_allclose(np.angle(np.exp(1j * (psi - yaw))), 0.0, atol=1e-12)


def test_magnetometer_stream_statistics():
    cfg = aiding.MagConfig()
    s = aiding.MagStream(cfg, np.random.default_rng(2), runs=4000)
    b0 = s.bias.copy()
    q = quat_from_euler(0.1, -0.05, 1.0)
    z = s.measure(q)
    m_b = Rotation.from_quat(q[[1, 2, 3, 0]]).inv().apply(cfg.field_ned)
    np.testing.assert_allclose(b0.std(axis=0), cfg.bias0, rtol=0.05)
    np.testing.assert_allclose((z - m_b - b0).std(axis=0), cfg.noise, rtol=0.05)
    for _ in range(499):
        s.measure(q)
    np.testing.assert_allclose((s.bias - b0).std(axis=0), cfg.bias_rw * np.sqrt(10.0), rtol=0.05)


def test_baro_stream_statistics():
    cfg = aiding.BARO_10HZ
    s = aiding.BaroStream(cfg, np.random.default_rng(3), runs=4000)
    z0 = s.measure(100.0)
    for _ in range(int(cfg.rate_hz * 60)):
        z = s.measure(100.0)
    total = np.hypot(cfg.drift_sigma, cfg.noise_sigma)
    np.testing.assert_allclose(z0.std(), total, rtol=0.05)
    np.testing.assert_allclose(z.std(), total, rtol=0.05)   # stationary
    assert abs(z.mean() - 100.0) < 4 * total / np.sqrt(4000)


# --- measurement Jacobians against numerical differentiation -------------------

FULL = aiding.AidingConfig(mag=aiding.MagConfig(), baro=aiding.BARO_10HZ, pitot=aiding.PitotConfig(),
                           wind=aiding.WindModel(), sideslip_sigma_deg=6.0)


def random_filter(seed):
    rng = np.random.default_rng(seed)
    ekf = ErrorStateEKF(imu.IMUConfig(), gnss.REALISTIC, 0.005, runs=1, model_gnss_bias=True, aiding=FULL)
    q = quat_from_euler(*rng.uniform([-0.5, -0.3, -np.pi], [0.5, 0.3, np.pi]))[None]
    v = np.array([[*(17.0 * np.array([np.cos(rng.uniform(-3, 3)), np.sin(rng.uniform(-3, 3))])),
                   rng.uniform(-2, 2)]])
    extra = rng.normal(0.0, 1.0, (1, ekf.n - 15))
    ekf.initialise(rng.normal(0, 50, (1, 3)), v, q, np.eye(ekf.n), rng.normal(0, 1e-3, (1, 3)),
                   rng.normal(0, 0.05, (1, 3)), extra)
    return ekf


def true_state(ekf, dx):
    """nominal (+) dx, with the filter's error convention retyped:
    x_true = x_nom + dx, R_true = exp(dtheta) R_nom (dtheta in NED)."""
    R = Rotation.from_rotvec(dx[6:9]) * Rotation.from_quat(ekf.q[0, [1, 2, 3, 0]])
    extra = {name: ekf.nominal(name)[0] + dx[s] for name, s in ekf.blocks.items() if s.start >= 15}
    return ekf.p[0] + dx[0:3], ekf.v[0] + dx[3:6], R, extra


def h_baro(p, v, R, x):
    return np.array([-p[2] + x["baro_bias"][0]])


EARTH_FIELD = np.asarray(aiding.MagConfig().field_ned)


def h_mag(p, v, R, x):
    return R.inv().apply(EARTH_FIELD) + x["mag_bias"]


def h_airspeed(p, v, R, x):
    return np.array([np.linalg.norm(v - np.r_[x["wind"], 0.0])])


def h_sideslip(p, v, R, x):
    v_air = v - np.r_[x["wind"], 0.0]
    return np.array([R.inv().apply(v_air)[1] / np.linalg.norm(v_air)])


def captured_update(ekf, call):
    """Run one update and return the (y, H) the filter used."""
    seen = {}
    real = ekf.update

    def spy(y, H, R, frozen=None, gate=None):
        seen["y"], seen["H"] = np.array(y), np.broadcast_to(H, (ekf.runs, *np.shape(H)[-2:])).copy()
        return real(y, H, R, frozen, gate)

    ekf.update = spy
    call(ekf)
    return seen["y"][0], seen["H"][0]


CASES = {
    "baro": (h_baro, lambda e, z: e.update_baro(z[None, 0])),
    "mag": (h_mag, lambda e, z: e.update_mag(z[None])),
    "airspeed": (h_airspeed, lambda e, z: e.update_airspeed(z[None, 0])),
    "sideslip": (h_sideslip, lambda e, z: e.update_sideslip(6.0 * DEG)),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_measurement_jacobian_matches_numerical_derivative(name, seed):
    h, call = CASES[name]
    ekf = random_filter(seed)
    z_nom = h(*true_state(ekf, np.zeros(ekf.n)))
    z = z_nom if name != "sideslip" else np.zeros(1)
    y, H = captured_update(ekf, lambda e: call(e, z))
    ekf = random_filter(seed)  # the update above changed the state
    np.testing.assert_allclose(y, z - z_nom, atol=1e-12)          # predicted measurement
    eps = 1e-6
    H_num = np.empty_like(H)
    for i in range(ekf.n):
        d = np.zeros(ekf.n)
        d[i] = eps
        H_num[:, i] = (h(*true_state(ekf, d)) - h(*true_state(ekf, -d))) / (2 * eps)
    np.testing.assert_allclose(H, H_num, atol=1e-7 * max(1.0, np.abs(H_num).max()))


# --- Schmidt gating of the magnetometer bias ------------------------------------

def test_frozen_bias_is_a_consider_state():
    """learn_bias=False: the bias keeps its value and its variance, but its
    uncertainty still widens the innovation (it is not ignored)."""
    ekf = random_filter(5)
    sl = ekf.blocks["mag_bias"]
    P_bb, b = ekf.P[0, sl, sl].copy(), ekf.nominal("mag_bias").copy()
    z = h_mag(*true_state(ekf, np.zeros(ekf.n))) + 0.05
    _, S, _ = ekf.update_mag(z[None], learn_bias=np.array([False]))
    np.testing.assert_array_equal(ekf.P[0, sl, sl], P_bb)
    np.testing.assert_array_equal(ekf.nominal("mag_bias"), b)
    assert np.all(np.diag(S[0]) > np.diag(P_bb))
    _, _, _ = ekf.update_mag(z[None], learn_bias=np.array([True]))
    assert np.all(np.diag(ekf.P[0, sl, sl]) < np.diag(P_bb))


@pytest.fixture(scope="module")
def before_first_turn():
    """68 s of straight flight in wind, magnetometer, with and without gating."""
    tr = trajectory3d.generate(dataclasses.replace(trajectory3d.MISSION_WIND, duration=68.0))
    init = fusion.InitConfig(yaw_source="magnetometer", yaw_sigma_deg=15.0)
    out = {}
    for key, rate in (("gated", 5.0), ("always", None)):
        ad = aiding.AidingConfig(mag=aiding.MagConfig(learn_bias_min_rate_deg_s=rate))
        out[key] = fusion.run(tr, imu.IMUConfig(), gnss.REALISTIC, init, 24, np.random.default_rng(7),
                              model_gnss_bias=True, latency_mode="delayed", aiding=ad)
    return out


def test_gated_mag_bias_keeps_the_heading_consistent(before_first_turn):
    lg = before_first_turn["gated"]
    w = lg.t > 5.0
    assert lg.nees["attitude"][:, w].mean() / 3 < 1.3
    assert lg.nees["mag_bias"][:, w].mean() / 3 < 1.5


def test_learning_mag_bias_in_straight_flight_is_overconfident(before_first_turn):
    """The failure the gating fixes: keep it visible, so that removing the
    gating (or breaking it) shows up as a failing test."""
    lg = before_first_turn["always"]
    w = lg.t > 5.0
    assert lg.nees["mag_bias"][:, w].mean() / 3 > 2.0


# --- system level: wind, barometer ------------------------------------------------

@pytest.fixture(scope="module")
def full_aiding():
    """200 s in wind (first turn and two loiter circles), all step 5 sensors,
    and the same flight with the same draws but without the barometer."""
    tr = trajectory3d.generate(dataclasses.replace(trajectory3d.MISSION_WIND, duration=200.0))
    init = fusion.InitConfig(yaw_source="magnetometer", yaw_sigma_deg=15.0)
    ad = dataclasses.replace(FULL, wind=aiding.WindModel(rw=0.1))
    logs = {}
    for key, cfg in (("all", ad), ("no_baro", dataclasses.replace(ad, baro=None))):
        logs[key] = fusion.run(tr, imu.IMUConfig(), gnss.REALISTIC, init, 16, np.random.default_rng(11),
                               model_gnss_bias=True, latency_mode="delayed", aiding=cfg)
    return tr, logs["all"], logs["no_baro"]


def test_wind_is_estimated_and_consistent(full_aiding):
    tr, lg, _ = full_aiding
    late = lg.t > 120.0
    err = lg.err[:, late][..., lg.blocks["wind"]]
    assert np.sqrt((err**2).mean()) < 0.4                          # [m/s], truth is 5 m/s
    nees = lg.nees["wind"][:, late].mean() / 2
    assert 0.2 < nees < 1.5


def test_every_block_stays_consistent(full_aiding):
    _, lg, _ = full_aiding
    late = lg.t > 120.0
    for name, s in lg.blocks.items():
        dof = s.stop - s.start
        assert lg.nees[name][:, late].mean() / dof < 1.6, name
    for name, (t, v) in lg.aux_nis.items():
        dof = 3 if name == "mag" else 1
        assert v[:, t > 120.0].mean() / dof < 1.3, name


def test_barometer_improves_the_vertical(full_aiding):
    """Same IMU, GNSS, magnetometer and Pitot draws with and without the
    barometer (common random numbers), so the difference is the barometer."""
    _, lg, ref = full_aiding
    late = lg.t > 120.0
    rms = [np.sqrt((x.err[:, late, 2] ** 2).mean()) for x in (lg, ref)]
    assert rms[0] < 0.8 * rms[1]


# --- initial wind from the first airspeed sample ------------------------------------

def test_initial_wind_covariance_matches_monte_carlo():
    """Draw the true state, build the filter's view of it (heading, GNSS
    velocity, Pitot), and compare the empirical covariance of the actual
    errors (velocity, heading, wind) to the P0 the filter derives, cross
    terms included: a missing correlation would leave the means right."""
    rng = np.random.default_rng(4)
    runs, n = 40000, 21
    blocks = {"wind": slice(19, 21)}
    psi_t = 0.3 + rng.normal(0.0, 0.02, runs)
    tas_t, w_t = 17.0, np.array([1.0, 5.0])
    sig_v, sig_psi, sig_tas, sig_beta = 0.1, np.radians(3.0), 0.3, np.radians(2.0)
    course_air = psi_t + rng.normal(0.0, sig_beta, runs)      # heading + sideslip
    v_t = np.c_[tas_t * np.cos(course_air) + w_t[0], tas_t * np.sin(course_air) + w_t[1], np.zeros(runs)]
    P0 = np.eye(n)
    P0[3:6, 3:6] = sig_v**2 * np.eye(3)
    P0[8, 8] = sig_psi**2
    dv = rng.normal(0.0, sig_v, (runs, 3))
    dpsi = rng.normal(0.0, sig_psi, runs)
    v0 = v_t - dv
    q0 = quat_from_euler(np.zeros(runs), np.zeros(runs), psi_t - dpsi)
    tas0 = tas_t + rng.normal(0.0, sig_tas, runs)
    P, extra = fusion.wind_from_first_airspeed(blocks, n, v0, q0, tas0, P0, sig_tas, sig_beta)
    dw = w_t - extra[:, 4:6]
    errs = np.c_[dv[:, :2], dpsi, dw]
    idx = [3, 4, 8, 19, 20]
    P_mean = P.mean(axis=0)[np.ix_(idx, idx)]
    emp = np.cov(errs.T)
    scale = np.sqrt(np.outer(np.diag(P_mean), np.diag(P_mean)))
    np.testing.assert_allclose(emp / scale, P_mean / scale, atol=0.03)
    assert abs(dw.mean(axis=0)).max() < 0.05                      # no bias at 3 deg of heading error
