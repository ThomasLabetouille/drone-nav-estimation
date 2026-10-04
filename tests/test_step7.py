"""Step 7: replay of recorded sensor data.

The real logs are not in the repository, so the replay is tested on
simulated logs, built with the quirks of a real one: uneven IMU intervals,
GNSS accuracy reported by the receiver, an autopilot mounted at an angle.
There the truth is known. The ULog reader is tested on a small real log
from the pyulog repository.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from navsim import aiding, gnss, imu, replay, trajectory3d
from navsim.eskf import ErrorStateEKF
from navsim.rotations import (quat_conj, quat_from_euler, quat_from_rotvec, quat_mul, quat_normalize, quat_rotate,
                              rotvec_from_quat)
from navsim.sensors import BaroConfig

DEG = np.pi / 180
SAMPLE = Path(__file__).parent / "data" / "pyulog_sample_log_small.ulg"


# --- filter extensions ---------------------------------------------------------

def test_rediscretisation_matches_a_filter_built_at_that_interval():
    ad = aiding.AidingConfig(baro=aiding.BARO_10HZ, pitot=aiding.PitotConfig(), wind=aiding.WindModel())
    nominal = ErrorStateEKF(imu.IMUConfig(), gnss.REALISTIC, 0.005, 1, model_gnss_bias=True, aiding=ad)
    for dt in (0.004, 0.01, 0.0123):
        built = ErrorStateEKF(imu.IMUConfig(), gnss.REALISTIC, dt, 1, model_gnss_bias=True, aiding=ad)
        Qd, phi = nominal._discrete(dt)
        np.testing.assert_allclose(np.diag(Qd), np.diag(built.Qd), rtol=1e-10, atol=1e-20)
        np.testing.assert_allclose(phi, built.phi_diag, rtol=1e-12)
    Qd, phi = nominal._discrete(0.005)
    assert Qd is nominal.Qd                      # nominal interval: exactly the cached values


def test_config_like_ekf2_converts_per_step_noises_to_densities():
    data = replay.LogData(name="x", t_imu=np.zeros(1), dt_imu=np.full(1, 0.005), dtheta=np.zeros((1, 3)),
                          dvel=np.zeros((1, 3)), t_gnss=np.zeros(1), gnss_p=np.zeros((1, 3)),
                          gnss_v=np.zeros((1, 3)), gnss_sigma=np.zeros((1, 3)),
                          info={"params": {"EKF2_GYR_NOISE": 0.015, "EKF2_ACC_NOISE": 0.35,
                                           "EKF2_PREDICT_US": 10000, "EKF2_HEAD_NOISE": 0.3}})
    cfg = replay.config_like_ekf2(data, air_data=False, mag=None)
    assert cfg.imu.gyro_noise == pytest.approx(0.0015)
    assert cfg.imu.accel_noise == pytest.approx(0.035)


def test_receiver_accuracy_becomes_per_axis_sigmas():
    s = replay._sigma(np.array([2.0, 3.0, 0.6]), (0.5, 0.75, 0.3))
    np.testing.assert_allclose(s, [np.sqrt(2), np.sqrt(2), 3.0, 0.6 / np.sqrt(3)] + [0.6 / np.sqrt(3)] * 2)
    s = replay._sigma(np.array([0.1, 0.1, 0.01]), (0.5, 0.75, 0.3))       # floors
    np.testing.assert_allclose(s, [0.5, 0.5, 0.75, 0.3, 0.3, 0.3])


def make_filter(pitch, yaw, seed=0):
    ad = aiding.AidingConfig(mag=aiding.MagConfig(fusion="heading"), pitot=aiding.PitotConfig(scale_sigma0=0.3),
                             wind=aiding.WindModel(), sideslip_sigma_deg=6.0, sideslip_offset_sigma0_deg=15.0)
    ekf = ErrorStateEKF(imu.IMUConfig(), gnss.REALISTIC, 0.005, 1, aiding=ad)
    rng = np.random.default_rng(seed)
    q = quat_from_euler(0.2, pitch, yaw)[None]
    extra = rng.normal(0.0, 0.5, (1, ekf.n - 15)) * 0.1
    ekf.initialise(np.zeros((1, 3)), np.array([[16.0, 5.0, -1.0]]), q, np.eye(ekf.n), extra=extra)
    return ekf


def spy_update(ekf, call):
    seen = {}
    real = ekf.update

    def spy(y, H, R, frozen=None, gate=None):
        seen["y"], seen["H"] = np.array(y), np.broadcast_to(H, (1, *np.shape(H)[-2:])).copy()
        return real(y, H, R, frozen, gate)

    ekf.update = spy
    call(ekf)
    return seen["y"][0], seen["H"][0, 0]


def perturbed(ekf, dx):
    R = Rotation.from_rotvec(dx[6:9]) * Rotation.from_quat(ekf.q[0, [1, 2, 3, 0]])
    ex = {k: ekf.nominal(k)[0] + dx[s] for k, s in ekf.blocks.items() if s.start >= 15}
    return ekf.v[0] + dx[3:6], R, ex


def numeric_row(ekf, h):
    eps = 1e-6
    row = np.empty(ekf.n)
    for i in range(ekf.n):
        d = np.zeros(ekf.n)
        d[i] = eps
        row[i] = (h(*perturbed(ekf, d)) - h(*perturbed(ekf, -d))) / (2 * eps)
    return row


@pytest.mark.parametrize("pitch", [0.0, 0.3, -0.25])
def test_heading_jacobian_includes_the_pitch_coupling(pitch):
    """Predicted heading = yaw angle of the estimate (ZYX, scipy)."""
    ekf = make_filter(pitch, 1.0)
    m_b = Rotation.from_quat(ekf.q[0, [1, 2, 3, 0]]).inv().apply(aiding.earth_field())
    _, H = spy_update(ekf, lambda e: e.update_mag_heading(m_b[None], 0.2, np.radians(1.5)))
    ekf = make_filter(pitch, 1.0)
    yaw = lambda v, R, ex: R.as_euler("ZYX")[0]
    np.testing.assert_allclose(H, numeric_row(ekf, yaw), atol=1e-6)


def test_airspeed_jacobian_with_scale_state():
    def h(v, R, ex):
        return (1.0 + ex["tas_scale"][0]) * np.linalg.norm(v - np.r_[ex["wind"], 0.0])

    ekf = make_filter(0.05, 0.4)
    z = h(*perturbed(ekf, np.zeros(ekf.n)))
    y, H = spy_update(ekf, lambda e: e.update_airspeed(np.array([z])))
    assert abs(y[0]) < 1e-12
    ekf = make_filter(0.05, 0.4)
    np.testing.assert_allclose(H, numeric_row(ekf, h), atol=1e-6)


def test_sideslip_jacobian_with_offset_state():
    def h(v, R, ex):
        v_air = v - np.r_[ex["wind"], 0.0]
        return R.inv().apply(v_air)[1] / np.linalg.norm(v_air) - ex["beta_offset"][0]

    ekf = make_filter(0.05, 0.4)
    y, H = spy_update(ekf, lambda e: e.update_sideslip(0.1))
    ekf = make_filter(0.05, 0.4)
    assert y[0] == pytest.approx(-h(*perturbed(ekf, np.zeros(ekf.n))), abs=1e-12)
    np.testing.assert_allclose(H, numeric_row(ekf, h), atol=1e-6)


# --- replay of simulated logs ----------------------------------------------------

PLAN = dataclasses.replace(trajectory3d.MISSION_WIND, duration=240.0)
SENSORS = dict(mag_cfg=aiding.MagConfig(), baro_cfg=aiding.BARO_10HZ, pitot_cfg=aiding.PitotConfig())


def full_config(**kw):
    base = dict(baro=aiding.BARO_10HZ, mag=aiding.MagConfig(), pitot=aiding.PitotConfig(),
                wind=aiding.WindModel(rw=0.1), sideslip_sigma_deg=6.0)
    base.update(kw)
    return replay.ReplayConfig(**base)


@pytest.fixture(scope="module")
def simulated():
    tr = trajectory3d.generate(PLAN)
    return replay.from_simulation(tr, imu.IMUConfig(), np.random.default_rng(5), **SENSORS)


def errors(log, data, after=100.0):
    c = replay.compare(log, data)
    m = c["t"] > after
    return {"horiz": np.sqrt((c["dp"][m, :2] ** 2).sum(1).mean()),
            "head": np.degrees(np.sqrt((c["datt"][m, 2] ** 2).mean())),
            "head_mean": np.degrees(c["datt"][m, 2].mean()),
            "wind": np.sqrt((c["dwind"][m] ** 2).sum(1).mean()) if "dwind" in c else None}


def test_simulated_log_has_uneven_intervals(simulated):
    assert set(np.round(np.unique(simulated.dt_imu), 6)) == {0.005, 0.01}
    np.testing.assert_allclose(simulated.dt_imu.sum(), simulated.t_imu[-1] - simulated.t_imu[0] + simulated.dt_imu[0],
                               rtol=1e-9)


def test_replay_follows_the_truth(simulated):
    log = replay.run(simulated, full_config())
    e = errors(log, simulated)
    assert e["horiz"] < 1.0
    assert e["head"] < 0.5
    assert e["wind"] < 0.4
    dof = {"gnss": 6, "mag": 3, "baro": 1, "pitot": 1}
    for name, (t, v, r) in log.nis.items():
        if name in dof:
            assert v[t > 30].mean() / dof[name] < 1.5, name
            assert r.mean() < 0.01, name


def test_replay_matches_the_simulation_loop(simulated):
    """GNSS only, nothing specific to the replay: the error must be that of
    the step 3 filter on a comparable flight (sub-metre with 1 m GNSS)."""
    log = replay.run(simulated, replay.ReplayConfig())
    assert errors(log, simulated)["horiz"] < 1.0


def test_outage_errors_are_consistent(simulated):
    windows = ((100.0, 130.0), (180.0, 210.0))
    log = replay.run(simulated, full_config(gnss_outage=windows))
    assert np.all(log.gnss_status[(log.t >= 100.0) & (log.t < 130.0)] == 2)
    e = replay.outage_errors(log, windows)
    assert e.shape == (2, 2)
    assert np.all(e[:, 0] < 3.0 * e[:, 1])       # within 3 sigma
    assert np.all(e[:, 0] < 20.0)


@pytest.fixture(scope="module")
def misaligned():
    tr = trajectory3d.generate(PLAN)
    return replay.from_simulation(tr, imu.IMUConfig(), np.random.default_rng(6), mount_yaw_deg=8.0, **SENSORS)


# IMU noise as EKF2 states it in the real logs (0.015 rad/s and 0.35 m/s^2 per
# 10 ms step): the heading then leans on the sideslip as much as on the GNSS.
EKF2_IMU = imu.IMUConfig(gyro_noise=0.0015, accel_noise=0.035, gyro_bias_rw=1e-4, accel_bias_rw=3e-4,
                         gyro_bias0=0.01, accel_bias0=0.2)


def test_misalignment_biases_the_heading_through_the_sideslip(misaligned):
    """The real logs of step 7 showed it: an autopilot turned by 8 deg makes
    the zero-sideslip measurement pull the heading (by ~3 deg there).
    Without magnetometer, as the real one was too poor to hold the heading."""
    e = errors(replay.run(misaligned, full_config(imu=EKF2_IMU, mag=None)), misaligned)
    assert e["head_mean"] > 1.5


def test_sideslip_offset_state_recovers_the_misalignment(misaligned):
    """The offset converges to -8 deg give or take the real sideslip of the
    simulated airframe in turns (up to 1.5 deg, mostly one way here)."""
    log = replay.run(misaligned, full_config(imu=EKF2_IMU, mag=None, sideslip_offset_sigma0_deg=15.0))
    e = errors(log, misaligned)
    sl = log.blocks["beta_offset"]
    assert np.degrees(log.extra[-1, sl.start - 15]) == pytest.approx(-8.0, abs=1.5)
    assert abs(e["head_mean"]) < 0.5


def test_airspeed_scale_state_recovers_a_calibration_error():
    tr = trajectory3d.generate(PLAN)
    data = replay.from_simulation(tr, imu.IMUConfig(), np.random.default_rng(7), **SENSORS)
    data.tas = data.tas * 1.2                       # Pitot reading 20 % high
    cfg = full_config(pitot=aiding.PitotConfig(scale_sigma0=0.3))
    log = replay.run(data, cfg)
    s = log.extra[-1, log.blocks["tas_scale"].start - 15]
    assert s == pytest.approx(0.2, abs=0.03)
    assert errors(log, data)["wind"] < 0.6


def test_heading_mode_ignores_the_field_intensity_and_inclination():
    """Measured on the real flights: field 15 % weaker and 5 deg less inclined
    than the WMM. The heading fusion only uses the declination: scaling the
    measurement or tilting the field in the vertical plane changes nothing."""
    ekf = make_filter(0.1, 0.7)
    R = Rotation.from_quat(ekf.q[0, [1, 2, 3, 0]])
    outs = []
    for total, incl in ((0.47, 64.0), (0.40, 59.0)):
        e = make_filter(0.1, 0.7)
        m_b = R.inv().apply(aiding.earth_field(total, incl, 1.5)) + [0.002, -0.001, 0.0]
        y, H = spy_update(e, lambda f, m_b=m_b: f.update_mag_heading(m_b[None], 0.2, np.radians(1.5)))
        outs.append((y, H))
    np.testing.assert_allclose(outs[0][0], outs[1][0], atol=1e-3)
    np.testing.assert_array_equal(outs[0][1], outs[1][1])


# --- ULog reader -------------------------------------------------------------------

def test_geodetic_to_ned_against_ecef():
    """Our local frame (PX4's azimuthal equidistant projection on a sphere)
    against an exact conversion through Cartesian coordinates on the same
    sphere, 2 km away: they differ by d^3 / (6 R^2), well below 1 mm."""
    ulog_reader = pytest.importorskip("navsim.ulog_reader")
    a, e2 = ulog_reader.R_EARTH, 0.0

    def ecef(lat, lon, h):
        la, lo = np.radians(lat), np.radians(lon)
        n = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
        return np.array([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                         (n * (1 - e2) + h) * np.sin(la)])

    lat0, lon0, h0 = 49.09, 2.11, 0.0
    la, lo = np.radians(lat0), np.radians(lon0)
    Rm = np.array([[-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                   [-np.sin(lo), np.cos(lo), 0.0]])
    for dlat, dlon in ((0.018, 0.0), (0.0, 0.027), (-0.012, 0.02)):
        ned = ulog_reader.geodetic_to_ned(np.array([lat0 + dlat]), np.array([lon0 + dlon]), np.array([h0]),
                                          lat0, lon0, h0)[0]
        ne = Rm @ (ecef(lat0 + dlat, lon0 + dlon, h0) - ecef(lat0, lon0, h0))
        np.testing.assert_allclose(ned[:2], ne, atol=0.002)


def test_ulog_reader_on_a_real_log():
    pytest.importorskip("pyulog")
    pytest.importorskip("pygeomag")
    from navsim import ulog_reader

    d = ulog_reader.load(SAMPLE)
    assert d.info["sys"]["ver_hw"] == "CUBEPILOT_CUBEORANGE"
    assert len(d.t_gnss) == 32 and len(d.t_mag) > 0 and len(d.t_baro) > 0 and len(d.t_tas) > 0
    assert d.info["ekf2_origin"] is False             # this log has no EKF2 global origin
    np.testing.assert_allclose(d.gnss_p[0], 0.0, atol=1e-9)
    assert np.all(np.diff(d.t_imu) > 0) and np.all(d.dt_imu > 0)
    # magnetic field of Norway (63.4 N, 10.4 E) in 2021: ~0.52 G, inclination ~76 deg
    f = d.field_ned
    assert np.linalg.norm(f) == pytest.approx(0.52, abs=0.02)
    assert np.degrees(np.arctan2(f[2], np.hypot(f[0], f[1]))) == pytest.approx(76.0, abs=2.0)
    # the IMU increments are those of the logged rates
    from pyulog import ULog
    u = ULog(str(SAMPLE), ["sensor_combined", "vehicle_gps_position"])
    sc = [x for x in u.data_list if x.name == "sensor_combined"][0].data
    # each GNSS fix is dated back by EKF2_GPS_DELAY (110 ms in this log)
    g = [x for x in u.data_list if x.name == "vehicle_gps_position"][0].data
    ok = g["fix_type"] >= 3
    np.testing.assert_allclose(d.t_gnss, (g["timestamp"][ok] - u.start_timestamp) * 1e-6 - 0.110, atol=1e-9)
    k = 10
    np.testing.assert_allclose(d.dtheta[k], [sc[f"gyro_rad[{i}]"][k] * sc["gyro_integral_dt"][k] * 1e-6
                                              for i in range(3)], rtol=1e-6)


def test_baro_config_in_replay_needs_the_drift_state():
    with pytest.raises(ValueError):
        ErrorStateEKF(imu.IMUConfig(), gnss.GNSSConfig(), 0.005, 1,
                      aiding=aiding.AidingConfig(baro=BaroConfig(drift_sigma=0.0)))


# --- simulated logs: the generator itself -------------------------------------------

PERFECT = imu.IMUConfig(gyro_noise=0.0, gyro_bias0=0.0, gyro_bias_rw=0.0,
                        accel_noise=0.0, accel_bias0=0.0, accel_bias_rw=0.0)


@pytest.mark.parametrize("mount_yaw_deg", [0.0, 8.0])
def test_simulated_log_with_a_perfect_imu_reproduces_its_reference(mount_yaw_deg):
    """The generator merges increments and turns them into the autopilot
    frame; the reference attitude is that of the autopilot. A strapdown
    written here, with the uneven intervals of the log, must then follow the
    reference: an increment merged wrongly, or measured in one frame and
    referenced in another, shows up at once (8 deg of mounting yaw gives an
    8 deg error)."""
    tr = trajectory3d.generate(dataclasses.replace(trajectory3d.MISSION_WIND, duration=120.0))
    mag = aiding.MagConfig(noise=0.0, bias0=0.0, bias_rw=0.0)
    d = replay.from_simulation(tr, PERFECT, np.random.default_rng(0), mag_cfg=mag, mount_yaw_deg=mount_yaw_deg)
    g = np.array([0.0, 0.0, replay.G])
    q, v, p = d.ref_q[0].copy(), d.ref_v[0].copy(), d.ref_p[0].copy()
    i_ref = np.searchsorted(d.t_ref, d.t_imu)
    np.testing.assert_allclose(d.t_ref[i_ref], d.t_imu, atol=1e-9)
    worst = np.zeros(3)
    for k in range(len(d.t_imu)):
        dth, dv, dt = d.dtheta[k], d.dvel[k], d.dt_imu[k]
        v_new = v + quat_rotate(q, dv + 0.5 * np.cross(dth, dv)) + g * dt
        p = p + 0.5 * (v + v_new) * dt
        v = v_new
        q = quat_normalize(quat_mul(q, quat_from_rotvec(dth)))
        i = i_ref[k]
        worst = np.maximum(worst, [np.linalg.norm(rotvec_from_quat(quat_mul(quat_conj(d.ref_q[i]), q))),
                                   np.linalg.norm(v - d.ref_v[i]), np.linalg.norm(p - d.ref_p[i])])
    assert worst[0] < np.radians(0.001)        # 120 s: 5e-5 deg in practice
    assert worst[1] < 0.005                    # m/s
    assert worst[2] < 0.05                     # m
    # a perfect magnetometer measures the field in the autopilot frame
    i_mag = np.searchsorted(d.t_ref, d.t_mag)
    expected = quat_rotate(quat_conj(d.ref_q[i_mag]), np.broadcast_to(d.field_ned, (len(i_mag), 3)))
    np.testing.assert_allclose(d.mag, expected, atol=1e-12)


# --- comparison with the reference, outages -----------------------------------------

def fake_replay_log(t, p, v, q, sigma, gnss, extra=None, blocks=None):
    E = len(t)
    return replay.ReplayLog(t=np.asarray(t, float), p=np.asarray(p, float), v=np.asarray(v, float),
                            q=np.asarray(q, float), sigma=np.asarray(sigma, float), blocks=blocks or {},
                            extra=np.zeros((E, 0)) if extra is None else np.asarray(extra, float),
                            bg=np.zeros((E, 3)), ba=np.zeros((E, 3)), gnss=np.asarray(gnss, float),
                            gnss_status=np.zeros(E, np.int8), nis={}, wind_started=None, info={})


def test_compare_sign_conventions():
    """Position, velocity, Euler angles and wind: estimate minus reference.
    datt is the rotation from the estimate to the reference in NED, so its
    z component is the reference heading minus the estimated one (the step 7
    script uses -datt[:, 2] as the heading error). The reference is
    interpolated at the replay epochs, the heading difference is wrapped."""
    t_ref = np.arange(0.0, 11.0)
    ref_p = np.column_stack([10.0 * t_ref, -2.0 * t_ref, np.full(11, -100.0)])
    ref_v = np.tile([10.0, -2.0, 0.0], (11, 1))
    yaw_ref = np.radians(179.0)
    ref_q = np.tile(quat_from_euler(0.02, -0.03, yaw_ref), (11, 1))
    ref_wind = np.tile([3.0, -1.0], (11, 1))
    data = replay.LogData(name="x", t_imu=np.zeros(1), dt_imu=np.ones(1), dtheta=np.zeros((1, 3)),
                          dvel=np.zeros((1, 3)), t_gnss=np.zeros(1), gnss_p=np.zeros((1, 3)),
                          gnss_v=np.zeros((1, 3)), gnss_sigma=np.zeros((1, 3)), t_ref=t_ref, ref_p=ref_p,
                          ref_v=ref_v, ref_q=ref_q, ref_wind=ref_wind)
    t = np.array([2.5, 3.5, 20.0])            # the last epoch is after the reference: dropped
    d_yaw = np.radians(2.0)                   # estimate 2 deg to the right: 181 deg = -179 deg
    q_est = quat_mul(quat_from_euler(0.0, 0.0, d_yaw), ref_q[0])       # rotation about NED down
    log = fake_replay_log(t, np.column_stack([10.0 * t + 1.0, -2.0 * t, np.full(3, -103.0)]),
                          np.tile([10.5, -2.0, 0.0], (3, 1)), np.tile(q_est, (3, 1)), np.ones((3, 17)),
                          np.zeros((3, 6)), extra=np.tile([3.5, -1.0], (3, 1)), blocks={"wind": slice(15, 17)})
    c = replay.compare(log, data)
    np.testing.assert_array_equal(c["t"], [2.5, 3.5])
    np.testing.assert_allclose(c["dp"], [[1.0, 0.0, -3.0]] * 2, atol=1e-12)
    np.testing.assert_allclose(c["dv"], [[0.5, 0.0, 0.0]] * 2, atol=1e-12)
    np.testing.assert_allclose(c["dyaw"], d_yaw, atol=1e-9)
    np.testing.assert_allclose(c["datt"], [[0.0, 0.0, -d_yaw]] * 2, atol=1e-9)
    np.testing.assert_allclose(c["dwind"], [[0.5, 0.0]] * 2, atol=1e-12)


def test_outage_windows():
    w = replay.outage_windows((100.0, 500.0))
    assert w == ((130.0, 160.0), (220.0, 250.0), (310.0, 340.0), (400.0, 430.0))
    assert all(100.0 + 30.0 <= a and b <= 500.0 - 30.0 for a, b in w)
    assert replay.outage_windows((100.0, 180.0)) == ()            # too short for one outage
    assert replay.outage_windows((0.0, 200.0), duration=20.0, spacing=50.0, margin=10.0) == (
        (10.0, 30.0), (60.0, 80.0), (110.0, 130.0), (160.0, 180.0))


def test_outage_errors_on_a_known_log():
    """Error at the end of an outage: the last dead reckoning estimate,
    carried to the time of the first fix after the outage by its velocity,
    against that fix. Announced sigma: horizontal, from that last estimate."""
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    p = [[0, 0, 0], [1, 0, 0], [2, 1, 0], [9, 9, 9], [9, 9, 9]]
    v = [[1, 0, 0], [1, 0, 0], [1, 2, 0], [0, 0, 0], [0, 0, 0]]
    sigma = np.ones((5, 15))
    sigma[2, :2] = [3.0, 4.0]
    gnss = np.zeros((5, 6))
    gnss[3, :3] = [3.0, 7.0, 50.0]         # vertical ignored
    log = fake_replay_log(t, p, v, np.tile([1.0, 0, 0, 0], (5, 1)), sigma, gnss)
    e = replay.outage_errors(log, ((1.5, 3.0), (3.5, 10.0)))   # the second ends after the log: skipped
    # estimate carried to t=3: (2, 1) + (1, 2) * 1 = (3, 3); fix (3, 7): error 4
    np.testing.assert_allclose(e, [[4.0, 5.0]])


# --- replay loop: thresholds and protections ----------------------------------------

def test_airspeed_below_the_minimum_is_not_fused(simulated):
    """No Pitot or sideslip update below min_airspeed (taxi, take-off run),
    and the wind starts on the first airspeed above it."""
    d = dataclasses.replace(simulated, tas=simulated.tas.copy())
    slow = (d.t_tas < 30.0) | ((d.t_tas > 100.0) & (d.t_tas < 120.0))
    d.tas[slow] = 5.0
    log = replay.run(d, full_config(mag=aiding.MagConfig(fusion="heading")))
    for name in ("pitot", "sideslip"):
        t = log.nis[name][0]
        assert len(t) > 1000
        assert not np.any(t < 30.0) and not np.any((t > 100.0) & (t < 120.0)), name
    assert log.wind_started == d.t_tas[d.t_tas >= 30.0][0]


def test_gnss_reset_in_the_replay(simulated):
    """A 25 m jump of the GNSS for 40 s with a 10 s reset delay: rejected
    for 10 s, then the filter restarts on the fix, with the accuracy of the
    fix; same when the jump ends."""
    d = dataclasses.replace(simulated, gnss_p=simulated.gnss_p.copy())
    jump = (d.t_gnss >= 80.0) & (d.t_gnss < 120.0)
    d.gnss_p[jump] += [0.0, 25.0, 0.0]
    cfg = full_config(mag=aiding.MagConfig(fusion="heading"), gnss_reset_after_s=10.0)
    log = replay.run(d, cfg)
    st = lambda a, b: log.gnss_status[(log.t >= a) & (log.t < b)]
    assert np.all(st(80.0, 89.9) == 1)
    assert np.mean(st(92.0, 120.0) == 0) > 0.95
    assert np.all(st(120.0, 129.9) == 1)
    assert np.mean(st(132.0, 240.0) == 0) > 0.95
    resets = np.flatnonzero((log.gnss_status == 1) & np.all(log.p == log.gnss[:, :3], axis=1))
    np.testing.assert_allclose(log.t[resets], [90.0, 130.0], atol=0.21)
    for j in resets:
        s = replay._sigma(d.gnss_sigma[0], cfg.gnss_floor)
        np.testing.assert_allclose(log.sigma[j, :6], s, rtol=1e-12)
        np.testing.assert_array_equal(log.v[j], log.gnss[j, 3:])


def test_config_like_ekf2_with_air_data_and_magnetometer():
    t = lambda hz, n: np.arange(n) / hz
    data = replay.LogData(name="x", t_imu=np.zeros(1), dt_imu=np.full(1, 0.004), dtheta=np.zeros((1, 3)),
                          dvel=np.zeros((1, 3)), t_gnss=np.zeros(1), gnss_p=np.zeros((1, 3)),
                          gnss_v=np.zeros((1, 3)), gnss_sigma=np.zeros((1, 3)),
                          t_baro=t(20.0, 201), t_mag=t(50.0, 501), t_tas=t(10.0, 101),
                          field_ned=np.array([0.2, 0.01, 0.45]),
                          info={"params": {"EKF2_BARO_NOISE": 2.0, "EKF2_HEAD_NOISE": 0.2, "EKF2_EAS_NOISE": 1.0,
                                           "EKF2_GPS_P_NOISE": 0.4, "EKF2_GPS_V_NOISE": 0.2,
                                           "EKF2_MAG_NOISE": 0.04}})
    cfg = replay.config_like_ekf2(data)
    assert cfg.baro.rate_hz == pytest.approx(20.0) and cfg.baro.noise_sigma == 2.0
    assert cfg.mag.fusion == "heading" and cfg.mag.rate_hz == pytest.approx(50.0)
    assert cfg.mag.heading_sigma_deg == pytest.approx(np.degrees(0.2))
    assert cfg.mag.field_ned == (0.2, 0.01, 0.45) and cfg.mag.noise == 0.04
    assert cfg.pitot.rate_hz == pytest.approx(10.0) and cfg.pitot.noise == 1.0 and cfg.pitot.scale_sigma0 == 0.3
    assert cfg.gnss_floor == pytest.approx((0.4, 0.6, 0.2))
    assert cfg.sideslip_sigma_deg == 6.0 and cfg.sideslip_offset_sigma0_deg == 15.0
    assert cfg.wind.rw == 0.1
    cfg = replay.config_like_ekf2(data, tas_scale=False, sideslip_sigma_deg=None, mag="3d", gate_prob=None)
    assert cfg.pitot.scale_sigma0 is None and cfg.sideslip_offset_sigma0_deg is None
    assert cfg.mag.fusion == "3d" and cfg.gate_prob is None
    cfg = replay.config_like_ekf2(data, air_data=False, mag=None)
    assert cfg.baro is None and cfg.mag is None and cfg.pitot is None and cfg.wind is None


# --- ULog reader on synthetic logs --------------------------------------------------
#
# The real sample log has no EKF2 origin, the old position format and a single
# airspeed topic. These fake logs go through the other branches, with values
# known by construction. pyulog is replaced by a stand-in object with the same
# attributes, so these tests need neither pyulog nor pygeomag.

def px4_reproject(north, east, lat0, lon0):
    """Inverse of PX4's azimuthal equidistant projection (map_projection_reproject)."""
    from navsim.ulog_reader import R_EARTH

    la0, lo0 = np.radians(lat0), np.radians(lon0)
    x, y = np.asarray(north) / R_EARTH, np.asarray(east) / R_EARTH
    c = np.hypot(x, y)
    sc = np.where(c > 0, np.sin(c) / np.where(c > 0, c, 1.0), 1.0)
    lat = np.arcsin(np.cos(c) * np.sin(la0) + x * sc * np.cos(la0))
    lon = lo0 + np.arctan2(y * sc, np.cos(la0) * np.cos(c) - x * sc * np.sin(la0))
    return np.degrees(lat), np.degrees(lon)


@pytest.mark.parametrize("lat0, lon0", [(63.4, 10.4), (-45.0, 170.0), (0.0, -179.9)])
def test_geodetic_to_ned_inverts_the_px4_reprojection(lat0, lon0):
    from navsim.ulog_reader import geodetic_to_ned

    ne = np.array([[0.0, 0.0], [1000.0, -500.0], [-30000.0, 25000.0], [12.3, 45.6]])
    lat, lon = px4_reproject(ne[:, 0], ne[:, 1], lat0, lon0)
    out = geodetic_to_ned(lat, lon, [100.0, 90.0, 150.0, 101.0], lat0, lon0, 100.0)
    np.testing.assert_allclose(out[:, :2], ne, atol=1e-6)
    np.testing.assert_allclose(out[:, 2], [0.0, 10.0, -50.0, -1.0])


class FakeULog:
    def __init__(self, topics, params, start=1_000_000):
        self.data_list = [type("D", (), {"name": n, "multi_id": 0, "data": d})() for n, d in topics.items()]
        self.initial_parameters = params
        self.start_timestamp = start
        self.msg_info_dict = {"ver_hw": "FAKE", "unrelated": "x"}


ORIGIN = (45.0, 5.0, 300.0)                 # the last EKF2 origin
PARAMS = {"EKF2_GPS_DELAY": 110.0, "EKF2_BARO_DELAY": 20.0, "EKF2_MAG_DELAY": 5.0, "EKF2_ASP_DELAY": 100.0,
          "EKF2_GYR_NOISE": 0.02, "OTHER": 1.0}
UTC_2023_07_02 = 1688256000 * 1e6          # 2023-07-02 00:00 UTC, day 183


def fake_topics(position_format="int", ekf2_origin=True, airspeed_counts=(60, 20)):
    us = lambda s: (1_000_000 + np.asarray(s) * 1e6).astype(np.uint64)
    n = 500                                  # IMU, 250 Hz
    t = np.arange(n) * 0.004
    dt_us = np.full(n, 4000)
    dt_us[[10, 11]] = 0                      # dropped samples
    dt_us[20] = 60000                        # a gap: dropped
    sc = {"timestamp": us(t), "gyro_integral_dt": dt_us, "accelerometer_integral_dt": np.full(n, 4000)}
    for i, (w, a) in enumerate(zip((0.01, 0.02, -0.03), (0.1, 0.2, -9.8), strict=True)):
        sc[f"gyro_rad[{i}]"], sc[f"accelerometer_m_s2[{i}]"] = np.full(n, w), np.full(n, a)

    tg = np.arange(10) * 0.2                 # GNSS, 5 Hz
    ne = np.column_stack([100.0 * tg, -40.0 * tg])
    lat, lon = px4_reproject(ne[:, 0], ne[:, 1], *ORIGIN[:2])
    alt = ORIGIN[2] + 10.0 + tg
    g = {"timestamp": us(tg), "timestamp_sample": np.zeros(10), "fix_type": np.array([2] + [3] * 9),
         "vel_ned_valid": np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1]), "vel_n_m_s": np.full(10, 100.0),
         "vel_e_m_s": np.full(10, -40.0), "vel_d_m_s": np.full(10, -1.0), "eph": np.full(10, 1.5),
         "epv": np.full(10, 2.5), "s_variance_m_s": np.full(10, 0.4),
         "time_utc_usec": np.concatenate([[0.0], np.full(9, UTC_2023_07_02)])}
    if position_format == "int":
        g.update(lat=np.round(lat * 1e7), lon=np.round(lon * 1e7), alt=np.round(alt * 1e3))
    else:
        g.update(latitude_deg=lat, longitude_deg=lon, altitude_msl_m=alt)

    tl = np.arange(20) * 0.1                 # EKF2 local position: origin moved once at t = 1 s
    valid = np.array([0] * 3 + [1] * 17, dtype=bool) & ekf2_origin
    lp = {"timestamp": us(tl), "xy_global": valid, "z_global": valid,
          "ref_lat": np.where(tl < 1.0, 44.0, ORIGIN[0]), "ref_lon": np.where(tl < 1.0, 4.0, ORIGIN[1]),
          "ref_alt": np.where(tl < 1.0, 0.0, ORIGIN[2])}
    for i, c in enumerate("xyz"):
        lp[c], lp["v" + c] = tl * (i + 1), np.full(20, float(i))
    att = {"timestamp": us(tl)}
    for i in range(4):
        att[f"q[{i}]"] = np.full(20, (1.0, 0.0, 0.0, 0.0)[i])
    wind = {"timestamp": us(tl), "windspeed_north": tl, "windspeed_east": -tl}

    tb = np.arange(20) * 0.1
    air = {"timestamp": us(tb), "timestamp_sample": us(tb) - 3000, "baro_alt_meter": ORIGIN[2] + 7.0 + tb}
    mag = {"timestamp": us(tb), "timestamp_sample": us(tb)[::-1]}     # not increasing: timestamp used
    for i in range(3):
        mag[f"magnetometer_ga[{i}]"] = np.full(20, 0.1 * (i + 1))
    topics = {"sensor_combined": sc, "vehicle_gps_position": g, "vehicle_local_position": lp,
              "vehicle_attitude": att, "wind": wind, "vehicle_air_data": air, "vehicle_magnetometer": mag,
              "vehicle_land_detected": {"timestamp": us([0.0, 0.5, 1.5, 1.9]), "landed": np.array([1, 0, 0, 1])}}
    for name, count in zip(("airspeed_validated", "airspeed"), airspeed_counts, strict=True):
        ta = np.arange(count) * 0.03
        tas = np.full(count, 20.0 if name == "airspeed_validated" else 30.0)
        tas[1] = np.nan
        topics[name] = {"timestamp": us(ta), "true_airspeed_m_s": tas}
    return topics, ne, alt


def load_fake(monkeypatch, topics, params=PARAMS):
    import sys
    import types

    from navsim import ulog_reader

    monkeypatch.setitem(sys.modules, "pyulog", types.SimpleNamespace(ULog=lambda path: FakeULog(topics, params)))
    calls = []
    monkeypatch.setattr(ulog_reader, "wmm_field", lambda *a: calls.append(a) or np.array([0.2, 0.0, 0.4]))
    return ulog_reader.load("fake_flight.ulg"), calls


@pytest.mark.parametrize("position_format", ["int", "deg"])
def test_ulog_reader_on_a_synthetic_log(monkeypatch, position_format):
    topics, ne, alt = fake_topics(position_format)
    d, calls = load_fake(monkeypatch, topics)
    assert d.name == "fake_flight" and d.info["sys"] == {"ver_hw": "FAKE"}
    assert d.info["params"] == {"EKF2_ASP_DELAY": 100.0, "EKF2_BARO_DELAY": 20.0, "EKF2_GPS_DELAY": 110.0,
                                "EKF2_GYR_NOISE": 0.02, "EKF2_MAG_DELAY": 5.0}
    # IMU: samples with a zero or too long interval dropped, rates turned into increments
    assert len(d.t_imu) == 497
    np.testing.assert_allclose(d.dt_imu, 0.004)
    np.testing.assert_allclose(d.dtheta, np.tile([0.01, 0.02, -0.03], (497, 1)) * 0.004)
    np.testing.assert_allclose(d.dvel, np.tile([0.1, 0.2, -9.8], (497, 1)) * 0.004)
    assert d.info["imu_rate_hz"] == pytest.approx(250.0)
    # EKF2 origin: the last one, and the EKF2 positions are kept
    assert d.info["ekf2_origin"] is True and d.info["origin"] == ORIGIN
    np.testing.assert_allclose(d.ref_p[:, 0], topics["vehicle_local_position"]["x"])
    # GNSS: fixes without 3D fix or valid velocity dropped; timestamp_sample all
    # zero -> timestamp; dated back by EKF2_GPS_DELAY; NED in the EKF2 frame
    keep = [1, 3, 4, 5, 6, 7, 8, 9]
    np.testing.assert_allclose(d.t_gnss, np.array(keep) * 0.2 - 0.110, atol=1e-9)
    tol = 0.02 if position_format == "int" else 1e-6     # 1e-7 deg and mm quantisation
    np.testing.assert_allclose(d.gnss_p[:, :2], ne[keep], atol=tol)
    np.testing.assert_allclose(d.gnss_p[:, 2], -(alt[keep] - ORIGIN[2]), atol=1e-3)
    np.testing.assert_allclose(d.gnss_v, np.tile([100.0, -40.0, -1.0], (8, 1)))
    np.testing.assert_allclose(d.gnss_sigma, np.tile([1.5, 2.5, 0.4], (8, 1)))
    # baro: timestamp_sample when it increases, delay, altitude above the origin
    np.testing.assert_allclose(d.t_baro, np.arange(20) * 0.1 - 0.003 - 0.020, atol=1e-9)
    np.testing.assert_allclose(d.baro_alt, 7.0 + np.arange(20) * 0.1)
    # magnetometer: timestamp_sample not increasing -> timestamp
    np.testing.assert_allclose(d.t_mag, np.arange(20) * 0.1 - 0.005, atol=1e-9)
    np.testing.assert_allclose(d.mag, np.tile([0.1, 0.2, 0.3], (20, 1)))
    # airspeed: the topic with more samples, NaN dropped
    assert d.info["airspeed_topic"] == "airspeed_validated"
    assert len(d.tas) == 59 and np.all(d.tas == 20.0)
    np.testing.assert_allclose(d.t_tas[:2], [0.0 - 0.1, 0.06 - 0.1], atol=1e-9)
    # wind of EKF2, interpolated at the reference times; flight interval
    np.testing.assert_allclose(d.ref_wind, np.column_stack([d.t_ref, -d.t_ref]), atol=1e-12)
    assert d.info["airborne"] == pytest.approx((0.5, 1.5))
    # Earth field: WMM at the origin, on the UTC date of the flight
    (lat0, lon0, alt0, year), = calls
    assert (lat0, lon0, alt0) == ORIGIN
    assert year == pytest.approx(2023 + 182 / 365.25)


def test_ulog_reader_without_ekf2_origin_and_with_raw_airspeed(monkeypatch):
    topics, ne, alt = fake_topics(ekf2_origin=False, airspeed_counts=(10, 40))
    del topics["wind"]
    d, calls = load_fake(monkeypatch, topics, params={})
    assert d.info["ekf2_origin"] is False
    assert np.all(np.isnan(d.ref_p))
    # origin: the first fix with a 3D fix and a valid velocity (index 1)
    np.testing.assert_allclose(d.info["origin"][:2], [topics["vehicle_gps_position"]["lat"][1] * 1e-7,
                                                       topics["vehicle_gps_position"]["lon"][1] * 1e-7])
    np.testing.assert_allclose(d.gnss_p[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(d.gnss_p[1, :2], ne[3] - ne[1], atol=0.03)
    # no delay parameters: times as logged
    np.testing.assert_allclose(d.t_gnss[0], 0.2, atol=1e-9)
    assert d.info["airspeed_topic"] == "airspeed" and np.all(d.tas == 30.0)
    assert d.ref_wind is None
