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
from navsim.rotations import quat_from_euler
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
