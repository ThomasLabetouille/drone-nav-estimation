"""Step 8: the C++ port against the Python reference.

The C++ filter is built from cpp/ in a fresh directory, then both versions
replay the same simulated logs. They must agree to rounding: same states at
every GNSS epoch, same NIS, same gate decisions, same resets. Simulated logs
are used so that every branch of the loop is exercised (outage, rejections,
GNSS and wind resets); the real logs are compared by scripts/step8_cpp.py.
"""

from __future__ import annotations

import dataclasses
import subprocess

import numpy as np
import pytest
from scipy.stats import chi2

from navsim import aiding, cpp_bridge, imu, replay, trajectory3d
from navsim.rotations import quat_conj, quat_mul, rotvec_from_quat

pytestmark = pytest.mark.skipif(not cpp_bridge.available(), reason="needs cmake and a C++ compiler")


@pytest.fixture(scope="module")
def build_dir(tmp_path_factory):
    try:
        return cpp_bridge.build(tmp_path_factory.mktemp("navcpp_build"))
    except subprocess.CalledProcessError as e:     # Eigen missing, compile error...
        pytest.fail(f"C++ build failed:\n{e.stdout}\n{e.stderr}")


def r3(**kw):
    base = dict(baro=aiding.BARO_10HZ, mag=aiding.MagConfig(fusion="heading", heading_sigma_deg=10.0),
                pitot=aiding.PitotConfig(scale_sigma0=0.3), wind=aiding.WindModel(rw=0.1), sideslip_sigma_deg=6.0,
                sideslip_offset_sigma0_deg=15.0)
    base.update(kw)
    return replay.ReplayConfig(**base)


SENSORS = dict(mag_cfg=aiding.MagConfig(), baro_cfg=aiding.BARO_10HZ, pitot_cfg=aiding.PitotConfig())


def simulated(seed, duration=200.0, **kw):
    tr = trajectory3d.generate(dataclasses.replace(trajectory3d.MISSION_WIND, duration=duration))
    return replay.from_simulation(tr, imu.IMUConfig(), np.random.default_rng(seed), **SENSORS, **kw)


def assert_same(py, cc):
    np.testing.assert_array_equal(py.t, cc["t"])
    np.testing.assert_array_equal(py.gnss_status, cc["gnss_status"])
    np.testing.assert_allclose(cc["p"], py.p, rtol=0, atol=1e-8)
    np.testing.assert_allclose(cc["v"], py.v, rtol=0, atol=1e-9)
    datt = rotvec_from_quat(quat_mul(py.q, quat_conj(cc["q"])))
    assert np.abs(datt).max() < 1e-11
    np.testing.assert_allclose(cc["extra"], py.extra, rtol=0, atol=1e-9)
    np.testing.assert_allclose(cc["bg"], py.bg, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cc["ba"], py.ba, rtol=0, atol=1e-11)
    np.testing.assert_allclose(cc["sigma"], py.sigma, rtol=1e-9)
    assert set(cc["nis"]) == set(py.nis)
    for name, (t, v, r) in py.nis.items():
        tc, vc, rc = cc["nis"][name]
        np.testing.assert_array_equal(t, tc)
        np.testing.assert_allclose(vc, v, rtol=1e-6, atol=1e-12)
        np.testing.assert_array_equal(r, rc)


def test_cpp_unit_tests(build_dir):
    out = subprocess.run([str(build_dir / "test_eskf")], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "all tests passed" in out.stdout


def test_same_states_on_a_misaligned_flight_with_an_outage(build_dir):
    data = simulated(1, mount_yaw_deg=8.0)
    cfg = r3(gnss_outage=((120.0, 150.0),))
    py, cc = replay.run(data, cfg), cpp_bridge.run(data, cfg, build_dir)
    assert (py.gnss_status == 2).sum() > 100
    assert_same(py, cc)


def test_same_gate_decisions_and_resets(build_dir):
    """A GNSS jump longer than the reset delay, and a Pitot that reads 30 %
    high from mid-flight on: every rejection and reset branch runs."""
    data = simulated(2)
    jump = (data.t_gnss >= 80.0) & (data.t_gnss < 120.0)
    data.gnss_p[jump] += [0.0, 25.0, 0.0]
    data.tas[data.t_tas > 140.0] *= 1.3
    cfg = r3(gnss_reset_after_s=10.0, pitot=aiding.PitotConfig(scale_sigma0=None))
    with pytest.raises(ValueError):            # the C++ port needs the scale state
        cpp_bridge.run(data, cfg, build_dir)
    cfg = r3(gnss_reset_after_s=10.0, wind=aiding.WindModel(rw=0.01, reset_after_rejected_s=2.0))
    py, cc = replay.run(data, cfg), cpp_bridge.run(data, cfg, build_dir)
    assert (py.gnss_status == 1).any()
    assert py.nis["pitot"][2].any()
    assert_same(py, cc)


def test_cpp_is_faster(build_dir):
    data = simulated(3, duration=60.0)
    cc = cpp_bridge.run(data, r3(), build_dir)
    assert cc["stats"]["us_per_event"] < 200.0           # a few microseconds on a desktop


# --- what Python hands over to the C++ program --------------------------------------

def test_events_array_follows_the_replay_loop():
    """Oracle: the order of replay.run, written as its loop (predict IMU
    sample k, then every measurement whose time is not after it), against
    the vectorised merge of events_array. Same rows, same order, payloads
    and outage flags those of the data."""
    data = simulated(4, duration=60.0)
    cfg = r3(gnss_outage=((20.0, 30.0),))
    st = replay.setup(data, cfg, None, 50.0)
    ev = cpp_bridge.events_array(data, cfg, st)
    rows, e = [], 0
    events = st["events"]
    k = st["k0"]
    while k < len(data.t_imu) and data.t_imu[k] <= st["t1"]:
        rows.append((1, data.t_imu[k]))
        while e < len(events) and events[e][0] <= data.t_imu[k]:
            rows.append(({"gnss": 2, "baro": 3, "mag": 4, "tas": 5}[events[e][1]], events[e][0]))
            e += 1
        k += 1
    np.testing.assert_array_equal(ev[:, :2], np.array(rows))
    assert ev[-1, 1] <= 50.0
    imu_rows = ev[ev[:, 0] == 1]
    ks = np.searchsorted(data.t_imu, imu_rows[:, 1])
    np.testing.assert_array_equal(imu_rows[:, 2], data.dt_imu[ks])
    np.testing.assert_array_equal(imu_rows[:, 3:9], np.hstack([data.dtheta[ks], data.dvel[ks]]))
    g = ev[ev[:, 0] == 2]
    i = np.searchsorted(data.t_gnss, g[:, 1])
    np.testing.assert_array_equal(g[:, 2:11], np.hstack([data.gnss_p[i], data.gnss_v[i], data.gnss_sigma[i]]))
    np.testing.assert_array_equal(g[:, 11] == 1.0, (g[:, 1] >= 20.0) & (g[:, 1] < 30.0))
    m = ev[ev[:, 0] == 4]
    np.testing.assert_array_equal(m[:, 2:5], data.mag[np.searchsorted(data.t_mag, m[:, 1])])
    for code, t, x in ((3, data.t_baro, data.baro_alt), (5, data.t_tas, data.tas)):
        r = ev[ev[:, 0] == code]
        np.testing.assert_array_equal(r[:, 2], x[np.searchsorted(t, r[:, 1])])


def parse_config(text):
    out = {}
    for line in text.splitlines():
        key, *vals = line.split()
        out[key] = [float(v) for v in vals]
    return out


def test_config_text_is_exact():
    """Every number reaches the C++ program unchanged (repr round trip, no
    rounding by the text format); a missing initial magnetometer or baro
    sample is written as nan."""
    data = simulated(5, duration=30.0)
    cfg = r3(gate_prob=0.99, gnss_reset_after_s=None)
    st = replay.setup(data, cfg, None, None)
    c = parse_config(cpp_bridge.config_text(data, cfg, st))
    assert c["gyro_noise"] == [cfg.imu.gyro_noise] and c["t0"] == [st["t0"]]
    assert c["heading_sigma"] == [np.radians(10.0)]
    assert c["gate_gnss"] == [chi2.ppf(0.99, 6)] and c["gate_pitot"] == [chi2.ppf(0.99, 1)]
    assert c["gnss_reset_after"] == [-1.0] and c["pitot_reset_after"] == [cfg.wind.reset_after_rejected_s]
    assert c["init_fb"] == list(st["f_b"]) and c["init_mag"] == list(st["m_b"])
    assert c["init_p"] == list(data.gnss_p[st["g0"]]) and c["init_baro"] == [st["baro0"]]
    no_mag = dataclasses.replace(data, t_mag=np.zeros(0), mag=np.zeros((0, 3)), t_baro=np.zeros(0),
                                 baro_alt=np.zeros(0))
    c = parse_config(cpp_bridge.config_text(no_mag, cfg, replay.setup(no_mag, cfg, None, None)))
    assert np.all(np.isnan(c["init_mag"])) and np.isnan(c["init_baro"][0])
    c = parse_config(cpp_bridge.config_text(data, r3(gate_prob=None), st))
    assert all(c[k] == [0.0] for k in c if k.startswith("gate_"))      # 0: no gate in the C++ filter


@pytest.mark.parametrize("change", [
    dict(baro=None), dict(mag=None), dict(mag=aiding.MagConfig(fusion="3d")), dict(pitot=None),
    dict(pitot=aiding.PitotConfig(scale_sigma0=None)), dict(wind=None), dict(sideslip_sigma_deg=None),
    dict(sideslip_offset_sigma0_deg=None), dict(gnss_delay=0.1)])
def test_configurations_the_port_does_not_cover_are_refused(change):
    with pytest.raises(ValueError):
        cpp_bridge.check_config(r3(**change))
    cpp_bridge.check_config(r3())


# --- more branches of the loop, against Python --------------------------------------

def test_same_states_without_gate_and_without_resets(build_dir):
    """gate_prob None (threshold 0 in C++) and both resets off (-1): the
    jump and the Pitot error stay rejected in the second case, and nothing is
    rejected in the first."""
    data = simulated(6)
    jump = (data.t_gnss >= 80.0) & (data.t_gnss < 120.0)
    data.gnss_p[jump] += [0.0, 25.0, 0.0]
    data.tas[data.t_tas > 140.0] *= 1.3
    cfg = r3(gate_prob=None)
    py, cc = replay.run(data, cfg), cpp_bridge.run(data, cfg, build_dir)
    assert not any(r.any() for _, _, r in py.nis.values())
    assert_same(py, cc)
    cfg = r3(gnss_reset_after_s=None, wind=aiding.WindModel(rw=0.01, reset_after_rejected_s=None))
    py, cc = replay.run(data, cfg), cpp_bridge.run(data, cfg, build_dir)
    assert np.all(py.gnss_status[(py.t >= 85.0) & (py.t < 120.0)] == 1)
    assert py.nis["pitot"][2][py.nis["pitot"][0] > 150.0].mean() > 0.9
    assert_same(py, cc)


def test_same_states_without_magnetometer_or_baro_at_the_start(build_dir):
    """No magnetometer sample (initial heading 0, no heading update) and no
    baro sample (bias started at 0): the nan path of the initialisation."""
    data = simulated(7, duration=120.0)
    data = dataclasses.replace(data, t_mag=np.zeros(0), mag=np.zeros((0, 3)), t_baro=np.zeros(0),
                               baro_alt=np.zeros(0))
    cfg = r3(yaw_sigma_deg=60.0)
    py, cc = replay.run(data, cfg), cpp_bridge.run(data, cfg, build_dir)
    assert "mag_heading" not in py.nis and "baro" not in py.nis
    assert_same(py, cc)


def test_same_states_on_a_time_window_with_slow_airspeed(build_dir):
    """A replay window inside the log, and airspeeds below min_airspeed at
    its start and in its middle: no air data update there, wind started on
    the first sample above."""
    data = simulated(8, duration=150.0)
    data.tas[(data.t_tas < 55.0) | ((data.t_tas > 80.0) & (data.t_tas < 90.0))] = 5.0
    cfg = r3()
    py = replay.run(data, cfg, t_start=40.0, t_end=110.0)
    cc = cpp_bridge.run(data, cfg, build_dir, t_start=40.0, t_end=110.0)
    assert 40.0 <= py.t[0] < 41.0 and py.t[-1] <= 110.0
    assert py.wind_started >= 55.0
    t = py.nis["pitot"][0]
    assert not np.any((t > 80.0) & (t < 90.0))
    assert_same(py, cc)


# --- nav_replay, used wrongly -------------------------------------------------------

def nav_replay(build_dir, *args):
    return subprocess.run([str(build_dir / "nav_replay"), *map(str, args)], capture_output=True, text=True)


def test_nav_replay_reports_errors(build_dir, tmp_path):
    out = nav_replay(build_dir, "only_one_argument")
    assert out.returncode == 2 and "usage" in out.stderr

    data = simulated(9, duration=20.0)
    cfg = r3()
    st = replay.setup(data, cfg, None, None)
    text = cpp_bridge.config_text(data, cfg, st)
    ev = cpp_bridge.events_array(data, cfg, st)
    files = [tmp_path / n for n in ("config.txt", "events.bin", "states.bin", "nis.bin")]

    files[0].write_text("\n".join(ln for ln in text.splitlines() if not ln.startswith("wind_rw")))
    ev.astype("<f8").tofile(files[1])
    out = nav_replay(build_dir, *files)
    assert out.returncode == 2 and "missing config key wind_rw" in out.stderr

    files[0].write_text(text.replace("init_fb", "init_fb_x"))
    out = nav_replay(build_dir, *files)
    assert out.returncode == 2 and "init_fb" in out.stderr

    files[0].write_text(text)
    bad = ev.copy()
    bad[5, 0] = 7.0
    bad.astype("<f8").tofile(files[1])
    out = nav_replay(build_dir, *files)
    assert out.returncode == 2 and "unknown event type 7" in out.stderr

    ev.astype("<f8").tofile(files[1])
    out = nav_replay(build_dir, *files)
    assert out.returncode == 0, out.stderr
    stats = dict(zip(out.stdout.split()[::2], map(float, out.stdout.split()[1::2]), strict=True))
    assert stats["events"] == len(ev) and stats["imu"] == (ev[:, 0] == 1).sum()
    assert files[2].stat().st_size == 8 * 43 * (1 + (ev[:, 0] == 2).sum())
