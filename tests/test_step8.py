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
