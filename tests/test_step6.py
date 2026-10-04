"""Step 6: faults, innovation gating, and the protections around it."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.stats import binomtest, chi2

from navsim import aiding, fusion, gnss, imu, trajectory3d
from navsim.eskf import ErrorStateEKF
from navsim.faults import NONE as NO_FAULTS
from navsim.faults import Faults

INIT_MAG = fusion.InitConfig(yaw_source="magnetometer", yaw_sigma_deg=15.0)
INIT_COURSE = fusion.InitConfig(yaw_source="gnss_course")


def full(**wind):
    return aiding.AidingConfig(mag=aiding.MagConfig(), baro=aiding.BARO_10HZ, pitot=aiding.PitotConfig(),
                               wind=aiding.WindModel(**wind), sideslip_sigma_deg=6.0)


def fly(plan, init, ad, faults=NO_FAULTS, gate=0.999, runs=12, seed=3, gnss_reset=45.0):
    tr = trajectory3d.generate(plan)
    lg = fusion.run(tr, imu.IMUConfig(), gnss.REALISTIC, init, runs, np.random.default_rng(seed),
                    model_gnss_bias=True, latency_mode="delayed", aiding=ad, faults=faults, gate_prob=gate,
                    gnss_reset_after_s=gnss_reset)
    return tr, lg


def horiz(lg, t):
    k = int(np.searchsorted(lg.t, t))
    return np.linalg.norm(lg.out_err[:, k, :2], axis=1), lg.nees["position"][:, k].mean() / 3


# --- faults and gate, unit level ---------------------------------------------------

def test_fault_intervals():
    f = Faults(gnss_outage=((10.0, 20.0),), gnss_jump=((5.0, 8.0, (1.0, 2.0, 3.0)),),
               mag_disturbance=((0.0, 1.0, (0.1, 0.0, 0.0)), (0.5, 2.0, (0.0, 0.2, 0.0))))
    assert not f.gnss_lost(9.99) and f.gnss_lost(10.0) and not f.gnss_lost(20.0)
    np.testing.assert_array_equal(f.gnss_offset(6.0), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(f.gnss_offset(8.0), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(f.mag_offset(0.7), [0.1, 0.2, 0.0])   # overlapping faults add up


def filter_at_rest(runs, seed=0):
    ekf = ErrorStateEKF(imu.IMUConfig(), gnss.REALISTIC, 0.005, runs, model_gnss_bias=True)
    rng = np.random.default_rng(seed)
    ekf.initialise(rng.normal(0, 1, (runs, 3)), rng.normal(0, 1, (runs, 3)),
                   np.tile([1.0, 0, 0, 0], (runs, 1)), np.eye(ekf.n) * 4.0)
    return ekf


def test_gate_rejects_outliers_and_leaves_the_filter_untouched():
    ekf = filter_at_rest(2)
    ekf.gates = {"gnss": chi2.ppf(0.999, 6)}
    p, P = ekf.p.copy(), ekf.P.copy()
    z = np.concatenate([ekf.p + ekf.bgnss, ekf.v], axis=1)
    z[0, 0] += 100.0                       # run 0: 100 m outlier; run 1: perfect measurement
    ekf.update_gnss(z)
    np.testing.assert_array_equal(ekf.rejected, [True, False])
    np.testing.assert_array_equal(ekf.p[0], p[0])
    np.testing.assert_array_equal(ekf.P[0], P[0])
    assert np.trace(ekf.P[1]) < np.trace(P[1])


def test_gate_false_alarm_rate_matches_its_probability():
    """Innovations drawn from the covariance the filter predicts must be
    rejected with the chosen probability, no more: this checks the NIS, the
    threshold and the degrees of freedom together."""
    runs = 20000
    ekf = filter_at_rest(runs)
    H, R = ekf.H_gnss, ekf.R_gnss
    S = H @ ekf.P[0] @ H.T + R
    y = np.random.default_rng(1).multivariate_normal(np.zeros(6), S, runs)
    ekf.update(y, H, R, gate=chi2.ppf(0.99, 6))
    assert binomtest(int(ekf.rejected.sum()), runs, 0.01).pvalue > 0.001


def test_random_walk_can_change_per_run():
    ad = full()
    ekf = ErrorStateEKF(imu.IMUConfig(), gnss.REALISTIC, 0.005, 3, model_gnss_bias=True, aiding=ad)
    s = ekf.blocks["wind"]
    ekf.set_random_walk("wind", np.array([0.1, 0.01, 0.1]))
    q = np.diagonal(ekf.Qd, axis1=1, axis2=2)[:, s]
    np.testing.assert_allclose(q[:, 0], np.array([0.1, 0.01, 0.1]) ** 2 * 0.005)
    np.testing.assert_allclose(q[:, 1], q[:, 0])


def filter_with_correlated_states(model_gnss_bias, seed=0):
    """A filter whose P is a dense random covariance: every state correlated
    with every other one, as after a long flight."""
    ekf = ErrorStateEKF(imu.IMUConfig(), gnss.REALISTIC, 0.005, 1, model_gnss_bias=model_gnss_bias)
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 0.5, (ekf.n, ekf.n))
    ekf.initialise(rng.normal(0, 1, (1, 3)), rng.normal(0, 1, (1, 3)), np.array([[1.0, 0, 0, 0]]), (A @ A.T)[None])
    if model_gnss_bias:
        ekf.extra[0, ekf.blocks["gnss_bias"].start - 15:ekf.blocks["gnss_bias"].stop - 15] = [0.5, -0.3, 1.2]
    return ekf, rng


@pytest.mark.parametrize("model_gnss_bias", [False, True])
def test_reset_to_gnss_gives_the_covariance_of_the_fix(model_gnss_bias):
    """Monte Carlo oracle, independent of the algebra in reset_to_gnss: draw
    errors from P and fix noises from R, apply the reset to each draw
    (p = z_p - b, v = z_v, the rest unchanged), and compare the empirical
    covariance of the new errors with the P the filter announces. Until this
    test was written, the position lost its correlations with the states the
    GNSS bias is correlated with, and P could have negative eigenvalues."""
    ekf, rng = filter_with_correlated_states(model_gnss_bias)
    P_old = ekf.P[0].copy()
    z = np.array([[10.0, -20.0, 5.0, 1.0, 2.0, -0.5]])
    ekf.reset_to_gnss(np.array([True]), z)
    np.testing.assert_allclose(ekf.p[0], z[0, :3] - ekf.bgnss[0])
    np.testing.assert_allclose(ekf.v[0], z[0, 3:])

    N = 400_000
    e = rng.multivariate_normal(np.zeros(ekf.n), P_old, N)
    noise = rng.multivariate_normal(np.zeros(6), ekf.R_gnss, N)
    new = e.copy()
    new[:, 0:3] = noise[:, 0:3] - (e[:, ekf.blocks["gnss_bias"]] if model_gnss_bias else 0.0)
    new[:, 3:6] = noise[:, 3:6]
    C = np.cov(new.T)
    scale = np.sqrt(np.outer(np.diag(C), np.diag(C)))
    assert np.abs(ekf.P[0] - C).max() / scale.max() < 0.01
    assert np.abs((ekf.P[0] - C) / scale).max() < 0.02     # correlations within 0.02
    assert np.linalg.eigvalsh(ekf.P[0]).min() > 0.0
    np.testing.assert_array_equal(ekf.P[0], ekf.P[0].T)
    # the states other than position and velocity are untouched
    np.testing.assert_array_equal(ekf.P[0][6:, 6:], P_old[6:, 6:])


def test_reset_to_gnss_only_touches_the_runs_in_the_mask():
    ekf = filter_at_rest(3)
    ekf.P[:] = ekf.P[0] + 0.1                    # correlated
    p, v, P = ekf.p.copy(), ekf.v.copy(), ekf.P.copy()
    z = np.zeros((3, 6))
    ekf.reset_to_gnss(np.array([False, True, False]), z)
    for i in (0, 2):
        np.testing.assert_array_equal(ekf.p[i], p[i])
        np.testing.assert_array_equal(ekf.v[i], v[i])
        np.testing.assert_array_equal(ekf.P[i], P[i])
    assert not np.array_equal(ekf.P[1], P[1])
    ekf.reset_to_gnss(np.array([False, False, False]), z)    # nothing to do


def test_false_alarm_rate_of_each_sensor_without_faults(outage):
    """Before the outage nothing is wrong, so each gate at 99.9 % must reject
    about 0.1 % of its measurements: magnetometer, baro and Pitot alike. The
    sideslip is a pseudo-measurement (beta = 0) with a deliberately large
    sigma (6 deg against an actual sideslip of a fraction of a degree): its
    gate must never trip without a fault. The magnetometer rate also checks
    the Joseph form: with the bias states frozen (consider states), the gain
    is not optimal, and the short form (I - KH) P gives no rejection at all."""
    lg = outage["best"]
    for name in ("mag", "baro", "pitot"):
        t = lg.aux_nis[name][0]
        r = lg.aux_rejected[name][:, (t > 5.0) & (t < 40.0)]
        assert r.size > 3000, name
        assert binomtest(int(r.sum()), r.size, 0.001).pvalue > 0.001, (name, r.mean())
    t = lg.aux_nis["sideslip"][0]
    assert lg.aux_rejected["sideslip"][:, (t > 5.0) & (t < 40.0)].sum() == 0


# --- system level -----------------------------------------------------------------

OUTAGE_PLAN = dataclasses.replace(trajectory3d.MISSION_WIND, duration=110.0)
OUTAGE = Faults(gnss_outage=((40.0, 100.0),))


@pytest.fixture(scope="module")
def outage():
    return {key: fly(OUTAGE_PLAN, init, ad, OUTAGE)[1] for key, init, ad in (
        ("imu", INIT_COURSE, None),
        ("step5", INIT_MAG, full(rw=0.1)),
        ("best", INIT_MAG, full(rw=0.1, rw_without_gnss=0.01)),
    )}


def test_outage_is_applied_and_gnss_is_accepted_again(outage):
    lg = outage["best"]
    during = (lg.t >= 40.0) & (lg.t < 100.0)
    assert np.all(lg.gnss_status[:, during] == 2)
    assert np.all(np.isnan(lg.nis[:, during]))
    after = lg.t >= 100.0
    assert (lg.gnss_status[:, after] == 1).mean() < 0.02      # no lock-out after 60 s


def test_gnss_false_alarm_rate_in_flight(outage):
    """Before the outage, nothing is wrong: the gate at 99.9 % must reject
    about 0.1 % of the fixes (here ~2 of 2300). A gate on the wrong number of
    degrees of freedom (3 instead of 6) would reject ~1 %."""
    lg = outage["best"]
    before = (lg.t > 5.0) & (lg.t < 40.0)
    status = lg.gnss_status[:, before]
    assert binomtest(int((status == 1).sum()), status.size, 0.001).pvalue > 0.001


def test_dead_reckoning_with_air_data_beats_inertial(outage):
    e_imu, _ = horiz(outage["imu"], 99.8)
    e_step5, _ = horiz(outage["step5"], 99.8)
    e_best, nees = horiz(outage["best"], 99.8)
    rms = lambda e: np.sqrt(np.mean(e**2))
    assert rms(e_best) < rms(e_step5) < rms(e_imu)
    assert rms(e_best) < rms(e_imu) / 3
    assert nees < 2.0                                          # still consistent at the end


def test_wind_lockout_and_its_protection():
    """A wind change the random walk cannot follow: the Pitot is rejected by
    the gate, and without protection it stays rejected for good."""
    plan = dataclasses.replace(trajectory3d.MISSION_WIND, duration=150.0, wind_n=((60.0, 70.0, -4.0),),
                               wind_e=())
    errs = {}
    for key, ad in (("off", full(rw=0.01, reset_after_rejected_s=None)), ("on", full(rw=0.01))):
        _, lg = fly(plan, INIT_MAG, ad, runs=8)
        late = lg.t > 120.0
        errs[key] = np.sqrt((lg.err[:, late][..., lg.blocks["wind"]] ** 2).sum(-1).mean())
        if key == "on":
            assert sum(n for _, n in lg.wind_resets) > 0
    assert errs["off"] > 1.0
    assert errs["on"] < 0.3


def test_gnss_jump_is_rejected_by_the_gate():
    plan = dataclasses.replace(trajectory3d.MISSION_WIND, duration=110.0)
    faults = Faults(gnss_jump=((60.0, 90.0, (0.0, 15.0, 0.0)),))
    ad = full(rw=0.1)
    _, gated = fly(plan, INIT_MAG, ad, faults)
    _, open_ = fly(plan, INIT_MAG, ad, faults, gate=None)
    w = (gated.t >= 60.0) & (gated.t < 90.0)
    assert (gated.gnss_status[:, w] == 1).mean() > 0.9
    assert gated.nees["position"][:, w].mean() / 3 < 2.5
    assert open_.nees["position"][:, w].mean() / 3 > 3.0


def test_strong_magnetic_disturbance_is_rejected():
    plan = dataclasses.replace(trajectory3d.MISSION_WIND, duration=70.0)
    faults = Faults(mag_disturbance=((20.0, 60.0, (0.05, 0.03, 0.0)),))
    ad = full(rw=0.1)
    _, gated = fly(plan, INIT_MAG, ad, faults)
    _, open_ = fly(plan, INIT_MAG, ad, faults, gate=None)
    t_m = gated.aux_nis["mag"][0]
    inside = (t_m >= 20.0) & (t_m < 60.0)
    assert gated.aux_rejected["mag"][:, inside].mean() > 0.95
    w = (gated.t > 30.0) & (gated.t < 60.0)
    yaw = lambda lg: np.degrees(np.sqrt((lg.err[:, w, 8] ** 2).mean()))
    assert yaw(open_) > 3.0
    assert yaw(gated) < 3.0


def test_gnss_lockout_and_its_protection():
    """The wind turns while the GNSS is off by 15 m. With the wind nearly
    frozen without GNSS, the dead reckoning drifts, the filter stays sure of
    itself, and the gate then rejects GNSS for good. The reset on GNSS after
    45 s of rejection gets it back."""
    plan = dataclasses.replace(trajectory3d.MISSION_WIND, duration=160.0, wind_n=((50.0, 110.0, -4.0),),
                               wind_e=())
    faults = Faults(gnss_jump=((60.0, 90.0, (0.0, 15.0, 0.0)),))
    ad = full(rw=0.1, rw_without_gnss=0.01)
    late = {}
    for key, reset in (("off", None), ("on", 45.0)):
        _, lg = fly(plan, INIT_MAG, ad, faults, runs=8, gnss_reset=reset)
        w = lg.t > 130.0
        late[key] = (np.sqrt((lg.out_err[:, w, :2] ** 2).sum(-1).mean()), (lg.gnss_status[:, w] == 1).mean())
    assert late["off"][0] > 20.0 and late["off"][1] > 0.9
    assert late["on"][0] < 5.0 and late["on"][1] < 0.05
