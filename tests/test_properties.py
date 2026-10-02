"""Property-based and metamorphic tests.

Property-based (hypothesis): instead of a few hand-picked inputs, hypothesis
draws hundreds and shrinks any failure to the simplest counter-example.
Invariants checked: what must hold for ANY input (a rotation keeps lengths,
a covariance stays symmetric positive definite, a measurement never adds
uncertainty...).

Metamorphic: no reference answer is needed, only a relation between two
runs. Turning the whole flight by 90 degrees must turn the errors by 90
degrees; doubling a bias must double the error it causes.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from navsim import gnss, imu, strapdown, trajectory3d
from navsim.eskf import ErrorStateEKF
from navsim.kf_altitude import AltitudeKF, van_loan
from navsim.rotations import (attitude_error, quat_conj, quat_from_rotvec, quat_mul, quat_normalize,
                              quat_rotate, rotvec_from_quat)
from navsim.sensors import AccelConfig, BaroConfig

PROFILE = settings(max_examples=60, deadline=None, derandomize=True)

finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
vec3 = arrays(np.float64, 3, elements=finite)
unit_scale = st.floats(-3.0, 3.0, allow_nan=False)
rotvec = arrays(np.float64, 3, elements=st.floats(-1.8, 1.8, allow_nan=False))  # |r| < pi
quat = arrays(np.float64, 4, elements=st.floats(-1, 1, allow_nan=False)).filter(
    lambda q: np.linalg.norm(q) > 0.1).map(quat_normalize)


@PROFILE
@given(q=quat, v=vec3)
def test_rotation_preserves_length(q, v):
    np.testing.assert_allclose(np.linalg.norm(quat_rotate(q, v)), np.linalg.norm(v), rtol=1e-12, atol=1e-9)


@PROFILE
@given(q1=quat, q2=quat, v=vec3)
def test_product_composes_rotations(q1, q2, v):
    np.testing.assert_allclose(quat_rotate(quat_mul(q1, q2), v), quat_rotate(q1, quat_rotate(q2, v)),
                               atol=1e-9)


@PROFILE
@given(q=quat, v=vec3)
def test_conjugate_is_inverse(q, v):
    np.testing.assert_allclose(quat_rotate(quat_conj(q), quat_rotate(q, v)), v, atol=1e-9)


@PROFILE
@given(r=rotvec)
def test_exp_log_roundtrip(r):
    np.testing.assert_allclose(rotvec_from_quat(quat_from_rotvec(r)), r, atol=1e-10)


@PROFILE
@given(q=quat, e=arrays(np.float64, 3, elements=st.floats(-0.5, 0.5, allow_nan=False)))
def test_attitude_error_recovers_the_perturbation(q, e):
    np.testing.assert_allclose(attitude_error(q, quat_mul(quat_from_rotvec(e), q)), e, atol=1e-10)


def random_spd(rng, n, scale):
    A = rng.normal(size=(n, n)) * scale[:, None]
    return A @ A.T + np.diag(scale**2) * 1e-3


@PROFILE
@given(seed=st.integers(0, 2**31 - 1), z_off=arrays(np.float64, 6, elements=unit_scale))
def test_gnss_update_keeps_covariance_valid(seed, z_off):
    """Any prior covariance, any measurement: the posterior covariance must
    stay symmetric positive definite."""
    rng = np.random.default_rng(seed)
    scale = np.array([3] * 3 + [0.5] * 3 + [0.05] * 3 + [5e-4] * 3 + [0.05] * 3, dtype=float)
    P = random_spd(rng, 15, scale)
    cfg = gnss.GNSSConfig()
    ekf = ErrorStateEKF(imu.IMUConfig(), cfg, 0.005, runs=1)
    ekf.initialise(np.zeros((1, 3)), np.array([[17.0, 0, 0]]), np.array([[1.0, 0, 0, 0]]), P)
    z = np.concatenate([ekf.p, ekf.v], axis=1) + z_off * np.array([3, 3, 3, 0.3, 0.3, 0.3])
    ekf.update_gnss(z)
    Pp = ekf.P[0]
    np.testing.assert_allclose(Pp, Pp.T, atol=1e-14)
    assert np.linalg.eigvalsh(Pp).min() > 0


@PROFILE
@given(seed=st.integers(0, 2**31 - 1))
def test_measurement_never_adds_uncertainty(seed):
    """With zero innovation (no reset), P_prior - P_post must be positive
    semi-definite: a measurement can only remove uncertainty."""
    rng = np.random.default_rng(seed)
    scale = np.array([3] * 3 + [0.5] * 3 + [0.05] * 3 + [5e-4] * 3 + [0.05] * 3, dtype=float)
    P = random_spd(rng, 15, scale)
    ekf = ErrorStateEKF(imu.IMUConfig(), gnss.GNSSConfig(), 0.005, runs=1)
    ekf.initialise(np.zeros((1, 3)), np.array([[17.0, 0, 0]]), np.array([[1.0, 0, 0, 0]]), P)
    ekf.update_gnss(np.concatenate([ekf.p, ekf.v], axis=1))
    d = P - ekf.P[0]
    assert np.linalg.eigvalsh(0.5 * (d + d.T)).min() > -1e-10 * np.abs(P).max()


@PROFILE
@given(noise=st.floats(1e-4, 1.0), rw=st.floats(1e-6, 1e-2), baro_sigma=st.floats(0.05, 5.0),
       drift=st.one_of(st.just(0.0), st.floats(1e-6, 5.0)), tau=st.floats(5.0, 1000.0),
       seed=st.integers(0, 2**31 - 1))
def test_altitude_filter_covariance_stays_valid(noise, rw, baro_sigma, drift, tau, seed):
    accel = AccelConfig(noise_density=noise, bias_rw=rw)
    baro = BaroConfig(noise_sigma=baro_sigma, drift_sigma=drift, drift_tau=tau)
    kf = AltitudeKF(accel, baro, 0.01)
    assert np.linalg.eigvalsh(kf.Q).min() >= -1e-18
    kf.initialise(10.0)
    rng = np.random.default_rng(seed)
    for k in range(200):
        kf.predict(rng.normal())
        if k % 4 == 3:
            kf.update_baro(10.0 + rng.normal())
    np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-12)
    assert np.linalg.eigvalsh(kf.P).min() > 0


def test_degenerate_drift_is_rejected():
    """Regression for the case hypothesis found: a tiny positive drift std made
    the drift state's variance underflow and P singular."""
    import pytest

    with pytest.raises(ValueError):
        AltitudeKF(AccelConfig(), BaroConfig(drift_sigma=2.3e-164), 0.01)


@PROFILE
@given(dt=st.floats(1e-4, 1.0), q=st.floats(1e-6, 10.0))
def test_van_loan_scalar_random_walk(dt, q):
    """dx/dt = w, E[w w] = q: Qd = q dt for any dt."""
    Phi, Qd = van_loan(np.zeros((1, 1)), np.ones((1, 1)), np.array([[q]]), dt)
    assert Qd[0, 0] == np.float64(Qd[0, 0]) and abs(Qd[0, 0] - q * dt) <= 1e-9 * q * dt


# ---------------------------------------------------------------------------
# Metamorphic
# ---------------------------------------------------------------------------

def strapdown_error(plan, dtheta_err, dvel_err):
    tr = trajectory3d.generate(plan)
    sol = strapdown.integrate(tr.p_n[0], tr.v_n[0], tr.q_nb[0], tr.dtheta + dtheta_err, tr.dvel + dvel_err,
                              tr.dt)
    return sol.p_n - tr.p_n, tr


def test_turning_the_flight_turns_the_errors():
    """Same flight flown with an initial heading of 0 and of 90 deg, same IMU
    errors (they live in the body frame, which does not see the heading). The
    navigation errors must be the same vectors, turned by 90 deg."""
    base = trajectory3d.FlightPlan(duration=90.0, bank_deg=((20, 24, 30), (50, 54, -30)),
                                   gamma_deg=((10, 15, 5), (30, 35, -5)))
    n = int(round(base.duration * 200))
    rng = np.random.default_rng(3)
    meas = imu.simulate(np.zeros((n, 3)), np.zeros((n, 3)), imu.IMUConfig(), rng)
    e0, _ = strapdown_error(base, meas.dtheta, meas.dvel)
    e90, _ = strapdown_error(trajectory3d.FlightPlan(**{**base.__dict__, "heading0_deg": 90.0}),
                             meas.dtheta, meas.dvel)
    np.testing.assert_allclose(e90[:, 0], -e0[:, 1], atol=1e-6 * np.abs(e0).max())
    np.testing.assert_allclose(e90[:, 1], e0[:, 0], atol=1e-6 * np.abs(e0).max())
    np.testing.assert_allclose(e90[:, 2], e0[:, 2], atol=1e-6 * np.abs(e0).max())
    assert np.abs(e0).max() > 10.0  # the errors are not trivially zero


def test_error_scales_with_bias():
    """In the linear regime, doubling a sensor bias doubles the error it causes."""
    plan = trajectory3d.FlightPlan(duration=30.0, bank_deg=((10, 14, 25),))
    n = int(round(plan.duration * 200))
    dt = 1 / 200
    b_g = np.radians(np.array([0.01, -0.02, 0.015])) * dt
    b_a = np.array([0.02, 0.01, -0.03]) * dt
    e1, _ = strapdown_error(plan, np.tile(b_g, (n, 1)), np.tile(b_a, (n, 1)))
    e2, _ = strapdown_error(plan, np.tile(2 * b_g, (n, 1)), np.tile(2 * b_a, (n, 1)))
    np.testing.assert_allclose(e2[-1], 2 * e1[-1], rtol=0.01)
