"""Run the filter over a simulated flight and collect everything needed for
evaluation (estimates, covariances, innovations, errors w.r.t. truth)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kf_altitude import AltitudeKF
from .sensors import (
    AccelConfig,
    AccelMeasurements,
    BaroConfig,
    BaroMeasurements,
    simulate_accel,
    simulate_baro,
)
from .trajectory import VerticalProfile


@dataclass
class RunResult:
    t: np.ndarray
    x: np.ndarray  # (N, n) estimates after each step
    P: np.ndarray  # (N, n, n)
    nis: np.ndarray  # one per baro update
    nees_hv: np.ndarray  # NEES on [h, v], one per IMU step
    accel: AccelMeasurements
    baro: BaroMeasurements


def run_filter(
    profile: VerticalProfile,
    accel_meas: AccelMeasurements,
    baro_meas: BaroMeasurements,
    accel_model: AccelConfig,
    baro_model: BaroConfig,
) -> RunResult:
    kf = AltitudeKF(accel_model, baro_model, profile.dt)
    N = len(profile.t)
    xs = np.empty((N, kf.n))
    Ps = np.empty((N, kf.n, kf.n))
    nis = np.empty(len(baro_meas.idx))

    is_baro = np.zeros(N, dtype=int) - 1
    is_baro[baro_meas.idx] = np.arange(len(baro_meas.idx))

    kf.initialise(baro_meas.z[0])
    xs[0], Ps[0] = kf.x, kf.P
    nis[0] = np.nan  # the first sample is used for initialisation
    for k in range(1, N):
        # a_meas[k-1] is held over [t_{k-1}, t_k] (zero-order hold)
        kf.predict(accel_meas.a[k - 1])
        j = is_baro[k]
        if j >= 0:
            _, _, nis[j] = kf.update_baro(baro_meas.z[j])
        xs[k], Ps[k] = kf.x, kf.P

    err = np.stack([xs[:, 0] - profile.h, xs[:, 1] - profile.v], axis=1)
    Phv = Ps[:, :2, :2]
    nees = np.einsum("ni,nij,nj->n", err, np.linalg.inv(Phv), err)
    return RunResult(profile.t, xs, Ps, nis, nees, accel_meas, baro_meas)


def simulate_and_run(
    profile: VerticalProfile,
    accel_truth: AccelConfig,
    baro_truth: BaroConfig,
    accel_model: AccelConfig,
    baro_model: BaroConfig,
    rng,
) -> RunResult:
    accel = simulate_accel(profile, accel_truth, rng)
    baro = simulate_baro(profile, baro_truth, rng)
    return run_filter(profile, accel, baro, accel_model, baro_model)


def dead_reckoning(profile: VerticalProfile, a_meas: np.ndarray):
    """Integrate the accelerometer alone from the true initial state."""
    dt = profile.dt
    h = np.empty_like(a_meas)
    v = np.empty_like(a_meas)
    h[0], v[0] = profile.h[0], profile.v[0]
    for k in range(1, len(a_meas)):
        a = a_meas[k - 1]
        h[k] = h[k - 1] + v[k - 1] * dt + 0.5 * a * dt * dt
        v[k] = v[k - 1] + a * dt
    return h, v
