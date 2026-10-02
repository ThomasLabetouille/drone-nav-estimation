"""Sensor error models.

Step 1 simplification: the accelerometer is a single vertical axis with
gravity already removed and attitude assumed perfect. The full strapdown
model (specific force, rotation, gravity) comes in step 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trajectory import VerticalProfile


@dataclass(frozen=True)
class AccelConfig:
    rate_hz: float = 100.0
    # White noise density [m/s^2/sqrt(Hz)]. Much higher than a datasheet
    # value (~0.001) to account for airframe and propeller vibration.
    noise_density: float = 0.02
    # Turn-on bias, drawn once per run [m/s^2] (0.05 m/s^2 ~ 5 mg).
    bias_sigma0: float = 0.05
    # Bias random walk [m/s^2/sqrt(s)].
    bias_rw: float = 5e-4

    @property
    def noise_sigma(self) -> float:
        """Standard deviation of one discrete sample."""
        return self.noise_density * np.sqrt(self.rate_hz)


@dataclass(frozen=True)
class BaroConfig:
    rate_hz: float = 25.0
    noise_sigma: float = 0.4  # white noise [m]
    # Slow error (weather, temperature, static pressure error), modelled as
    # a first-order Gauss-Markov process.
    drift_sigma: float = 1.5  # steady-state std [m]
    drift_tau: float = 120.0  # correlation time [s]


@dataclass(frozen=True)
class AccelMeasurements:
    a: np.ndarray  # measured vertical acceleration [m/s^2]
    bias: np.ndarray  # true bias, for evaluation only


@dataclass(frozen=True)
class BaroMeasurements:
    idx: np.ndarray  # index into the IMU time vector
    t: np.ndarray
    z: np.ndarray  # measured altitude [m]
    drift: np.ndarray  # true slow error, for evaluation only


def simulate_accel(profile: VerticalProfile, cfg: AccelConfig, rng) -> AccelMeasurements:
    dt = profile.dt
    if not np.isclose(1.0 / dt, cfg.rate_hz):
        raise ValueError("profile sampling must match accelerometer rate")
    n = len(profile.t)
    bias = np.empty(n)
    bias[0] = rng.normal(0.0, cfg.bias_sigma0)
    steps = rng.normal(0.0, cfg.bias_rw * np.sqrt(dt), n - 1)
    bias[1:] = bias[0] + np.cumsum(steps)
    noise = rng.normal(0.0, cfg.noise_sigma, n)
    # Like a real IMU delta-velocity output, sample k is the mean acceleration
    # over [t_k, t_k+1], i.e. (v_k+1 - v_k) / dt. Sampling a(t_k) instead would
    # add a small model error that shows up as a slightly optimistic filter.
    a_mean = np.empty(n)
    a_mean[:-1] = np.diff(profile.v) / dt
    a_mean[-1] = profile.a[-1]
    return AccelMeasurements(a=a_mean + bias + noise, bias=bias)


def simulate_baro(profile: VerticalProfile, cfg: BaroConfig, rng) -> BaroMeasurements:
    decim = int(round(1.0 / (cfg.rate_hz * profile.dt)))
    idx = np.arange(0, len(profile.t), decim)
    t = profile.t[idx]
    dtb = decim * profile.dt

    drift = np.zeros(len(idx))
    if cfg.drift_sigma > 0:
        # Exact discretisation of the Gauss-Markov process.
        phi = np.exp(-dtb / cfg.drift_tau)
        q = cfg.drift_sigma * np.sqrt(1.0 - phi**2)
        drift[0] = rng.normal(0.0, cfg.drift_sigma)
        w = rng.normal(0.0, q, len(idx) - 1)
        for k in range(1, len(idx)):
            drift[k] = phi * drift[k - 1] + w[k - 1]

    z = profile.h[idx] + drift + rng.normal(0.0, cfg.noise_sigma, len(idx))
    return BaroMeasurements(idx=idx, t=t, z=z, drift=drift)
