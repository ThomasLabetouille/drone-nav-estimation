"""6-axis IMU error model, applied to ideal increments.

Each axis of each sensor gets, independently:
- white noise (angle / velocity random walk), given as a density;
- a turn-on bias, drawn once per run;
- a bias random walk.

Values are those of an uncalibrated-in-flight consumer MEMS IMU on a small
airframe, with vibration included in the noise densities. They are about one
order of magnitude above a datasheet, on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IMUConfig:
    rate_hz: float = 200.0
    gyro_noise: float = np.radians(0.01)        # [rad/s/sqrt(Hz)]  0.6 deg/sqrt(h)
    gyro_bias0: float = np.radians(0.02)        # [rad/s]           72 deg/h
    gyro_bias_rw: float = 2e-5                  # [rad/s/sqrt(s)]
    accel_noise: float = 0.02                   # [m/s^2/sqrt(Hz)]
    accel_bias0: float = 0.05                   # [m/s^2]           ~5 mg
    accel_bias_rw: float = 5e-4                 # [m/s^2/sqrt(s)]


PERFECT = IMUConfig(gyro_noise=0.0, gyro_bias0=0.0, gyro_bias_rw=0.0,
                    accel_noise=0.0, accel_bias0=0.0, accel_bias_rw=0.0)


@dataclass(frozen=True)
class IMUMeasurements:
    dtheta: np.ndarray      # (..., K-1, 3) measured delta-angles
    dvel: np.ndarray        # (..., K-1, 3) measured delta-velocities
    gyro_bias: np.ndarray   # (..., K-1, 3) true bias over each interval
    accel_bias: np.ndarray  # (..., K-1, 3)


def _bias(rng, shape, sigma0, rw, dt):
    b0 = rng.normal(0.0, sigma0, shape[:-2] + (1, 3))
    if rw == 0.0:
        return np.broadcast_to(b0, shape).copy()
    steps = rng.normal(0.0, rw * np.sqrt(dt), shape)
    steps[..., 0, :] = 0.0
    return b0 + np.cumsum(steps, axis=-2)


def simulate(dtheta: np.ndarray, dvel: np.ndarray, cfg: IMUConfig, rng, runs: int | None = None,
             dt: float | None = None) -> IMUMeasurements:
    """Corrupt ideal increments. With `runs`, returns a leading batch axis of
    independent draws (for Monte-Carlo)."""
    dt = dt if dt is not None else 1.0 / cfg.rate_hz
    shape = ((runs,) if runs else ()) + dtheta.shape
    bg = _bias(rng, shape, cfg.gyro_bias0, cfg.gyro_bias_rw, dt)
    ba = _bias(rng, shape, cfg.accel_bias0, cfg.accel_bias_rw, dt)
    # White noise of density N integrated over dt has std N * sqrt(dt).
    ng = rng.normal(0.0, cfg.gyro_noise * np.sqrt(dt), shape) if cfg.gyro_noise else 0.0
    na = rng.normal(0.0, cfg.accel_noise * np.sqrt(dt), shape) if cfg.accel_noise else 0.0
    return IMUMeasurements(
        dtheta=dtheta + bg * dt + ng,
        dvel=dvel + ba * dt + na,
        gyro_bias=bg,
        accel_bias=ba,
    )
