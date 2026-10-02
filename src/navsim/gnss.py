"""GNSS receiver model: position and velocity in NED.

Position error = white noise + an optional first-order Gauss-Markov error per
axis (atmosphere, multipath, ephemeris: errors that persist for tens of
seconds). Velocity (Doppler) errors are white. The fix may be delivered with a
latency: the measurement describes the state at t, but reaches the filter at
t + latency.

The default configuration is the step 3 one (all white, no latency).
REALISTIC keeps the same total position error but makes most of it correlated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GNSSConfig:
    rate_hz: float = 5.0
    pos_sigma_h: float = 1.5   # [m] horizontal, per axis
    pos_sigma_v: float = 3.0   # [m] vertical
    vel_sigma_h: float = 0.10  # [m/s]
    vel_sigma_v: float = 0.20  # [m/s]
    corr_sigma_h: float = 0.0  # [m] steady-state std of the correlated part
    corr_sigma_v: float = 0.0  # [m]
    corr_tau: float = 60.0     # [s] correlation time
    latency_s: float = 0.0     # [s]

    @property
    def R(self) -> np.ndarray:
        """White part only."""
        return np.diag([self.pos_sigma_h, self.pos_sigma_h, self.pos_sigma_v,
                        self.vel_sigma_h, self.vel_sigma_h, self.vel_sigma_v]) ** 2

    @property
    def R_total(self) -> np.ndarray:
        """White + correlated variance: what a filter that treats the whole
        error as white noise would use."""
        return self.R + np.diag([self.corr_sigma_h**2, self.corr_sigma_h**2, self.corr_sigma_v**2,
                                 0.0, 0.0, 0.0])

    @property
    def corr_sigma(self) -> np.ndarray:
        return np.array([self.corr_sigma_h, self.corr_sigma_h, self.corr_sigma_v])


# Same total position error as the default (1.5 m / 3 m), but mostly
# correlated over a minute, and 150 ms of latency.
REALISTIC = GNSSConfig(pos_sigma_h=0.5, pos_sigma_v=1.0, corr_sigma_h=1.4, corr_sigma_v=2.8,
                       corr_tau=60.0, latency_s=0.15)


class CorrelatedError:
    """Gauss-Markov position error of each run, advanced once per fix."""

    def __init__(self, cfg: GNSSConfig, rng, runs: int):
        self.cfg = cfg
        self.rng = rng
        self.runs = runs
        self.phi = np.exp(-1.0 / (cfg.rate_hz * cfg.corr_tau))
        self.value = rng.normal(0.0, 1.0, (runs, 3)) * cfg.corr_sigma
        self._started = False

    def next(self) -> np.ndarray:
        """Error of the next fix. The first call returns the initial draw."""
        if self._started:
            w = self.rng.normal(0.0, 1.0, (self.runs, 3)) * self.cfg.corr_sigma
            self.value = self.phi * self.value + np.sqrt(1.0 - self.phi**2) * w
        self._started = True
        return self.value


def measure(p_true: np.ndarray, v_true: np.ndarray, cfg: GNSSConfig, rng, runs: int,
            pos_bias: np.ndarray | None = None) -> np.ndarray:
    """One fix for each run: (runs, 6) = [p_N, p_E, p_D, v_N, v_E, v_D]."""
    sigma = np.sqrt(np.diag(cfg.R))
    truth = np.concatenate([p_true, v_true])
    z = truth + rng.normal(0.0, 1.0, (runs, 6)) * sigma
    if pos_bias is not None:
        z[:, 0:3] += pos_bias
    return z
