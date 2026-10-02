"""GNSS receiver model: position and velocity in NED.

Step 3 keeps it simple: white errors, no latency, no outage. Real receiver
errors are correlated over tens of seconds (multipath, atmosphere), and the
measurement is delayed by 100-200 ms; both come in later steps.
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

    @property
    def R(self) -> np.ndarray:
        return np.diag([self.pos_sigma_h, self.pos_sigma_h, self.pos_sigma_v,
                        self.vel_sigma_h, self.vel_sigma_h, self.vel_sigma_v]) ** 2


def measure(p_true: np.ndarray, v_true: np.ndarray, cfg: GNSSConfig, rng, runs: int) -> np.ndarray:
    """One fix for each run: (runs, 6) = [p_N, p_E, p_D, v_N, v_E, v_D]."""
    sigma = np.sqrt(np.diag(cfg.R))
    truth = np.concatenate([p_true, v_true])
    return truth + rng.normal(0.0, 1.0, (runs, 6)) * sigma
