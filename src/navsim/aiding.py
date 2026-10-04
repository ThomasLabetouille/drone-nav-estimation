"""Aiding sensors of step 5: barometer, magnetometer, Pitot tube, and the
wind model the filter needs to use the Pitot.

Each sensor has a configuration (what the hardware is like) and an error
stream that produces one measurement per run at its own rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rotations import quat_conj, quat_rotate
from .sensors import BaroConfig


def earth_field(total_gauss: float = 0.47, inclination_deg: float = 61.0, declination_deg: float = 1.5):
    """Earth magnetic field in NED. Default: roughly the south of France."""
    i, d = np.radians(inclination_deg), np.radians(declination_deg)
    return total_gauss * np.array([np.cos(i) * np.cos(d), np.cos(i) * np.sin(d), np.sin(i)])


@dataclass(frozen=True)
class MagConfig:
    rate_hz: float = 50.0
    field_ned: tuple = tuple(earth_field())  # [gauss]
    noise: float = 0.003      # [gauss] per sample
    bias0: float = 0.01       # [gauss] residual hard-iron after calibration, per axis
    bias_rw: float = 1e-5     # [gauss/sqrt(s)]
    # Learn the bias only while the body turns faster than this; below, the
    # bias states are "consider" states (Schmidt-Kalman). None: always learn.
    # Without it, the filter gains false confidence in straight flight (see README).
    learn_bias_min_rate_deg_s: float | None = 5.0
    # Step 7: "3d" fuses the three axes against field_ned, with bias states;
    # "heading" fuses only the tilt-compensated heading, with heading_sigma_deg
    # and no bias state: it only needs the declination, not the intensity and
    # inclination of the field, which a real calibration often gets wrong.
    fusion: str = "3d"
    heading_sigma_deg: float = 10.0


@dataclass(frozen=True)
class PitotConfig:
    rate_hz: float = 10.0
    noise: float = 0.3        # [m/s] true airspeed
    # Step 7: estimate a scale error of the airspeed, TAS_meas = (1 + s) |v - w|,
    # with this initial sigma on s. None: no scale state.
    scale_sigma0: float | None = None


@dataclass(frozen=True)
class WindModel:
    """How the filter models the horizontal wind: a random walk."""
    sigma0: float = 5.0       # [m/s] initial uncertainty per axis
    rw: float = 0.2           # [m/s/sqrt(s)]
    # Step 6: random walk used while no GNSS fix has been fused for more than
    # 1 s. The wind is then unobservable, and a large random walk only lets
    # the filter explain its own IMU drift as wind changes. None: keep rw.
    rw_without_gnss: float | None = None
    # Start from the first Pitot sample (ground velocity minus airspeed along
    # the heading) instead of zero +- sigma0. See fusion.wind_from_first_airspeed.
    init_from_airspeed: bool = True
    init_sideslip_sigma_deg: float = 2.0   # doubt on "air velocity along the heading"
    # Step 6, protection against gating lock-out: if the Pitot is rejected by
    # the innovation gate for this long without a break, the wind estimate is
    # assumed wrong and restarted from the current airspeed sample, the same
    # way as at start-up. None: never.
    reset_after_rejected_s: float | None = 5.0


BARO_10HZ = BaroConfig(rate_hz=10.0)


@dataclass(frozen=True)
class AidingConfig:
    baro: BaroConfig | None = None
    mag: MagConfig | None = None
    pitot: PitotConfig | None = None
    wind: WindModel | None = field(default=None)
    # Synthetic zero-sideslip measurement, fused with each Pitot sample. Its
    # sigma covers the real sideslip of the airframe (here up to ~1.5 deg in
    # turns, from the trim angle). None: not used.
    sideslip_sigma_deg: float | None = None
    # Step 7: estimate a constant sideslip offset b (yaw misalignment of the
    # autopilot, or a real trim sideslip): the pseudo-measurement becomes
    # beta - b = 0. Initial sigma of b; None: no offset state.
    sideslip_offset_sigma0_deg: float | None = None

    def __post_init__(self):
        if self.pitot is not None and self.wind is None:
            raise ValueError("a Pitot tube needs a wind model (airspeed = |v - wind|)")
        if self.sideslip_sigma_deg is not None and self.pitot is None:
            raise ValueError("the sideslip pseudo-measurement is fused with the Pitot samples")
        if self.sideslip_offset_sigma0_deg is not None and self.sideslip_sigma_deg is None:
            raise ValueError("a sideslip offset needs the sideslip pseudo-measurement")


class BaroStream:
    """Altitude + slow Gauss-Markov drift + white noise, per run."""

    def __init__(self, cfg: BaroConfig, rng, runs: int):
        self.cfg, self.rng, self.runs = cfg, rng, runs
        self.phi = np.exp(-1.0 / (cfg.rate_hz * cfg.drift_tau))
        self.drift = rng.normal(0.0, cfg.drift_sigma, runs)
        self._started = False

    def measure(self, altitude: float) -> np.ndarray:
        if self._started:
            w = self.rng.normal(0.0, self.cfg.drift_sigma * np.sqrt(1 - self.phi**2), self.runs)
            self.drift = self.phi * self.drift + w
        self._started = True
        return altitude + self.drift + self.rng.normal(0.0, self.cfg.noise_sigma, self.runs)


class MagStream:
    """Earth field seen in the body frame + hard-iron bias + white noise."""

    def __init__(self, cfg: MagConfig, rng, runs: int):
        self.cfg, self.rng, self.runs = cfg, rng, runs
        self.bias = rng.normal(0.0, cfg.bias0, (runs, 3))

    def measure(self, q_nb: np.ndarray, disturbance_ned=None) -> np.ndarray:
        """disturbance_ned: local field added to the Earth field (step 6 faults)."""
        field = np.asarray(self.cfg.field_ned)
        if disturbance_ned is not None:
            field = field + disturbance_ned
        m_b = quat_rotate(quat_conj(q_nb), field)
        z = m_b + self.bias + self.rng.normal(0.0, self.cfg.noise, (self.runs, 3))
        self.bias = self.bias + self.rng.normal(0.0, self.cfg.bias_rw / np.sqrt(self.cfg.rate_hz),
                                                (self.runs, 3))
        return z


class PitotStream:
    def __init__(self, cfg: PitotConfig, rng, runs: int):
        self.cfg, self.rng, self.runs = cfg, rng, runs

    def measure(self, airspeed: float) -> np.ndarray:
        return airspeed + self.rng.normal(0.0, self.cfg.noise, self.runs)


def heading_from_magnetometer(m_b: np.ndarray, roll: np.ndarray, pitch: np.ndarray,
                              declination_rad: float) -> np.ndarray:
    """Tilt-compensated magnetic heading, the way an autopilot initialises it:
    rotate the measured field by the estimated roll and pitch to level it,
    then read the angle of its horizontal part and add the declination."""
    from .rotations import quat_from_euler

    m_level = quat_rotate(quat_from_euler(roll, pitch, np.zeros_like(roll)), m_b)
    return declination_rad + np.arctan2(-m_level[..., 1], m_level[..., 0])
