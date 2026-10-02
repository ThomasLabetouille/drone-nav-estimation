"""Reference (truth) trajectories.

Step 1 only needs the vertical channel of a fixed-wing flight: altitude,
vertical speed and vertical acceleration, all analytic so that the truth is
exactly self-consistent (a = dv/dt, v = dh/dt).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VerticalProfile:
    t: np.ndarray  # [s]
    h: np.ndarray  # altitude above launch point [m]
    v: np.ndarray  # vertical speed, positive up [m/s]
    a: np.ndarray  # vertical acceleration, positive up [m/s^2]

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


def min_jerk_step(t: np.ndarray, t0: float, t1: float):
    """Smooth 0 -> 1 step between t0 and t1 (quintic, minimum jerk).

    Returns s(t), ds/dt, d2s/dt2. Position, velocity and acceleration are all
    continuous, so the accelerometer never sees a discontinuity.
    """
    T = t1 - t0
    tau = np.clip((t - t0) / T, 0.0, 1.0)
    inside = (t > t0) & (t < t1)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = np.where(inside, (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T, 0.0)
    dds = np.where(inside, (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2, 0.0)
    return s, ds, dds


# (start [s], end [s], altitude change [m])
DEFAULT_LEGS = (
    (10.0, 60.0, +120.0),   # launch and climb to 120 m
    (120.0, 150.0, +30.0),  # step climb to 150 m
    (210.0, 260.0, -120.0),  # descent to 30 m
)

# Vertical turbulence: (amplitude [m], period [s]). Each component peaks at
# A * (2*pi/T)^2, i.e. 0.4 to 2 m/s^2 here (up to ~3.3 m/s^2 combined).
DEFAULT_TURBULENCE = ((1.5, 12.0), (0.6, 5.0), (0.2, 2.0))


def fixed_wing_vertical_profile(
    duration: float = 300.0,
    dt: float = 0.01,
    legs=DEFAULT_LEGS,
    turbulence=DEFAULT_TURBULENCE,
    launch_time: float = 10.0,
    seed: int | None = 0,
) -> VerticalProfile:
    """Altitude profile of a short fixed-wing mission.

    The aircraft sits still on the catapult for `launch_time` seconds, then
    climbs, cruises, steps up, cruises and descends. Turbulence is a sum of
    sinusoids with random phases, faded in after launch.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration + 0.5 * dt, dt)
    h = np.zeros_like(t)
    v = np.zeros_like(t)
    a = np.zeros_like(t)

    for t0, t1, dh in legs:
        s, ds, dds = min_jerk_step(t, t0, t1)
        h += dh * s
        v += dh * ds
        a += dh * dds

    if turbulence:
        # Envelope e(t) fades turbulence in after launch; product rule below.
        e, de, dde = min_jerk_step(t, launch_time, launch_time + 10.0)
        g = np.zeros_like(t)
        dg = np.zeros_like(t)
        ddg = np.zeros_like(t)
        for amp, period in turbulence:
            w = 2 * np.pi / period
            phi = rng.uniform(0, 2 * np.pi)
            g += amp * np.sin(w * t + phi)
            dg += amp * w * np.cos(w * t + phi)
            ddg += -amp * w**2 * np.sin(w * t + phi)
        h += e * g
        v += de * g + e * dg
        a += dde * g + 2 * de * dg + e * ddg

    return VerticalProfile(t=t, h=h, v=v, a=a)
