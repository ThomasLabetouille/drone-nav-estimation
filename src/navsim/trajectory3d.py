"""3D reference trajectory of a fixed-wing drone, and the matching ideal IMU.

Kinematic model, no wind, no sideslip, coordinated turns:
- three commanded channels, each a sum of smooth (minimum-jerk) steps:
  bank angle phi, flight-path angle gamma, airspeed V;
- turn rate of a coordinated turn: dpsi/dt = g tan(phi) / V;
- pitch = gamma + constant trim angle of attack.

Because the turn rate formula ignores the trim angle, a turning body sees a
small side force g sin(phi)(1 - cos(theta)), about 7 mm/s^2 at 30 deg of bank.
It does not matter here: what counts is that the IMU and the truth are exactly
consistent with each other, which the tests check.

What the IMU measures (body rates and specific force in the body frame) does
not depend on the heading, so both are analytic functions of time. Only the
heading and the position need a numerical integration, done on a fine grid
(8x the IMU rate) with Simpson's rule.

The IMU output is given as increments over each sample interval, like a real
IMU: delta-angle (integral of the body rate) and delta-velocity (integral of
the specific force), both in the body frame. Note that the delta-angle is the
integral of the rate, not the exact attitude increment: the two differ when
the rotation axis moves (coning), which is the strapdown algorithm's problem,
not the sensor's.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_simpson

from .rotations import quat_conj, quat_from_euler, quat_rotate
from .trajectory import min_jerk_step

G = 9.80665


@dataclass(frozen=True)
class FlightPlan:
    duration: float = 480.0
    airspeed0: float = 17.0  # [m/s]
    altitude0: float = 50.0  # [m] above the launch point
    heading0_deg: float = 0.0
    alpha_trim_deg: float = 3.0
    # (start [s], end [s], change) for each channel
    bank_deg: tuple = ()
    gamma_deg: tuple = ()
    airspeed: tuple = ()
    # Horizontal wind in NED (the direction it blows towards), and its changes
    wind_n0: float = 0.0  # [m/s]
    wind_e0: float = 0.0  # [m/s]
    wind_n: tuple = ()
    wind_e: tuple = ()


# 8 minutes: climb, right turn, two loiter circles, speed dash, S-turns,
# descent, left loiter.
MISSION = FlightPlan(
    bank_deg=(
        (70, 74, +25), (80, 84, -25),        # right turn
        (110, 114, +30), (150, 154, -30),    # loiter, about two circles
        (270, 274, -25), (282, 288, +50),    # S-turns
        (296, 300, -25),
        (400, 404, -30), (440, 444, +30),    # left loiter
    ),
    gamma_deg=(
        (15, 20, +6), (55, 60, -6),          # climb to ~120 m
        (330, 335, -5), (385, 390, +5),      # descent
    ),
    airspeed=(
        (190, 200, +5), (240, 250, -5),      # dash at 22 m/s
    ),
)

STRAIGHT_LEVEL = FlightPlan(duration=120.0, altitude0=120.0)

# Same mission in a 5 m/s wind from the west, which veers to a 6 m/s
# north-westerly between 300 and 360 s.
MISSION_WIND = dataclasses.replace(
    MISSION, wind_n0=0.0, wind_e0=5.0,
    wind_n=((300, 360, -4.24),), wind_e=((300, 360, -0.76),),
)


@dataclass(frozen=True)
class Trajectory3D:
    t: np.ndarray        # (K,)  [s]
    p_n: np.ndarray      # (K,3) position NED [m]
    v_n: np.ndarray      # (K,3) velocity NED [m/s]
    a_n: np.ndarray      # (K,3) acceleration NED [m/s^2]
    q_nb: np.ndarray     # (K,4) attitude, body to NED
    euler: np.ndarray    # (K,3) roll, pitch, yaw [rad]
    omega_b: np.ndarray  # (K,3) body rates [rad/s]
    f_b: np.ndarray      # (K,3) specific force, body [m/s^2]
    dtheta: np.ndarray   # (K-1,3) ideal delta-angle over [t_k, t_k+1]
    dvel: np.ndarray     # (K-1,3) ideal delta-velocity over [t_k, t_k+1]
    plan: FlightPlan = field(repr=False)
    wind_n: np.ndarray | None = field(default=None, repr=False)    # (K,3) wind in NED
    airspeed: np.ndarray | None = field(default=None, repr=False)  # (K,) true airspeed

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


def _channel(t, base, steps, scale=1.0):
    x = np.full_like(t, base)
    dx = np.zeros_like(t)
    for t0, t1, delta in steps:
        s, ds, _ = min_jerk_step(t, t0, t1)
        x += scale * delta * s
        dx += scale * delta * ds
    return x, dx


def body_kinematics(phi, dphi, gamma, dgamma, V, dV, alpha):
    """Body rates and specific force, independent of heading."""
    theta = gamma + alpha
    dpsi = G * np.tan(phi) / V
    p = dphi - dpsi * np.sin(theta)
    q = dgamma * np.cos(phi) + dpsi * np.cos(theta) * np.sin(phi)
    r = dpsi * np.cos(theta) * np.cos(phi) - dgamma * np.sin(phi)
    omega_b = np.stack([p, q, r], axis=-1)

    # Acceleration in the heading frame (x along the horizontal track).
    cg, sg = np.cos(gamma), np.sin(gamma)
    a_h = np.stack([
        dV * cg - V * dgamma * sg,
        dpsi * V * cg,
        -dV * sg - V * dgamma * cg,
    ], axis=-1)
    f_h = a_h - np.array([0.0, 0.0, G])

    # Heading frame -> body: pitch then roll (frame rotations).
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    fx = ct * f_h[:, 0] - st * f_h[:, 2]
    fz1 = st * f_h[:, 0] + ct * f_h[:, 2]
    fy = cp * f_h[:, 1] + sp * fz1
    fz = -sp * f_h[:, 1] + cp * fz1
    f_b = np.stack([fx, fy, fz], axis=-1)
    return omega_b, f_b, a_h, dpsi, theta


def generate(plan: FlightPlan = MISSION, rate_hz: float = 200.0, oversample: int = 8) -> Trajectory3D:
    dt_f = 1.0 / (rate_hz * oversample)
    n_f = int(round(plan.duration / dt_f)) + 1
    tf = np.arange(n_f) * dt_f
    alpha = np.radians(plan.alpha_trim_deg)

    phi, dphi = _channel(tf, 0.0, plan.bank_deg, np.pi / 180)
    gamma, dgamma = _channel(tf, 0.0, plan.gamma_deg, np.pi / 180)
    V, dV = _channel(tf, plan.airspeed0, plan.airspeed)

    omega_b, f_b, a_h, dpsi, theta = body_kinematics(phi, dphi, gamma, dgamma, V, dV, alpha)

    psi = np.radians(plan.heading0_deg) + cumulative_simpson(dpsi, x=tf, initial=0.0)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    # Air-relative velocity and acceleration (the turn is coordinated in the
    # air mass, so the attitude follows the air-relative heading)...
    v_n = np.stack([V * np.cos(gamma) * cpsi, V * np.cos(gamma) * spsi, -V * np.sin(gamma)], axis=-1)
    a_n = np.stack([cpsi * a_h[:, 0] - spsi * a_h[:, 1],
                    spsi * a_h[:, 0] + cpsi * a_h[:, 1],
                    a_h[:, 2]], axis=-1)
    # ...plus the wind, which adds to the ground velocity, and whose changes
    # add an inertial acceleration that the accelerometer feels.
    wn, dwn = _channel(tf, plan.wind_n0, plan.wind_n)
    we, dwe = _channel(tf, plan.wind_e0, plan.wind_e)
    wind = np.stack([wn, we, np.zeros_like(wn)], axis=-1)
    dwind = np.stack([dwn, dwe, np.zeros_like(wn)], axis=-1)
    v_n = v_n + wind
    a_n = a_n + dwind
    if np.any(dwind):
        f_b = f_b + quat_rotate(quat_conj(quat_from_euler(phi, theta, psi)), dwind)
    p_n = np.array([0.0, 0.0, -plan.altitude0]) + cumulative_simpson(v_n, x=tf, axis=0, initial=0.0)

    int_omega = cumulative_simpson(omega_b, x=tf, axis=0, initial=0.0)
    int_f = cumulative_simpson(f_b, x=tf, axis=0, initial=0.0)

    idx = np.arange(0, n_f, oversample)
    euler = np.stack([phi[idx], theta[idx], psi[idx]], axis=-1)
    return Trajectory3D(
        t=tf[idx],
        p_n=p_n[idx],
        v_n=v_n[idx],
        a_n=a_n[idx],
        q_nb=quat_from_euler(phi[idx], theta[idx], psi[idx]),
        euler=euler,
        omega_b=omega_b[idx],
        f_b=f_b[idx],
        dtheta=np.diff(int_omega[idx], axis=0),
        dvel=np.diff(int_f[idx], axis=0),
        plan=plan,
        wind_n=wind[idx],
        airspeed=V[idx],
    )
