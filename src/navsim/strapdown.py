"""Strapdown inertial navigation from IMU increments.

Flat Earth, NED, constant gravity, no Earth rotation (the truth generator
makes the same assumptions, so a perfect IMU must reproduce the truth).

Three variants of the update, to measure what each correction buys:
- "naive": attitude q <- q * exp(dtheta), velocity v <- v + C(q_k) dvel + g dt;
- "rotcomp": adds the rotation compensation 0.5 * dtheta x dvel, which
  accounts for the body turning during the interval;
- "full": rotcomp plus the two-sample coning correction on the attitude and
  the two-sample sculling correction on the velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotations import quat_from_rotvec, quat_mul, quat_normalize, quat_rotate
from .trajectory3d import G

METHODS = ("naive", "rotcomp", "full")


@dataclass(frozen=True)
class NavSolution:
    p_n: np.ndarray   # (..., K, 3)
    v_n: np.ndarray   # (..., K, 3)
    q_nb: np.ndarray  # (..., K, 4)


def integrate(p0, v0, q0, dtheta, dvel, dt, method: str = "full") -> NavSolution:
    """Integrate increments of shape (..., K-1, 3) from an initial state of
    shape (..., 3) / (..., 4). Leading dimensions are carried through, so a
    batch of Monte-Carlo runs is integrated in one pass."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    n = dtheta.shape[-2]
    batch = dtheta.shape[:-2]
    p = np.empty(batch + (n + 1, 3))
    v = np.empty(batch + (n + 1, 3))
    q = np.empty(batch + (n + 1, 4))
    p[..., 0, :] = p0
    v[..., 0, :] = v0
    q[..., 0, :] = q0
    g_dt = np.array([0.0, 0.0, G * dt])

    prev_dth = np.zeros(batch + (3,))
    prev_dv = np.zeros(batch + (3,))
    for k in range(n):
        dth = dtheta[..., k, :]
        dv = dvel[..., k, :]
        rot = dth
        dv_b = dv
        if method in ("rotcomp", "full"):
            dv_b = dv + 0.5 * np.cross(dth, dv)
        if method == "full" and k > 0:
            rot = dth + np.cross(prev_dth, dth) / 12.0
            dv_b = dv_b + (np.cross(prev_dth, dv) + np.cross(prev_dv, dth)) / 12.0
        qk = q[..., k, :]
        v[..., k + 1, :] = v[..., k, :] + quat_rotate(qk, dv_b) + g_dt
        p[..., k + 1, :] = p[..., k, :] + 0.5 * (v[..., k, :] + v[..., k + 1, :]) * dt
        q[..., k + 1, :] = quat_normalize(quat_mul(qk, quat_from_rotvec(rot)))
        prev_dth, prev_dv = dth, dv
    return NavSolution(p, v, q)


def error_budget(t, cfg, g: float = G) -> dict[str, np.ndarray]:
    """Standard deviation of the horizontal position error, PER AXIS, of an
    unaided strapdown started from the true state, in straight and level
    flight. One term per error source; they are independent, so the total is
    the root sum of squares.

    A tilt error d makes the accelerometer read g*d sideways, so gyro errors
    enter the position through two extra integrations (and a factor g).
    """
    t = np.asarray(t, dtype=float)
    terms = {
        "biais accéléro": cfg.accel_bias0 * t**2 / 2,
        "biais gyro": g * cfg.gyro_bias0 * t**3 / 6,
        "bruit accéléro (VRW)": cfg.accel_noise * np.sqrt(t**3 / 3),
        "bruit gyro (ARW)": g * cfg.gyro_noise * np.sqrt(t**5 / 20),
        "dérive biais accéléro": cfg.accel_bias_rw * np.sqrt(t**5 / 20),
        "dérive biais gyro": g * cfg.gyro_bias_rw * np.sqrt(t**7 / 252),
    }
    terms["total"] = np.sqrt(sum(v**2 for v in terms.values()))
    return terms
