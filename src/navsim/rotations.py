"""Quaternions and rotations.

Conventions (the usual aerospace ones, same as PX4):
- navigation frame NED (north, east, down), body frame FRD (forward, right, down);
- q = [w, x, y, z], Hamilton product, q_nb rotates a body vector into NED:
  v_n = q_nb * v_b * conj(q_nb);
- Euler angles ZYX: yaw psi, then pitch theta, then roll phi.

Every function accepts leading batch dimensions, so a Monte-Carlo can carry
N attitudes at once: q has shape (..., 4), vectors (..., 3).
"""

from __future__ import annotations

import numpy as np


def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pw, px, py, pz = np.moveaxis(p, -1, 0)
    qw, qx, qy, qz = np.moveaxis(q, -1, 0)
    return np.stack([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_from_rotvec(r: np.ndarray) -> np.ndarray:
    """exp map: rotation vector (axis * angle) to quaternion."""
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    half = 0.5 * angle
    # sin(a/2)/a with its Taylor expansion near 0
    k = np.where(angle > 1e-8, np.sin(half) / np.where(angle > 1e-8, angle, 1.0),
                 0.5 - angle**2 / 48.0)
    return np.concatenate([np.cos(half), k * r], axis=-1)


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """log map, returns the shortest rotation (angle in [0, pi])."""
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., :1])
    k = np.where(s > 1e-12, angle / np.where(s > 1e-12, s, 1.0), 2.0)
    return k * v


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by q (body to nav for q_nb)."""
    qv = q[..., 1:]
    w = q[..., :1]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_from_euler(phi, theta, psi) -> np.ndarray:
    cr, sr = np.cos(phi / 2), np.sin(phi / 2)
    cp, sp = np.cos(theta / 2), np.sin(theta / 2)
    cy, sy = np.cos(psi / 2), np.sin(psi / 2)
    return np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)


def euler_from_quat(q: np.ndarray):
    w, x, y, z = np.moveaxis(q, -1, 0)
    phi = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    theta = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    psi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return phi, theta, psi


def attitude_error(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Small-angle attitude error, as a rotation vector expressed in NED:
    q_est = exp(err) * q_true. Its x/y components are the tilt errors, z the
    heading error."""
    return rotvec_from_quat(quat_mul(q_est, quat_conj(q_true)))
