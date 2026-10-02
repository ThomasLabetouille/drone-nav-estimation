"""Error-state extended Kalman filter, IMU + GNSS, 15 states.

The nominal state (position, velocity, attitude quaternion, gyro bias,
accelerometer bias) is propagated by the strapdown of step 2 with
bias-corrected increments. The filter estimates the error of that nominal
state, defined as truth minus nominal:

    dx = [dp, dv, dtheta, dbg, dba]                  (15,)
    q_true = exp(dtheta) * q_nominal                 dtheta in NED

Linearised error dynamics (flat Earth, no Earth rotation, as in step 2),
with f_n the specific force in NED and C = C_nb:

    d(dp)/dt     = dv
    d(dv)/dt     = -[f_n x] dtheta - C dba + C n_a
    d(dtheta)/dt = -C dbg + C n_g
    d(dbg)/dt    = w_bg
    d(dba)/dt    = w_ba

-[f_n x] dtheta is the coupling that makes attitude observable from
position/velocity measurements: a tilt error makes the accelerometer project
part of the specific force on the wrong axis. In straight unaccelerated
flight f_n is vertical, so a heading error (rotation about vertical) has no
effect: heading only becomes observable when the aircraft accelerates or
turns.

Optionally (step 4), three more states model the slowly varying error of
the GNSS position as a first-order Gauss-Markov process:

    d(db_gnss)/dt = -db_gnss / tau + w          z_pos = p + b_gnss + n

Every array carries a leading batch axis (runs), so a Monte-Carlo runs all
its filters in one pass.
"""

from __future__ import annotations

import numpy as np

from . import strapdown
from .gnss import GNSSConfig
from .imu import IMUConfig
from .rotations import dcm_from_quat, quat_from_rotvec, quat_mul, quat_normalize, skew

BLOCKS = {
    "position": slice(0, 3),
    "velocity": slice(3, 6),
    "attitude": slice(6, 9),
    "gyro_bias": slice(9, 12),
    "accel_bias": slice(12, 15),
}
GNSS_BIAS = slice(15, 18)
N = 15


class ErrorStateEKF:
    def __init__(self, imu_cfg: IMUConfig, gnss_cfg: GNSSConfig, dt: float, runs: int,
                 model_gnss_bias: bool = False):
        """With model_gnss_bias, the filter carries the three GNSS bias states
        and uses only the white part of the GNSS noise as R. Without it, the
        whole GNSS error (white + correlated) is treated as white noise."""
        self.dt = dt
        self.runs = runs
        self.model_gnss_bias = model_gnss_bias
        n = 18 if model_gnss_bias else 15
        self.n = n
        # Discrete process noise. C (n_a) C^T = N_a^2 I because C is a
        # rotation, so Qd is constant and diagonal.
        qd = np.zeros(n)
        qd[3:6] = imu_cfg.accel_noise**2 * dt
        qd[6:9] = imu_cfg.gyro_noise**2 * dt
        qd[9:12] = imu_cfg.gyro_bias_rw**2 * dt
        qd[12:15] = imu_cfg.accel_bias_rw**2 * dt
        self.H = np.zeros((6, n))
        self.H[:, :6] = np.eye(6)
        if model_gnss_bias:
            phi = np.exp(-dt / gnss_cfg.corr_tau)
            self.gnss_phi = phi
            sig2 = np.array([gnss_cfg.corr_sigma_h, gnss_cfg.corr_sigma_h, gnss_cfg.corr_sigma_v]) ** 2
            qd[15:18] = sig2 * (1.0 - phi**2)
            self.H[0:3, 15:18] = np.eye(3)
            self.R = gnss_cfg.R
        else:
            self.R = gnss_cfg.R_total
        self.Qd = np.diag(qd)
        self.eye = np.eye(n)

        self.p = np.zeros((runs, 3))
        self.v = np.zeros((runs, 3))
        self.q = np.tile([1.0, 0, 0, 0], (runs, 1))
        self.bg = np.zeros((runs, 3))
        self.ba = np.zeros((runs, 3))
        self.bgnss = np.zeros((runs, 3))
        self.P = np.tile(np.eye(n), (runs, 1, 1))
        self._prev = None

    def initialise(self, p, v, q, P0, bg=None, ba=None):
        self.p = np.array(p, dtype=float)
        self.v = np.array(v, dtype=float)
        self.q = np.array(q, dtype=float)
        self.bg = np.zeros((self.runs, 3)) if bg is None else np.array(bg, dtype=float)
        self.ba = np.zeros((self.runs, 3)) if ba is None else np.array(ba, dtype=float)
        self.bgnss = np.zeros((self.runs, 3))
        self.P = np.broadcast_to(P0, (self.runs, self.n, self.n)).copy()
        self._prev = None

    def copy_nominal(self):
        return self.p.copy(), self.v.copy(), self.q.copy(), self.bg.copy(), self.ba.copy()

    def predict(self, dth_meas: np.ndarray, dv_meas: np.ndarray):
        dt, n = self.dt, self.n
        dth = dth_meas - self.bg * dt
        dv = dv_meas - self.ba * dt
        C = dcm_from_quat(self.q)
        f_n = np.einsum("rij,rj->ri", C, dv) / dt

        first = self._prev is None
        prev_dth, prev_dv = (dth, dv) if first else self._prev
        self.p, self.v, self.q = strapdown.step(self.p, self.v, self.q, dth, dv,
                                                prev_dth, prev_dv, dt, "full", first)
        self._prev = (dth, dv)
        if self.model_gnss_bias:
            self.bgnss = self.bgnss * self.gnss_phi

        # Phi = I + F dt + (F dt)^2 / 2, written out block by block (F is sparse).
        fx = skew(f_n)
        Phi = np.broadcast_to(self.eye, (self.runs, n, n)).copy()
        Phi[:, 0:3, 3:6] += np.eye(3) * dt
        Phi[:, 3:6, 6:9] += -fx * dt
        Phi[:, 3:6, 12:15] += -C * dt
        Phi[:, 6:9, 9:12] += -C * dt
        h = 0.5 * dt * dt
        Phi[:, 0:3, 6:9] += -fx * h
        Phi[:, 0:3, 12:15] += -C * h
        Phi[:, 3:6, 9:12] += (fx @ C) * h
        if self.model_gnss_bias:
            Phi[:, 15:18, 15:18] = np.eye(3) * self.gnss_phi  # exact for Gauss-Markov
        self.Phi = Phi  # kept for inspection and tests
        P = Phi @ self.P @ np.swapaxes(Phi, 1, 2) + self.Qd
        self.P = 0.5 * (P + np.swapaxes(P, 1, 2))

    def update_gnss(self, z: np.ndarray):
        """z: (runs, 6) = position and velocity in NED.

        Returns innovations (runs, 6), their covariance (runs, 6, 6) and NIS."""
        H, Ht = self.H, self.H.T
        pred = np.concatenate([self.p + self.bgnss, self.v], axis=1)
        y = z - pred
        PHt = self.P @ Ht
        S = H @ PHt + self.R
        S_inv = np.linalg.inv(S)
        K = PHt @ S_inv
        dx = np.einsum("rij,rj->ri", K, y)
        # Joseph form: (I - KH) P (I - KH)^T + K R K^T
        IKH = self.eye - K @ H
        P = IKH @ self.P @ np.swapaxes(IKH, 1, 2) + K @ self.R @ np.swapaxes(K, 1, 2)

        # Inject the estimated error into the nominal state, then reset it to
        # zero. The reset rotates the attitude covariance by
        # G = I - [dtheta/2 x] (first-order reset Jacobian).
        self.p = self.p + dx[:, 0:3]
        self.v = self.v + dx[:, 3:6]
        self.q = quat_normalize(quat_mul(quat_from_rotvec(dx[:, 6:9]), self.q))
        self.bg = self.bg + dx[:, 9:12]
        self.ba = self.ba + dx[:, 12:15]
        if self.model_gnss_bias:
            self.bgnss = self.bgnss + dx[:, 15:18]
        G = np.broadcast_to(self.eye, (self.runs, self.n, self.n)).copy()
        G[:, 6:9, 6:9] -= 0.5 * skew(dx[:, 6:9])
        P = G @ P @ np.swapaxes(G, 1, 2)
        self.P = 0.5 * (P + np.swapaxes(P, 1, 2))

        nis = np.einsum("ri,rij,rj->r", y, S_inv, y)
        return y, S, nis
