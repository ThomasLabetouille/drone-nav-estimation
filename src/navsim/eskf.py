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

Step 5 adds optional states for the aiding sensors (see aiding.py):

    baro_bias (1)   slow baro drift, Gauss-Markov       z_baro = -p_D + b + n
    mag_bias  (3)   magnetometer hard-iron bias, random walk
                    z_mag = C_nb^T m_n + b_mag + n
    wind      (2)   horizontal wind, random walk        z_pitot = |v - w| + n

The state vector is laid out at construction from what is enabled, in this
order: the 15 core states, then gnss_bias, baro_bias, mag_bias, wind.

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
GNSS_BIAS = slice(15, 18)  # when the GNSS bias is modelled (it always comes first)
N = 15
MIN_SIGMA = 1e-6


def reset_jacobian(dtheta_hat: np.ndarray) -> np.ndarray:
    """d(dtheta+)/d(dtheta) when the estimate dtheta_hat is injected.

    The attitude error is GLOBAL (q_true = exp(dtheta) q, dtheta in NED), so
    exp(dtheta+) = exp(dtheta) exp(-dtheta_hat), and to first order
    G = I + [dtheta_hat/2 x]. The often-quoted I - [dtheta_hat/2 x] is for a
    LOCAL error (q_true = q exp(dtheta)). This project used the local sign
    until step 5 (see the README); tests/test_oracles.py now checks G
    against a numerical derivative of the exact rotation composition."""
    return np.eye(3) + 0.5 * skew(dtheta_hat)


class ErrorStateEKF:
    def __init__(self, imu_cfg: IMUConfig, gnss_cfg: GNSSConfig, dt: float, runs: int,
                 model_gnss_bias: bool = False, aiding=None):
        """With model_gnss_bias, the filter carries the three GNSS bias states
        and uses only the white part of the GNSS noise as R. Without it, the
        whole GNSS error (white + correlated) is treated as white noise.

        aiding: an aiding.AidingConfig; each sensor it enables adds its states."""
        self.dt = dt
        self.runs = runs
        self.model_gnss_bias = model_gnss_bias
        self.aiding = aiding
        blocks = dict(BLOCKS)
        sizes = []
        if model_gnss_bias:
            sizes.append(("gnss_bias", 3))
        if aiding is not None and aiding.baro is not None:
            sizes.append(("baro_bias", 1))
        if aiding is not None and aiding.mag is not None:
            sizes.append(("mag_bias", 3))
        if aiding is not None and aiding.wind is not None:
            sizes.append(("wind", 2))
        n = 15
        for name, size in sizes:
            blocks[name] = slice(n, n + size)
            n += size
        self.blocks = blocks
        self.n = n

        # Discrete process noise. C (n_a) C^T = N_a^2 I because C is a
        # rotation, so Qd is constant and diagonal.
        qd = np.zeros(n)
        qd[3:6] = imu_cfg.accel_noise**2 * dt
        qd[6:9] = imu_cfg.gyro_noise**2 * dt
        qd[9:12] = imu_cfg.gyro_bias_rw**2 * dt
        qd[12:15] = imu_cfg.accel_bias_rw**2 * dt
        self.phi_diag = np.ones(n)  # transition of the extra states (Gauss-Markov or random walk)
        self.H_gnss = np.zeros((6, n))
        self.H_gnss[:, :6] = np.eye(6)
        if model_gnss_bias:
            if min(gnss_cfg.corr_sigma_h, gnss_cfg.corr_sigma_v) <= 0.0:
                raise ValueError("model_gnss_bias needs corr_sigma_h and corr_sigma_v > 0")
            s = blocks["gnss_bias"]
            phi = np.exp(-dt / gnss_cfg.corr_tau)
            self.phi_diag[s] = phi
            sig2 = np.array([gnss_cfg.corr_sigma_h, gnss_cfg.corr_sigma_h, gnss_cfg.corr_sigma_v]) ** 2
            qd[s] = sig2 * (1.0 - phi**2)
            self.H_gnss[0:3, s] = np.eye(3)
            self.R_gnss = gnss_cfg.R
        else:
            self.R_gnss = gnss_cfg.R_total
        if "baro_bias" in blocks:
            b = aiding.baro
            if b.drift_sigma < MIN_SIGMA:
                raise ValueError("the baro bias state needs drift_sigma > 0")
            s = blocks["baro_bias"]
            phi = np.exp(-dt / b.drift_tau)
            self.phi_diag[s] = phi
            qd[s] = b.drift_sigma**2 * (1.0 - phi**2)
            self.R_baro = np.array([[b.noise_sigma**2]])
        if "mag_bias" in blocks:
            m = aiding.mag
            qd[blocks["mag_bias"]] = m.bias_rw**2 * dt
            self.R_mag = np.eye(3) * m.noise**2
            self.m_n = np.asarray(m.field_ned, dtype=float)
        if "wind" in blocks:
            qd[blocks["wind"]] = aiding.wind.rw**2 * dt
        if aiding is not None and aiding.pitot is not None:
            self.R_pitot = np.array([[aiding.pitot.noise**2]])
        self.Qd = np.diag(qd)
        self.eye = np.eye(n)
        # kept for backward compatibility with step 4
        self.H, self.R = self.H_gnss, self.R_gnss
        if model_gnss_bias:
            self.gnss_phi = self.phi_diag[blocks["gnss_bias"]][0]

        self.p = np.zeros((runs, 3))
        self.v = np.zeros((runs, 3))
        self.q = np.tile([1.0, 0, 0, 0], (runs, 1))
        self.bg = np.zeros((runs, 3))
        self.ba = np.zeros((runs, 3))
        self.extra = np.zeros((runs, n - 15))  # nominal values of the extra states
        self.P = np.tile(np.eye(n), (runs, 1, 1))
        self._prev = None

    # nominal values of the extra states, by name
    def nominal(self, name):
        s = self.blocks[name]
        return self.extra[:, s.start - 15:s.stop - 15]

    @property
    def bgnss(self):
        return self.nominal("gnss_bias") if self.model_gnss_bias else np.zeros((self.runs, 3))

    def initialise(self, p, v, q, P0, bg=None, ba=None, extra=None):
        self.p = np.array(p, dtype=float)
        self.v = np.array(v, dtype=float)
        self.q = np.array(q, dtype=float)
        self.bg = np.zeros((self.runs, 3)) if bg is None else np.array(bg, dtype=float)
        self.ba = np.zeros((self.runs, 3)) if ba is None else np.array(ba, dtype=float)
        self.extra = np.zeros((self.runs, self.n - 15)) if extra is None else np.array(extra, dtype=float)
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

        self.omega = dth / dt  # bias-corrected body rate, used to detect manoeuvres
        first = self._prev is None
        prev_dth, prev_dv = (dth, dv) if first else self._prev
        self.p, self.v, self.q = strapdown.step(self.p, self.v, self.q, dth, dv,
                                                prev_dth, prev_dv, dt, "full", first)
        self._prev = (dth, dv)
        self.extra = self.extra * self.phi_diag[15:]

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
        idx = np.arange(15, n)
        Phi[:, idx, idx] = self.phi_diag[15:]  # exact for Gauss-Markov and random walk
        self.Phi = Phi  # kept for inspection and tests
        P = Phi @ self.P @ np.swapaxes(Phi, 1, 2) + self.Qd
        self.P = 0.5 * (P + np.swapaxes(P, 1, 2))

    # --- measurement updates ------------------------------------------------

    def update(self, y: np.ndarray, H: np.ndarray, R: np.ndarray, frozen=None):
        """Generic update. y: (runs, m) innovations, H: (m, n) or (runs, m, n),
        R: (m, m). Joseph form, then injection and reset. Returns
        (innovations, their covariance, NIS).

        frozen: optional (slice, mask) — for the runs where mask is True, the
        states in the slice are "consider" states (Schmidt-Kalman): their
        uncertainty is counted in S, but their gain is set to zero, so the
        update does not learn them. The Joseph form keeps P exact for any gain."""
        Ht = np.swapaxes(H, -1, -2)
        PHt = self.P @ Ht
        S = H @ PHt + R
        S_inv = np.linalg.inv(S)
        K = PHt @ S_inv
        if frozen is not None:
            sl, mask = frozen
            K = K.copy()
            K[mask, sl, :] = 0.0
        dx = np.einsum("rij,rj->ri", K, y)
        # Joseph form: (I - KH) P (I - KH)^T + K R K^T
        IKH = self.eye - K @ H
        P = IKH @ self.P @ np.swapaxes(IKH, 1, 2) + K @ R @ np.swapaxes(K, 1, 2)

        # Inject the estimated error into the nominal state, then reset it to
        # zero; the reset rotates the attitude covariance (see reset_jacobian).
        self.p = self.p + dx[:, 0:3]
        self.v = self.v + dx[:, 3:6]
        self.q = quat_normalize(quat_mul(quat_from_rotvec(dx[:, 6:9]), self.q))
        self.bg = self.bg + dx[:, 9:12]
        self.ba = self.ba + dx[:, 12:15]
        self.extra = self.extra + dx[:, 15:]
        G = np.broadcast_to(self.eye, (self.runs, self.n, self.n)).copy()
        G[:, 6:9, 6:9] = reset_jacobian(dx[:, 6:9])
        P = G @ P @ np.swapaxes(G, 1, 2)
        self.P = 0.5 * (P + np.swapaxes(P, 1, 2))

        nis = np.einsum("ri,rij,rj->r", y, S_inv, y)
        return y, S, nis

    def update_gnss(self, z: np.ndarray):
        """z: (runs, 6) = position and velocity in NED."""
        y = z - np.concatenate([self.p + self.bgnss, self.v], axis=1)
        return self.update(y, self.H_gnss, self.R_gnss)

    def update_baro(self, z: np.ndarray):
        """z: (runs,) altitude. Altitude is -p_D, so H has -1 on dp_D."""
        y = (z - (-self.p[:, 2] + self.nominal("baro_bias")[:, 0]))[:, None]
        H = np.zeros((1, self.n))
        H[0, 2] = -1.0
        H[0, self.blocks["baro_bias"]] = 1.0
        return self.update(y, H, self.R_baro)

    def update_mag(self, z: np.ndarray, learn_bias=None):
        """z: (runs, 3) magnetic field in the body frame.

        Predicted: C^T m_n + b. With q_true = exp(dtheta) q, the true C^T is
        C^T (I - [dtheta x]), so the attitude columns of H are C^T [m_n x]."""
        C = dcm_from_quat(self.q)
        Ct = np.swapaxes(C, 1, 2)
        y = z - (np.einsum("rij,j->ri", Ct, self.m_n) + self.nominal("mag_bias"))
        H = np.zeros((self.runs, 3, self.n))
        H[:, :, 6:9] = Ct @ skew(self.m_n)
        H[:, :, self.blocks["mag_bias"]] = np.eye(3)
        frozen = None if learn_bias is None else (self.blocks["mag_bias"], ~np.asarray(learn_bias))
        return self.update(y, H, self.R_mag, frozen)

    def update_airspeed(self, z: np.ndarray):
        """z: (runs,) true airspeed = |v - w|, w = (w_N, w_E, 0).

        H = u^T on the velocity and -u_h^T on the wind, u the unit vector of
        the air-relative velocity."""
        w = np.concatenate([self.nominal("wind"), np.zeros((self.runs, 1))], axis=1)
        v_air = self.v - w
        speed = np.linalg.norm(v_air, axis=1)
        u = v_air / speed[:, None]
        y = (z - speed)[:, None]
        H = np.zeros((self.runs, 1, self.n))
        H[:, 0, 3:6] = u
        H[:, 0, self.blocks["wind"]] = -u[:, :2]
        return self.update(y, H, self.R_pitot)

    def update_sideslip(self, sigma_rad: float):
        """Synthetic measurement of zero sideslip: a fixed-wing flies with its
        air-relative velocity in its plane of symmetry, so the lateral
        component of v - w in the body frame is ~0. Without it, the Pitot only
        sees the wind component along the flight path, and the crosswind is
        unobservable in straight flight.

        beta ~ (C^T (v - w))_y / V. With C_true^T = C^T (I - [dtheta x]):
        d/d(dv) = (C^T_y - beta u) / V, d/d(dw) = minus its horizontal part,
        d/d(dtheta) = (C^T [(v - w) x])_y / V, with u the unit air velocity."""
        w = np.concatenate([self.nominal("wind"), np.zeros((self.runs, 1))], axis=1)
        v_air = self.v - w
        V = np.linalg.norm(v_air, axis=1)
        Ct = np.swapaxes(dcm_from_quat(self.q), 1, 2)
        beta = np.einsum("rj,rj->r", Ct[:, 1, :], v_air) / V
        y = (0.0 - beta)[:, None]
        # derivative of beta w.r.t. the air velocity, including that of 1/V
        d_vair = (Ct[:, 1, :] - beta[:, None] * v_air / V[:, None]) / V[:, None]
        H = np.zeros((self.runs, 1, self.n))
        H[:, 0, 3:6] = d_vair
        H[:, 0, self.blocks["wind"]] = -d_vair[:, :2]
        H[:, 0, 6:9] = (Ct @ skew(v_air))[:, 1, :] / V[:, None]
        return self.update(y, H, np.array([[sigma_rad**2]]))

