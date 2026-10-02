"""Linear Kalman filter for the vertical channel: accelerometer + barometer.

State:
    x = [h, v, b_a]          altitude, vertical speed, accelerometer bias
    x = [h, v, b_a, d]       + slow baro error, if the filter models it

Continuous model (the accelerometer is used as a control input):
    dh/dt   = v
    dv/dt   = a_meas - b_a - n_a        n_a : accel white noise
    db_a/dt = w_b                       w_b : bias random walk
    dd/dt   = -d / tau + w_d            w_d : drives the Gauss-Markov drift

Baro measurement:
    z = h + d + n_z

The continuous model is discretised exactly (Van Loan), which keeps the
filter tuning expressed in physical units (noise densities, random walks)
instead of per-sample magic numbers.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .sensors import AccelConfig, BaroConfig


def van_loan(A: np.ndarray, L: np.ndarray, Qc: np.ndarray, dt: float):
    """Exact discretisation of dx/dt = A x + L w, E[w w^T] = Qc delta(t).

    Returns (Phi, Qd) such that x_{k+1} = Phi x_k + w_k, Cov(w_k) = Qd.
    """
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = L @ Qc @ L.T
    M[n:, n:] = A.T
    E = expm(M * dt)
    Phi = E[n:, n:].T
    Qd = Phi @ E[:n, n:]
    return Phi, 0.5 * (Qd + Qd.T)


def input_matrix(A: np.ndarray, B: np.ndarray, dt: float) -> np.ndarray:
    """Discrete input matrix for a zero-order-hold input."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    return expm(M * dt)[:n, n:]


class AltitudeKF:
    def __init__(self, accel: AccelConfig, baro: BaroConfig, dt: float):
        """`accel` and `baro` describe what the filter *believes* about the
        sensors. If baro.drift_sigma > 0 the slow baro error is added to the
        state; otherwise the filter treats the baro as white noise only."""
        self.dt = dt
        self.models_drift = baro.drift_sigma > 0
        n = 4 if self.models_drift else 3
        self.n = n

        A = np.zeros((n, n))
        A[0, 1] = 1.0
        A[1, 2] = -1.0
        B = np.zeros((n, 1))
        B[1, 0] = 1.0

        noises = [accel.noise_density**2, accel.bias_rw**2]
        L = np.zeros((n, 3 if self.models_drift else 2))
        L[1, 0] = -1.0
        L[2, 1] = 1.0
        if self.models_drift:
            A[3, 3] = -1.0 / baro.drift_tau
            L[3, 2] = 1.0
            noises.append(2.0 * baro.drift_sigma**2 / baro.drift_tau)
        Qc = np.diag(noises)

        self.Phi, self.Q = van_loan(A, L, Qc, dt)
        self.Gamma = input_matrix(A, B, dt)[:, 0]
        self.H = np.zeros(n)
        self.H[0] = 1.0
        if self.models_drift:
            self.H[3] = 1.0
        self.R = baro.noise_sigma**2

        self._accel = accel
        self._baro = baro
        self.x = np.zeros(n)
        self.P = np.eye(n)

    def initialise(self, z0: float, v0: float = 0.0, v0_sigma: float = 0.05):
        """Start from the first baro sample, aircraft at rest on the catapult.

        With h_hat = z0, the altitude error is (n_z + d) and the drift error
        is -d, so the two are negatively correlated. Writing that correlation
        into P0 is what keeps the filter consistent from the first step.
        """
        n = self.n
        self.x = np.zeros(n)
        self.x[0] = z0
        self.x[1] = v0
        P = np.zeros((n, n))
        P[0, 0] = self._baro.noise_sigma**2
        P[1, 1] = v0_sigma**2
        P[2, 2] = self._accel.bias_sigma0**2
        if self.models_drift:
            sd2 = self._baro.drift_sigma**2
            P[0, 0] += sd2
            P[3, 3] = sd2
            P[0, 3] = P[3, 0] = -sd2
        self.P = P

    def predict(self, a_meas: float):
        self.x = self.Phi @ self.x + self.Gamma * a_meas
        self.P = self.Phi @ self.P @ self.Phi.T + self.Q

    def update_baro(self, z: float):
        """Scalar update in Joseph form. Returns (innovation, S, NIS)."""
        H = self.H
        y = z - H @ self.x
        PHt = self.P @ H
        S = H @ PHt + self.R
        K = PHt / S
        self.x = self.x + K * y
        I_KH = np.eye(self.n) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + self.R * np.outer(K, K)
        return y, S, y * y / S
