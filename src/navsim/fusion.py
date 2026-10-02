"""Run the IMU + GNSS error-state EKF over a simulated flight, for a batch of
Monte-Carlo runs, and log everything needed for evaluation at each GNSS
epoch: true error, covariance, NEES per block, NIS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gnss as gnss_mod
from .eskf import BLOCKS, ErrorStateEKF
from .imu import IMUConfig, IMUStream
from .rotations import quat_conj, quat_from_rotvec, quat_mul, rotvec_from_quat
from .trajectory3d import Trajectory3D


@dataclass(frozen=True)
class InitConfig:
    """Initial attitude uncertainty. In flight, roll and pitch come from a
    short accelerometer levelling, heading from the GNSS course (no wind, no
    sideslip here, so the course is the heading up to the error below)."""
    tilt_sigma_deg: float = 2.0
    yaw_sigma_deg: float = 5.0


@dataclass
class FusionLog:
    t: np.ndarray            # (E,) GNSS epochs, plus t=0
    err: np.ndarray          # (runs, E, 15) truth minus estimate
    sigma: np.ndarray        # (runs, E, 15) sqrt(diag(P))
    nees: dict               # block name -> (runs, E); plus "total"
    nis: np.ndarray          # (runs, E) NaN at t=0
    gyro_bias: np.ndarray    # (runs, E, 3) truth
    accel_bias: np.ndarray   # (runs, E, 3) truth
    gyro_bias_est: np.ndarray
    accel_bias_est: np.ndarray


def initial_covariance(imu_cfg: IMUConfig, gnss_cfg: gnss_mod.GNSSConfig, init: InitConfig):
    tilt, yaw = np.radians(init.tilt_sigma_deg), np.radians(init.yaw_sigma_deg)
    d = np.concatenate([
        np.diag(gnss_cfg.R),
        [tilt**2, tilt**2, yaw**2],
        [imu_cfg.gyro_bias0**2] * 3,
        [imu_cfg.accel_bias0**2] * 3,
    ])
    return np.diag(d)


def run(traj: Trajectory3D, imu_cfg: IMUConfig, gnss_cfg: gnss_mod.GNSSConfig,
        init: InitConfig, runs: int, rng) -> FusionLog:
    dt = traj.dt
    decim = int(round(1.0 / (gnss_cfg.rate_hz * dt)))
    n_steps = len(traj.dtheta)
    epochs = [0] + list(range(decim, n_steps + 1, decim))
    E = len(epochs)

    stream = IMUStream(imu_cfg, rng, runs, dt)
    ekf = ErrorStateEKF(imu_cfg, gnss_cfg, dt, runs)

    # Initial state: first GNSS fix, attitude perturbed by the alignment error.
    z0 = gnss_mod.measure(traj.p_n[0], traj.v_n[0], gnss_cfg, rng, runs)
    tilt, yaw = np.radians(init.tilt_sigma_deg), np.radians(init.yaw_sigma_deg)
    dtheta0 = rng.normal(0.0, 1.0, (runs, 3)) * np.array([tilt, tilt, yaw])
    q0 = quat_mul(quat_from_rotvec(-dtheta0), np.broadcast_to(traj.q_nb[0], (runs, 4)))
    P0 = initial_covariance(imu_cfg, gnss_cfg, init)
    ekf.initialise(z0[:, :3], z0[:, 3:], q0, P0)

    log = FusionLog(
        t=traj.t[epochs],
        err=np.empty((runs, E, 15)), sigma=np.empty((runs, E, 15)),
        nees={name: np.empty((runs, E)) for name in list(BLOCKS) + ["total"]},
        nis=np.full((runs, E), np.nan),
        gyro_bias=np.empty((runs, E, 3)), accel_bias=np.empty((runs, E, 3)),
        gyro_bias_est=np.empty((runs, E, 3)), accel_bias_est=np.empty((runs, E, 3)),
    )

    def record(j, k):
        e = np.concatenate([
            traj.p_n[k] - ekf.p,
            traj.v_n[k] - ekf.v,
            rotvec_from_quat(quat_mul(np.broadcast_to(traj.q_nb[k], (runs, 4)), quat_conj(ekf.q))),
            stream.gyro_bias - ekf.bg,
            stream.accel_bias - ekf.ba,
        ], axis=1)
        log.err[:, j] = e
        log.sigma[:, j] = np.sqrt(np.einsum("rii->ri", ekf.P))
        for name, s in list(BLOCKS.items()) + [("total", slice(0, 15))]:
            Pb = ekf.P[:, s, s]
            log.nees[name][:, j] = np.einsum("ri,rij,rj->r", e[:, s], np.linalg.inv(Pb), e[:, s])
        log.gyro_bias[:, j] = stream.gyro_bias
        log.accel_bias[:, j] = stream.accel_bias
        log.gyro_bias_est[:, j] = ekf.bg
        log.accel_bias_est[:, j] = ekf.ba

    record(0, 0)
    j = 1
    for k in range(n_steps):
        dth, dv = stream.sample(traj.dtheta[k], traj.dvel[k])
        ekf.predict(dth, dv)
        if (k + 1) % decim == 0:
            z = gnss_mod.measure(traj.p_n[k + 1], traj.v_n[k + 1], gnss_cfg, rng, runs)
            _, _, nis = ekf.update_gnss(z)
            log.nis[:, j] = nis
            record(j, k + 1)
            j += 1
    return log
