"""Run the IMU + GNSS error-state EKF over a simulated flight, for a batch of
Monte-Carlo runs, and log everything needed for evaluation at each GNSS
epoch: true error, covariance, NEES per block, NIS, and the error of the
state actually delivered at real time (which differs from the filter state
when the filter runs on a delayed horizon)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import gnss as gnss_mod
from . import strapdown
from .eskf import BLOCKS, GNSS_BIAS, ErrorStateEKF
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


LATENCY_MODES = ("ignore", "compensate", "delayed")


@dataclass
class FusionLog:
    t: np.ndarray            # (E,) filter time of each logged epoch (GNSS fix time)
    t_out: np.ndarray        # (E,) real time of the output state
    err: np.ndarray          # (runs, E, n) truth minus estimate, at filter time
    sigma: np.ndarray        # (runs, E, n) sqrt(diag(P))
    nees: dict               # block name -> (runs, E); plus "total"
    nis: np.ndarray          # (runs, E) NaN at t=0
    out_err: np.ndarray      # (runs, E, 9) position, velocity, attitude error of the
                             # output state at real time (what a controller would use)
    gyro_bias: np.ndarray    # (runs, E, 3) truth
    accel_bias: np.ndarray   # (runs, E, 3) truth
    gyro_bias_est: np.ndarray
    accel_bias_est: np.ndarray


def initial_covariance(imu_cfg: IMUConfig, gnss_cfg: gnss_mod.GNSSConfig, init: InitConfig,
                       model_gnss_bias: bool = False):
    """Initial covariance for an initialisation on the first GNSS fix.

    With the GNSS bias modelled, the position error at t=0 is -(b + n) and the
    bias error is b: the two are negatively correlated, as for the baro drift
    of step 1, and P0 says so."""
    tilt, yaw = np.radians(init.tilt_sigma_deg), np.radians(init.yaw_sigma_deg)
    pos_var = np.diag(gnss_cfg.R if model_gnss_bias else gnss_cfg.R_total)
    d = np.concatenate([
        pos_var,
        [tilt**2, tilt**2, yaw**2],
        [imu_cfg.gyro_bias0**2] * 3,
        [imu_cfg.accel_bias0**2] * 3,
    ])
    if not model_gnss_bias:
        return np.diag(d)
    corr2 = gnss_cfg.corr_sigma**2
    P = np.diag(np.concatenate([d, corr2]))
    P[0:3, 0:3] += np.diag(corr2)
    P[0:3, 15:18] = P[15:18, 0:3] = -np.diag(corr2)
    return P


def run(traj: Trajectory3D, imu_cfg: IMUConfig, gnss_cfg: gnss_mod.GNSSConfig,
        init: InitConfig, runs: int, rng, model_gnss_bias: bool = False,
        latency_mode: str = "ignore") -> FusionLog:
    """latency_mode, used when gnss_cfg.latency_s > 0:
    - "ignore": fuse each fix when it arrives, as if it described the present;
    - "compensate": same, but shift the position measurement forward by
      v * latency (the velocity measurement is left as is);
    - "delayed": run the filter on a delayed time horizon, where each fix
      arrives exactly at its own time, and produce the present state by
      re-integrating the buffered IMU samples (output predictor)."""
    if latency_mode not in LATENCY_MODES:
        raise ValueError(f"latency_mode must be one of {LATENCY_MODES}")
    dt = traj.dt
    decim = int(round(1.0 / (gnss_cfg.rate_hz * dt)))
    D = int(round(gnss_cfg.latency_s / dt))
    n_steps = len(traj.dtheta)
    E_max = n_steps // decim + 1

    stream = IMUStream(imu_cfg, rng, runs, dt)
    ekf = ErrorStateEKF(imu_cfg, gnss_cfg, dt, runs, model_gnss_bias=model_gnss_bias)
    n = ekf.n
    correlated = gnss_cfg.corr_sigma_h > 0 or gnss_cfg.corr_sigma_v > 0
    corr = gnss_mod.CorrelatedError(gnss_cfg, rng, runs) if correlated else None
    gnss_bias_truth = np.zeros((runs, 3))

    def fix(m):
        nonlocal gnss_bias_truth
        bias = corr.next() if corr is not None else None
        if bias is not None:
            gnss_bias_truth = bias.copy()
        return gnss_mod.measure(traj.p_n[m], traj.v_n[m], gnss_cfg, rng, runs, bias)

    # Initial state: first GNSS fix, attitude perturbed by the alignment error.
    z0 = fix(0)
    tilt, yaw = np.radians(init.tilt_sigma_deg), np.radians(init.yaw_sigma_deg)
    dtheta0 = rng.normal(0.0, 1.0, (runs, 3)) * np.array([tilt, tilt, yaw])
    q0 = quat_mul(quat_from_rotvec(-dtheta0), np.broadcast_to(traj.q_nb[0], (runs, 4)))
    P0 = initial_covariance(imu_cfg, gnss_cfg, init, model_gnss_bias)
    ekf.initialise(z0[:, :3], z0[:, 3:], q0, P0)

    blocks = dict(BLOCKS)
    if model_gnss_bias:
        blocks["gnss_bias"] = GNSS_BIAS
    log = FusionLog(
        t=np.empty(E_max), t_out=np.empty(E_max),
        err=np.empty((runs, E_max, n)), sigma=np.empty((runs, E_max, n)),
        nees={name: np.empty((runs, E_max)) for name in list(blocks) + ["total"]},
        nis=np.full((runs, E_max), np.nan),
        out_err=np.empty((runs, E_max, 9)),
        gyro_bias=np.empty((runs, E_max, 3)), accel_bias=np.empty((runs, E_max, 3)),
        gyro_bias_est=np.empty((runs, E_max, 3)), accel_bias_est=np.empty((runs, E_max, 3)),
    )

    def truth_err(k, p, v, q):
        return np.concatenate([
            traj.p_n[k] - p,
            traj.v_n[k] - v,
            rotvec_from_quat(quat_mul(np.broadcast_to(traj.q_nb[k], (runs, 4)), quat_conj(q))),
        ], axis=1)

    def record(j, kf, ko, bg_true, ba_true, out=None):
        parts = [truth_err(kf, ekf.p, ekf.v, ekf.q), bg_true - ekf.bg, ba_true - ekf.ba]
        if model_gnss_bias:
            parts.append(gnss_bias_truth - ekf.bgnss)
        e = np.concatenate(parts, axis=1)
        log.t[j], log.t_out[j] = traj.t[kf], traj.t[ko]
        log.err[:, j] = e
        log.sigma[:, j] = np.sqrt(np.einsum("rii->ri", ekf.P))
        for name, s in list(blocks.items()) + [("total", slice(0, n))]:
            log.nees[name][:, j] = np.einsum("ri,rij,rj->r", e[:, s], np.linalg.inv(ekf.P[:, s, s]),
                                             e[:, s])
        log.out_err[:, j] = e[:, :9] if out is None else truth_err(ko, *out)
        log.gyro_bias[:, j], log.accel_bias[:, j] = bg_true, ba_true
        log.gyro_bias_est[:, j], log.accel_bias_est[:, j] = ekf.bg, ekf.ba

    def output_predictor(samples):
        """Propagate a copy of the nominal state through the buffered samples."""
        p, v, q, bg, ba = ekf.copy_nominal()
        prev = ekf._prev
        for dth_m, dv_m, _, _ in samples:
            dth, dv = dth_m - bg * dt, dv_m - ba * dt
            first = prev is None
            p, v, q = strapdown.step(p, v, q, dth, dv, *(prev or (dth, dv)), dt, "full", first)
            prev = (dth, dv)
        return p, v, q

    record(0, 0, 0, stream.gyro_bias, stream.accel_bias)
    j = 1
    fifo = deque()
    for k in range(n_steps):
        dth, dv = stream.sample(traj.dtheta[k], traj.dvel[k])
        if latency_mode == "delayed":
            fifo.append((dth, dv, stream.gyro_bias, stream.accel_bias))
            if len(fifo) <= D:
                continue
            dth_f, dv_f, bg_t, ba_t = fifo.popleft()
            ekf.predict(dth_f, dv_f)
            f = k + 1 - D
            if f % decim == 0:
                _, _, nis = ekf.update_gnss(fix(f))
                log.nis[:, j] = nis
                record(j, f, k + 1, bg_t, ba_t, output_predictor(fifo))
                j += 1
        else:
            ekf.predict(dth, dv)
            m = k + 1 - D
            if m > 0 and m % decim == 0:
                z = fix(m)
                if latency_mode == "compensate":
                    z[:, 0:3] += ekf.v * gnss_cfg.latency_s
                _, _, nis = ekf.update_gnss(z)
                log.nis[:, j] = nis
                record(j, k + 1, k + 1, stream.gyro_bias, stream.accel_bias)
                j += 1

    for name in ("t", "t_out"):
        setattr(log, name, getattr(log, name)[:j])
    for name in ("err", "sigma", "nis", "out_err", "gyro_bias", "accel_bias",
                 "gyro_bias_est", "accel_bias_est"):
        setattr(log, name, getattr(log, name)[:, :j])
    log.nees = {k_: v[:, :j] for k_, v in log.nees.items()}
    return log
