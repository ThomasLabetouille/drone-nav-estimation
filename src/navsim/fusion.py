"""Run the IMU + GNSS error-state EKF over a simulated flight, for a batch of
Monte-Carlo runs, and log everything needed for evaluation at each GNSS
epoch: true error, covariance, NEES per block, NIS, and the error of the
state actually delivered at real time (which differs from the filter state
when the filter runs on a delayed horizon)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from . import gnss as gnss_mod
from . import strapdown
from .eskf import ErrorStateEKF
from .imu import IMUConfig, IMUStream
from . import aiding as aid
from .faults import NONE as NO_FAULTS
from .faults import Faults
from .rotations import euler_from_quat, quat_conj, quat_from_euler, quat_from_rotvec, quat_mul, rotvec_from_quat
from .trajectory3d import Trajectory3D


@dataclass(frozen=True)
class InitConfig:
    """Initial attitude uncertainty. In flight, roll and pitch come from a
    short accelerometer levelling, heading from the GNSS course (no wind, no
    sideslip here, so the course is the heading up to the error below)."""
    tilt_sigma_deg: float = 2.0
    yaw_sigma_deg: float = 5.0
    # Where the initial heading comes from (step 5):
    #   "perturbed"    truth plus an error drawn with yaw_sigma (steps 3-4);
    #   "gnss_course"  direction of the first GNSS velocity: equal to the
    #                  heading only without wind and sideslip;
    #   "magnetometer" tilt-compensated heading from the first magnetometer sample.
    yaw_source: str = "perturbed"


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
    blocks: dict             # state layout of the filter (name -> slice)
    aux_nis: dict            # sensor name -> (times (U,), NIS (runs, U)) of its updates
    aux_rejected: dict       # sensor name -> (runs, U) True where the gate rejected it
    gnss_status: np.ndarray  # (runs, E) 0 fused, 1 rejected by the gate, 2 no fix (outage)
    wind_resets: list        # (time, number of runs) of each wind reset after a Pitot lock-out
    gnss_resets: list        # (time, number of runs) of each reset on GNSS after a GNSS lock-out
    gyro_bias: np.ndarray    # (runs, E, 3) truth
    accel_bias: np.ndarray   # (runs, E, 3) truth
    gyro_bias_est: np.ndarray
    accel_bias_est: np.ndarray


def initial_covariance(imu_cfg: IMUConfig, gnss_cfg: gnss_mod.GNSSConfig, init: InitConfig,
                       model_gnss_bias: bool = False, aiding: aid.AidingConfig | None = None):
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
    extra = []
    if model_gnss_bias:
        extra.append(gnss_cfg.corr_sigma**2)
    if aiding is not None and aiding.baro is not None:
        extra.append([aiding.baro.drift_sigma**2])
    if aiding is not None and aiding.mag is not None and aiding.mag.fusion == "3d":
        extra.append([aiding.mag.bias0**2] * 3)
    if aiding is not None and aiding.wind is not None:
        extra.append([aiding.wind.sigma0**2] * 2)
    P = np.diag(np.concatenate([d, *extra])) if extra else np.diag(d)
    if model_gnss_bias:
        corr2 = gnss_cfg.corr_sigma**2
        P[0:3, 0:3] += np.diag(corr2)
        P[0:3, 15:18] = P[15:18, 0:3] = -np.diag(corr2)
    return P


def wind_from_first_airspeed(blocks, n, v0, q0, tas0, P0, tas_sigma, sideslip_sigma):
    """Initial wind from the first Pitot sample, as EKF2 does when it starts
    fusing airspeed: wind = ground velocity - air velocity, with the air
    velocity taken along the heading (zero sideslip, level flight at start).

    Linearised: dw = dv_h + n_tas u - TAS u_perp (dpsi + beta), with u the
    heading direction, u_perp its normal, dpsi = dtheta_D (attitude error
    about the vertical) and beta the sideslip, zero on average, of standard
    deviation sideslip_sigma [rad]. The wind is then correlated with the
    velocity and the heading, and P0 says so: the filter knows the crosswind
    is only as good as the heading. The sideslip term also keeps P0 positive
    definite (without it, the crosswind would be an exact combination of the
    velocity and heading errors).

    Also used to restart the wind after a Pitot lock-out (step 6)."""
    runs = len(tas0)
    psi = euler_from_quat(q0)[2]
    u = np.stack([np.cos(psi), np.sin(psi)], axis=1)
    u_perp = np.stack([-np.sin(psi), np.cos(psi)], axis=1)
    sl = blocks["wind"]
    J = np.zeros((runs, 2, n))
    J[:, :, 3:5] = np.eye(2)
    J[:, :, 8] = -tas0[:, None] * u_perp
    P = np.broadcast_to(P0, (runs, n, n)).copy()
    P[:, sl, :] = 0.0
    P[:, :, sl] = 0.0
    cross = J @ P                     # cov(wind, x): J has no wind column
    P[:, sl, :] = cross
    P[:, :, sl] = np.swapaxes(cross, 1, 2)
    P[:, sl, sl] = (cross @ np.swapaxes(J, 1, 2) + tas_sigma**2 * u[:, :, None] * u[:, None, :]
                    + (tas0 * sideslip_sigma)[:, None, None] ** 2 * u_perp[:, :, None] * u_perp[:, None, :])
    extra = np.zeros((runs, n - 15))
    extra[:, sl.start - 15:sl.stop - 15] = v0[:, :2] - tas0[:, None] * u
    return P, extra


def run(traj: Trajectory3D, imu_cfg: IMUConfig, gnss_cfg: gnss_mod.GNSSConfig,
        init: InitConfig, runs: int, rng, model_gnss_bias: bool = False,
        latency_mode: str = "ignore", aiding: aid.AidingConfig | None = None,
        faults: Faults = NO_FAULTS, gate_prob: float | None = None,
        gnss_reset_after_s: float | None = 45.0) -> FusionLog:
    """latency_mode, used when gnss_cfg.latency_s > 0:
    - "ignore": fuse each fix when it arrives, as if it described the present;
    - "compensate": same, but shift the position measurement forward by
      v * latency (the velocity measurement is left as is);
    - "delayed": run the filter on a delayed time horizon, where each fix
      arrives exactly at its own time, and produce the present state by
      re-integrating the buffered IMU samples (output predictor).

    faults (step 6): GNSS outages and jumps, magnetic disturbances, injected
    into the measurements; the filter is not told.
    gate_prob: if set, every measurement goes through a chi-square
    innovation test at this probability (0.999: a consistent filter rejects
    0.1 % of good measurements) and is rejected per run when it fails.
    gnss_reset_after_s: if a GNSS fix is rejected and none has been fused for
    this long, position and velocity restart on the fix (None: never). A gate
    alone can lock the filter out of GNSS for good once it has drifted. The
    delay is the longest dead reckoning the filter is trusted for: shorter,
    a GNSS fault that lasts longer is accepted; longer, a real drift is
    corrected later."""
    if latency_mode not in LATENCY_MODES:
        raise ValueError(f"latency_mode must be one of {LATENCY_MODES}")
    dt = traj.dt
    decim = int(round(1.0 / (gnss_cfg.rate_hz * dt)))
    D = int(round(gnss_cfg.latency_s / dt))
    n_steps = len(traj.dtheta)
    E_max = n_steps // decim + 1

    stream = IMUStream(imu_cfg, rng, runs, dt)
    ekf = ErrorStateEKF(imu_cfg, gnss_cfg, dt, runs, model_gnss_bias=model_gnss_bias, aiding=aiding)
    n = ekf.n
    if gate_prob is not None:
        dof = {"gnss": 6, "baro": 1, "mag": 3, "pitot": 1, "sideslip": 1}
        ekf.gates = {name: chi2.ppf(gate_prob, d) for name, d in dof.items()}
    correlated = gnss_cfg.corr_sigma_h > 0 or gnss_cfg.corr_sigma_v > 0
    corr = gnss_mod.CorrelatedError(gnss_cfg, rng, runs) if correlated else None
    gnss_bias_truth = np.zeros((runs, 3))

    def fix(m):
        nonlocal gnss_bias_truth
        bias = corr.next() if corr is not None else None
        if bias is not None:
            gnss_bias_truth = bias.copy()
        z = gnss_mod.measure(traj.p_n[m], traj.v_n[m], gnss_cfg, rng, runs, bias)
        z[:, 0:3] += faults.gnss_offset(traj.t[m])
        return z

    last_gnss_fused = np.zeros(runs)

    def fuse_gnss(j, m, z):
        """GNSS update at fix index m, unless the receiver has no fix then."""
        t = traj.t[m]
        if faults.gnss_lost(t):
            log.gnss_status[:, j] = 2
        else:
            _, _, nis = ekf.update_gnss(z)
            log.nis[:, j] = nis
            rejected = ekf.rejected.copy()
            log.gnss_status[:, j] = np.where(rejected, 1, 0)
            last_gnss_fused[~rejected] = t
            if gnss_reset_after_s is not None:
                reset = rejected & (t - last_gnss_fused >= gnss_reset_after_s)
                if reset.any():
                    ekf.reset_to_gnss(reset, z)
                    last_gnss_fused[reset] = t
                    log.gnss_resets.append((t, int(reset.sum())))
        if aiding is not None and aiding.wind is not None and aiding.wind.rw_without_gnss is not None:
            dead_reckoning = t - last_gnss_fused > 1.0
            ekf.set_random_walk("wind", np.where(dead_reckoning, aiding.wind.rw_without_gnss, aiding.wind.rw))

    # Initial state: first GNSS fix, attitude perturbed by the alignment error.
    z0 = fix(0)
    tilt, yaw = np.radians(init.tilt_sigma_deg), np.radians(init.yaw_sigma_deg)
    dtheta0 = rng.normal(0.0, 1.0, (runs, 3)) * np.array([tilt, tilt, yaw])
    q0 = quat_mul(quat_from_rotvec(-dtheta0), np.broadcast_to(traj.q_nb[0], (runs, 4)))

    # Aiding sensors (step 5) draw from their own generators, spawned from the
    # seed without advancing the main one. The IMU and GNSS errors are then
    # exactly the same with or without aiding (common random numbers), so two
    # configurations differ only by what the filter does, and steps 3 and 4
    # keep their random sequence.
    ad = aiding
    rng_baro, rng_mag, rng_pitot = (np.random.default_rng(s) for s in rng.bit_generator.seed_seq.spawn(3))
    baro_s = aid.BaroStream(ad.baro, rng_baro, runs) if ad is not None and ad.baro is not None else None
    mag_s = aid.MagStream(ad.mag, rng_mag, runs) if ad is not None and ad.mag is not None else None
    pitot_s = aid.PitotStream(ad.pitot, rng_pitot, runs) if ad is not None and ad.pitot is not None else None

    if init.yaw_source != "perturbed":
        roll, pitch, _ = euler_from_quat(quat_mul(quat_from_rotvec(-dtheta0 * [1, 1, 0]),
                                                  np.broadcast_to(traj.q_nb[0], (runs, 4))))
        if init.yaw_source == "gnss_course":
            heading = np.arctan2(z0[:, 4], z0[:, 3])
        elif init.yaw_source == "magnetometer":
            if mag_s is None:
                raise ValueError("yaw_source='magnetometer' needs a magnetometer")
            m_n = np.asarray(ad.mag.field_ned)
            heading = aid.heading_from_magnetometer(mag_s.measure(traj.q_nb[0]), roll, pitch,
                                                    np.arctan2(m_n[1], m_n[0]))
        else:
            raise ValueError(f"unknown yaw_source {init.yaw_source!r}")
        q0 = quat_from_euler(roll, pitch, heading)
    P0 = initial_covariance(imu_cfg, gnss_cfg, init, model_gnss_bias, aiding)
    extra0 = None
    if pitot_s is not None:
        tas0 = pitot_s.measure(traj.airspeed[0])  # drawn in every Pitot configuration: same draws
        if ad.wind.init_from_airspeed:
            P0, extra0 = wind_from_first_airspeed(ekf.blocks, n, z0[:, 3:], q0, tas0, P0, ad.pitot.noise,
                                                  np.radians(ad.wind.init_sideslip_sigma_deg))
    ekf.initialise(z0[:, :3], z0[:, 3:], q0, P0, extra=extra0)

    blocks = ekf.blocks
    log = FusionLog(
        t=np.empty(E_max), t_out=np.empty(E_max),
        err=np.empty((runs, E_max, n)), sigma=np.empty((runs, E_max, n)),
        nees={name: np.empty((runs, E_max)) for name in list(blocks) + ["total"]},
        nis=np.full((runs, E_max), np.nan),
        out_err=np.empty((runs, E_max, 9)), blocks=dict(blocks), aux_nis={}, aux_rejected={},
        gnss_status=np.zeros((runs, E_max), dtype=np.int8), wind_resets=[], gnss_resets=[],
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
        if baro_s is not None:
            parts.append(baro_s.drift[:, None] - ekf.nominal("baro_bias"))
        if mag_s is not None:
            parts.append(mag_s.bias - ekf.nominal("mag_bias"))
        if "wind" in blocks:
            parts.append(traj.wind_n[kf, :2] - ekf.nominal("wind"))
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

    aux_log = {}

    def learn_mag_bias():
        thr = ad.mag.learn_bias_min_rate_deg_s
        return None if thr is None else np.linalg.norm(ekf.omega, axis=1) > np.radians(thr)

    pitot_rejected_since = np.full(runs, np.nan)

    def pitot_update(kf):
        """Airspeed update, with the lock-out protection of the wind states."""
        tas = pitot_s.measure(traj.airspeed[kf])
        out = ekf.update_airspeed(tas)
        rejected = ekf.rejected.copy()
        t = traj.t[kf]
        start = rejected & np.isnan(pitot_rejected_since)
        pitot_rejected_since[start] = t
        pitot_rejected_since[~rejected] = np.nan
        timeout = ad.wind.reset_after_rejected_s
        if timeout is not None:
            reset = rejected & (t - pitot_rejected_since >= timeout)
            if reset.any():
                P, extra = wind_from_first_airspeed(ekf.blocks, n, ekf.v[reset], ekf.q[reset], tas[reset],
                                                    ekf.P[reset], ad.pitot.noise,
                                                    np.radians(ad.wind.init_sideslip_sigma_deg))
                sl = ekf.blocks["wind"]
                ekf.P[reset] = P
                ekf.extra[reset, sl.start - 15:sl.stop - 15] = extra[:, sl.start - 15:sl.stop - 15]
                pitot_rejected_since[reset] = np.nan
                log.wind_resets.append((t, int(reset.sum())))
        ekf.rejected = rejected
        return out

    def fuse_aiding(kf):
        """Fuse the aiding sensors whose sample falls at filter time index kf."""
        updates = (
            ("baro", baro_s, lambda: ekf.update_baro(baro_s.measure(-traj.p_n[kf, 2]))),
            ("mag", mag_s, lambda: ekf.update_mag(mag_s.measure(traj.q_nb[kf], faults.mag_offset(traj.t[kf])),
                                                  learn_mag_bias())),
            ("pitot", pitot_s, lambda: pitot_update(kf)),
        )
        if pitot_s is not None and ad.sideslip_sigma_deg is not None:
            updates = (*updates, ("sideslip", pitot_s,
                                  lambda: ekf.update_sideslip(np.radians(ad.sideslip_sigma_deg))))
        for name, s, do in updates:
            if s is None or kf % int(round(1.0 / (s.cfg.rate_hz * dt))) != 0:
                continue
            _, _, nis = do()
            aux_log.setdefault(name, ([], [], []))
            aux_log[name][0].append(traj.t[kf])
            aux_log[name][1].append(nis)
            aux_log[name][2].append(ekf.rejected)

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
            if ad is not None:
                fuse_aiding(f)
            if f % decim == 0:
                fuse_gnss(j, f, fix(f))
                record(j, f, k + 1, bg_t, ba_t, output_predictor(fifo))
                j += 1
        else:
            ekf.predict(dth, dv)
            if ad is not None:
                fuse_aiding(k + 1)
            m = k + 1 - D
            if m > 0 and m % decim == 0:
                z = fix(m)
                if latency_mode == "compensate":
                    z[:, 0:3] += ekf.v * gnss_cfg.latency_s
                fuse_gnss(j, m, z)
                record(j, k + 1, k + 1, stream.gyro_bias, stream.accel_bias)
                j += 1

    for name in ("t", "t_out"):
        setattr(log, name, getattr(log, name)[:j])
    for name in ("err", "sigma", "nis", "out_err", "gyro_bias", "accel_bias",
                 "gyro_bias_est", "accel_bias_est", "gnss_status"):
        setattr(log, name, getattr(log, name)[:, :j])
    log.nees = {k_: v[:, :j] for k_, v in log.nees.items()}
    log.aux_nis = {name: (np.array(t), np.array(v).T) for name, (t, v, _) in aux_log.items()}
    log.aux_rejected = {name: np.array(r).T for name, (_, _, r) in aux_log.items()}
    return log
