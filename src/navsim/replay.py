"""Step 7: run the error-state EKF on recorded sensor data instead of the
simulator, one flight at a time, with the timestamps of the log.

The input is a LogData: sensor samples with their own times, in the local
NED frame of the flight. It comes from a PX4 log (navsim.ulog_reader) or
from the simulator (from_simulation, used by the tests: there the truth is
known). The output is a ReplayLog sampled at every GNSS fix.

Differences with fusion.run: one flight (runs=1), uneven IMU intervals,
GNSS covariance taken from the accuracy the receiver reports, each
measurement fused at its own time of validity (its timestamp minus the delay
of the sensor), airspeed and sideslip fused only above a minimum airspeed,
and the wind started on the first such sample.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from . import aiding as aid
from .eskf import ErrorStateEKF
from .fusion import wind_from_first_airspeed
from .gnss import GNSSConfig
from .imu import IMUConfig
from .rotations import euler_from_quat, quat_conj, quat_from_euler, quat_mul, quat_rotate, rotvec_from_quat
from .sensors import BaroConfig

G = 9.80665


@dataclass
class LogData:
    name: str
    t_imu: np.ndarray            # (K,) [s] end of each IMU interval
    dt_imu: np.ndarray           # (K,) [s]
    dtheta: np.ndarray           # (K, 3) [rad] body frame (FRD)
    dvel: np.ndarray             # (K, 3) [m/s]
    t_gnss: np.ndarray           # (G,) [s] time of validity
    gnss_p: np.ndarray           # (G, 3) [m] local NED
    gnss_v: np.ndarray           # (G, 3) [m/s]
    gnss_sigma: np.ndarray       # (G, 3) [m, m, m/s] horizontal, vertical, speed accuracy (1 sigma)
    t_baro: np.ndarray = field(default_factory=lambda: np.zeros(0))
    baro_alt: np.ndarray = field(default_factory=lambda: np.zeros(0))   # [m] above the NED origin, up to a bias
    t_mag: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mag: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # [gauss] body frame
    t_tas: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tas: np.ndarray = field(default_factory=lambda: np.zeros(0))        # [m/s] true airspeed
    field_ned: np.ndarray = field(default_factory=lambda: np.array(aid.earth_field()))  # [gauss]
    # Reference to compare with: the onboard EKF2 for a real log, the truth
    # for a simulated one. Times (R,), position/velocity NED (R, 3),
    # quaternion (R, 4), horizontal wind (R, 2) or None.
    ref_name: str = ""
    t_ref: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ref_p: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    ref_v: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    ref_q: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    ref_wind: np.ndarray | None = None
    info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayConfig:
    imu: IMUConfig = IMUConfig()
    gnss_floor: tuple = (0.5, 1.0, 0.3)       # minimum (horizontal, vertical, speed) sigma [m, m, m/s]
    gnss_delay: float = 0.0                    # already applied in LogData when 0
    baro: BaroConfig | None = None
    mag: aid.MagConfig | None = None
    pitot: aid.PitotConfig | None = None
    wind: aid.WindModel | None = None
    sideslip_sigma_deg: float | None = None
    sideslip_offset_sigma0_deg: float | None = None
    min_airspeed: float = 8.0                  # [m/s] below, no airspeed or sideslip fusion
    tilt_sigma_deg: float = 2.0
    yaw_sigma_deg: float = 15.0
    gate_prob: float | None = 0.999
    gnss_reset_after_s: float | None = 45.0
    # GNSS outages to simulate on the recorded data: ((t0, t1), ...)
    gnss_outage: tuple = ()
    init_window_s: float = 1.0                 # accelerometer averaging for the initial roll and pitch

    @property
    def aiding(self) -> aid.AidingConfig | None:
        if self.baro is None and self.mag is None and self.pitot is None:
            return None
        return aid.AidingConfig(baro=self.baro, mag=self.mag, pitot=self.pitot, wind=self.wind,
                                sideslip_sigma_deg=self.sideslip_sigma_deg,
                                sideslip_offset_sigma0_deg=self.sideslip_offset_sigma0_deg)


@dataclass
class ReplayLog:
    t: np.ndarray                 # (E,) GNSS epochs
    p: np.ndarray                 # (E, 3) estimate
    v: np.ndarray
    q: np.ndarray                 # (E, 4)
    sigma: np.ndarray             # (E, n)
    blocks: dict
    extra: np.ndarray             # (E, n - 15) nominal extra states (biases, wind)
    bg: np.ndarray
    ba: np.ndarray
    gnss: np.ndarray              # (E, 6) the fix itself (position, velocity)
    gnss_status: np.ndarray       # (E,) 0 fused, 1 rejected, 2 outage
    nis: dict                     # sensor -> (times, NIS, rejected)
    wind_started: float | None    # time the wind states started
    info: dict


def _quat_from_accel_mag(f_b, m_b, declination):
    """Initial attitude, aircraft at rest or in steady flight: roll and pitch
    from the specific force (it points up in the body frame), heading from the
    tilt-compensated magnetometer."""
    fx, fy, fz = f_b
    roll = np.arctan2(-fy, -fz)
    pitch = np.arctan2(fx, np.hypot(fy, fz))
    if m_b is None:
        return quat_from_euler(roll, pitch, 0.0), roll, pitch, None
    yaw = aid.heading_from_magnetometer(np.asarray(m_b), np.array(roll), np.array(pitch), declination)
    return quat_from_euler(roll, pitch, float(yaw)), roll, pitch, float(yaw)


def run(data: LogData, cfg: ReplayConfig, t_start: float | None = None, t_end: float | None = None) -> ReplayLog:
    """Replay one flight. Starts at the first GNSS fix after t_start."""
    ad = cfg.aiding
    dt0 = float(np.median(data.dt_imu))
    gnss_cfg = GNSSConfig(pos_sigma_h=cfg.gnss_floor[0], pos_sigma_v=cfg.gnss_floor[1],
                          vel_sigma_h=cfg.gnss_floor[2], vel_sigma_v=cfg.gnss_floor[2])
    ekf = ErrorStateEKF(cfg.imu, gnss_cfg, dt0, 1, aiding=ad)
    n = ekf.n
    if cfg.gate_prob is not None:
        dof = {"gnss": 6, "baro": 1, "mag": 3, "mag_heading": 1, "pitot": 1, "sideslip": 1}
        ekf.gates = {k: chi2.ppf(cfg.gate_prob, d) for k, d in dof.items()}

    t_gnss = data.t_gnss - cfg.gnss_delay
    # the first fix with at least init_window_s of IMU data before it
    t_min = data.t_imu[0] + cfg.init_window_s
    t0 = t_gnss[np.searchsorted(t_gnss, t_min if t_start is None else max(t_start, t_min))]
    t1 = data.t_imu[-1] if t_end is None else t_end
    g0 = int(np.searchsorted(t_gnss, t0))

    # Initial attitude from the accelerometer (averaged) and the magnetometer.
    w = (data.t_imu > t0 - cfg.init_window_s) & (data.t_imu <= t0)
    f_b = data.dvel[w].sum(axis=0) / data.dt_imu[w].sum()
    m_b = None
    if cfg.mag is not None and len(data.t_mag):
        km = int(np.clip(np.searchsorted(data.t_mag, t0), 0, len(data.t_mag) - 1))
        m_b = data.mag[km]
    m_n = np.asarray(data.field_ned if cfg.mag is None else cfg.mag.field_ned)
    q0, _, _, _ = _quat_from_accel_mag(f_b, m_b, np.arctan2(m_n[1], m_n[0]))

    tilt, yaw = np.radians(cfg.tilt_sigma_deg), np.radians(cfg.yaw_sigma_deg)
    s = _sigma(data.gnss_sigma[g0], cfg.gnss_floor)
    d = np.concatenate([s[:3] ** 2, s[3:] ** 2, [tilt**2, tilt**2, yaw**2],
                        [cfg.imu.gyro_bias0**2] * 3, [cfg.imu.accel_bias0**2] * 3])
    extra_var = []
    if "baro_bias" in ekf.blocks:
        extra_var.append([cfg.baro.drift_sigma**2])
    if "mag_bias" in ekf.blocks:
        extra_var.append([cfg.mag.bias0**2] * 3)
    if "wind" in ekf.blocks:
        extra_var.append([cfg.wind.sigma0**2] * 2)
    if "tas_scale" in ekf.blocks:
        extra_var.append([cfg.pitot.scale_sigma0**2])
    if "beta_offset" in ekf.blocks:
        extra_var.append([np.radians(cfg.sideslip_offset_sigma0_deg) ** 2])
    P0 = np.diag(np.concatenate([d, *extra_var])) if extra_var else np.diag(d)
    extra0 = np.zeros((1, n - 15))
    if "baro_bias" in ekf.blocks and len(data.t_baro):
        kb = int(np.clip(np.searchsorted(data.t_baro, t0), 0, len(data.t_baro) - 1))
        sl = ekf.blocks["baro_bias"]
        extra0[0, sl.start - 15] = data.baro_alt[kb] - (-data.gnss_p[g0, 2])
    ekf.initialise(data.gnss_p[g0][None], data.gnss_v[g0][None], q0[None], P0, extra=extra0)

    # Measurement queues (time of validity, kind, index)
    events = []
    for name, ts in (("baro", data.t_baro if cfg.baro else ()), ("mag", data.t_mag if cfg.mag else ()),
                     ("tas", data.t_tas if cfg.pitot else ())):
        ts = np.asarray(ts)
        idx = np.flatnonzero((ts > t0) & (ts <= t1))
        events += [(ts[i], name, i) for i in idx]
    gi = np.flatnonzero((t_gnss > t0) & (t_gnss <= t1))
    events += [(t_gnss[i], "gnss", i) for i in gi]
    events.sort(key=lambda e: e[0])

    E = len(gi) + 1
    log = ReplayLog(t=np.empty(E), p=np.empty((E, 3)), v=np.empty((E, 3)), q=np.empty((E, 4)),
                    sigma=np.empty((E, n)), blocks=dict(ekf.blocks), extra=np.empty((E, n - 15)),
                    bg=np.empty((E, 3)), ba=np.empty((E, 3)), gnss=np.empty((E, 6)),
                    gnss_status=np.zeros(E, dtype=np.int8), nis={}, wind_started=None,
                    info={"t0": float(t0), "n_states": n})

    def record(j, t, z):
        log.t[j] = t
        log.p[j], log.v[j], log.q[j] = ekf.p[0], ekf.v[0], ekf.q[0]
        log.sigma[j] = np.sqrt(np.diag(ekf.P[0]))
        log.extra[j] = ekf.extra[0]
        log.bg[j], log.ba[j] = ekf.bg[0], ekf.ba[0]
        log.gnss[j] = z

    def note(name, t, nis):
        log.nis.setdefault(name, ([], [], []))
        log.nis[name][0].append(t)
        log.nis[name][1].append(float(nis[0]))
        log.nis[name][2].append(bool(ekf.rejected[0]))

    record(0, t0, np.concatenate([data.gnss_p[g0], data.gnss_v[g0]]))
    j = 1
    wind_on = False
    last_fused = t0
    pitot_rejected_since = None
    k = int(np.searchsorted(data.t_imu, t0, side="right"))
    ev = 0
    while k < len(data.t_imu) and data.t_imu[k] <= t1:
        ekf.predict(data.dtheta[k][None], data.dvel[k][None], data.dt_imu[k])
        tk = data.t_imu[k]
        while ev < len(events) and events[ev][0] <= tk:
            t_e, kind, i = events[ev]
            ev += 1
            if kind == "gnss":
                z = np.concatenate([data.gnss_p[i], data.gnss_v[i]])
                if any(a <= t_e < b for a, b in cfg.gnss_outage):
                    log.gnss_status[j] = 2
                else:
                    s = _sigma(data.gnss_sigma[i], cfg.gnss_floor)
                    _, _, nis = ekf.update_gnss(z[None], np.diag(s**2))
                    note("gnss", t_e, nis)
                    rejected = bool(ekf.rejected[0])
                    log.gnss_status[j] = int(rejected)
                    if not rejected:
                        last_fused = t_e
                    elif cfg.gnss_reset_after_s is not None and t_e - last_fused >= cfg.gnss_reset_after_s:
                        ekf.R_gnss = np.diag(s**2)
                        ekf.reset_to_gnss(np.array([True]), z[None])
                        last_fused = t_e
                record(j, t_e, z)
                j += 1
            elif kind == "baro":
                _, _, nis = ekf.update_baro(np.array([data.baro_alt[i]]))
                note("baro", t_e, nis)
            elif kind == "mag" and cfg.mag.fusion == "heading":
                _, _, nis = ekf.update_mag_heading(data.mag[i][None], np.radians(cfg.mag.heading_sigma_deg),
                                                   np.arctan2(m_n[1], m_n[0]))
                note("mag_heading", t_e, nis)
            elif kind == "mag":
                thr = cfg.mag.learn_bias_min_rate_deg_s
                learn = None if thr is None else np.linalg.norm(ekf.omega, axis=1) > np.radians(thr)
                _, _, nis = ekf.update_mag(data.mag[i][None], learn)
                note("mag", t_e, nis)
            elif kind == "tas":
                tas = np.array([data.tas[i]])
                if tas[0] < cfg.min_airspeed:
                    continue
                if not wind_on:
                    P, extra = wind_from_first_airspeed(ekf.blocks, n, ekf.v, ekf.q, tas / (1.0 + ekf.tas_scale()), ekf.P,
                                                        cfg.pitot.noise, np.radians(cfg.wind.init_sideslip_sigma_deg))
                    sl = ekf.blocks["wind"]
                    ekf.P = P
                    ekf.extra[:, sl.start - 15:sl.stop - 15] = extra[:, sl.start - 15:sl.stop - 15]
                    wind_on = True
                    log.wind_started = float(t_e)
                    continue
                _, _, nis = ekf.update_airspeed(tas)
                note("pitot", t_e, nis)
                if ekf.rejected[0]:
                    pitot_rejected_since = t_e if pitot_rejected_since is None else pitot_rejected_since
                    timeout = cfg.wind.reset_after_rejected_s
                    if timeout is not None and t_e - pitot_rejected_since >= timeout:
                        P, extra = wind_from_first_airspeed(ekf.blocks, n, ekf.v, ekf.q, tas / (1.0 + ekf.tas_scale()), ekf.P,
                                                            cfg.pitot.noise,
                                                            np.radians(cfg.wind.init_sideslip_sigma_deg))
                        sl = ekf.blocks["wind"]
                        ekf.P = P
                        ekf.extra[:, sl.start - 15:sl.stop - 15] = extra[:, sl.start - 15:sl.stop - 15]
                        pitot_rejected_since = None
                else:
                    pitot_rejected_since = None
                if cfg.sideslip_sigma_deg is not None:
                    _, _, nis = ekf.update_sideslip(np.radians(cfg.sideslip_sigma_deg))
                    note("sideslip", t_e, nis)
        k += 1

    for name in ("t", "p", "v", "q", "sigma", "extra", "bg", "ba", "gnss", "gnss_status"):
        setattr(log, name, getattr(log, name)[:j])
    log.nis = {k_: (np.array(a), np.array(b), np.array(c)) for k_, (a, b, c) in log.nis.items()}
    return log


def _sigma(acc, floor):
    """(eph, epv, speed accuracy) -> sigmas of (pN, pE, pD, vN, vE, vD), with floors.
    eph is a horizontal accuracy: split equally on north and east."""
    h = max(acc[0] / np.sqrt(2.0), floor[0])
    v = max(acc[1], floor[1])
    s = max(acc[2] / np.sqrt(3.0), floor[2])
    return np.array([h, h, v, s, s, s])


# --- comparison with the reference -----------------------------------------------

def compare(log: ReplayLog, data: LogData) -> dict:
    """Differences between the replay and the reference (EKF2 or truth),
    interpolated at the replay epochs."""
    t = log.t
    ok = (t >= data.t_ref[0]) & (t <= data.t_ref[-1])
    t = t[ok]
    ref_p = np.column_stack([np.interp(t, data.t_ref, data.ref_p[:, i]) for i in range(3)])
    ref_v = np.column_stack([np.interp(t, data.t_ref, data.ref_v[:, i]) for i in range(3)])
    k = np.clip(np.searchsorted(data.t_ref, t), 0, len(data.t_ref) - 1)
    ref_q = data.ref_q[k]
    datt = rotvec_from_quat(quat_mul(ref_q, quat_conj(log.q[ok])))
    e_ref = np.array(euler_from_quat(ref_q)).T
    e_est = np.array(euler_from_quat(log.q[ok])).T
    dyaw = np.angle(np.exp(1j * (e_est[:, 2] - e_ref[:, 2])))
    out = {
        "t": t,
        "dp": log.p[ok] - ref_p,
        "dv": log.v[ok] - ref_v,
        "datt": datt,
        "droll": e_est[:, 0] - e_ref[:, 0],
        "dpitch": e_est[:, 1] - e_ref[:, 1],
        "dyaw": dyaw,
    }
    if data.ref_wind is not None and "wind" in log.blocks:
        sl = log.blocks["wind"]
        ref_w = np.column_stack([np.interp(t, data.t_ref, data.ref_wind[:, i]) for i in range(2)])
        out["dwind"] = log.extra[ok][:, sl.start - 15:sl.stop - 15] - ref_w
    return out


# --- simulated logs, for the tests -------------------------------------------------

def from_simulation(traj, imu_cfg: IMUConfig, rng, gnss_sigma=(1.0, 2.0, 0.15), mag_cfg=None, baro_cfg=None,
                    pitot_cfg=None, field_ned=None, mount_yaw_deg: float = 0.0) -> LogData:
    """A LogData built from a simulated flight, with the truth as reference.
    The IMU intervals are uneven (5 or 10 ms at random), by merging
    consecutive increments. GNSS at 5 Hz with white errors of the given
    sigmas (reported as the receiver accuracy).

    mount_yaw_deg: the autopilot is mounted turned by this angle about the
    vertical axis of the airframe. The IMU and the magnetometer measure in
    its frame, and the reference attitude is that of the autopilot, as in a
    real log. The air velocity, along the airframe, then shows a sideslip of
    -mount_yaw_deg in the autopilot frame."""
    from .imu import IMUStream

    K = len(traj.dtheta)
    stream = IMUStream(imu_cfg, rng, 1, traj.dt)
    q_m = quat_from_euler(0.0, 0.0, np.radians(mount_yaw_deg))
    to_imu = lambda v: quat_rotate(quat_conj(q_m), v)
    q_imu = quat_mul(traj.q_nb, np.broadcast_to(q_m, traj.q_nb.shape))
    dth = np.empty((K, 3))
    dv = np.empty((K, 3))
    for k in range(K):
        a, b = stream.sample(to_imu(traj.dtheta[k]), to_imu(traj.dvel[k]))
        dth[k], dv[k] = a[0], b[0]
    # merge consecutive samples by groups of 1 or 2 at random: uneven intervals
    groups = rng.integers(1, 3, K)
    ends = np.cumsum(groups)
    ends = ends[ends <= K]
    starts = np.concatenate([[0], ends[:-1]])
    cdth, cdv = np.cumsum(dth, axis=0), np.cumsum(dv, axis=0)
    pad = lambda c, i: np.where((i > 0)[:, None], c[np.maximum(i - 1, 0)], 0.0)
    dtheta = cdth[ends - 1] - pad(cdth, starts)
    dvel = cdv[ends - 1] - pad(cdv, starts)
    t_imu = traj.t[ends]
    dt_imu = (ends - starts) * traj.dt

    dec = int(round(0.2 / traj.dt))
    kg = np.arange(0, K + 1, dec)
    sh, sv, ss = gnss_sigma
    gp = traj.p_n[kg] + rng.normal(0.0, 1.0, (len(kg), 3)) * [sh / np.sqrt(2), sh / np.sqrt(2), sv]
    gv = traj.v_n[kg] + rng.normal(0.0, ss / np.sqrt(3), (len(kg), 3))
    data = LogData(name="simulation", t_imu=t_imu, dt_imu=dt_imu, dtheta=dtheta, dvel=dvel,
                   t_gnss=traj.t[kg], gnss_p=gp, gnss_v=gv, gnss_sigma=np.tile([sh, sv, ss], (len(kg), 1)),
                   ref_name="vérité", t_ref=traj.t, ref_p=traj.p_n, ref_v=traj.v_n, ref_q=q_imu,
                   ref_wind=traj.wind_n[:, :2] if hasattr(traj, "wind_n") else None)
    if field_ned is not None:
        data.field_ned = np.asarray(field_ned)
    if mag_cfg is not None:
        ms = aid.MagStream(dataclasses.replace(mag_cfg, field_ned=tuple(data.field_ned)), rng, 1)
        km = np.arange(0, K + 1, int(round(1.0 / (mag_cfg.rate_hz * traj.dt))))
        data.t_mag, data.mag = traj.t[km], np.array([ms.measure(q_imu[i])[0] for i in km])
        data.info["mag_bias"] = ms.bias[0]
    if baro_cfg is not None:
        bs = aid.BaroStream(baro_cfg, rng, 1)
        kb = np.arange(0, K + 1, int(round(1.0 / (baro_cfg.rate_hz * traj.dt))))
        data.t_baro, data.baro_alt = traj.t[kb], np.array([bs.measure(-traj.p_n[i, 2])[0] for i in kb])
    if pitot_cfg is not None:
        ps = aid.PitotStream(pitot_cfg, rng, 1)
        kp = np.arange(0, K + 1, int(round(1.0 / (pitot_cfg.rate_hz * traj.dt))))
        data.t_tas, data.tas = traj.t[kp], np.array([ps.measure(traj.airspeed[i])[0] for i in kp])
    return data


# --- tuning ------------------------------------------------------------------------

def config_like_ekf2(data: LogData, air_data: bool = True, mag: str | None = "heading",
                     sideslip_sigma_deg: float | None = 6.0, sideslip_offset: bool = True,
                     tas_scale: bool = True, wind_rw: float = 0.1, **overrides) -> ReplayConfig:
    """A ReplayConfig with the noise levels EKF2 used in that flight.

    EKF2 states its IMU noises per prediction step (EKF2_PREDICT_US, 10 ms):
    a gyro noise of 0.015 rad/s means an angle increment of standard
    deviation 0.015 x 0.01 rad per step, i.e. a density of
    0.015 x sqrt(0.01) = 0.0015 rad/s/sqrt(Hz). Same for the accelerometer
    and the bias random walks. Measurement noises (GNSS floors, baro,
    airspeed, magnetometer heading) are standard deviations and are used as
    they are. The wind random walk is that of steps 5 and 6 (0.1 m/s/sqrt(s)),
    not EKF2_WIND_NSD, and the baro drift is ours (5 m over 10 min)."""
    p = data.info.get("params", {})
    get = lambda k, d: float(p.get(k, d))
    sq = np.sqrt(get("EKF2_PREDICT_US", 10000.0) * 1e-6)
    imu_cfg = IMUConfig(gyro_noise=get("EKF2_GYR_NOISE", 0.015) * sq, accel_noise=get("EKF2_ACC_NOISE", 0.35) * sq,
                        gyro_bias_rw=get("EKF2_GYR_B_NOISE", 0.001) * sq,
                        accel_bias_rw=get("EKF2_ACC_B_NOISE", 0.003) * sq,
                        gyro_bias0=get("EKF2_GBIAS_INIT", 0.1), accel_bias0=get("EKF2_ABIAS_INIT", 0.2))
    gp, gv = get("EKF2_GPS_P_NOISE", 0.5), get("EKF2_GPS_V_NOISE", 0.3)
    kw = dict(imu=imu_cfg, gnss_floor=(gp, 1.5 * gp, gv))
    rate = lambda t: float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 2 else 1.0
    if air_data or mag is not None:
        kw["baro"] = BaroConfig(rate_hz=rate(data.t_baro), noise_sigma=get("EKF2_BARO_NOISE", 3.5),
                                drift_sigma=5.0, drift_tau=600.0)
    if mag is not None:
        kw["mag"] = aid.MagConfig(rate_hz=rate(data.t_mag), field_ned=tuple(data.field_ned),
                                  noise=get("EKF2_MAG_NOISE", 0.05), bias0=0.05, bias_rw=1e-5, fusion=mag,
                                  heading_sigma_deg=np.degrees(get("EKF2_HEAD_NOISE", 0.3)))
    if air_data:
        kw["pitot"] = aid.PitotConfig(rate_hz=rate(data.t_tas), noise=get("EKF2_EAS_NOISE", 1.4),
                                      scale_sigma0=0.3 if tas_scale else None)
        kw["wind"] = aid.WindModel(rw=wind_rw)
        kw["sideslip_sigma_deg"] = sideslip_sigma_deg
        kw["sideslip_offset_sigma0_deg"] = 15.0 if (sideslip_offset and sideslip_sigma_deg is not None) else None
    kw.update(overrides)
    return ReplayConfig(**kw)


def outage_windows(t_air: tuple, duration: float = 30.0, spacing: float = 90.0, margin: float = 30.0) -> tuple:
    """GNSS outages to simulate in a recorded flight: every `spacing` s while
    airborne, leaving `margin` s after take-off and before landing."""
    a, b = t_air
    starts = np.arange(a + margin, b - margin - duration, spacing)
    return tuple((float(s), float(s + duration)) for s in starts)


def outage_errors(log: ReplayLog, windows: tuple) -> np.ndarray:
    """For each outage: (horizontal error, announced 1 sigma) at the end of
    the outage, against the first GNSS fix after it (the fix is accurate to
    a few tens of cm, the dead reckoning error is metres)."""
    out = []
    for _a, b in windows:
        k = int(np.searchsorted(log.t, b))           # first epoch after the outage
        if k >= len(log.t) or k < 1:
            continue
        j = k - 1                                    # last epoch of dead reckoning
        err = np.linalg.norm(log.p[j, :2] - log.gnss[k, :2] + log.v[j, :2] * (log.t[k] - log.t[j]))
        sig = np.hypot(log.sigma[j, 0], log.sigma[j, 1])
        out.append((err, sig))
    return np.array(out)
