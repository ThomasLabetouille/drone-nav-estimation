"""Step 8: run the C++ port (cpp/) on the same data as the Python replay.

The Python side decides everything that happens before the first
prediction (replay.setup: start time, initial data, order of the
measurements) and writes it to files: a config, and the exact stream of
events the filter must process. The C++ program runs the filter loop and
writes its states at every GNSS epoch, in the layout of a ReplayLog.

Needs CMake, a C++17 compiler and Eigen 3.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .replay import LogData, ReplayConfig, setup

CPP_DIR = Path(__file__).resolve().parents[2] / "cpp"
KINDS = {1: "gnss", 2: "baro", 3: "mag_heading", 4: "pitot", 5: "sideslip"}


def available() -> bool:
    return shutil.which("cmake") is not None and (shutil.which("c++") or shutil.which("g++")) is not None


def build(build_dir: Path | None = None) -> Path:
    """Configure and build cpp/ (Release), return the build directory."""
    build_dir = Path(build_dir or CPP_DIR / "build")
    subprocess.run(["cmake", "-S", str(CPP_DIR), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
                   check=True, capture_output=True, text=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--parallel"], check=True, capture_output=True, text=True)
    return build_dir


def check_config(cfg: ReplayConfig):
    """The C++ port covers the configuration of the real logs (R3)."""
    ok = (cfg.baro is not None and cfg.mag is not None and cfg.mag.fusion == "heading"
          and cfg.pitot is not None and cfg.pitot.scale_sigma0 is not None and cfg.wind is not None
          and cfg.sideslip_sigma_deg is not None and cfg.sideslip_offset_sigma0_deg is not None
          and cfg.gnss_delay == 0.0)
    if not ok:
        raise ValueError("the C++ port needs baro, magnetometer heading, Pitot with scale state, wind, "
                         "and sideslip with offset state")


def _fmt(x) -> str:
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def config_text(data: LogData, cfg: ReplayConfig, st: dict) -> str:
    check_config(cfg)
    gate = (lambda d: chi2.ppf(cfg.gate_prob, d)) if cfg.gate_prob is not None else (lambda d: 0.0)
    g0 = st["g0"]
    m_n = st["m_n"]
    vals = {
        "gyro_noise": cfg.imu.gyro_noise, "accel_noise": cfg.imu.accel_noise,
        "gyro_bias_rw": cfg.imu.gyro_bias_rw, "accel_bias_rw": cfg.imu.accel_bias_rw,
        "gyro_bias0": cfg.imu.gyro_bias0, "accel_bias0": cfg.imu.accel_bias0,
        "gnss_floor_h": cfg.gnss_floor[0], "gnss_floor_v": cfg.gnss_floor[1], "gnss_floor_s": cfg.gnss_floor[2],
        "baro_noise": cfg.baro.noise_sigma, "baro_drift_sigma": cfg.baro.drift_sigma,
        "baro_drift_tau": cfg.baro.drift_tau,
        "heading_sigma": np.radians(cfg.mag.heading_sigma_deg), "declination": np.arctan2(m_n[1], m_n[0]),
        "pitot_noise": cfg.pitot.noise, "scale_sigma0": cfg.pitot.scale_sigma0,
        "sideslip_sigma": np.radians(cfg.sideslip_sigma_deg),
        "sideslip_offset_sigma0": np.radians(cfg.sideslip_offset_sigma0_deg),
        "wind_rw": cfg.wind.rw, "wind_sigma0": cfg.wind.sigma0,
        "init_sideslip_sigma": np.radians(cfg.wind.init_sideslip_sigma_deg),
        "calib_rw": 1e-4, "min_airspeed": cfg.min_airspeed,
        "tilt_sigma": np.radians(cfg.tilt_sigma_deg), "yaw_sigma": np.radians(cfg.yaw_sigma_deg),
        "gate_gnss": gate(6), "gate_baro": gate(1), "gate_heading": gate(1), "gate_pitot": gate(1),
        "gate_sideslip": gate(1),
        "gnss_reset_after": -1.0 if cfg.gnss_reset_after_s is None else cfg.gnss_reset_after_s,
        "pitot_reset_after": -1.0 if cfg.wind.reset_after_rejected_s is None else cfg.wind.reset_after_rejected_s,
        "t0": st["t0"], "init_baro": st["baro0"],
    }
    lines = [f"{k} {_fmt(v)}" for k, v in vals.items()]
    vec = {"init_p": data.gnss_p[g0], "init_v": data.gnss_v[g0], "init_acc": data.gnss_sigma[g0],
           "init_fb": st["f_b"], "init_mag": st["m_b"] if st["m_b"] is not None else [np.nan] * 3}
    lines += [f"{k} " + " ".join(_fmt(x) for x in v) for k, v in vec.items()]
    return "\n".join(lines) + "\n"


def events_array(data: LogData, cfg: ReplayConfig, st: dict) -> np.ndarray:
    """The records the C++ loop processes, in the order of replay.run: each
    IMU sample, then the measurements whose time of validity is not after it."""
    k0 = st["k0"]
    k_end = int(np.searchsorted(data.t_imu, st["t1"], side="right"))
    ks = np.arange(k0, k_end)
    imu = np.zeros((len(ks), 12))
    imu[:, 0] = 1
    imu[:, 1] = data.t_imu[ks]
    imu[:, 2] = data.dt_imu[ks]
    imu[:, 3:6] = data.dtheta[ks]
    imu[:, 6:9] = data.dvel[ks]
    recs, attach = [], []
    for t_e, kind, i in st["events"]:
        k = max(k0, int(np.searchsorted(data.t_imu, t_e, side="left")))
        if k >= k_end:
            continue
        r = np.zeros(12)
        r[1] = t_e
        if kind == "gnss":
            r[0] = 2
            r[2:5], r[5:8], r[8:11] = data.gnss_p[i], data.gnss_v[i], data.gnss_sigma[i]
            r[11] = float(any(a <= t_e < b for a, b in cfg.gnss_outage))
        elif kind == "baro":
            r[0], r[2] = 3, data.baro_alt[i]
        elif kind == "mag":
            r[0], r[2:5] = 4, data.mag[i]
        else:
            r[0], r[2] = 5, data.tas[i]
        recs.append(r)
        attach.append(k)
    # merge: IMU sample k first, then the events attached to it, in queue order
    order_imu = np.stack([ks, np.zeros(len(ks)), np.arange(len(ks))], axis=1)
    order_ev = np.stack([np.array(attach, dtype=float), np.ones(len(attach)), np.arange(len(attach))], axis=1)
    keys = np.concatenate([order_imu, order_ev])
    rows = np.concatenate([imu, np.array(recs).reshape(-1, 12)])
    idx = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    return rows[idx]


def run(data: LogData, cfg: ReplayConfig, build_dir: Path | None = None, t_start=None, t_end=None) -> dict:
    """Replay with the C++ filter. Returns the states at every GNSS epoch
    (same layout as ReplayLog), the NIS of every update, and the time spent
    in the filter."""
    build_dir = Path(build_dir or CPP_DIR / "build")
    exe = build_dir / "nav_replay"
    if not exe.exists():
        build(build_dir)
    st = setup(data, cfg, t_start, t_end)
    with tempfile.TemporaryDirectory(prefix="navcpp_") as tmp:
        tmp = Path(tmp)
        (tmp / "config.txt").write_text(config_text(data, cfg, st))
        events_array(data, cfg, st).astype("<f8").tofile(tmp / "events.bin")
        out = subprocess.run([str(exe), str(tmp / "config.txt"), str(tmp / "events.bin"), str(tmp / "states.bin"),
                              str(tmp / "nis.bin")], check=True, capture_output=True, text=True).stdout
        S = np.fromfile(tmp / "states.bin", dtype="<f8").reshape(-1, 43)
        Nn = np.fromfile(tmp / "nis.bin", dtype="<f8").reshape(-1, 4)
    stats = dict(zip(out.split()[::2], (float(x) for x in out.split()[1::2]), strict=True))
    nis = {}
    for code, name in KINDS.items():
        m = Nn[:, 0] == code
        if m.any():
            nis[name] = (Nn[m, 1], Nn[m, 2], Nn[m, 3].astype(bool))
    return {"t": S[:, 0], "gnss_status": S[:, 1].astype(np.int8), "p": S[:, 2:5], "v": S[:, 5:8], "q": S[:, 8:12],
            "bg": S[:, 12:15], "ba": S[:, 15:18], "extra": S[:, 18:23], "sigma": S[:, 23:43], "nis": nis,
            "stats": stats}
