"""Step 7: read a PX4 flight log (ULog) into a replay.LogData.

Needs pyulog (log parsing) and pygeomag (World Magnetic Model): the
optional dependencies `pip install -e ".[replay]"`.

What is taken from the log:
- sensor_combined: gyro and accelerometer, turned into increments over
  their own integration intervals;
- vehicle_gps_position (or sensor_gps): position, velocity, and the
  accuracies the receiver reports;
- vehicle_air_data, vehicle_magnetometer, airspeed: barometric altitude,
  magnetic field (already calibrated by PX4), true airspeed;
- vehicle_local_position, vehicle_attitude, wind: the onboard EKF2, kept
  as the reference to compare with. Its NED origin is used as ours.

Each sensor time is shifted by the delay EKF2 was configured with in that
flight (EKF2_GPS_DELAY, EKF2_BARO_DELAY...), so that every measurement is
fused at its time of validity, as EKF2 does.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from .replay import LogData

R_EARTH = 6371000.0   # [m] radius PX4 uses for its local projection


def _field(d, *names):
    for n in names:
        if n in d:
            return np.asarray(d[n], dtype=float)
    raise KeyError(f"none of {names} in topic")


def _vec(d, base, n=3):
    return np.column_stack([np.asarray(d[f"{base}[{i}]"], dtype=float) for i in range(n)])


def _time(d):
    """timestamp_sample when the driver fills it, timestamp otherwise [us]."""
    ts = d.get("timestamp_sample")
    if ts is not None and np.all(np.diff(np.asarray(ts, dtype=float)) > 0):
        return ts
    return d["timestamp"]


def _topic(ulog, *names, multi_id=0):
    for name in names:
        for d in ulog.data_list:
            if d.name == name and d.multi_id == multi_id:
                return d.data
    return None


def geodetic_to_ned(lat, lon, alt, lat0, lon0, alt0):
    """Local frame of PX4 around (lat0, lon0, alt0): azimuthal equidistant
    projection on a sphere of radius R_EARTH (PX4 map_projection), down =
    minus the altitude difference. Using EKF2's own projection keeps the two
    filters in the same frame; it differs from a WGS84 tangent plane by
    about 0.2 % in scale."""
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    la, lo = np.radians(np.asarray(lat, dtype=float)), np.radians(np.asarray(lon, dtype=float))
    cos_dlon = np.cos(lo - lo0)
    arg = np.clip(np.sin(la0) * np.sin(la) + np.cos(la0) * np.cos(la) * cos_dlon, -1.0, 1.0)
    c = np.arccos(arg)
    k = np.where(c > 1e-12, c / np.sin(np.where(c > 1e-12, c, 1.0)), 1.0)
    north = k * (np.cos(la0) * np.sin(la) - np.sin(la0) * np.cos(la) * cos_dlon) * R_EARTH
    east = k * np.cos(la) * np.sin(lo - lo0) * R_EARTH
    return np.column_stack([north, east, -(np.asarray(alt, dtype=float) - alt0)])


def wmm_field(lat, lon, alt_m, decimal_year):
    """Earth magnetic field in NED [gauss] from the World Magnetic Model."""
    from pygeomag import GeoMag

    cof = "WMM_2025.COF" if decimal_year >= 2025 else ("WMM_2020.COF" if decimal_year >= 2020 else
                                                          "WMM_2015v2.COF")
    r = GeoMag(coefficients_file=f"wmm/{cof}").calculate(glat=lat, glon=lon, alt=alt_m / 1000.0,
                                                          time=decimal_year, allow_date_outside_lifespan=True)
    return np.array([r.x, r.y, r.z]) * 1e-5


def load(path, name: str | None = None) -> LogData:
    from pyulog import ULog

    path = Path(path)
    ulog = ULog(str(path))
    params = ulog.initial_parameters
    t_start = ulog.start_timestamp

    def sec(us):
        return (np.asarray(us, dtype=float) - t_start) * 1e-6

    def delay(p):
        return float(params.get(p, 0.0)) * 1e-3

    # IMU: rates averaged over their integration interval -> increments
    sc = _topic(ulog, "sensor_combined")
    t_gyro = sec(sc["timestamp"])
    dt_g = np.asarray(sc["gyro_integral_dt"], dtype=float) * 1e-6
    dt_a = np.asarray(sc["accelerometer_integral_dt"], dtype=float) * 1e-6
    keep = (dt_g > 0) & (dt_a > 0) & (dt_g < 0.05)
    dtheta = _vec(sc, "gyro_rad")[keep] * dt_g[keep, None]
    dvel = _vec(sc, "accelerometer_m_s2")[keep] * dt_a[keep, None]
    t_imu, dt_imu = t_gyro[keep], dt_g[keep]

    # EKF2 reference and NED origin
    lp = _topic(ulog, "vehicle_local_position")
    t_lp = sec(lp["timestamp"])
    valid = np.asarray(lp["xy_global"], dtype=bool) & np.asarray(lp["z_global"], dtype=bool)
    g = _topic(ulog, "vehicle_gps_position", "sensor_gps")
    if "latitude_deg" in g:
        lat, lon, alt = _field(g, "latitude_deg"), _field(g, "longitude_deg"), _field(g, "altitude_msl_m")
    else:
        lat, lon, alt = _field(g, "lat") * 1e-7, _field(g, "lon") * 1e-7, _field(g, "alt") * 1e-3
    fix = np.asarray(g["fix_type"])
    ok = (fix >= 3) & np.asarray(g.get("vel_ned_valid", np.ones_like(fix)), dtype=bool)
    ref_aligned = bool(valid.any())
    if ref_aligned:
        # EKF2's own origin, so that both filters work in the same frame. The
        # origin of the last reset is kept (EKF2 moves it at most once, in flight).
        i_ref = int(np.flatnonzero(valid)[-1])
        lat0, lon0, alt0 = (float(lp["ref_lat"][i_ref]), float(lp["ref_lon"][i_ref]), float(lp["ref_alt"][i_ref]))
    else:
        # no global origin in EKF2 (no GPS lock in the air): first good fix, and
        # EKF2 positions are not comparable
        i0 = int(np.flatnonzero(ok)[0])
        lat0, lon0, alt0 = float(lat[i0]), float(lon[i0]), float(alt[i0])
    ref_p = np.column_stack([lp["x"], lp["y"], lp["z"]]).astype(float)
    if not ref_aligned:
        ref_p[:] = np.nan
    ref_v = np.column_stack([lp["vx"], lp["vy"], lp["vz"]]).astype(float)
    att = _topic(ulog, "vehicle_attitude")
    t_att = sec(att["timestamp"])
    q_att = _vec(att, "q", 4)
    k = np.clip(np.searchsorted(t_att, t_lp), 0, len(t_att) - 1)
    ref_q = q_att[k]
    w = _topic(ulog, "wind", "wind_estimate")
    ref_wind = None
    if w is not None:
        t_w = sec(w["timestamp"])
        wn, we = _field(w, "windspeed_north"), _field(w, "windspeed_east")
        ref_wind = np.column_stack([np.interp(t_lp, t_w, wn), np.interp(t_lp, t_w, we)])

    # GNSS
    t_g = sec(_time(g))[ok] - delay("EKF2_GPS_DELAY")
    gnss_p = geodetic_to_ned(lat[ok], lon[ok], alt[ok], lat0, lon0, alt0)
    gnss_v = np.column_stack([_field(g, "vel_n_m_s"), _field(g, "vel_e_m_s"), _field(g, "vel_d_m_s")])[ok]
    gnss_sigma = np.column_stack([_field(g, "eph"), _field(g, "epv"), _field(g, "s_variance_m_s")])[ok]

    data = LogData(name=name or path.stem, t_imu=t_imu, dt_imu=dt_imu, dtheta=dtheta, dvel=dvel,
                   t_gnss=t_g, gnss_p=gnss_p, gnss_v=gnss_v, gnss_sigma=gnss_sigma,
                   ref_name="EKF2", t_ref=t_lp, ref_p=ref_p, ref_v=ref_v, ref_q=ref_q, ref_wind=ref_wind)

    ad = _topic(ulog, "vehicle_air_data")
    if ad is not None:
        data.t_baro = sec(_time(ad)) - delay("EKF2_BARO_DELAY")
        data.baro_alt = _field(ad, "baro_alt_meter") - alt0
    mg = _topic(ulog, "vehicle_magnetometer")
    if mg is not None:
        data.t_mag = sec(_time(mg)) - delay("EKF2_MAG_DELAY")
        data.mag = _vec(mg, "magnetometer_ga")
    # airspeed_validated (the airspeed EKF2 uses) is usually logged more
    # often than the raw airspeed topic; take whichever has more samples.
    candidates = [(nm, a) for nm in ("airspeed_validated", "airspeed") if (a := _topic(ulog, nm)) is not None]
    airspeed_topic = None
    if candidates:
        airspeed_topic, asp = max(candidates, key=lambda c: len(c[1]["timestamp"]))
        tas = _field(asp, "true_airspeed_m_s")
        good = np.isfinite(tas)
        data.t_tas = sec(_time(asp))[good] - delay("EKF2_ASP_DELAY")
        data.tas = tas[good]

    # Earth field at the origin, on the date of the flight
    utc = np.asarray(g.get("time_utc_usec", np.zeros(1)), dtype=float)
    utc = utc[utc > 0]
    if len(utc):
        d = _dt.datetime.fromtimestamp(utc[0] * 1e-6, tz=_dt.timezone.utc)
        year = d.year + (d.timetuple().tm_yday - 1) / 365.25
    else:
        year = 2024.0
    data.field_ned = wmm_field(lat0, lon0, alt0, year)

    status = _topic(ulog, "vehicle_land_detected")
    airborne = None
    if status is not None:
        t_s = sec(status["timestamp"])
        landed = np.asarray(status["landed"], dtype=bool)
        if (~landed).any():
            airborne = (float(t_s[np.flatnonzero(~landed)[0]]), float(t_s[np.flatnonzero(~landed)[-1]]))
    data.info = {
        "airspeed_topic": airspeed_topic,
        "file": path.name, "origin": (lat0, lon0, alt0), "ekf2_origin": ref_aligned, "decimal_year": year,
        "airborne": airborne, "duration_s": float(t_imu[-1] - t_imu[0]),
        "imu_rate_hz": float(1.0 / np.median(dt_imu)),
        "gnss_rate_hz": float((len(t_g) - 1) / (t_g[-1] - t_g[0])) if len(t_g) > 2 else 0.0,
        "airspeed_rate_hz": float((len(data.t_tas) - 1) / (data.t_tas[-1] - data.t_tas[0])) if len(data.t_tas) > 2
        else 0.0,
        "mag_rate_hz": float((len(data.t_mag) - 1) / (data.t_mag[-1] - data.t_mag[0])) if len(data.t_mag) > 2 else 0.0,
        "params": {k_: params[k_] for k_ in sorted(params) if k_.startswith("EKF2_")},
        "sys": {k_: v for k_, v in ulog.msg_info_dict.items() if k_ in ("ver_hw", "ver_sw_release_str",
                                                                         "sys_name", "ver_sw")},
    }
    return data
