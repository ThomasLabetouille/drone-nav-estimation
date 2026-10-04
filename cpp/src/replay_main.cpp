// nav_replay: runs the C++ filter on an event stream exported by Python
// (navsim.cpp_bridge.export), the loop of navsim.replay.run.
//
//   nav_replay config.txt events.bin out_states.bin out_nis.bin
//
// config.txt: "key value [value...]" lines (Config fields, then the initial
// data). events.bin: records of 12 float64 [type, t, payload...] in the
// order the filter must process them:
//   1 IMU   [dt, dtheta(3), dvel(3)]
//   2 GNSS  [p(3), v(3), eph, epv, speed accuracy, outage flag]
//   3 baro  [altitude]
//   4 mag   [field(3)]
//   5 tas   [true airspeed]
// out_states.bin: one record of 43 float64 per GNSS epoch
//   [t, status, p(3), v(3), q(4), bg(3), ba(3), extra(5), sigma(20)].
// out_nis.bin: one record of 4 float64 per update [kind, t, nis, rejected].
//
// Reading and writing use the heap; the filter itself does not.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "navcpp/eskf.hpp"

using navcpp::Vec3;

namespace {

std::map<std::string, std::vector<double>> read_config(const char* path) {
    std::ifstream f(path);
    std::map<std::string, std::vector<double>> kv;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string key, tok;
        if (!(ss >> key)) continue;
        std::vector<double> vals;
        while (ss >> tok) vals.push_back(tok == "nan" ? NAN : std::strtod(tok.c_str(), nullptr));
        kv[key] = vals;
    }
    return kv;
}

double get(const std::map<std::string, std::vector<double>>& kv, const std::string& k) {
    auto it = kv.find(k);
    if (it == kv.end() || it->second.empty()) {
        std::cerr << "missing config key " << k << "\n";
        std::exit(2);
    }
    return it->second[0];
}

Vec3 get3(const std::map<std::string, std::vector<double>>& kv, const std::string& k) {
    auto it = kv.find(k);
    if (it == kv.end() || it->second.size() != 3) {
        std::cerr << "missing config key " << k << "\n";
        std::exit(2);
    }
    return Vec3(it->second[0], it->second[1], it->second[2]);
}

enum Kind { KGNSS = 1, KBARO = 2, KHEADING = 3, KPITOT = 4, KSIDESLIP = 5 };

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: nav_replay config.txt events.bin out_states.bin out_nis.bin\n";
        return 2;
    }
    const auto kv = read_config(argv[1]);
    navcpp::Config c;
    c.gyro_noise = get(kv, "gyro_noise");
    c.accel_noise = get(kv, "accel_noise");
    c.gyro_bias_rw = get(kv, "gyro_bias_rw");
    c.accel_bias_rw = get(kv, "accel_bias_rw");
    c.gyro_bias0 = get(kv, "gyro_bias0");
    c.accel_bias0 = get(kv, "accel_bias0");
    c.gnss_floor_h = get(kv, "gnss_floor_h");
    c.gnss_floor_v = get(kv, "gnss_floor_v");
    c.gnss_floor_s = get(kv, "gnss_floor_s");
    c.baro_noise = get(kv, "baro_noise");
    c.baro_drift_sigma = get(kv, "baro_drift_sigma");
    c.baro_drift_tau = get(kv, "baro_drift_tau");
    c.heading_sigma = get(kv, "heading_sigma");
    c.declination = get(kv, "declination");
    c.pitot_noise = get(kv, "pitot_noise");
    c.scale_sigma0 = get(kv, "scale_sigma0");
    c.sideslip_sigma = get(kv, "sideslip_sigma");
    c.sideslip_offset_sigma0 = get(kv, "sideslip_offset_sigma0");
    c.wind_rw = get(kv, "wind_rw");
    c.wind_sigma0 = get(kv, "wind_sigma0");
    c.init_sideslip_sigma = get(kv, "init_sideslip_sigma");
    c.calib_rw = get(kv, "calib_rw");
    c.min_airspeed = get(kv, "min_airspeed");
    c.tilt_sigma = get(kv, "tilt_sigma");
    c.yaw_sigma = get(kv, "yaw_sigma");
    c.gate_gnss = get(kv, "gate_gnss");
    c.gate_baro = get(kv, "gate_baro");
    c.gate_heading = get(kv, "gate_heading");
    c.gate_pitot = get(kv, "gate_pitot");
    c.gate_sideslip = get(kv, "gate_sideslip");
    c.gnss_reset_after = get(kv, "gnss_reset_after");
    c.pitot_reset_after = get(kv, "pitot_reset_after");

    navcpp::Eskf ekf(c);
    const Vec3 m0 = get3(kv, "init_mag");
    const double baro0 = get(kv, "init_baro");
    const Vec3 p0 = get3(kv, "init_p"), v0 = get3(kv, "init_v"), acc0 = get3(kv, "init_acc");
    ekf.initialise(p0, v0, acc0, get3(kv, "init_fb"), !std::isnan(m0[0]), m0, !std::isnan(baro0), baro0);
    const double t0 = get(kv, "t0");

    std::ifstream ev(argv[2], std::ios::binary);
    std::vector<double> buf;
    {
        ev.seekg(0, std::ios::end);
        const auto bytes = static_cast<size_t>(ev.tellg());
        ev.seekg(0);
        buf.resize(bytes / sizeof(double));
        ev.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(bytes));
    }
    const size_t n_ev = buf.size() / 12;

    std::vector<double> states, nis_out;
    auto record = [&](double t, double status) {
        double rec[43];
        rec[0] = t;
        rec[1] = status;
        for (int i = 0; i < 3; ++i) {
            rec[2 + i] = ekf.p()[i];
            rec[5 + i] = ekf.v()[i];
            rec[12 + i] = ekf.bg()[i];
            rec[15 + i] = ekf.ba()[i];
        }
        for (int i = 0; i < 4; ++i) rec[8 + i] = ekf.q()[i];
        for (int i = 0; i < 5; ++i) rec[18 + i] = ekf.x()[15 + i];
        for (int i = 0; i < navcpp::N; ++i) rec[23 + i] = std::sqrt(ekf.P()(i, i));
        states.insert(states.end(), rec, rec + 43);
    };
    auto note = [&](int kind, double t, double nis) {
        nis_out.push_back(kind);
        nis_out.push_back(t);
        nis_out.push_back(nis);
        nis_out.push_back(ekf.rejected() ? 1.0 : 0.0);
    };

    record(t0, 0.0);
    bool wind_on = false;
    double last_fused = t0;
    bool pitot_rejecting = false;
    double pitot_rejected_since = 0.0;
    size_t n_imu = 0;
    double filter_seconds = 0.0;
    for (size_t e = 0; e < n_ev; ++e) {
        const double* r = &buf[12 * e];
        const int type = static_cast<int>(r[0]);
        const double t = r[1];
        const auto start = std::chrono::steady_clock::now();
        double status = -1.0;
        switch (type) {
            case 1:
                ekf.predict(Vec3(r[3], r[4], r[5]), Vec3(r[6], r[7], r[8]), r[2]);
                ++n_imu;
                break;
            case 2: {
                const Vec3 pos(r[2], r[3], r[4]), vel(r[5], r[6], r[7]), acc(r[8], r[9], r[10]);
                if (r[11] != 0.0) {
                    status = 2.0;
                } else {
                    note(KGNSS, t, ekf.update_gnss(pos, vel, acc));
                    status = ekf.rejected() ? 1.0 : 0.0;
                    if (!ekf.rejected()) {
                        last_fused = t;
                    } else if (c.gnss_reset_after >= 0.0 && t - last_fused >= c.gnss_reset_after) {
                        ekf.reset_to_gnss(pos, vel, acc);
                        last_fused = t;
                    }
                }
                break;
            }
            case 3:
                note(KBARO, t, ekf.update_baro(r[2]));
                break;
            case 4:
                note(KHEADING, t, ekf.update_mag_heading(Vec3(r[2], r[3], r[4])));
                break;
            case 5: {
                const double tas = r[2];
                if (tas < c.min_airspeed) break;
                if (!wind_on) {
                    ekf.wind_from_airspeed(tas / (1.0 + ekf.tas_scale()));
                    wind_on = true;
                    break;
                }
                note(KPITOT, t, ekf.update_airspeed(tas));
                if (ekf.rejected()) {
                    if (!pitot_rejecting) {
                        pitot_rejecting = true;
                        pitot_rejected_since = t;
                    }
                    if (c.pitot_reset_after >= 0.0 && t - pitot_rejected_since >= c.pitot_reset_after) {
                        ekf.wind_from_airspeed(tas / (1.0 + ekf.tas_scale()));
                        pitot_rejecting = false;
                    }
                } else {
                    pitot_rejecting = false;
                }
                note(KSIDESLIP, t, ekf.update_sideslip());
                break;
            }
            default:
                std::cerr << "unknown event type " << type << "\n";
                return 2;
        }
        filter_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (type == 2) record(t, status);
    }

    std::ofstream(argv[3], std::ios::binary)
        .write(reinterpret_cast<const char*>(states.data()), static_cast<std::streamsize>(states.size() * 8));
    std::ofstream(argv[4], std::ios::binary)
        .write(reinterpret_cast<const char*>(nis_out.data()), static_cast<std::streamsize>(nis_out.size() * 8));
    std::printf("events %zu imu %zu filter_seconds %.6f us_per_event %.3f\n", n_ev, n_imu, filter_seconds,
                1e6 * filter_seconds / static_cast<double>(n_ev));
    return 0;
}
