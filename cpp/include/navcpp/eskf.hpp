// Error-state EKF of steps 3 to 7, ported from src/navsim/eskf.py for the
// configuration flown on the real logs of step 7 (R3): 15 core states,
// barometer drift, horizontal wind, airspeed scale and sideslip offset,
// magnetometer fused as a heading.
//
// Every matrix has a size known at compile time: no heap allocation after
// construction (checked by cpp/tests/test_eskf.cpp with
// EIGEN_RUNTIME_NO_MALLOC).
#pragma once

#include <Eigen/Core>

#include "navcpp/rotations.hpp"

namespace navcpp {

constexpr int N = 20;
using MatN = Eigen::Matrix<double, N, N>;
using VecN = Eigen::Matrix<double, N, 1>;

// State layout (same order as the Python blocks)
enum : int { IP = 0, IV = 3, ITH = 6, IBG = 9, IBA = 12, IBARO = 15, IWIND = 16, ISCALE = 18, IBETA = 19 };

constexpr double G = 9.80665;

struct Config {
    // IMU (densities, as in navsim.imu.IMUConfig)
    double gyro_noise = 0.0, accel_noise = 0.0, gyro_bias_rw = 0.0, accel_bias_rw = 0.0;
    double gyro_bias0 = 0.0, accel_bias0 = 0.0;
    // GNSS floors on (horizontal, vertical, speed) sigma
    double gnss_floor_h = 0.5, gnss_floor_v = 0.75, gnss_floor_s = 0.3;
    // barometer
    double baro_noise = 3.5, baro_drift_sigma = 5.0, baro_drift_tau = 600.0;
    // magnetometer heading
    double heading_sigma = 0.3, declination = 0.0;
    // air data
    double pitot_noise = 1.4, scale_sigma0 = 0.3, sideslip_sigma = 0.1, sideslip_offset_sigma0 = 0.26;
    double wind_rw = 0.1, wind_sigma0 = 5.0, init_sideslip_sigma = 0.035, calib_rw = 1e-4;
    double min_airspeed = 8.0;
    // initial attitude uncertainty [rad]
    double tilt_sigma = 0.035, yaw_sigma = 0.26;
    // innovation gates (NIS thresholds, <= 0: no gate)
    double gate_gnss = 0.0, gate_baro = 0.0, gate_heading = 0.0, gate_pitot = 0.0, gate_sideslip = 0.0;
    // lock-out protections [s] (< 0: never)
    double gnss_reset_after = 45.0, pitot_reset_after = 5.0;
};

class Eskf {
public:
    explicit Eskf(const Config& cfg) : cfg_(cfg) {}

    // Initial state from the first GNSS fix, the mean specific force over the
    // initial window and one magnetometer sample (has_mag false: heading 0).
    void initialise(const Vec3& p0, const Vec3& v0, const Vec3& gnss_acc, const Vec3& f_b, bool has_mag,
                    const Vec3& m_b, bool has_baro, double baro0);

    void predict(const Vec3& dtheta, const Vec3& dvel, double dt);

    // Each update returns the NIS and sets rejected().
    double update_gnss(const Vec3& pos, const Vec3& vel, const Vec3& acc);
    double update_baro(double alt);
    double update_mag_heading(const Vec3& m_b);
    double update_airspeed(double tas);
    double update_sideslip();
    void reset_to_gnss(const Vec3& pos, const Vec3& vel, const Vec3& acc);
    void wind_from_airspeed(double tas);  // tas already corrected for the estimated scale

    bool rejected() const { return rejected_; }
    double tas_scale() const { return x_[ISCALE]; }

    // nominal state
    const Vec3& p() const { return p_; }
    const Vec3& v() const { return v_; }
    const Quat& q() const { return q_; }
    const Vec3& bg() const { return bg_; }
    const Vec3& ba() const { return ba_; }
    // extra states (index 15..19), as in the Python `extra` array
    const VecN& x() const { return x_; }
    const MatN& P() const { return P_; }

    static Eigen::Matrix<double, 6, 1> gnss_sigma(const Vec3& acc, double fh, double fv, double fs);

private:
    template <int M>
    double update(const Eigen::Matrix<double, M, 1>& y, const Eigen::Matrix<double, M, N>& H,
                  const Eigen::Matrix<double, M, M>& R, double gate);

    Config cfg_;
    Vec3 p_ = Vec3::Zero(), v_ = Vec3::Zero(), bg_ = Vec3::Zero(), ba_ = Vec3::Zero();
    Quat q_ = Quat(1, 0, 0, 0);
    VecN x_ = VecN::Zero();  // only 15..19 are used (the error-state extras' nominal values)
    MatN P_ = MatN::Identity();
    Vec3 prev_dth_ = Vec3::Zero(), prev_dv_ = Vec3::Zero();
    bool have_prev_ = false;
    bool rejected_ = false;
};

}  // namespace navcpp
