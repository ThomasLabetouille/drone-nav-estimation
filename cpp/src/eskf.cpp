// See navcpp/eskf.hpp. Each function follows its Python counterpart line by
// line (src/navsim/eskf.py, replay.py, fusion.py); the comments name it.
#include "navcpp/eskf.hpp"

#include <Eigen/LU>
#include <cmath>

namespace navcpp {

Eigen::Matrix<double, 6, 1> Eskf::gnss_sigma(const Vec3& acc, double fh, double fv, double fs) {
    // replay._sigma
    const double h = std::max(acc[0] / std::sqrt(2.0), fh);
    const double v = std::max(acc[1], fv);
    const double s = std::max(acc[2] / std::sqrt(3.0), fs);
    Eigen::Matrix<double, 6, 1> out;
    out << h, h, v, s, s, s;
    return out;
}

void Eskf::initialise(const Vec3& p0, const Vec3& v0, const Vec3& gnss_acc, const Vec3& f_b, bool has_mag,
                      const Vec3& m_b, bool has_baro, double baro0) {
    // replay._quat_from_accel_mag
    const double roll = std::atan2(-f_b[1], -f_b[2]);
    const double pitch = std::atan2(f_b[0], std::hypot(f_b[1], f_b[2]));
    const double yaw = has_mag ? heading_from_magnetometer(m_b, roll, pitch, cfg_.declination) : 0.0;
    q_ = quat_from_euler(roll, pitch, yaw);
    p_ = p0;
    v_ = v0;
    bg_.setZero();
    ba_.setZero();
    x_.setZero();
    if (has_baro) x_[IBARO] = baro0 - (-p0[2]);

    // replay.run: P0
    const auto s = gnss_sigma(gnss_acc, cfg_.gnss_floor_h, cfg_.gnss_floor_v, cfg_.gnss_floor_s);
    VecN d;
    d << s[0] * s[0], s[1] * s[1], s[2] * s[2], s[3] * s[3], s[4] * s[4], s[5] * s[5],
        cfg_.tilt_sigma * cfg_.tilt_sigma, cfg_.tilt_sigma * cfg_.tilt_sigma, cfg_.yaw_sigma * cfg_.yaw_sigma,
        cfg_.gyro_bias0 * cfg_.gyro_bias0, cfg_.gyro_bias0 * cfg_.gyro_bias0, cfg_.gyro_bias0 * cfg_.gyro_bias0,
        cfg_.accel_bias0 * cfg_.accel_bias0, cfg_.accel_bias0 * cfg_.accel_bias0, cfg_.accel_bias0 * cfg_.accel_bias0,
        cfg_.baro_drift_sigma * cfg_.baro_drift_sigma, cfg_.wind_sigma0 * cfg_.wind_sigma0,
        cfg_.wind_sigma0 * cfg_.wind_sigma0, cfg_.scale_sigma0 * cfg_.scale_sigma0,
        cfg_.sideslip_offset_sigma0 * cfg_.sideslip_offset_sigma0;
    P_ = d.asDiagonal();
    have_prev_ = false;
}

void Eskf::predict(const Vec3& dth_meas, const Vec3& dv_meas, double dt) {
    // ErrorStateEKF.predict + strapdown.step("full")
    const Vec3 dth = dth_meas - bg_ * dt;
    const Vec3 dv = dv_meas - ba_ * dt;
    const Mat3 C = dcm_from_quat(q_);
    const Vec3 f_n = C * dv / dt;

    Vec3 rot = dth;
    Vec3 dv_b = dv + 0.5 * dth.cross(dv);
    if (have_prev_) {
        rot = dth + prev_dth_.cross(dth) / 12.0;
        dv_b = dv_b + (prev_dth_.cross(dv) + prev_dv_.cross(dth)) / 12.0;
    }
    const Vec3 v_new = v_ + quat_rotate(q_, dv_b) + Vec3(0.0, 0.0, G * dt);
    p_ = p_ + 0.5 * (v_ + v_new) * dt;
    v_ = v_new;
    q_ = quat_normalize(quat_mul(q_, quat_from_rotvec(rot)));
    prev_dth_ = dth;
    prev_dv_ = dv;
    have_prev_ = true;

    // ErrorStateEKF._discrete: the baro drift is the only Gauss-Markov state
    const double phi_b = std::exp(-dt / cfg_.baro_drift_tau);
    x_[IBARO] *= phi_b;

    // Phi = I + F dt + (F dt)^2 / 2, block by block
    const Mat3 fx = skew(f_n);
    const Mat3 I3 = Mat3::Identity();
    MatN Phi = MatN::Identity();
    Phi.block<3, 3>(IP, IV) += I3 * dt;
    Phi.block<3, 3>(IV, ITH) += -fx * dt;
    Phi.block<3, 3>(IV, IBA) += -C * dt;
    Phi.block<3, 3>(ITH, IBG) += -C * dt;
    const double h = 0.5 * dt * dt;
    Phi.block<3, 3>(IP, ITH) += -fx * h;
    Phi.block<3, 3>(IP, IBA) += -C * h;
    Phi.block<3, 3>(IV, IBG) += (fx * C) * h;
    Phi(IBARO, IBARO) = phi_b;

    VecN qd = VecN::Zero();
    qd.segment<3>(IV).setConstant(cfg_.accel_noise * cfg_.accel_noise * dt);
    qd.segment<3>(ITH).setConstant(cfg_.gyro_noise * cfg_.gyro_noise * dt);
    qd.segment<3>(IBG).setConstant(cfg_.gyro_bias_rw * cfg_.gyro_bias_rw * dt);
    qd.segment<3>(IBA).setConstant(cfg_.accel_bias_rw * cfg_.accel_bias_rw * dt);
    qd[IBARO] = cfg_.baro_drift_sigma * cfg_.baro_drift_sigma * (1.0 - phi_b * phi_b);
    qd.segment<2>(IWIND).setConstant(cfg_.wind_rw * cfg_.wind_rw * dt);
    qd[ISCALE] = cfg_.calib_rw * cfg_.calib_rw * dt;
    qd[IBETA] = cfg_.calib_rw * cfg_.calib_rw * dt;

    MatN P = Phi * P_ * Phi.transpose();
    P.diagonal() += qd;
    P_ = 0.5 * (P + P.transpose());
}

template <int M>
double Eskf::update(const Eigen::Matrix<double, M, 1>& y, const Eigen::Matrix<double, M, N>& H,
                    const Eigen::Matrix<double, M, M>& R, double gate) {
    // ErrorStateEKF.update: Joseph form, gate, injection, reset
    using MatNM = Eigen::Matrix<double, N, M>;
    using MatMM = Eigen::Matrix<double, M, M>;
    const MatNM PHt = P_ * H.transpose();
    const MatMM S = H * PHt + R;
    const MatMM S_inv = S.inverse();
    MatNM K = PHt * S_inv;
    const double nis = y.dot(S_inv * y);
    rejected_ = gate > 0.0 && nis > gate;
    if (rejected_) K.setZero();
    const VecN dx = K * y;
    const MatN IKH = MatN::Identity() - K * H;
    MatN P = IKH * P_ * IKH.transpose() + K * R * K.transpose();

    p_ += dx.segment<3>(IP);
    v_ += dx.segment<3>(IV);
    const Vec3 dth = dx.segment<3>(ITH);
    q_ = quat_normalize(quat_mul(quat_from_rotvec(dth), q_));
    bg_ += dx.segment<3>(IBG);
    ba_ += dx.segment<3>(IBA);
    x_.tail<N - 15>() += dx.tail<N - 15>();
    // reset_jacobian: G = I + [dtheta/2 x] (global attitude error)
    MatN Gr = MatN::Identity();
    Gr.block<3, 3>(ITH, ITH) += 0.5 * skew(dth);
    P = Gr * P * Gr.transpose();
    P_ = 0.5 * (P + P.transpose());
    return nis;
}

double Eskf::update_gnss(const Vec3& pos, const Vec3& vel, const Vec3& acc) {
    Eigen::Matrix<double, 6, 1> y;
    y << pos - p_, vel - v_;
    Eigen::Matrix<double, 6, N> H = Eigen::Matrix<double, 6, N>::Zero();
    H.leftCols<6>().setIdentity();
    const auto s = gnss_sigma(acc, cfg_.gnss_floor_h, cfg_.gnss_floor_v, cfg_.gnss_floor_s);
    const Eigen::Matrix<double, 6, 6> R = s.cwiseProduct(s).asDiagonal();
    return update<6>(y, H, R, cfg_.gate_gnss);
}

void Eskf::reset_to_gnss(const Vec3& pos, const Vec3& vel, const Vec3& acc) {
    // ErrorStateEKF.reset_to_gnss (no GNSS bias state in this configuration)
    const auto s = gnss_sigma(acc, cfg_.gnss_floor_h, cfg_.gnss_floor_v, cfg_.gnss_floor_s);
    P_.topRows<6>().setZero();
    P_.leftCols<6>().setZero();
    P_.topLeftCorner<6, 6>() = s.cwiseProduct(s).asDiagonal();
    p_ = pos;
    v_ = vel;
}

double Eskf::update_baro(double alt) {
    Eigen::Matrix<double, 1, 1> y(alt - (-p_[2] + x_[IBARO]));
    Eigen::Matrix<double, 1, N> H = Eigen::Matrix<double, 1, N>::Zero();
    H(0, 2) = -1.0;
    H(0, IBARO) = 1.0;
    return update<1>(y, H, Eigen::Matrix<double, 1, 1>(cfg_.baro_noise * cfg_.baro_noise), cfg_.gate_baro);
}

double Eskf::update_mag_heading(const Vec3& m_b) {
    const Vec3 e = euler_from_quat(q_);
    const double psi = heading_from_magnetometer(m_b, e[0], e[1], cfg_.declination);
    Eigen::Matrix<double, 1, 1> y(wrap_angle(psi - e[2]));
    Eigen::Matrix<double, 1, N> H = Eigen::Matrix<double, 1, N>::Zero();
    H(0, ITH) = std::tan(e[1]) * std::cos(e[2]);
    H(0, ITH + 1) = std::tan(e[1]) * std::sin(e[2]);
    H(0, ITH + 2) = 1.0;
    return update<1>(y, H, Eigen::Matrix<double, 1, 1>(cfg_.heading_sigma * cfg_.heading_sigma),
                     cfg_.gate_heading);
}

double Eskf::update_airspeed(double tas) {
    const Vec3 w(x_[IWIND], x_[IWIND + 1], 0.0);
    const Vec3 v_air = v_ - w;
    const double speed = v_air.norm();
    const Vec3 u = v_air / speed;
    const double k = 1.0 + x_[ISCALE];
    Eigen::Matrix<double, 1, 1> y(tas - k * speed);
    Eigen::Matrix<double, 1, N> H = Eigen::Matrix<double, 1, N>::Zero();
    H.block<1, 3>(0, IV) = k * u.transpose();
    H(0, IWIND) = -k * u[0];
    H(0, IWIND + 1) = -k * u[1];
    H(0, ISCALE) = speed;
    return update<1>(y, H, Eigen::Matrix<double, 1, 1>(cfg_.pitot_noise * cfg_.pitot_noise), cfg_.gate_pitot);
}

double Eskf::update_sideslip() {
    const Vec3 w(x_[IWIND], x_[IWIND + 1], 0.0);
    const Vec3 v_air = v_ - w;
    const double V = v_air.norm();
    const Mat3 Ct = dcm_from_quat(q_).transpose();
    const double beta = Ct.row(1).dot(v_air) / V;
    Eigen::Matrix<double, 1, 1> y(0.0 - (beta - x_[IBETA]));
    const Eigen::RowVector3d d_vair = (Ct.row(1) - beta * v_air.transpose() / V) / V;
    Eigen::Matrix<double, 1, N> H = Eigen::Matrix<double, 1, N>::Zero();
    H.block<1, 3>(0, IV) = d_vair;
    H(0, IWIND) = -d_vair[0];
    H(0, IWIND + 1) = -d_vair[1];
    H.block<1, 3>(0, ITH) = (Ct * skew(v_air)).row(1) / V;
    H(0, IBETA) = -1.0;
    return update<1>(y, H, Eigen::Matrix<double, 1, 1>(cfg_.sideslip_sigma * cfg_.sideslip_sigma),
                     cfg_.gate_sideslip);
}

void Eskf::wind_from_airspeed(double tas) {
    // fusion.wind_from_first_airspeed
    const double psi = euler_from_quat(q_)[2];
    const Eigen::Vector2d u(std::cos(psi), std::sin(psi));
    const Eigen::Vector2d u_perp(-std::sin(psi), std::cos(psi));
    Eigen::Matrix<double, 2, N> J = Eigen::Matrix<double, 2, N>::Zero();
    J.block<2, 2>(0, IV).setIdentity();
    J.col(ITH + 2) = -tas * u_perp;
    P_.middleRows<2>(IWIND).setZero();
    P_.middleCols<2>(IWIND).setZero();
    const Eigen::Matrix<double, 2, N> cross = J * P_;
    P_.middleRows<2>(IWIND) = cross;
    P_.middleCols<2>(IWIND) = cross.transpose();
    const double sb = tas * cfg_.init_sideslip_sigma;
    P_.block<2, 2>(IWIND, IWIND) = cross * J.transpose() + cfg_.pitot_noise * cfg_.pitot_noise * u * u.transpose() +
                                    sb * sb * u_perp * u_perp.transpose();
    x_[IWIND] = v_[0] - tas * u[0];
    x_[IWIND + 1] = v_[1] - tas * u[1];
}

template double Eskf::update<1>(const Eigen::Matrix<double, 1, 1>&, const Eigen::Matrix<double, 1, N>&,
                                const Eigen::Matrix<double, 1, 1>&, double);
template double Eskf::update<6>(const Eigen::Matrix<double, 6, 1>&, const Eigen::Matrix<double, 6, N>&,
                                const Eigen::Matrix<double, 6, 6>&, double);

}  // namespace navcpp
