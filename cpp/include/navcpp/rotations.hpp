// Quaternions and rotations, same conventions as src/navsim/rotations.py:
// q = [w, x, y, z], Hamilton product, q_nb rotates a body vector into NED,
// Euler angles ZYX (yaw, pitch, roll). Fixed-size Eigen types only.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>  // cross product
#include <cmath>

namespace navcpp {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Quat = Eigen::Vector4d;  // w, x, y, z

inline Quat quat_mul(const Quat& p, const Quat& q) {
    return Quat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]);
}

inline Quat quat_normalize(const Quat& q) { return q / q.norm(); }

// exp map: rotation vector to quaternion, Taylor expansion of sin(a/2)/a near 0
inline Quat quat_from_rotvec(const Vec3& r) {
    const double angle = r.norm();
    const double half = 0.5 * angle;
    const double k = angle > 1e-8 ? std::sin(half) / angle : 0.5 - angle * angle / 48.0;
    return Quat(std::cos(half), k * r[0], k * r[1], k * r[2]);
}

inline Vec3 quat_rotate(const Quat& q, const Vec3& v) {
    const Vec3 qv(q[1], q[2], q[3]);
    const Vec3 t = 2.0 * qv.cross(v);
    return v + q[0] * t + qv.cross(t);
}

inline Quat quat_from_euler(double phi, double theta, double psi) {
    const double cr = std::cos(phi / 2), sr = std::sin(phi / 2);
    const double cp = std::cos(theta / 2), sp = std::sin(theta / 2);
    const double cy = std::cos(psi / 2), sy = std::sin(psi / 2);
    return Quat(cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy);
}

// roll, pitch, yaw
inline Vec3 euler_from_quat(const Quat& q) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    double s = 2 * (w * y - z * x);
    s = s > 1.0 ? 1.0 : (s < -1.0 ? -1.0 : s);
    return Vec3(std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), std::asin(s),
                std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)));
}

// C_nb, body to NED
inline Mat3 dcm_from_quat(const Quat& q) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    Mat3 C;
    C << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);
    return C;
}

inline Mat3 skew(const Vec3& v) {
    Mat3 S;
    S << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
    return S;
}

// Tilt-compensated magnetic heading (aiding.heading_from_magnetometer)
inline double heading_from_magnetometer(const Vec3& m_b, double roll, double pitch, double declination) {
    const Vec3 m_level = quat_rotate(quat_from_euler(roll, pitch, 0.0), m_b);
    return declination + std::atan2(-m_level[1], m_level[0]);
}

inline double wrap_angle(double a) { return std::atan2(std::sin(a), std::cos(a)); }

}  // namespace navcpp
