// Unit tests of the C++ port that do not need Python. The comparison with
// the Python reference, sample by sample, is tests/test_step8.py.
//
// The main check here: the filter never touches the heap once built. Two
// independent ways: Eigen's own guard (EIGEN_RUNTIME_NO_MALLOC, set for this
// target), and a global operator new that counts every allocation.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

#include <Eigen/Eigenvalues>

#include "navcpp/eskf.hpp"

static std::atomic<long> g_allocations{0};
void* operator new(std::size_t n) {
    ++g_allocations;
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

static int g_failures = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

using namespace navcpp;

static Config config() {
    Config c;
    c.gyro_noise = 0.0015;
    c.accel_noise = 0.035;
    c.gyro_bias_rw = 1e-4;
    c.accel_bias_rw = 3e-4;
    c.gyro_bias0 = 0.1;
    c.accel_bias0 = 0.2;
    c.gate_gnss = 22.46;
    c.gate_baro = c.gate_heading = c.gate_pitot = c.gate_sideslip = 10.83;
    return c;
}

// Level flight north at 17 m/s, no wind: the specific force is -g along z.
static void fly(Eskf& ekf, int steps, double dt) {
    for (int k = 0; k < steps; ++k) ekf.predict(Vec3::Zero(), Vec3(0.0, 0.0, -G * dt), dt);
}

static void test_allocation_counter_works() {
    const long before = g_allocations.load();
    {
        std::vector<double> v(16);
        v[3] = 1.0;
        CHECK(v[3] == 1.0);
    }
    CHECK(g_allocations.load() == before + 1);
}

static void test_no_heap_allocation() {
    Eskf ekf(config());
    ekf.initialise(Vec3::Zero(), Vec3(17, 0, 0), Vec3(1, 2, 0.3), Vec3(0, 0, -G), true, Vec3(0.2, 0.0, 0.43),
                   true, 0.0);
    const long before = g_allocations.load();
    Eigen::internal::set_is_malloc_allowed(false);
    for (int k = 0; k < 200; ++k) {
        fly(ekf, 40, 0.005);
        ekf.update_gnss(ekf.p(), ekf.v(), Vec3(1, 2, 0.3));
        ekf.update_baro(-ekf.p()[2]);
        ekf.update_mag_heading(Vec3(0.2, 0.0, 0.43));
        if (k == 0) ekf.wind_from_airspeed(17.0);
        ekf.update_airspeed(17.0);
        ekf.update_sideslip();
    }
    ekf.reset_to_gnss(ekf.p(), ekf.v(), Vec3(1, 2, 0.3));
    Eigen::internal::set_is_malloc_allowed(true);
    CHECK(g_allocations.load() == before);
}

static void test_rest_stays_at_rest() {
    Eskf ekf(config());
    ekf.initialise(Vec3::Zero(), Vec3::Zero(), Vec3(1, 2, 0.3), Vec3(0, 0, -G), false, Vec3::Zero(), false, 0.0);
    for (int k = 0; k < 2000; ++k) ekf.predict(Vec3::Zero(), Vec3(0.0, 0.0, -G * 0.005), 0.005);
    CHECK(ekf.v().norm() < 1e-9);
    CHECK(ekf.p().norm() < 1e-9);
    CHECK(std::abs(ekf.q().norm() - 1.0) < 1e-12);
}

static void test_gate_rejects_an_outlier() {
    Eskf ekf(config());
    ekf.initialise(Vec3::Zero(), Vec3(17, 0, 0), Vec3(1, 2, 0.3), Vec3(0, 0, -G), false, Vec3::Zero(), false, 0.0);
    fly(ekf, 40, 0.005);
    const Vec3 p = ekf.p();
    const MatN P = ekf.P();
    const double nis = ekf.update_gnss(p + Vec3(100, 0, 0), ekf.v(), Vec3(1, 2, 0.3));
    CHECK(ekf.rejected());
    CHECK(nis > 22.46);
    CHECK((ekf.p() - p).norm() == 0.0);
    CHECK((ekf.P() - P).norm() < 1e-12);
    ekf.update_gnss(p, ekf.v(), Vec3(1, 2, 0.3));
    CHECK(!ekf.rejected());
}

static void test_covariance_stays_symmetric_positive() {
    Eskf ekf(config());
    ekf.initialise(Vec3::Zero(), Vec3(17, 0, 0), Vec3(1, 2, 0.3), Vec3(0, 0, -G), true, Vec3(0.2, 0.0, 0.43),
                   true, 0.0);
    ekf.wind_from_airspeed(17.0);
    for (int k = 0; k < 100; ++k) {
        fly(ekf, 40, 0.005);
        ekf.update_gnss(ekf.p(), ekf.v(), Vec3(1, 2, 0.3));
        ekf.update_airspeed(17.0);
        ekf.update_sideslip();
    }
    CHECK((ekf.P() - ekf.P().transpose()).norm() == 0.0);
    Eigen::SelfAdjointEigenSolver<MatN> es(ekf.P());
    CHECK(es.eigenvalues().minCoeff() > 0.0);
}

static void timing() {
    Eskf ekf(config());
    ekf.initialise(Vec3::Zero(), Vec3(17, 0, 0), Vec3(1, 2, 0.3), Vec3(0, 0, -G), false, Vec3::Zero(), false, 0.0);
    const int n = 20000;
    const auto t0 = std::chrono::steady_clock::now();
    fly(ekf, n, 0.005);
    const double us = 1e6 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / n;
    const auto t1 = std::chrono::steady_clock::now();
    for (int k = 0; k < 2000; ++k) ekf.update_gnss(ekf.p(), ekf.v(), Vec3(1, 2, 0.3));
    const double us_gnss = 1e6 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() / 2000;
    std::printf("predict %.2f us, GNSS update %.2f us (20 states, this machine)\n", us, us_gnss);
}

int main() {
    test_allocation_counter_works();
    test_no_heap_allocation();
    test_rest_stays_at_rest();
    test_gate_rejects_an_outlier();
    test_covariance_stays_symmetric_positive();
    timing();
    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
