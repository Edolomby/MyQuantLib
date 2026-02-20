#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Include your header
#include <myql/math/interpolation/splines.hpp>

using namespace myql::math::interpolation;

// =============================================================================
// UTILS
// =============================================================================
double get_max_error(const MonotoneCubicSpline &spline, double x_min,
                     double x_max, double (*func)(double)) {
  double max_err = 0.0;
  int steps = 1000;
  double dx = (x_max - x_min) / steps;

  for (int i = 0; i <= steps; ++i) {
    double x = x_min + i * dx;
    double exact = func(x);
    double interp = spline(x);
    max_err = std::max(max_err, std::abs(exact - interp));
  }
  return max_err;
}

// =============================================================================
// TEST 1: INVERSE CDF SIMULATION (Sigmoid)
// mimics the shape of a CDF inverse: Flat -> Steep -> Flat
// =============================================================================
void test_inverse_cdf_shape() {
  std::cout << "[TEST] Inverse CDF Shape (Sigmoid)... ";

  // We use a sigmoid: y = 1 / (1 + exp(-x)) -> x = -ln(1/y - 1)
  // We map uniform u in [0.01, 0.99] to x
  double u_min = 0.01;
  double u_max = 0.99;
  size_t n_points = 50;

  std::vector<double> x_vals(n_points);
  for (size_t i = 0; i < n_points; ++i) {
    double u = u_min + i * (u_max - u_min) / (n_points - 1);
    // Inverse Sigmoid
    x_vals[i] = -std::log(1.0 / u - 1.0);
  }

  MonotoneCubicSpline spline(u_min, u_max, x_vals);

  // Verify Monotonicity is preserved even in steep regions
  double prev_val = spline(u_min);
  for (double u = u_min; u <= u_max; u += 0.001) {
    double val = spline(u);
    if (val < prev_val) {
      std::cerr << "Monotonicity Failed at u=" << u << std::endl;
      exit(1);
    }
    prev_val = val;
  }
  std::cout << "PASSED (Monotonicity Preserved)" << std::endl;
}

// =============================================================================
// TEST 2: CONVERGENCE RATE
// Does error decrease as we add points?
// =============================================================================
void test_convergence() {
  std::cout << "[TEST] Convergence Rate (Exponential)... ";

  auto target_func = [](double x) { return std::exp(x); };
  double x_min = 0.0, x_max = 2.0;

  // Grid sizes to test
  std::vector<size_t> grids = {10, 20, 40, 80};
  std::vector<double> errors;

  for (size_t n : grids) {
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i) {
      double x = x_min + i * (x_max - x_min) / (n - 1);
      y[i] = target_func(x);
    }

    MonotoneCubicSpline spline(x_min, x_max, y);
    errors.push_back(get_max_error(spline, x_min, x_max, target_func));
  }

  // Check that error decreases significantly (roughly O(h^3) or O(h^4))
  // We expect error to drop by ~8x-16x when grid doubles
  for (size_t i = 1; i < errors.size(); ++i) {
    double ratio = errors[i - 1] / errors[i];
    if (ratio < 4.0) { // Loose bound, usually higher
      std::cerr << "Convergence too slow! Ratio: " << ratio << std::endl;
      exit(1);
    }
  }

  std::cout << "PASSED (Error drops " << std::fixed << std::setprecision(1)
            << (errors[0] / errors[1]) << "x -> " << (errors[1] / errors[2])
            << "x -> " << (errors[2] / errors[3]) << "x)" << std::endl;
}

// =============================================================================
// TEST 3: BOUNDARY & CLAMPING
// =============================================================================
void test_boundaries() {
  std::cout << "[TEST] Boundary Clamping... ";

  std::vector<double> y = {0.0, 1.0, 8.0, 27.0}; // x^3 on [0,3]
  MonotoneCubicSpline spline(0.0, 3.0, y);

  // Exact Bounds
  assert(std::abs(spline(0.0) - 0.0) < 1e-12);
  assert(std::abs(spline(3.0) - 27.0) < 1e-12);

  // Out of Bounds (Should Clamp)
  assert(std::abs(spline(-1.0) - 0.0) < 1e-12); // Clamped to y[0]
  assert(std::abs(spline(5.0) - 27.0) < 1e-12); // Clamped to y[n-1]

  std::cout << "PASSED" << std::endl;
}

// =============================================================================
// TEST 4: RANDOMIZED STRESS TEST (Monte Carlo Simulation Proxy)
// =============================================================================
void test_random_access() {
  std::cout << "[TEST] Randomized Stress (10M Lookups)... ";

  // This simulates the NCI scheme access pattern:
  // Millions of random uniform numbers hitting the inverse CDF spline
  size_t n = 1000;
  std::vector<double> y(n);
  for (size_t i = 0; i < n; ++i)
    y[i] = std::sqrt((double)i); // Monotone

  MonotoneCubicSpline spline(0.0, 1.0, y);

  std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(0.0, 1.0);

  auto start = std::chrono::high_resolution_clock::now();

  double checksum = 0.0;
  int N_OPS = 10'000'000;

  for (int i = 0; i < N_OPS; ++i) {
    double u = dis(gen);
    checksum += spline(u);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  double mops = (N_OPS / 1e6) / diff.count();

  std::cout << "Done in " << diff.count() << "s (" << mops
            << " Mops/sec). Checksum: " << (int)checksum << " -> PASSED"
            << std::endl;
}

int main() {
  std::cout << "==========================================" << std::endl;
  std::cout << "  ADVANCED SPLINE TEST SUITE" << std::endl;
  std::cout << "==========================================" << std::endl;

  test_inverse_cdf_shape();
  test_convergence();
  test_boundaries();
  test_random_access();

  std::cout << "==========================================" << std::endl;
  std::cout << "  ALL ADVANCED TESTS PASSED" << std::endl;
  std::cout << "==========================================" << std::endl;

  return 0;
}