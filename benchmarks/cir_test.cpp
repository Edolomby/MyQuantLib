#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// =============================================================================
// IMPORTING MY QUANTLIB ARCHITECTURE
// =============================================================================
#include <myql/models/asvj/data/ModelParams.hpp> // Assuming HestonParams is defined here
#include <myql/models/asvj/policies/VolSchemes.hpp>

// =============================================================================
// ANALYTICS (EXACT MOMENTS)
// =============================================================================
struct Analytics {
  // Standard Mean (1st Raw Moment)
  static double mean(double v0, double k, double theta, double T) {
    return theta + (v0 - theta) * std::exp(-k * T);
  }

  // 2nd Raw Moment: E[V^2] = Var(V) + E[V]^2
  static double second_moment(double v0, double k, double theta, double s,
                              double T) {
    double m1 = mean(v0, k, theta, T);
    double ekt = std::exp(-k * T);

    double var = v0 * (s * s / k) * (ekt - ekt * ekt) +
                 (theta * s * s / (2.0 * k)) * std::pow(1.0 - ekt, 2);

    return var + m1 * m1;
  }

  // 3rd Raw Moment: E[V^3] = E[(V - E[V])^3] + 3*E[V]*E[V^2] - 2*E[V]^3
  static double third_moment(double v0, double k, double theta, double s,
                             double T) {
    double m1 = mean(v0, k, theta, T);
    double m2 = second_moment(v0, k, theta, s, T);

    double ekt = std::exp(-k * T);
    double om_ekt = 1.0 - ekt;

    double c = (s * s * om_ekt) / (4.0 * k);
    double d = 4.0 * k * theta / (s * s);
    double lambda = 4.0 * k * v0 * ekt / (s * s * om_ekt);

    double mu3 = std::pow(c, 3) * 8.0 * (d + 3.0 * lambda);

    return mu3 + 3.0 * m1 * m2 - 2.0 * std::pow(m1, 3);
  }
};

// =============================================================================
// THE BENCHMARK RUNNER
// =============================================================================
template <typename Scheme>
void run_test(std::string name, const HestonParams &p, double T, size_t N_steps,
              size_t M_paths) {

  double dt = T / static_cast<double>(N_steps);

  // 1. BUILD GLOBAL WORKSPACE (TIMED)
  // This triggers the OpenMP parallel spline generation for NCI
  auto start_prep = std::chrono::high_resolution_clock::now();

  typename Scheme::GlobalWorkspace gw = Scheme::build_global_workspace(p, dt);

  auto end_prep = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_prep = end_prep - start_prep;

  // 2. PREPARE LOCAL WORKSPACE (Zero-cost pointer linking)
  typename Scheme::Workspace w;
  Scheme::prepare(w, gw, p, dt);

  // SIMULATION
  std::mt19937_64 rng(42);
  double s1 = 0, s2 = 0, s3 = 0, s4 = 0, s6 = 0;

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < M_paths; ++i) {
    double v = p.v0;
    for (size_t j = 0; j < N_steps; ++j) {

      // 4. EVOLVE
      if constexpr (std::is_same_v<Scheme, SchemeExact>) {
        v = Scheme::evolve(v, rng, w);
      } else {
        v = Scheme::evolve(v, rng, w);
      }
    }

    // Accumulate moments
    double v2 = v * v;
    double v3 = v2 * v;
    s1 += v;
    s2 += v2;
    s3 += v3;
    s4 += v2 * v2;
    s6 += v3 * v3;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  // -------------------------------------------------------------------------
  // RESULTS & PRINTING
  // -------------------------------------------------------------------------
  double inv_M = 1.0 / static_cast<double>(M_paths);
  double mc_m1 = s1 * inv_M;
  double mc_m2 = s2 * inv_M;
  double mc_m3 = s3 * inv_M;
  double mc_m4 = s4 * inv_M;
  double mc_m6 = s6 * inv_M;

  double ex_m1 = Analytics::mean(p.v0, p.kappa, p.theta, T);
  double ex_m2 = Analytics::second_moment(p.v0, p.kappa, p.theta, p.sigma, T);
  double ex_m3 = Analytics::third_moment(p.v0, p.kappa, p.theta, p.sigma, T);

  double se_m1 = std::sqrt((mc_m2 - mc_m1 * mc_m1) * inv_M);
  double se_m2 = std::sqrt((mc_m4 - mc_m2 * mc_m2) * inv_M);
  double se_m3 = std::sqrt((mc_m6 - mc_m3 * mc_m3) * inv_M);

  double z1 = (mc_m1 - ex_m1) / se_m1;
  double z2 = (mc_m2 - ex_m2) / se_m2;
  double z3 = (mc_m3 - ex_m3) / se_m3;

  std::cout << "\n[" << name << " - Raw Moment Stress Test]\n";
  std::cout << std::scientific << std::setprecision(4);

  std::cout << "  M1 (Raw): " << mc_m1 << " | Exact: " << ex_m1
            << " | Z: " << std::fixed << std::setprecision(2) << std::setw(6)
            << z1 << "\n";
  std::cout << std::scientific << std::setprecision(4);

  std::cout << "  M2 (Raw): " << mc_m2 << " | Exact: " << ex_m2
            << " | Z: " << std::fixed << std::setprecision(2) << std::setw(6)
            << z2 << "\n";
  std::cout << std::scientific << std::setprecision(4);

  std::cout << "  M3 (Raw): " << mc_m3 << " | Exact: " << ex_m3
            << " | Z: " << std::fixed << std::setprecision(2) << std::setw(6)
            << z3 << "\n";

  double time_sim = elapsed.count();
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  Time    : " << time_sim << "s ("
            << (size_t)(M_paths / time_sim) << " p/s)\n";

  // Print Memory and Setup Time purely for NCI
  if constexpr (std::is_same_v<Scheme, SchemeNCI>) {
    // Read directly from the GlobalWorkspace 'gw'
    double mem_mb = (double)((gw.y_data.capacity() + gw.m_data.capacity()) *
                             sizeof(double)) /
                    (1024.0 * 1024.0);

    std::cout << "  Heap Cache: " << gw.n_max << " splines | " << std::fixed
              << std::setprecision(2) << mem_mb << " MB\n";
    std::cout << "  Setup Time: " << std::fixed << std::setprecision(6)
              << elapsed_prep.count() << "s (Spline Creation)\n";
    std::cout << "  Total Time: " << std::fixed << std::setprecision(6)
              << time_sim + elapsed_prep.count() << "s\n";
  }
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================
int main() {
  std::vector<HestonParams> test_cases = {
      {2.5, 0.06, 0.4, -0.7, 0.05}, // Case 0: Standard
      {0.5, 0.04, 1.00, 0.04, 0.0}, // Case 1: High Vol (Feller Violated)
      {0.3, 0.04, 0.30, 0.04, 0.0}, // Case 2: Low Vol (Feller Satisfied)
      {2.0, 0.08, 0.60, 0.05, 0.0}, // Case 3: High Mean Reversion
      {0.5, 0.10, 0.40, 0.01, 0.0}  // Case 4: Deep Startup (v0 << theta)
  };

  std::vector<double> time_horizons = {
      0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0};

  size_t M_paths = 100000;
  size_t N_steps = 1;

  std::cout << "============================================================\n";
  std::cout << "CIR MARGINAL STRESS TEST (" << M_paths << " paths)\n";
  std::cout << "============================================================\n";

  for (size_t i = 0; i < test_cases.size(); ++i) {
    const auto &p = test_cases[i];

    std::cout
        << "\n############################################################\n";
    std::cout << ">>> TEST CASE " << i << " <<<\n";
    std::cout << "  Params: kappa=" << p.kappa << ", theta=" << p.theta
              << ", sigma=" << p.sigma << ", v0=" << p.v0 << "\n";
    std::cout << "  Feller Ratio (2kTheta/s^2): "
              << (2.0 * p.kappa * p.theta) / (p.sigma * p.sigma) << "\n";
    std::cout
        << "############################################################\n";

    for (double T : time_horizons) {
      std::cout << "\n--- Horizon T = " << T << " ---\n";

      run_test<SchemeExact>("HighVol", p, T, N_steps, M_paths);
      run_test<SchemeNCI>("SchemeNCI", p, T, N_steps, M_paths);
    }
    std::cout
        << "------------------------------------------------------------\n";
  }

  return 0;
}