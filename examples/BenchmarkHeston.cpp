#include <algorithm>
#include <chrono> // For high-precision timing
#include <cmath>
#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h> // Include OpenMP
#include <string>
#include <vector>

// Includes
#include "models/HestonModel.hpp"
#include "pricers/FourierPricer.hpp"
#include "utils/TablePrinter.hpp"

using namespace std;
using Complex = std::complex<double>;
using namespace std::complex_literals;

// =============================================================================
// LEGACY CODE (Optimized "C-style" function)
// =============================================================================
namespace Legacy {

// ... (Keep existing helper functions exactly as they were) ...
Complex charact_func_heston(double v, double Y0, double a, double b, double s,
                            double logS0, double r, double rho, double lambda,
                            double t) {
  double s2 = s * s;
  Complex b_rhosv1i = b - rho * s * v * 1i;
  Complex Delta_root =
      std::sqrt(b_rhosv1i * b_rhosv1i + s2 * v * (v + 2.0 * lambda * 1i));
  Complex g = (b_rhosv1i - Delta_root) / (b_rhosv1i + Delta_root);
  Complex exp_Dt = std::exp(-Delta_root * t);
  Complex term_t = (b_rhosv1i - Delta_root) / s2;
  Complex phi = (r * v * 1i + a * term_t) * t -
                2.0 * a / s2 * std::log((1.0 - g * exp_Dt) / (1.0 - g));
  Complex psi =
      logS0 * v * 1i + Y0 * term_t * ((1.0 - exp_Dt) / (1.0 - g * exp_Dt));
  return std::exp(phi + psi);
}

double integrand_heston(double v, double Y0, double a, double b, double s,
                        double logS0, double r, double rho, double lambda,
                        double t, double K) {
  if (std::abs(v) < 1e-10) {
    double total_var = 0.0;
    if (std::abs(b) < 1e-9)
      total_var = Y0 * t + 0.5 * a * t * t;
    else {
      double lr = a / b;
      total_var = lr * t + (Y0 - lr) * (1.0 - std::exp(-b * t)) / b;
    }
    double sign = (lambda > 0.0) ? -1.0 : 1.0;
    return (logS0 + r * t + sign * 0.5 * total_var) - std::log(K);
  }
  Complex val = std::exp(-v * std::log(K) * 1i) *
                charact_func_heston(v, Y0, a, b, s, logS0, r, rho, lambda, t) /
                (v * 1i);
  return val.real();
}

double integrand_abs_val(double v, double Y0, double a, double b, double s,
                         double logS0, double r, double rho, double lambda,
                         double t, double strike) {
  if (std::abs(v) < 1e-10)
    return 1.0;
  Complex val = std::exp(-v * std::log(strike) * 1i) *
                charact_func_heston(v, Y0, a, b, s, logS0, r, rho, lambda, t) /
                (v * 1i);
  return std::abs(val);
}

double find_upper_bound(std::function<double(double)> func,
                        double tolerance = 1e-9) {
  double v = 20.0;
  double multiplier = 1.5;
  for (int i = 0; i < 100; ++i) {
    if (func(v) < tolerance)
      return v;
    v *= multiplier;
  }
  return v;
}

static double adaptive_simpson_impl(const std::function<double(double)> &f,
                                    double a, double b, double eps, double S,
                                    double fa, double fb, double fc,
                                    unsigned depth) {
  double c = (a + b) / 2.0;
  double h = b - a;
  double d = (a + c) / 2.0;
  double e = (c + b) / 2.0;
  double fd = f(d);
  double fe = f(e);
  double Sleft = (h / 12.0) * (fa + 4.0 * fd + fc);
  double Sright = (h / 12.0) * (fc + 4.0 * fe + fb);
  double S2 = Sleft + Sright;
  if (std::isnan(S2) || std::isinf(S2))
    return std::numeric_limits<double>::quiet_NaN();
  if (depth <= 0 || std::abs(S2 - S) <= 15.0 * eps)
    return S2 + (S2 - S) / 15.0;
  return adaptive_simpson_impl(f, a, c, eps / 2.0, Sleft, fa, fc, fd,
                               depth - 1) +
         adaptive_simpson_impl(f, c, b, eps / 2.0, Sright, fc, fb, fe,
                               depth - 1);
}
double adaptive_simpson(const std::function<double(double)> &f, double a,
                        double b, double eps, unsigned max_depth) {
  double c = (a + b) / 2.0;
  double h = b - a;
  double fa = f(a);
  double fb = f(b);
  double fc = f(c);
  double S = (h / 6.0) * (fa + 4.0 * fc + fb);
  return adaptive_simpson_impl(f, a, b, eps, S, fa, fb, fc, max_depth);
}

// -----------------------------------------------------------------------------
// UPDATED LEGACY MAIN FUNCTION: NOW WITH PARALLELISM
// -----------------------------------------------------------------------------
std::pair<double, double> EU_heston_delta(double Y0, double a, double b,
                                          double s, double S0, double r,
                                          double div, double rho, double T,
                                          double strike, bool is_call) {
  double logS0 = std::log(S0);

  // 1. Bound Search (Serial)
  auto func_Envelope = [&](double v) {
    return integrand_abs_val(v, Y0, a, b, s, logS0, r - div, rho, 0.5, T,
                             strike);
  };
  double right_limit = find_upper_bound(func_Envelope);

  // 2. Define Integrands (Thread-safe lambdas capture by value/ref const)
  auto func_P1 = [&](double v) {
    return integrand_heston(v, Y0, a, b - rho * s, s, logS0, r - div, rho, -0.5,
                            T, strike);
  };
  auto func_P2 = [&](double v) {
    return integrand_heston(v, Y0, a, b, s, logS0, r - div, rho, 0.5, T,
                            strike);
  };

  // 3. Chunking Logic
  double moneyness = std::abs(std::log(S0 / strike));
  double pre_chunk_size =
      (moneyness < 0.01) ? right_limit
                         : std::max(5.0, std::min(6.28318 / moneyness, 100.0));
  unsigned num_threads = omp_get_max_threads();
  unsigned target_chunks = std::max(8u, num_threads * 32u);
  double parallel_chunk_size = right_limit / target_chunks;

  // C. Winner Takes All (Smallest constraint wins)
  double chunk_size = std::min(pre_chunk_size, parallel_chunk_size);

  // Prepare Chunks Vector
  std::vector<std::pair<double, double>> chunks;
  double current = 0.0;
  while (current < right_limit) {
    double next = std::min(current + chunk_size, right_limit);
    chunks.push_back({current, next});
    current = next;
    chunk_size *= 1.4;
  }

  // 4. Parallel Integration
  double int1 = 0.0;
  double int2 = 0.0;

  // Linear scaling of tolerance (conservative assumption)
  double eps = 1e-9 / (double)chunks.size();

#pragma omp parallel for reduction(+ : int1, int2) schedule(dynamic, 1)
  for (size_t i = 0; i < chunks.size(); ++i) {
    int1 +=
        adaptive_simpson(func_P1, chunks[i].first, chunks[i].second, eps, 25);
    int2 +=
        adaptive_simpson(func_P2, chunks[i].first, chunks[i].second, eps, 25);
  }

  const double PI = 3.14159265358979323846;
  double P1 = 0.5 + int1 / PI;
  double P2 = 0.5 + int2 / PI;
  double discount_div = std::exp(-div * T);
  double discount_r = std::exp(-r * T);
  double delta = discount_div * P1;
  double call = S0 * delta - strike * discount_r * P2;
  return {call, delta};
}
} // namespace Legacy

// =============================================================================
// MAIN BENCHMARK WITH COMPREHENSIVE TEST SUITE
// =============================================================================

int main() {
  std::cout << "============================================================="
               "=====================\n";
  std::cout << "  HESTON FOURIER PRICER BENCHMARK: NEW ENGINE vs LEGACY "
               "(PARALLEL)\n";
  std::cout << "  Scenario Suite: Comprehensive Stress Test\n";
  std::cout << "============================================================="
               "=====================\n\n";

  // --- CONFIGURATION ---
  struct TestCase {
    std::string name;
    double S0, T, r, v0, kappa, theta, sigma, rho;
  };

  // Define Stress Scenarios (from provided snippet)
  double h_vol = 3.0, m_vol = 0.3;
  double h_rho = -0.9, m_rho = -0.7, l_rho = -0.5;
  double h_kappa = 2.0, m_kappa = 1.0;
  double s_T = 0.08, m_T = 1.0, l_T = 3.0, vl_T = 5.0;

  std::vector<TestCase> tests = {
      {"Baseline", 100.0, m_T, 0.05, 0.09, m_kappa, 0.09, m_vol, l_rho},
      {"Short Mat 1M", 100.0, s_T, 0.05, 0.09, m_kappa, 0.09, m_vol, l_rho},
      {"Extr Short -0.9", 100.0, s_T, 0.05, 0.09, h_kappa, 0.09, h_vol, h_rho},
      {"Extr Med -0.9", 100.0, m_T, 0.05, 0.09, h_kappa, 0.09, h_vol, h_rho},
      {"Extr Long -0.9", 100.0, l_T, 0.05, 0.09, h_kappa, 0.09, h_vol, h_rho},
      {"Extr V.Long -0.9", 100.0, vl_T, 0.05, 0.09, h_kappa, 0.09, h_vol,
       h_rho}};

  struct Moneyness {
    std::string label;
    double mult;
  };
  std::vector<Moneyness> moneys = {{"ITM", 0.9}, {"ATM", 1.0}, {"OTM", 1.1}};

  // Output Vectors for Table
  std::vector<std::string> t_scenario, t_moneyness;
  std::vector<double> t_price_legacy, t_price_new, t_diff;
  std::vector<double> t_time_legacy, t_time_new, t_speedup;

  // Config for New Engine
  FourierEngine::Config cfg;
  cfg.tolerance = 1e-9;
  cfg.start_bound_guess = 20.0;

  constexpr int ITERATIONS = 1000;
  volatile double sink_legacy = 0;
  volatile double sink_new = 0;

  // --- RUN TESTS ---
  for (const auto &t : tests) {

    // 1. Setup Model Parameters
    // New Engine Params
    HestonModel::Parameters p;
    p.kappa = t.kappa;
    p.theta = t.theta;
    p.sigma = t.sigma;
    p.rho = t.rho;
    p.v0 = t.v0;
    HestonModel new_model(p);

    // Legacy Params (Mapping)
    double param_a = t.kappa * t.theta;
    double param_b = t.kappa;
    double param_Y0 = t.v0;
    double param_s = t.sigma;
    double q = 0.0; // Assuming q=0 as per snippet

    for (const auto &m : moneys) {
      double K = t.S0 * m.mult;

      // --- A. BENCHMARK LEGACY ---
      double legacy_price = 0.0;
      double legacy_ns = 0.0;
      {
        // Warmup
        Legacy::EU_heston_delta(param_Y0, param_a, param_b, param_s, t.S0, t.r,
                                q, t.rho, t.T, K, true);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
          auto res =
              Legacy::EU_heston_delta(param_Y0, param_a, param_b, param_s, t.S0,
                                      t.r, q, t.rho, t.T, K, true);
          sink_legacy += res.first;
          if (i == 0)
            legacy_price = res.first;
        }
        auto end = std::chrono::high_resolution_clock::now();
        legacy_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count() /
            (double)ITERATIONS;
      }

      // --- B. BENCHMARK NEW ENGINE ---
      double new_price = 0.0;
      double new_ns = 0.0;
      {
        // Warmup
        price_european_fourier(new_model, t.S0, K, t.T, t.r, q, true, cfg);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
          double price = price_european_fourier(new_model, t.S0, K, t.T, t.r, q,
                                                true, cfg);
          sink_new += price;
          if (i == 0)
            new_price = price;
        }
        auto end = std::chrono::high_resolution_clock::now();
        new_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count() /
            (double)ITERATIONS;
      }

      // Store Results
      t_scenario.push_back(t.name);
      t_moneyness.push_back(m.label);
      t_price_legacy.push_back(legacy_price);
      t_price_new.push_back(new_price);
      t_diff.push_back(std::abs(legacy_price - new_price));
      t_time_legacy.push_back(legacy_ns);
      t_time_new.push_back(new_ns);
      t_speedup.push_back(legacy_ns / new_ns);
    }
  }

  // --- PRINT RESULTS TABLE ---
  utils::printVectors({"Scenario", "Moneyness", "Legacy Price", "New Price",
                       "Diff", "Legacy(ns)", "New(ns)", "Speedup"},
                      t_scenario, t_moneyness, t_price_legacy, t_price_new,
                      t_diff, t_time_legacy, t_time_new, t_speedup);

  std::cout << "\nNotes:\n";
  std::cout << " - Speedup > 1.0 means New Engine is faster.\n";
  std::cout << " - Diff should be negligible (< 1e-7).\n";
  std::cout << " - Sink values (ignore): " << sink_legacy << ", " << sink_new
            << "\n";

  return 0;
}