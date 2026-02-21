#pragma once
#include <cmath>
#include <myql/math/Integration.hpp> // Your adaptive_simpson
#include <omp.h>
#include <vector>

struct FourierConfig {
  double tolerance = 1e-8;
  double start_bound_guess = 20.0;
};

class FourierEngine {
public:
  using Config = FourierConfig;
  FourierEngine(Config cfg = Config()) : cfg_(cfg) {}

  // -------------------------------------------------------------------------
  // The Core Method: Integrate any function from 0 to Infinity
  // -------------------------------------------------------------------------
  // IntegrandConcept must have:
  // 1. double operator()(double xi) const;  -> The real part of the integrand
  // 2. double magnitude(double xi) const;   -> The envelope for bound finding
  // -------------------------------------------------------------------------
  template <typename Integrand>
  double calculate_integral(const Integrand &func, double residual_T) const {

    // Handle Singularity at 0 (Head)
    constexpr double LOW_BOUND = 1e-8;
    double total_integral = LOW_BOUND * func(LOW_BOUND);

    // Find Upper Bound
    double right_limit = find_upper_bound(func, residual_T);
    double interval_width = right_limit - LOW_BOUND;

    // SMART GEOMETRIC GRID
    // A. Determine Target N (Load Balancing)
    // 4 tasks per thread ensures stragglers don't block the queue.
    int num_threads = omp_get_max_threads();
    int N = std::max(8, num_threads * 4);

    // B. Geometric Multiplier
    const double multiplier = 1.08; // seems a good choice

    // C. Calculate Start Step (Exact Formula)
    // a = L * (r - 1) / (r^N - 1)
    double current_step =
        interval_width * (multiplier - 1.0) / (std::pow(multiplier, N) - 1.0);

    // D. Generate Chunks
    std::vector<std::pair<double, double>> chunks;
    chunks.reserve(N);

    double current_u = LOW_BOUND;

    for (int i = 0; i < N; ++i) {
      double next_u = std::min(current_u + current_step, right_limit);

      if (next_u <= current_u)
        break;

      chunks.push_back({current_u, next_u});
      current_u = next_u;

      // Break if we hit the limit (floating point safety)
      if (std::abs(current_u - right_limit) < 1e-9)
        break;

      current_step *= multiplier;
    }

    // Edge case: If floating point math left a tiny gap at the end
    if (current_u < right_limit - 1e-9) {
      chunks.push_back({current_u, right_limit});
    }

    // Parallel Integration - linear tolerance scaling
    double chunk_tol = (chunks.empty())
                           ? cfg_.tolerance
                           : cfg_.tolerance / (double)chunks.size();

#pragma omp parallel for reduction(+ : total_integral) schedule(dynamic, 1)
    for (size_t i = 0; i < chunks.size(); ++i) {
      total_integral += numerics::adaptive_simpson(
          func, chunks[i].first, chunks[i].second, chunk_tol, 25);
    }

    return total_integral;
  }

private:
  Config cfg_;

  // Helper: Your specific upper bound finder logic
  template <typename Integrand>
  double find_upper_bound(const Integrand &func, double T,
                          const double tol_sq = 1e-24,
                          const double multiplier = 1.5) const {
    double v = cfg_.start_bound_guess / (T < 1.0 ? std::sqrt(T) : 1.0);
    unsigned int safety = 0;

    // Simple search heuristic
    while (safety++ < 20) {
      if (func.magnitude_sq(v) < tol_sq)
        return v;
      v *= multiplier;
    }
    return v; // Fallback
  }
};