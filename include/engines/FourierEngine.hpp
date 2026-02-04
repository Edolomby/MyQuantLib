#pragma once
#include "numerics/Integration.hpp" // Your adaptive_simpson
#include <cmath>
#include <functional>
#include <iostream>
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
  double calculate_integral(const Integrand &func, double osc_freq_hint,
                            double residual_T) const {

    // 1. Find Upper Bound (Truncation)
    double right_limit = find_upper_bound(func, residual_T);

    // 2. Determine Chunk Size (Dual Constraint Strategy)
    // A. Oscillation Constraint (Capture the waves)
    // If moneyness is 0 (ATM), we set it to right_limit (no oscillation).
    // Otherwise, 2*PI / freq is one full wave.
    constexpr double TWO_PI = 2.0 * std::numbers::pi_v<double>;
    double osc_chunk_size =
        (osc_freq_hint < 0.01) ? right_limit : (TWO_PI / osc_freq_hint);

    // B. Load Balancing Constraint (Keep all cores busy)
    // Heuristic: Create at least 4 tasks per thread.
    // If one task is slow, free threads pick up the slack.
    unsigned num_threads = omp_get_max_threads();
    unsigned target_chunks = std::max(8u, num_threads * 4u);
    double parallel_chunk_size = right_limit / target_chunks;

    // C. Winner Takes All (Smallest constraint wins)
    double chunk_size = std::min(osc_chunk_size, parallel_chunk_size);

    // some debug prints
    /* std::cout << "Right Limit: " << right_limit << std::endl;
    std::cout << "Osc Chunk Size: " << osc_chunk_size << std::endl;
    std::cout << "Parallel Chunk Size: " << parallel_chunk_size << std::endl;
    std::cout << "Chunk Size: " << chunk_size << std::endl; */

    // D. Safety Floor (Avoid overhead from microscopic chunks)
    if (chunk_size < 1e-3)
      chunk_size = 1e-3;

    // 3. Generate Chunks
    std::vector<std::pair<double, double>> chunks;
    // Reserve memory to avoid reallocation
    chunks.reserve(static_cast<size_t>(right_limit / chunk_size) + 5);

    constexpr double LOW_BOUND = 1e-8; // avoid 0
    double current = LOW_BOUND;
    while (current < right_limit) {
      double next = std::min(current + chunk_size, right_limit);
      if (next <= current)
        break;
      chunks.push_back({current, next});
      current = next;
    }

    // 4. Parallel Integration (OpenMP)
    // right end point rule on the first small chunk
    double total_integral = LOW_BOUND * func(LOW_BOUND);

    // Scale tolerance per chunk to maintain total error budget
    double chunk_tol = (chunks.empty())
                           ? cfg_.tolerance
                           : cfg_.tolerance / (double)chunks.size();

#pragma omp parallel for reduction(+ : total_integral) schedule(dynamic, 1)
    for (size_t i = 0; i < chunks.size(); ++i) {
      // Your adaptive_simpson from Integration.hpp
      /* total_integral += numerics::adaptive_simpson(
          [&](double x) { return func(x); }, // Lambda wrapper if needed
          chunks[i].first, chunks[i].second, chunk_tol,
          25 // Max depth
      ); */
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