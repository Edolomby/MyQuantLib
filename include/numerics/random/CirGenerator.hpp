#pragma once
#include "numerics/random/Distributions.hpp" // For normcdfinv_as241
#include <array>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <cmath>

namespace numerics {
namespace random {

class FastCirGenerator {
private:
  // --- Pre-calculated CIR Constants ---
  double prm_lambda_factor; // prm[0]: 2*ex / (sigma^2 * psi)
  double prm_alpha_base;    // prm[1]: 2*a / sigma^2
  double prm_scale;         // 1/prm[2]: 0.5 * sigma^2 * psi

  // --- Optimized Stack Arrays for Gamma Generation ---
  // CACHE_SIZE = 32 handles 99.9% of steps where Poisson(lambda) is small.
  // std::array is stack-allocated, zero heap overhead.
  static const int CACHE_SIZE = 32;
  std::array<double, CACHE_SIZE> cached_d;
  std::array<double, CACHE_SIZE> cached_c;

  // --- Optimization for Alpha < 1 (only used if alpha_base < 1 and N=0) ---
  double small_alpha_p;           // p = e / (alpha + e)
  double small_alpha_inv_p;       // 1.0 / p (avoids division in loop)
  double small_alpha_one_minus_p; // 1.0 - p
  double one_minus_small_alpha;   // 1.0 - alpha (for hyperbolic calc)
  double small_inv_alpha;         // 1.0 / alpha

  // --- Boost Distribution Objects ---
  boost::random::poisson_distribution<int, double> pois_dist;

public:
  FastCirGenerator(double a, double b, double sigma, double h) {

    double ex = std::exp(-b * h);
    double psi;

    if (std::abs(b) < 1e-9)
      psi = h;
    else
      psi = (1.0 - ex) / b;

    double sigma_sq = sigma * sigma;

    // --- 1. Your Corrected Constants ---
    prm_lambda_factor = 2.0 * ex / (sigma_sq * psi);
    prm_alpha_base = 2.0 * a / sigma_sq;
    prm_scale = 0.5 * sigma_sq * psi;

    // --- 2. Pre-compute Gamma Constants (Stack Arrays) ---
    // If alpha_base is large, we might not need small_alpha logic,
    // but we always precompute d/c for the cached range.
    bool alpha_base_is_small = (prm_alpha_base <= 1.0);

    for (int k = 0; k < CACHE_SIZE; ++k) {
      double current_alpha = prm_alpha_base + k;

      // Only Marsaglia-Tsang needs d and c, requires alpha > 1
      if (current_alpha > 1.0) {
        double d = current_alpha - 1.0 / 3.0;
        cached_d[k] = d;
        cached_c[k] = 1.0 / std::sqrt(9.0 * d);
      }
    }

    // --- 3. Precompute constants for alpha <= 1 case ---
    // This is only used if N=0 (so alpha == alpha_base) AND alpha_base <= 1
    if (alpha_base_is_small) {
      const double e = 2.71828182845904523536;
      // Standard definition: p = e / (alpha + e)
      small_alpha_p = e / (prm_alpha_base + e);

      // Your requested optimizations:
      small_alpha_inv_p = 1.0 / small_alpha_p;
      small_alpha_one_minus_p = 1.0 - small_alpha_p;
      one_minus_small_alpha = 1.0 - prm_alpha_base;
      small_inv_alpha = 1.0 / prm_alpha_base;
    }
  }

  template <class Generator>
  double next(double X_t, Generator &gen,
              boost::random::uniform_real_distribution<double> &Uni,
              boost::random::exponential_distribution<double> &Exp) {

    // 1. Generate Poisson Number (N)
    double lambda = prm_lambda_factor * X_t;
    int N = pois_dist(
        gen,
        boost::random::poisson_distribution<int, double>::param_type(lambda));

    // 2. Generate Gamma
    double alpha = prm_alpha_base + N;
    double gamma_val;

    // --- BRANCH A: Alpha > 1 (Marsaglia-Tsang) ---
    if (alpha > 1.0) {
      double d, c;

      // Fast Path: Retrieve from stack cache
      if (N < CACHE_SIZE) {
        d = cached_d[N];
        c = cached_c[N];
      } else {
        // Slow Path: Calculate manually
        d = alpha - 1.0 / 3.0;
        c = 1.0 / std::sqrt(9.0 * d);
      }

      double v, G1, U1;
      for (;;) {
        do {
          G1 = numerics::random::normcdfinv_as241(Uni(gen));
          v = 1.0 + c * G1;
        } while (v <= 0.0);

        v = v * v * v;
        U1 = Uni(gen);

        if (U1 < 1.0 - 0.0331 * G1 * G1 * G1 * G1) {
          gamma_val = d * v;
          break;
        }

        if (std::log(U1) < 0.5 * G1 * G1 + d * (1.0 - v + std::log(v))) {
          gamma_val = d * v;
          break;
        }
      }
    }
    // --- BRANCH B: Alpha <= 1 (Rejection Sampling) ---
    else {
      // Optimization: We use precomputed divisions/subtractions
      for (;;) {
        double u = Uni(gen);
        double y = Exp(gen);
        double x;

        if (u < small_alpha_p) {
          // Case 1: Sample from lower head
          x = std::exp(-y * small_inv_alpha);

          // Optimized check: u < p * (1-x)  =>  u/p < 1-x
          if (u * small_alpha_inv_p < 1.0 - x) {
            gamma_val = x;
            break;
          }
          // q = p * exp(-x);
          if (u < small_alpha_p * std::exp(-x)) {
            gamma_val = x;
            break;
          }

        } else {
          // Case 2: Sample from upper tail
          x = 1.0 + y;

          // Optimized: 1.0 / (1.0 + (1.0 - alpha) * y)
          double inv_hyperbolic = 1.0 / (1.0 + one_minus_small_alpha * y);

          // Optimized: u < p + (1-p) * inv_hyp
          if (u < small_alpha_p + small_alpha_one_minus_p * inv_hyperbolic) {
            gamma_val = x;
            break;
          }

          // q = p + (1-p) * x^(a-1)
          if (u < small_alpha_p + small_alpha_one_minus_p *
                                      std::pow(x, -one_minus_small_alpha)) {
            gamma_val = x;
            break;
          }
        }
      }
    }

    // 3. Return Scaled
    return gamma_val * prm_scale;
  }
};

} // namespace random
} // namespace numerics