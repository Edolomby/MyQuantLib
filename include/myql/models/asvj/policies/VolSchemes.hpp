#pragma once
#include <array>
#include <cmath>
#include <myql/math/Numerics.hpp> // For normcdfinv_as241

// We need boost for the internal distributions used in HighVol
// (Poisson, Gamma logic)
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

// =============================================================================
// 1. SCHEME: LOW VOLATILITY (Ninomiya-Victoir)
// =============================================================================
struct SchemeLowVol {

  struct Workspace {};

  // Prepare is a no-op for LowVol (constants calculated in Stepper constructor)
  template <typename HestonParams>
  static void prepare(Workspace &, const HestonParams &) {}

  // The Evolve Function (Static, Inlined)
  template <typename RNG>
  static inline double
  evolve(double v_curr, const std::array<double, 8> &C, // Passed from Stepper
         double Z_V,                                    // Indip. Gaussian
         RNG &,                                         // RNG unused here
         const Workspace &) {
    // C[0]=ex, C[1]=D*psi, C[2]=s_star
    double root_term = std::sqrt(C[0] * v_curr + C[1]) + C[2] * Z_V;
    return C[0] * (root_term * root_term) + C[1];
  }
};

// =============================================================================
// 2. SCHEME: HIGH VOLATILITY (Exact / Gamma Mixture)
// =============================================================================
struct SchemeHighVol {

  // -------------------------------------------------------------------------
  // The Workspace (Gamma Cache)
  // -------------------------------------------------------------------------
  struct Workspace {
    static constexpr int CACHE_SIZE = 64;

    // CORRECTION: Use std::array instead of C-style arrays
    std::array<double, CACHE_SIZE> d;
    std::array<double, CACHE_SIZE> c;

    // Optimization variables for Alpha <= 1
    bool alpha_small;
    double sm_p, sm_inv_p, sm_1_minus_p, sm_1_minus_alpha, sm_inv_alpha;
    // Poisson distribution
    mutable boost::random::poisson_distribution<int, double> pois_dist;
    mutable boost::random::uniform_real_distribution<double> Uni;
    mutable boost::random::exponential_distribution<double> Exp;
  };

  // -------------------------------------------------------------------------
  // Preparation (Run once in Constructor)
  // -------------------------------------------------------------------------
  template <typename HestonParams>
  static void prepare(Workspace &w, const HestonParams &p) {
    double alpha_base = (2.0 * p.kappa * p.theta) / (p.sigma * p.sigma);

    // 1. Pre-compute Gamma Constants
    for (int k = 0; k < Workspace::CACHE_SIZE; ++k) {
      double alpha = alpha_base + k;
      if (alpha > 1.0) {
        double d_val = alpha - (1.0 / 3.0);
        w.d[k] = d_val;
        w.c[k] = 1.0 / std::sqrt(9.0 * d_val);
      }
    }

    // 2. Pre-compute constants for small alpha
    w.alpha_small = (alpha_base <= 1.0);
    if (w.alpha_small) {
      const double e = 2.71828182845904523536;
      w.sm_p = e / (alpha_base + e);
      w.sm_inv_p = 1.0 / w.sm_p;
      w.sm_1_minus_p = 1.0 - w.sm_p;
      w.sm_1_minus_alpha = 1.0 - alpha_base;
      w.sm_inv_alpha = 1.0 / alpha_base;
    }
  }

  // Optimized Gamma Generator
  template <typename RNG>
  static inline double generate_gamma(double alpha_base, int N, RNG &rng,
                                      const Workspace &w) {
    double alpha = alpha_base + N;
    double x_gamma = 0.0;

    // --- BRANCH A: Alpha > 1 (Marsaglia-Tsang) ---
    if (alpha > 1.0) {
      double d, c, U, Z, v;

      if (N < Workspace::CACHE_SIZE) {
        d = w.d[N];
        c = w.c[N];
      } else {
        d = alpha - (1.0 / 3.0);
        c = 1.0 / std::sqrt(9.0 * d);
      }
      for (;;) {
        do {
          Z = Numerics::normcdfinv_as241(w.Uni(rng));
          v = 1.0 + c * Z;
        } while (v <= 0.0);

        v = v * v * v;
        U = w.Uni(rng);
        // squeeze
        if (U < 1.0 - 0.0331 * Z * Z * Z * Z) {
          x_gamma = d * v;
          break;
        }
        if (std::log(U) < 0.5 * Z * Z + d * (1.0 - v + std::log(v))) {
          x_gamma = d * v;
          break;
        }
      }
    }

    // --- BRANCH B: Alpha <= 1 (Rejection) ---
    // Only used if alpha_base <= 1 AND N=0
    else {
      double u, y, x;
      for (;;) {
        u = w.Uni(rng);
        y = w.Exp(rng);

        if (u < w.sm_p) {
          x = std::exp(-y * w.sm_inv_alpha);
          if (u * w.sm_inv_p < 1.0 - x) {
            x_gamma = x;
            break;
          }
          if (u < w.sm_p * std::exp(-x)) {
            x_gamma = x;
            break;
          }
        } else {
          x = 1.0 + y;
          double inv_hyperbolic = 1.0 / (1.0 + w.sm_1_minus_alpha * y);

          // squeeze
          if (u < w.sm_p + w.sm_1_minus_p * inv_hyperbolic) {
            x_gamma = x;
            break;
          }

          // q = p + (1-p) * x^(a-1)
          if (u < w.sm_p + w.sm_1_minus_p * std::pow(x, -w.sm_1_minus_alpha)) {
            x_gamma = x;
            break;
          }
        }
      }
    }
    return x_gamma;
  }

  // 4. The Evolve Function
  template <typename RNG>
  static inline double
  evolve(double v_curr,
         const std::array<double, 8> &C, // Use std::array reference
         double, RNG &rng, const Workspace &w) {
    // 1. Poisson N
    // C[0] is lambda_factor
    double lambda = C[0] * v_curr;
    using PoisParam =
        boost::random::poisson_distribution<int, double>::param_type;
    int N = w.pois_dist(rng, PoisParam(lambda));

    // 2. Gamma
    // C[1] is alpha_base, C[2] is inv_scale
    double gamma_val = generate_gamma(C[1], N, rng, w);

    return gamma_val * C[2];
  }
};
