#pragma once
#include <array>
#include <cmath>
#include <myql/math/Numerics.hpp> // For normcdfinv_as241, poissinvcdf
#include <myql/math/interpolation/splines.hpp>
#include <myql/models/asvj/data/ModelParams.hpp>

// We need boost for the internal distributions used in HighVol
// (Poisson, Gamma logic)
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

// =============================================================================
// SCHEME: LOW VOLATILITY (Ninomiya-Victoir) only for sigma^2 <= 4*kappa*theta
// =============================================================================

struct NullVolScheme {
  struct GlobalWorkspace {};
  struct Workspace {};
};

struct SchemeNV {

  struct GlobalWorkspace {};

  static GlobalWorkspace build_global_workspace(const HestonParams &, double) {
    return {};
  }

  struct Workspace {
    mutable boost::random::uniform_real_distribution<double> Uni;
    std::array<double, 3> C; // discr_params ex, D*psi, s_star
  };

  template <typename HestonParams>
  static void prepare(Workspace &w, const GlobalWorkspace &,
                      const HestonParams &p, double time_step) {
    if ((p.sigma * p.sigma) > (4.0 * p.kappa * p.theta))
      throw std::runtime_error("SchemeNV violation: sigma^2 > 4*kappa*theta");
    double ex = std::exp(-p.kappa * time_step * 0.5);
    double psi =
        (std::abs(p.kappa) < 1e-10) ? time_step * 0.5 : (1.0 - ex) / p.kappa;
    double D = (p.kappa * p.theta) - (p.sigma * p.sigma * 0.25);
    w.C[0] = ex;
    w.C[1] = D * psi;
    w.C[2] = p.sigma * std::sqrt(time_step) * 0.5;
  }

  template <typename RNG>
  static inline double evolve(double v_curr, RNG &rng, const Workspace &w) {
    // C[0]=ex, C[1]=D*psi, C[2]=s_star
    double Z_V = Numerics::normcdfinv_as241(w.Uni(rng));
    double root_term = std::sqrt(w.C[0] * v_curr + w.C[1]) + w.C[2] * Z_V;
    return w.C[0] * (root_term * root_term) + w.C[1];
  }
};

// =============================================================================
// SCHEME: HIGH VOLATILITY (Exact / Gamma Mixture)
// =============================================================================
struct SchemeExact {

  // Global Workspace: Holds the pre-computed constants
  struct GlobalWorkspace {
    static constexpr int CACHE_SIZE = 128;
    std::array<double, CACHE_SIZE> d;
    std::array<double, CACHE_SIZE> c;
    bool alpha_small;
    double sm_p, sm_inv_p, sm_1_minus_p, sm_1_minus_alpha, sm_inv_alpha;
  };

  static GlobalWorkspace build_global_workspace(const HestonParams &p, double) {
    GlobalWorkspace gw;
    double alpha_base = (2.0 * p.kappa * p.theta) / (p.sigma * p.sigma);

    for (int k = 0; k < GlobalWorkspace::CACHE_SIZE; ++k) {
      double alpha = alpha_base + k;
      if (alpha > 1.0) {
        double d_val = alpha - (1.0 / 3.0);
        gw.d[k] = d_val;
        gw.c[k] = 1.0 / std::sqrt(9.0 * d_val);
      }
    }

    gw.alpha_small = (alpha_base <= 1.0);
    if (gw.alpha_small) {
      const double e = 2.71828182845904523536;
      gw.sm_p = e / (alpha_base + e);
      gw.sm_inv_p = 1.0 / gw.sm_p;
      gw.sm_1_minus_p = 1.0 - gw.sm_p;
      gw.sm_1_minus_alpha = 1.0 - alpha_base;
      gw.sm_inv_alpha = 1.0 / alpha_base;
    }
    return gw;
  }

  struct Workspace {
    const GlobalWorkspace *global = nullptr;
    std::array<double, 3> C;
    mutable boost::random::uniform_real_distribution<double> Uni;
    mutable boost::random::exponential_distribution<double> Exp;
  };

  // -------------------------------------------------------------------------
  // Preparation (Run once in Constructor)
  // -------------------------------------------------------------------------
  template <typename HestonParams>
  static void prepare(Workspace &w, const GlobalWorkspace &gw,
                      const HestonParams &p, double time_step) {
    w.global = &gw; // Just an 8-byte pointer assignment!
    double ex = std::exp(-p.kappa * time_step);
    double psi = (std::abs(p.kappa) < 1e-10) ? time_step : (1.0 - ex) / p.kappa;
    w.C[0] = (2.0 / (p.sigma * p.sigma * psi)) * ex;
    w.C[1] = (2.0 * p.kappa * p.theta) / (p.sigma * p.sigma);
    w.C[2] = 0.5 * (p.sigma * p.sigma * psi);
  }
  // Gamma Generator
  template <typename RNG>
  static inline double generate_gamma(double alpha_base, int N, RNG &rng,
                                      const Workspace &w) {
    double alpha = alpha_base + N;
    double x_gamma = 0.0;
    const GlobalWorkspace *g = w.global;

    // --- BRANCH A: Alpha > 1 (Marsaglia-Tsang) ---
    if (alpha > 1.0) {
      double d, c, U, Z, v;

      if (N < GlobalWorkspace::CACHE_SIZE) {
        d = g->d[N];
        c = g->c[N];
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

        if (u < g->sm_p) {
          x = std::exp(-y * g->sm_inv_alpha);
          if (u * g->sm_inv_p < 1.0 - x) {
            x_gamma = x;
            break;
          }
          if (u < g->sm_p * std::exp(-x)) {
            x_gamma = x;
            break;
          }
        } else {
          x = 1.0 + y;
          double inv_hyperbolic = 1.0 / (1.0 + g->sm_1_minus_alpha * y);

          // squeeze
          if (u < g->sm_p + g->sm_1_minus_p * inv_hyperbolic) {
            x_gamma = x;
            break;
          }

          // q = p + (1-p) * x^(a-1)
          if (u <
              g->sm_p + g->sm_1_minus_p * std::pow(x, -g->sm_1_minus_alpha)) {
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
  static inline double evolve(double v_curr, RNG &rng, const Workspace &w) {
    // 1. Poisson N
    // C[0] is lambda_factor
    double lambda = w.C[0] * v_curr;
    int N = Numerics::poissinvcdf(lambda, w.Uni(rng));

    // 2. Gamma
    // C[1] is alpha_base, C[2] is inv_scale
    double gamma_val = generate_gamma(w.C[1], N, rng, w);

    return gamma_val * w.C[2];
  }
};

// =============================================================================
// SCHEME: NCI (Non-Central Chi-Squared Inversion)
// =============================================================================
struct SchemeNCI {
  // 1. Global Workspace: The heavy lifting
  struct GlobalWorkspace {
    std::vector<double> y_data;
    std::vector<double> m_data;
    size_t n_max = 0;
    size_t grid_size = 1024;
    double dx = 0.0, inv_dx = 0.0;
  };

  static GlobalWorkspace build_global_workspace(const HestonParams &p,
                                                double dt) {
    GlobalWorkspace gw;
    double d = (4.0 * p.kappa * p.theta) / (p.sigma * p.sigma);

    double v_max = std::max(p.v0, 3.0 * p.theta); // could take a larger factor
    double ex = std::exp(-p.kappa * dt);
    double psi = (std::abs(p.kappa) < 1e-10) ? dt : (1.0 - ex) / p.kappa;
    double lambda_max = (2.0 * v_max * ex) / (p.sigma * p.sigma * psi);

    // 4 * sqrt(lambda_max) conservative considering the 3 * p.theta
    gw.n_max = std::ceil(lambda_max + 4.0 * std::sqrt(lambda_max));
    if (gw.n_max < 23)
      gw.n_max = 23;
    if (gw.n_max > 1023) // before was capped at 127
      gw.n_max = 1023;

    gw.grid_size = 10000; // before was 1024
    gw.dx = 0.999 / (gw.grid_size - 1);
    gw.inv_dx = 1.0 / gw.dx;

    size_t total_elements = (gw.n_max + 1) * gw.grid_size;
    gw.y_data.assign(total_elements, 0.0);
    gw.m_data.assign(total_elements, 0.0);

    // PARALLEL SPLINE GENERATION
#pragma omp parallel for schedule(dynamic)
    for (size_t k = 0; k <= gw.n_max; ++k) {
      double dof = d + 2.0 * static_cast<double>(k); // degrees of freedom
      boost::math::chi_squared_distribution<double> dist(dof);
      size_t offset = k * gw.grid_size;

      // Fill Y-data (Quantiles)
      for (size_t i = 0; i < gw.grid_size; ++i) {
        // Print the exact state right BEFORE Boost attempts the calculation
        /* #pragma omp critical
                {
                  std::cout << "Attempting -> k: " << k << " | dof: " << dof
                            << " | index i: " << i << " | target U: " << i *
           gw.dx
                            << std::endl;
                } */
        gw.y_data[offset + i] = boost::math::quantile(dist, i * gw.dx);
        /* #pragma omp critical
                {
                  std::cout << "Result -> k: " << k << " | dof: " << dof
                            << " | index i: " << i
                            << " | y_data: " << gw.y_data[offset + i] <<
           std::endl;
                } */
      }
      // Fill M-data (Derivatives for Hermite Spline)
      for (size_t i = 0; i < gw.grid_size; ++i) {
        if (i == 0) {
          gw.m_data[offset + i] =
              (gw.y_data[offset + 1] - gw.y_data[offset]) * gw.inv_dx;
        } else if (i == gw.grid_size - 1) {
          gw.m_data[offset + i] =
              (gw.y_data[offset + i] - gw.y_data[offset + i - 1]) * gw.inv_dx;
        } else {
          double d1 =
              (gw.y_data[offset + i] - gw.y_data[offset + i - 1]) * gw.inv_dx;
          double d2 =
              (gw.y_data[offset + i + 1] - gw.y_data[offset + i]) * gw.inv_dx;
          if (d1 * d2 > 0.0) {
            gw.m_data[offset + i] = 2.0 * d1 * d2 / (d1 + d2);
          } else {
            gw.m_data[offset + i] = 0.0;
          }
        }
      }
    } // End of OpenMP parallel region
    return gw;
  }

  // 2. Local Workspace: Just a pointer and an RNG!
  struct Workspace {
    const GlobalWorkspace *global = nullptr;
    std::array<double, 3> C;
    mutable boost::random::uniform_real_distribution<double> Uni{0.0, 1.0};
  };

  template <typename HestonParams>
  static void prepare(Workspace &w, const GlobalWorkspace &gw,
                      const HestonParams &p, double time_step) {
    w.global = &gw; // Zero allocation pointer assignment
    double ex = std::exp(-p.kappa * time_step);
    double psi = (std::abs(p.kappa) < 1e-10) ? time_step : (1.0 - ex) / p.kappa;
    w.C[0] = (2.0 / (p.sigma * p.sigma * psi)) * ex;
    w.C[1] = (4.0 * p.kappa * p.theta) / (p.sigma * p.sigma); // d
    w.C[2] = 0.25 * p.sigma * p.sigma * psi;
  }

  template <typename RNG>
  static inline double evolve(double v_curr, RNG &rng, const Workspace &w) {
    double lambda = w.C[0] * v_curr;
    int N = static_cast<int>(Numerics::poissinvcdf(lambda, w.Uni(rng)));
    // if (N < 0) N = 0; //assured by poissinvcdf

    double U = w.Uni(rng);
    const GlobalWorkspace *g = w.global;

    if (__builtin_expect(static_cast<size_t>(N) <= g->n_max && U < 0.999, 1)) {
      double pos = U * g->inv_dx;
      size_t i = static_cast<size_t>(pos);

      //
      if (__builtin_expect(i >= g->grid_size - 1, 0))
        i = g->grid_size - 2;

      double t = pos - i;
      size_t offset = N * g->grid_size + i;

      double y0 = g->y_data[offset], y1 = g->y_data[offset + 1];
      double m0 = g->m_data[offset], m1 = g->m_data[offset + 1];

      double t2 = t * t, t3 = t2 * t;
      return ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * g->dx * m0 +
              (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * g->dx * m1) *
             w.C[2];
    }
    /* #pragma omp critical
        {
          std::cout << "[CRASH WATCH] Fallback hit! N=" << N
                    << " | dof=" << (w.C[1] + 2.0 * N) << " | U=" << U
                    << " | v_curr=" << v_curr << " | C[0]=" << w.C[0]
                    << " | C[1]=" << w.C[1] << std::endl;
        } */
    boost::math::chi_squared_distribution<double> dist(w.C[1] + 2.0 * N);
    return boost::math::quantile(dist, U) * w.C[2];
  }
};