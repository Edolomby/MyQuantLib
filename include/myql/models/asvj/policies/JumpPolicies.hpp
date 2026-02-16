#pragma once
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <cmath>

#include <myql/math/Numerics.hpp> // For normcdfinv_as241
#include <myql/models/asvj/data/ModelParams.hpp>

// =============================================================================
// 1. NO JUMPS (Zero Cost)
// =============================================================================
struct NoJumps {
  using Params = NoJumpParams;

  struct Workspace {};

  static void prepare(Workspace &, const Params &, double) {}

  // Signature simplified: No 'dt' argument needed
  template <typename RNG>
  static inline double compute_log_jump(const Params &, RNG &, Workspace &) {
    return 0.0;
  }
};

// =============================================================================
// 2. MERTON JUMPS (Gaussian Jumps)
// =============================================================================
struct MertonJump {

  using Params = MertonParams;

  struct Workspace {
    // Persistent distributions
    boost::random::poisson_distribution<int, double> pois;
    boost::random::uniform_01<double> Uni;
    double drift_compensation;

    Workspace() : pois(1.0), drift_compensation(0.0) {}
  };

  // Called ONCE in the Constructor
  static void prepare(Workspace &w, const Params &p, double dt) {
    // 1. Setup Poisson for this specific time span dt
    w.pois = boost::random::poisson_distribution<int, double>(p.lambda * dt);

    // 2. Pre-calculate Drift Rate: lambda * (E[e^J] - 1)
    double k = std::exp(p.mu + 0.5 * p.delta * p.delta) - 1.0;
    w.drift_compensation = -p.lambda * k * dt;
  }

  template <typename RNG>
  static inline double compute_log_jump(const Params &p, RNG &rng,
                                        Workspace &w) {

    // 1. How many jumps? (Distribution already set for dt)
    int n_jumps = w.pois(rng);

    if (n_jumps == 0)
      return w.drift_compensation;

    // 2. Sum of N Normals -> Single Normal(N*mu, sqrt(N)*delta)
    double total_mu = n_jumps * p.mu;
    double total_delta = std::sqrt(static_cast<double>(n_jumps)) * p.delta;

    // 3. Generate Standard Normal using your fast Acklam algorithm
    double u = w.Uni(rng);
    double Z = Numerics::normcdfinv_as241(u);

    // Scale and shift
    return w.drift_compensation + total_mu + total_delta * Z;
  }
};

// =============================================================================
// 3. KOU JUMPS (Double Exponential)
// =============================================================================
struct KouJump {

  using Params = KouParams;

  struct Workspace {
    boost::random::poisson_distribution<int, double> pois;
    boost::random::uniform_01<double> Uni;
    boost::random::exponential_distribution<double> exp_up;
    boost::random::exponential_distribution<double> exp_down;
    double drift_compensation;

    Workspace()
        : pois(1.0), exp_up(1.0), exp_down(1.0), drift_compensation(0.0) {}
  };

  static void prepare(Workspace &w, const Params &p, double dt) {
    // Pre-calculate Poisson for fixed dt
    w.pois = boost::random::poisson_distribution<int, double>(p.lambda * dt);

    // Pre-calculate Exponentials
    w.exp_up = boost::random::exponential_distribution<double>(p.eta1);
    w.exp_down = boost::random::exponential_distribution<double>(p.eta2);

    double term_up = (p.p_up * p.eta1) / (p.eta1 - 1.0);
    double term_down = ((1.0 - p.p_up) * p.eta2) / (p.eta2 + 1.0);
    double k = term_up + term_down - 1.0;

    w.drift_compensation = -p.lambda * k * dt;
  }

  template <typename RNG>
  static inline double compute_log_jump(const Params &p, RNG &rng,
                                        Workspace &w) {

    // 1. Number of Jumps
    int n_jumps = w.pois(rng);

    if (n_jumps == 0)
      return w.drift_compensation;

    double total_jump = 0.0;

    // 2. Accumulate individual jumps
    for (int i = 0; i < n_jumps; ++i) {
      double u = w.Uni(rng);

      if (u < p.p_up) {
        total_jump += w.exp_up(rng);
      } else {
        total_jump -= w.exp_down(rng);
      }
    }

    return w.drift_compensation + total_jump;
  }
};