#pragma once
#include <boost/random.hpp>
#include <boost/random/xoshiro.hpp>
#include <cmath>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <omp.h>

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
struct MonteCarloConfig {
  size_t num_paths = 100000;
  size_t time_steps = 100;
  unsigned long seed = 42;
};

// -----------------------------------------------------------------------------
// HIGH-PERFORMANCE MONTE CARLO ENGINE
// -----------------------------------------------------------------------------
template <typename Model, typename Stepper, typename Instrument,
          typename RNG = boost::random::xoshiro256pp>
class MonteCarloEngine {
public:
  MonteCarloEngine(const Model &model, const MonteCarloConfig &cfg)
      : model_(model), cfg_(cfg) {}

  // Returns {Price, Standard Error}
  std::pair<double, double> calculate(double S0, double r, double q,
                                      const Instrument &instr) const {

    size_t steps = cfg_.time_steps;
    size_t M = cfg_.num_paths;
    double T = instr.get_maturity();
    double dt = T / static_cast<double>(steps);

    double sum_payoff = 0.0;
    double sum_sq_payoff = 0.0;

// -------------------------------------------------------
// PARALLEL REGION
// -------------------------------------------------------
#pragma omp parallel reduction(+ : sum_payoff, sum_sq_payoff)
    {
      // [FIX HERE]: Pass 'T' (Total Time) as the 5th argument!
      Stepper stepper(model_, dt, r, q, T);

      // Thread-Local RNG
      int tid = omp_get_thread_num();
      unsigned long my_seed = cfg_.seed + (tid * 999983);
      RNG rng(my_seed);

      typename Stepper::State state;

#pragma omp for schedule(static) nowait
      for (size_t i = 0; i < M; ++i) {

        // A. Reset State
        // This calls Tracker::init internally!
        stepper.start_path(state, S0);

        // B. Evolve Path
        stepper.multi_step(state, rng, steps);

        // C. Evaluate Payoff
        double val = instr.calculate(state);

        sum_payoff += val;
        sum_sq_payoff += val * val;
      }
    }

    // 5. Aggregation & Statistics
    double df = std::exp(-r * T);
    double mean = sum_payoff / M;

    // Biased Variance of the ESTIMATOR
    double variance = (sum_sq_payoff / M) - (mean) * (mean);
    double stderr = std::sqrt(variance / M);

    return {mean * df, stderr * df};
  }

private:
  const Model &model_;
  MonteCarloConfig cfg_;
};