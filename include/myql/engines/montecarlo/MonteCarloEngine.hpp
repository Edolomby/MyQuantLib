#pragma once
#include <boost/random.hpp>
#include <boost/random/xoshiro.hpp>
#include <cmath>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <omp.h>

// UTILS
#include <myql/utils/VectorOps.hpp>

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
struct MonteCarloConfig {
  size_t num_paths = 100000;
  size_t time_steps = 100;
  unsigned long seed = 42;
};

// -----------------------------------------------------------------------------
// MONTE CARLO ENGINE
// -----------------------------------------------------------------------------
template <typename Model, typename Stepper, typename Instrument,
          typename RNG = boost::random::xoshiro256pp>
class MonteCarloEngine {
public:
  // Adapts to 'double' or 'std::vector<double>' automatically
  using ResultType = typename Instrument::ResultType;

  MonteCarloEngine(const Model &model, const MonteCarloConfig &cfg)
      : model_(model), cfg_(cfg) {}

  // Returns {Price, Standard Error}
  std::pair<ResultType, ResultType> calculate(double S0, double r, double q,
                                              const Instrument &instr) const {

    const size_t steps = cfg_.time_steps;
    const size_t M = cfg_.num_paths;
    const double T = instr.get_maturity();
    const double dt = T / static_cast<double>(steps);
    const size_t n_contracts = instr.size();

    // 1. PRE-COMPUTE SHARED WORKSPACE
    // Generated exactly once, safely outside the parallel block
    typename Stepper::VolGlobalWorkspace shared_vol_wksp =
        Stepper::build_global_workspace(model_, dt);

    // GLOBAL ACCUMULATOR INITIALIZATION
    ResultType global_sum;
    ResultType global_sq_sum;

    if constexpr (std::is_same_v<ResultType, double>) {
      global_sum = 0.0;
      global_sq_sum = 0.0;
    } else {
      global_sum.assign(n_contracts, 0.0);
      global_sq_sum.assign(n_contracts, 0.0);
    }

    // -------------------------------------------------------
    // PARALLEL REGION
    // -------------------------------------------------------
#pragma omp parallel
    {
      Stepper stepper(model_, dt, r, q, T, shared_vol_wksp);

      // Thread-Local RNG
      int tid = omp_get_thread_num();
      unsigned long my_seed = cfg_.seed + (tid * 999983);
      RNG rng(my_seed);

      typename Stepper::State state;

      // THREAD-LOCAL BUFFER ALLOCATION
      ResultType local_sum;
      ResultType local_sq_sum;
      ResultType payoff_buffer; // Reusable memory for each path

      if constexpr (std::is_same_v<ResultType, double>) {
        local_sum = 0.0;
        local_sq_sum = 0.0;
        payoff_buffer = 0.0;
      } else {
        local_sum.assign(n_contracts, 0.0);
        local_sq_sum.assign(n_contracts, 0.0);
        payoff_buffer.resize(n_contracts);
      }

#pragma omp for schedule(static) nowait
      for (size_t i = 0; i < M; ++i) {

        // A. Reset & Evolve Path
        stepper.start_path(state, S0);
        stepper.multi_step(state, rng, steps);

        // B. CALCULATE PAYOFF TO BUFFER (Zero Allocation!)
        instr.calculate_to_buffer(state, payoff_buffer);

        // C. ACCUMULATE
        using namespace myql::utils;
        local_sum += payoff_buffer;
        local_sq_sum += payoff_buffer * payoff_buffer;
      }

      // MANUAL REDUCTION
#pragma omp critical
      {
        using namespace myql::utils;
        global_sum += local_sum;
        global_sq_sum += local_sq_sum;
      }
    } // End Parallel

    // Aggregation & Statistics
    double df = std::exp(-r * T);
    using namespace myql::utils;

    double inv_M = 1.0 / static_cast<double>(M);

    ResultType mean = global_sum * inv_M;
    ResultType mean_sq = mean * mean;
    ResultType avg_sq_sum = global_sq_sum * inv_M;
    ResultType variance = avg_sq_sum - mean_sq;

    ResultType stderr;
    if constexpr (std::is_same_v<ResultType, double>) {
      stderr = std::sqrt(variance * inv_M);
    } else {
      stderr = element_wise_sqrt(variance * inv_M);
    }

    return {mean * df, stderr * df};
  }

private:
  const Model &model_;
  MonteCarloConfig cfg_;
};