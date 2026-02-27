#pragma once
#include <boost/random.hpp>
#include <boost/random/xoshiro.hpp>
#include <cmath>
#include <omp.h>

#include <myql/core/PricingTypes.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/utils/VectorOps.hpp>

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

struct MonteCarloConfig {
  size_t num_paths = 100000;
  size_t time_steps = 100;
  unsigned long seed = 42;
  double fd_bump = 1e-4; // Finite Difference Bump (e.g., 1 basis point)
};

namespace detail {

// Helper trait to dispatch variance calculation to the proper free functions
template <typename ModelT>
inline double compute_expected_average_variance(const ModelT &model, double T) {
  using namespace myql::models::variance;
  double var = 0.0;

  if constexpr (requires { model.vol; }) {
    var += expected_average_variance(model.vol, T);
  } else if constexpr (requires { model.heston; }) {
    var += expected_average_variance(model.heston, T);
  } else if constexpr (requires { model.heston1; }) {
    var += expected_average_variance(model.heston1, T);
    var += expected_average_variance(model.heston2, T);
  }

  var += jump_variance(model.jump);
  return var;
}

// Helper function to dynamically calculate and inject optimal payoff smoothing
// bandwidth for discontinuous options using model-driven variance proxies.
template <GreekMode Mode, typename ModelT, typename InstrumentT>
inline void apply_automatic_smoothing(const ModelT &model, InstrumentT &instr,
                                      double S0) {
  if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
    using PayoffT = typename InstrumentT::PayoffType;
    if constexpr (PayoffT::needs_smoothing) {
      if (instr.get_payoff_mut().eps < 0.0) {
        // Automatic bandwidth selection
        const double T = instr.get_maturity();
        const double sigma_eff =
            std::sqrt(compute_expected_average_variance(model, T));
        // Sweet-spot: eps ~ sigma * K * sqrt(T) / 10
        if constexpr (std::is_scalar_v<typename InstrumentT::ResultType>) {
          const double K = instr.get_strikes();
          instr.get_payoff_mut().eps = sigma_eff * K * std::sqrt(T) / 10.0;
        } else {
          // For vectorized options we use the ATM strike (S0) as a proxy scale
          instr.get_payoff_mut().eps = sigma_eff * S0 * std::sqrt(T) / 10.0;
        }
      }
    }
  }
}
} // namespace detail

// -----------------------------------------------------------------------------
// MONTE CARLO PRICER (Greek-Aware)
// Symmetric API with FourierPricer:
//   MonteCarloPricer<Model, Stepper, Instrument, Mode> pricer(model, cfg);
//   auto res = pricer.calculate(S0, r, q, instr);
//   res.price, res.delta, etc... (depending on GreekMode)
// -----------------------------------------------------------------------------
template <typename Model, typename Stepper, typename Instrument,
          GreekMode Mode = GreekMode::None,
          typename RNG = boost::random::xoshiro256pp>
class MonteCarloPricer {
public:
  using ResultType = typename Instrument::ResultType;
  using ReturnStruct = MonteCarloResult<Mode, ResultType>;

  MonteCarloPricer(const Model &model, const MonteCarloConfig &cfg)
      : model_(model), cfg_(cfg) {}

  ReturnStruct calculate(double S0, double r, double q,
                         const Instrument &input_instr) const {

    // Create a local mutable copy for potential payoff smoothing
    Instrument instr = input_instr;
    const double T = instr.get_maturity();

    // Dynamic application of payoff smoothing (if requested and applicable)
    detail::apply_automatic_smoothing<Mode>(model_, instr, S0);

    const size_t steps = cfg_.time_steps;
    const size_t M = cfg_.num_paths;
    const double dt = T / static_cast<double>(steps);
    const double h = S0 * cfg_.fd_bump;
    const size_t n_contracts = instr.size();

    typename Stepper::VolGlobalWorkspace shared_vol_wksp =
        Stepper::build_global_workspace(model_, dt);

    // Fetch tracker configuration (passes Mode, S0, and h for equivalent
    // barriers)
    const auto tracker_cfg = instr.template get_tracker_config<Mode>(S0, h);

    // GLOBAL ACCUMULATORS
    ResultType g_sum_price, g_sq_sum_price;
    ResultType g_sum_delta, g_sq_sum_delta;
    ResultType g_sum_gamma, g_sq_sum_gamma;

    auto init_accumulators = [&](ResultType &sum, ResultType &sq_sum) {
      if constexpr (std::is_same_v<ResultType, double>) {
        sum = 0.0;
        sq_sum = 0.0;
      } else {
        sum.assign(n_contracts, 0.0);
        sq_sum.assign(n_contracts, 0.0);
      }
    };

    init_accumulators(g_sum_price, g_sq_sum_price);
    if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
      init_accumulators(g_sum_delta, g_sq_sum_delta);
      init_accumulators(g_sum_gamma, g_sq_sum_gamma);
    }

    // PARALLEL REGION
#pragma omp parallel
    {
      Stepper stepper(model_, dt, r, q, T, shared_vol_wksp);
      RNG rng(cfg_.seed + (omp_get_thread_num() * 999983));
      typename Stepper::State state;

      ResultType p_base, p_up, p_dn;
      ResultType l_sum_price, l_sq_sum_price;
      ResultType l_sum_delta, l_sq_sum_delta;
      ResultType l_sum_gamma, l_sq_sum_gamma;

      auto init_locals = [&](ResultType &sum, ResultType &sq, ResultType &buf) {
        if constexpr (std::is_same_v<ResultType, double>) {
          sum = 0.0;
          sq = 0.0;
          buf = 0.0;
        } else {
          sum.assign(n_contracts, 0.0);
          sq.assign(n_contracts, 0.0);
          buf.resize(n_contracts);
        }
      };

      init_locals(l_sum_price, l_sq_sum_price, p_base);
      if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
        init_locals(l_sum_delta, l_sq_sum_delta, p_up);
        init_locals(l_sum_gamma, l_sq_sum_gamma, p_dn);
      }

#pragma omp for schedule(static) nowait
      for (size_t i = 0; i < M; ++i) {

        // Single physics simulation
        stepper.start_path(state, tracker_cfg, S0);
        stepper.multi_step(state, tracker_cfg, rng, steps);

        // Instrument calculates all requested payoffs instantly
        instr.template calculate_to_buffer<Mode>(state, S0, h, p_base, p_up,
                                                 p_dn);

        // Accumulate price
        using namespace myql::utils;
        l_sum_price += p_base;
        l_sq_sum_price += p_base * p_base;

        // Calculate and accumulate pathwise greeks
        if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
          ResultType path_delta = (p_up - p_dn) * (1.0 / (2.0 * h));
          ResultType path_gamma =
              (p_up - p_base * 2.0 + p_dn) * (1.0 / (h * h));

          l_sum_delta += path_delta;
          l_sq_sum_delta += path_delta * path_delta;
          l_sum_gamma += path_gamma;
          l_sq_sum_gamma += path_gamma * path_gamma;
        }
      }

      // MANUAL REDUCTION
#pragma omp critical
      {
        using namespace myql::utils;
        g_sum_price += l_sum_price;
        g_sq_sum_price += l_sq_sum_price;
        if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
          g_sum_delta += l_sum_delta;
          g_sq_sum_delta += l_sq_sum_delta;
          g_sum_gamma += l_sum_gamma;
          g_sq_sum_gamma += l_sq_sum_gamma;
        }
      }
    } // End Parallel

    // Aggregation & Statistics
    double df = std::exp(-r * T);
    double inv_M = 1.0 / static_cast<double>(M);
    using namespace myql::utils;

    auto compute_stats = [&](const ResultType &sum, const ResultType &sq_sum,
                             ResultType &val, ResultType &err) {
      ResultType mean = sum * inv_M;
      ResultType var = (sq_sum * inv_M) - (mean * mean);
      val = mean * df;
      if constexpr (std::is_same_v<ResultType, double>)
        err = std::sqrt(var * inv_M) * df;
      else
        err = element_wise_sqrt(var * inv_M) * df;
    };

    ReturnStruct res;
    compute_stats(g_sum_price, g_sq_sum_price, res.price, res.price_std_err);

    if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
      compute_stats(g_sum_delta, g_sq_sum_delta, res.delta, res.delta_std_err);
      compute_stats(g_sum_gamma, g_sq_sum_gamma, res.gamma, res.gamma_std_err);
    }

    return res;
  }

private:
  const Model &model_;
  MonteCarloConfig cfg_;
};
