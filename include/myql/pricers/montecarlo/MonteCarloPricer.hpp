#pragma once
#include <boost/random.hpp>
#include <boost/random/xoshiro.hpp>
#include <cmath>
#include <omp.h>

#include <myql/core/PricingTypes.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/AffineTraits.hpp>
#include <myql/utils/VectorOps.hpp>

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

struct MonteCarloConfig {
  size_t num_paths = 100000;
  size_t time_steps = 100;
  unsigned long seed = 42;
  double fd_bump = 1e-4; // Finite Difference Bump for spot (Delta/Gamma)

  // Full Greek bumps
  double vol_bump = 1e-4; // For Vega (v0 bump)
  double t_bump = 1e-4;   // For Theta (T bump)
  double r_bump = 1e-4;   // For Rho (r bump)
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
        // Sweet-spot: eps ~ sigma * K * sqrt(T) / 10 (good trade-off between
        // bias and variance)
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

    std::array<ResultType, 2> g_sum_vega, g_sq_sum_vega;
    ResultType g_sum_theta, g_sq_sum_theta;
    ResultType g_sum_rho, g_sq_sum_rho;

    if constexpr (Mode == GreekMode::Full) {
      init_accumulators(g_sum_vega[0], g_sq_sum_vega[0]);
      init_accumulators(g_sum_vega[1], g_sq_sum_vega[1]);
      init_accumulators(g_sum_theta, g_sq_sum_theta);
      init_accumulators(g_sum_rho, g_sq_sum_rho);
    }

    // PARALLEL REGION
#pragma omp parallel
    {
      Stepper stepper_base(model_, dt, r, q, T, shared_vol_wksp);
      RNG rng(cfg_.seed + (omp_get_thread_num() * 999983));
      typename Stepper::State state_base;

      ResultType p_base, p_up, p_dn;
      ResultType l_sum_price, l_sq_sum_price;
      ResultType l_sum_delta, l_sq_sum_delta;
      ResultType l_sum_gamma, l_sq_sum_gamma;
      std::array<ResultType, 2> l_sum_vega, l_sq_sum_vega;
      ResultType l_sum_theta, l_sq_sum_theta;
      ResultType l_sum_rho, l_sq_sum_rho;

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
      if constexpr (Mode == GreekMode::Full) {
        ResultType dummy_buf;
        init_locals(l_sum_vega[0], l_sq_sum_vega[0], dummy_buf);
        init_locals(l_sum_vega[1], l_sq_sum_vega[1], dummy_buf);
        init_locals(l_sum_theta, l_sq_sum_theta, dummy_buf);
        init_locals(l_sum_rho, l_sq_sum_rho, dummy_buf);
      }

      auto run_path_only = [&](auto &step_obj, const auto &t_cfg, RNG &path_rng,
                               ResultType &out_p) {
        typename Stepper::State st;
        step_obj.start_path(st, t_cfg, S0);
        step_obj.multi_step(st, t_cfg, path_rng, steps);
        ResultType d1, d2;
        instr.template calculate_to_buffer<Mode>(st, S0, h, out_p, d1, d2);
      };

      if constexpr (Mode != GreekMode::Full) {
        // ---------------------------------------------------------------------
        // NON-FULL MODE (Price, Essential)
        // ---------------------------------------------------------------------
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < M; ++i) {
          stepper_base.start_path(state_base, tracker_cfg, S0);
          stepper_base.multi_step(state_base, tracker_cfg, rng, steps);
          instr.template calculate_to_buffer<Mode>(state_base, S0, h, p_base,
                                                   p_up, p_dn);

          using namespace myql::utils;
          l_sum_price += p_base;
          l_sq_sum_price += p_base * p_base;

          if constexpr (Mode == GreekMode::Essential) {
            ResultType path_delta = (p_up - p_dn) * (1.0 / (2.0 * h));
            ResultType path_gamma =
                (p_up - p_base * 2.0 + p_dn) * (1.0 / (h * h));
            l_sum_delta += path_delta;
            l_sq_sum_delta += path_delta * path_delta;
            l_sum_gamma += path_gamma;
            l_sq_sum_gamma += path_gamma * path_gamma;
          }
        }
      } else {
        // ---------------------------------------------------------------------
        // FULL MODE: Instantiate Bumped Steppers
        // ---------------------------------------------------------------------
        using Traits = AffineTraits<Model>;
        constexpr int nf = Model::num_variance_factors;

        // Vega 0 Steppers
        auto m_vup0 = Traits::template bump_v0<0>(model_, cfg_.vol_bump);
        auto m_vdn0 = Traits::template bump_v0<0>(model_, -cfg_.vol_bump);
        Stepper stp_vup0(m_vup0, dt, r, q, T,
                         Stepper::build_global_workspace(m_vup0, dt));
        Stepper stp_vdn0(m_vdn0, dt, r, q, T,
                         Stepper::build_global_workspace(m_vdn0, dt));

        // Theta Steppers (bump T -> bump dt to keep steps constant for CRN)
        double Tup = T + cfg_.t_bump;
        double Tdn = T - cfg_.t_bump;
        auto tcfg_up = instr.template get_tracker_config<Mode>(S0, h, Tup);
        auto tcfg_dn = instr.template get_tracker_config<Mode>(S0, h, Tdn);
        Stepper stp_Tup(model_, Tup / steps, r, q, Tup,
                        Stepper::build_global_workspace(model_, Tup / steps));
        Stepper stp_Tdn(model_, Tdn / steps, r, q, Tdn,
                        Stepper::build_global_workspace(model_, Tdn / steps));

        // Rho Steppers (bump r)
        Stepper stp_rup(model_, dt, r + cfg_.r_bump, q, T, shared_vol_wksp);
        Stepper stp_rdn(model_, dt, r - cfg_.r_bump, q, T, shared_vol_wksp);

        double inv_2dv = 1.0 / (2.0 * cfg_.vol_bump);
        double inv_2dT = 1.0 / (2.0 * cfg_.t_bump);
        double inv_2dr = 1.0 / (2.0 * cfg_.r_bump);
        double c_v0 = Traits::template vega_chain_factor<0>(model_);

#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < M; ++i) {
          RNG rng_start = rng; // Snapshot RNG for CRN

          // 1. Base + Essential
          stepper_base.start_path(state_base, tracker_cfg, S0);
          stepper_base.multi_step(state_base, tracker_cfg, rng, steps);
          instr.template calculate_to_buffer<Mode>(state_base, S0, h, p_base,
                                                   p_up, p_dn);

          using namespace myql::utils;
          l_sum_price += p_base;
          l_sq_sum_price += p_base * p_base;

          ResultType path_delta = (p_up - p_dn) * (1.0 / (2.0 * h));
          ResultType path_gamma =
              (p_up - p_base * 2.0 + p_dn) * (1.0 / (h * h));
          l_sum_delta += path_delta;
          l_sq_sum_delta += path_delta * path_delta;
          l_sum_gamma += path_gamma;
          l_sq_sum_gamma += path_gamma * path_gamma;

          // 2. Vega Factor 0
          ResultType p_vup0, p_vdn0;
          RNG rng_v0up = rng_start;
          run_path_only(stp_vup0, tracker_cfg, rng_v0up, p_vup0);
          RNG rng_v0dn = rng_start;
          run_path_only(stp_vdn0, tracker_cfg, rng_v0dn, p_vdn0);
          ResultType path_vega0 = (p_vup0 - p_vdn0) * inv_2dv * c_v0;
          l_sum_vega[0] += path_vega0;
          l_sq_sum_vega[0] += path_vega0 * path_vega0;

          // 3. Vega Factor 1 (nested for performance)
          if constexpr (nf >= 2) {
            auto m_vup1 = Traits::template bump_v0<1>(model_, cfg_.vol_bump);
            auto m_vdn1 = Traits::template bump_v0<1>(model_, -cfg_.vol_bump);
            Stepper stp_vup1(m_vup1, dt, r, q, T,
                             Stepper::build_global_workspace(m_vup1, dt));
            Stepper stp_vdn1(m_vdn1, dt, r, q, T,
                             Stepper::build_global_workspace(m_vdn1, dt));
            double c_v1 = Traits::template vega_chain_factor<1>(model_);

            ResultType p_vup1, p_vdn1;
            RNG rng_v1up = rng_start;
            run_path_only(stp_vup1, tracker_cfg, rng_v1up, p_vup1);
            RNG rng_v1dn = rng_start;
            run_path_only(stp_vdn1, tracker_cfg, rng_v1dn, p_vdn1);
            ResultType path_vega1 = (p_vup1 - p_vdn1) * inv_2dv * c_v1;
            l_sum_vega[1] += path_vega1;
            l_sq_sum_vega[1] += path_vega1 * path_vega1;
          }

          // 4. Theta (Requires differential discounting relative to base T)
          ResultType p_Tup, p_Tdn;
          RNG rng_Tup = rng_start;
          run_path_only(stp_Tup, tcfg_up, rng_Tup, p_Tup);
          RNG rng_Tdn = rng_start;
          run_path_only(stp_Tdn, tcfg_dn, rng_Tdn, p_Tdn);
          double df_Tup = std::exp(-r * cfg_.t_bump); // e^{-r(T+dT)} / e^{-rT}
          double df_Tdn = std::exp(r * cfg_.t_bump);  // e^{-r(T-dT)} / e^{-rT}
          ResultType path_theta = (p_Tup * df_Tup - p_Tdn * df_Tdn) * inv_2dT;
          l_sum_theta += path_theta;
          l_sq_sum_theta += path_theta * path_theta;

          // 5. Rho (Requires differential discounting relative to base r)
          ResultType p_rup, p_rdn;
          RNG rng_rup = rng_start;
          run_path_only(stp_rup, tracker_cfg, rng_rup, p_rup);
          RNG rng_rdn = rng_start;
          run_path_only(stp_rdn, tracker_cfg, rng_rdn, p_rdn);
          double df_rup = std::exp(-cfg_.r_bump * T); // e^{-(r+dr)T} / e^{-rT}
          double df_rdn = std::exp(cfg_.r_bump * T);  // e^{-(r-dr)T} / e^{-rT}
          ResultType path_rho = (p_rup * df_rup - p_rdn * df_rdn) * inv_2dr;
          l_sum_rho += path_rho;
          l_sq_sum_rho += path_rho * path_rho;
        }
      }

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
        if constexpr (Mode == GreekMode::Full) {
          g_sum_vega[0] += l_sum_vega[0];
          g_sq_sum_vega[0] += l_sq_sum_vega[0];
          g_sum_vega[1] += l_sum_vega[1];
          g_sq_sum_vega[1] += l_sq_sum_vega[1];
          g_sum_theta += l_sum_theta;
          g_sq_sum_theta += l_sq_sum_theta;
          g_sum_rho += l_sum_rho;
          g_sq_sum_rho += l_sq_sum_rho;
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
    if constexpr (Mode == GreekMode::Full) {
      compute_stats(g_sum_vega[0], g_sq_sum_vega[0], res.vega[0],
                    res.vega_std_err[0]);
      compute_stats(g_sum_vega[1], g_sq_sum_vega[1], res.vega[1],
                    res.vega_std_err[1]);
      compute_stats(g_sum_theta, g_sq_sum_theta, res.theta, res.theta_std_err);
      compute_stats(g_sum_rho, g_sq_sum_rho, res.rho, res.rho_std_err);
    }

    return res;
  }

private:
  const Model &model_;
  MonteCarloConfig cfg_;
};
