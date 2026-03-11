#pragma once
#include <algorithm>
#include <boost/random.hpp>
#include <boost/random/xoshiro.hpp>
#include <myql/core/PricingTypes.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/models/asvj/core/CoupledASVJStepper.hpp>
#include <myql/utils/VectorOps.hpp>
#include <omp.h>
#include <vector>

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

struct MLMCConfig {
  double epsilon = 0.005; // Target RMSE accuracy (e.g., 0.005)
  size_t L_min = 2;       // Minimum levels to simulate
  size_t L_max = 15;      // Maximum allowed levels
  size_t N0 = 10000;      // Initial pilot paths per level
  size_t base_steps = 1;  // Steps at level 0 (e.g., 2 steps per year)

  // Convergence parameters
  double alpha = 2.0;         // Weak convergence order (bias)
  double beta = 1.0;          // Strong convergence order
  bool adaptive_rates = true; // If true, estimate alpha and beta during pilot

  unsigned long seed = 42;
  double fd_bump = 1e-4; // Finite Difference Bump for spot (Delta/Gamma)
};

// -----------------------------------------------------------------------------
// MULTI-LEVEL MONTE CARLO PRICER
// -----------------------------------------------------------------------------
template <typename Model, typename CoupledStepperType, typename Instrument,
          GreekMode Mode = GreekMode::None,
          typename RNG = boost::random::xoshiro256pp>
class MultiLevelMonteCarloPricer {
public:
  using ResultType = typename Instrument::ResultType;
  using ReturnStruct = MlmcResult<Mode, ResultType>;

  MultiLevelMonteCarloPricer(const Model &model, const MLMCConfig &cfg)
      : model_(model), cfg_(cfg) {}

  ReturnStruct calculate(double S0, double r, double q,
                         const Instrument &input_instr) const {

    Instrument instr = input_instr;
    const double T = instr.get_maturity();
    // To implement smooth payoff if needed (detail::apply_automatic_smoothing)
    // ...

    // Initialize Thread-Local RNG Pool
    int max_threads = omp_get_max_threads();
    std::vector<RNG> rng_pool;
    rng_pool.reserve(max_threads);
    for (int t = 0; t < max_threads; ++t) {
      // Seed each thread's RNG uniquely based on the config seed and thread ID
      rng_pool.emplace_back(cfg_.seed + t * 999983);
    }

    // Initialize MLMC Level tracking
    size_t L = cfg_.L_min;
    std::vector<size_t> N_l(cfg_.L_max + 1, 0); // Evaluated paths per level
    std::vector<double> sum_dP(cfg_.L_max + 1, 0.0);    // Sum of Delta P
    std::vector<double> sum_sq_dP(cfg_.L_max + 1, 0.0); // Sum of (Delta P)^2
    std::vector<double> V_l(cfg_.L_max + 1, 0.0); // Sample variance per level
    std::vector<double> cost_l(cfg_.L_max + 1, 0.0); // Cost per path at level l

    // Initialize costs: C_l = 2^l * base_steps
    for (size_t l = 0; l <= cfg_.L_max; ++l) {
      cost_l[l] = static_cast<double>(cfg_.base_steps * (1 << l));
    }

    double alpha = cfg_.alpha;
    double beta = cfg_.beta;
    bool converged = false;

    while (!converged) {
      // 1. Evaluate paths for each active level until optimal N_l is reached
      for (size_t l = 0; l <= L; ++l) {
        // Giles' optimal N_l formula (will be updated at step 2)
        // For the first iteration, we use the pilot N
        double sum_sqrt_VC_pilot = 0.0;
        for (size_t k = 0; k <= L; ++k)
          sum_sqrt_VC_pilot += std::sqrt(V_l[k] * cost_l[k]);

        size_t optimal_N = 0;
        if (V_l[l] > 1e-18) {
          optimal_N = static_cast<size_t>(
              std::ceil(2.0 * std::pow(cfg_.epsilon, -2.0) *
                        std::sqrt(V_l[l] / cost_l[l]) * sum_sqrt_VC_pilot));
        }

        // To avoid excessive computational cost at deep levels, the initial
        // pilot paths must decay geometrically: N_pilot(l) = N0 / 2^l, bounded
        // by a hard minimum (e.g., 20) to ensure stable variance estimation.
        size_t pilot_N = std::max<size_t>(20, cfg_.N0 / (1 << l));
        size_t target_N = std::max(pilot_N, optimal_N);

        if (N_l[l] < target_N) {
          size_t dN = target_N - N_l[l];
          evaluate_level(l, dN, S0, r, q, T, instr, sum_dP[l], sum_sq_dP[l],
                         rng_pool);
          N_l[l] += dN;

          // Update variance estimate
          if (N_l[l] > 1) {
            double mean = sum_dP[l] / N_l[l];
            V_l[l] = std::max(1e-18, (sum_sq_dP[l] / N_l[l]) - (mean * mean));
          }
        }
      }

      // 2. [Optional] Adaptive Estimation of alpha and beta (Moving Window)
      if (cfg_.adaptive_rates && L >= 2) {
        // Estimate alpha from |mean dP_l| ~ 2^(-alpha * l)
        // Estimate beta from V_l ~ 2^(-beta * l)
        // Use a moving window over the last 3 levels: [L-2, L-1, L]
        auto estimate_order = [&](const std::vector<double> &values) {
          double sum_l = 0, sum_logV = 0, sum_l_logV = 0, sum_l2 = 0;
          int count = 0;
          size_t start_l = (L >= 2) ? L - 2 : 1;
          for (size_t l = start_l; l <= L; ++l) {
            if (values[l] > 1e-16) {
              double y = std::log(values[l]);
              // Using relative x-coordinates (0, 1, 2) makes the regression
              // more stable numerically
              double x = static_cast<double>(l - start_l);
              sum_l += x;
              sum_logV += y;
              sum_l_logV += x * y;
              sum_l2 += x * x;
              count++;
            }
          }
          if (count < 2)
            return -1.0;
          double slope = (count * sum_l_logV - sum_l * sum_logV) /
                         (count * sum_l2 - sum_l * sum_l);
          return -slope / std::log(2.0);
        };

        std::vector<double> abs_means(L + 1);
        for (size_t l = 0; l <= L; ++l)
          abs_means[l] = std::abs(sum_dP[l] / N_l[l]);

        double est_alpha = estimate_order(abs_means);
        double est_beta = estimate_order(V_l);

        // Cap at 2.0 to avoid being too aggressive with bias correction
        if (est_alpha > 0.3)
          alpha = std::min(2.0, est_alpha);
        if (est_beta > 0.3)
          beta = std::min(2.0, est_beta);
      }

      // 3. Re-evaluate optimal N_l using Giles' formula
      double sum_sqrt_VC = 0.0;
      for (size_t l = 0; l <= L; ++l) {
        sum_sqrt_VC += std::sqrt(V_l[l] * cost_l[l]);
      }

      bool needs_more_paths = false;
      for (size_t l = 0; l <= L; ++l) {
        // Here beta could be used if we had a more complex allocation,
        // but Giles standard uses observed V_l directly.
        size_t optimal_N = static_cast<size_t>(
            std::ceil(2.0 * std::pow(cfg_.epsilon, -2.0) *
                      std::sqrt(V_l[l] / cost_l[l]) * sum_sqrt_VC));

        if (optimal_N > N_l[l]) {
          // In the next iteration of the while(!converged) loop,
          // level l will simulate more paths because target_N will be
          // optimal_N.
          needs_more_paths = true;
          // We don't update N_l[l] here! evaluate_level will do it.
        }
      }

      if (needs_more_paths)
        continue;

      // 4. Check weak convergence (Bias) with Giles' Zero-Crossing Heuristic
      double diff_mean_L = std::abs(sum_dP[L] / N_l[L]);

      // Protect against accidental zero-crossings by comparing with previous
      // level
      double effective_diff = diff_mean_L;
      if (L > 0) {
        double diff_mean_L_minus_1 = std::abs(sum_dP[L - 1] / N_l[L - 1]);
        effective_diff = std::max(diff_mean_L, 0.5 * diff_mean_L_minus_1);
      }

      // Bias estimate uses the weak order alpha: |E[P_L - P_{L-1}]| / (M^alpha
      // - 1)
      double bias_estimate = effective_diff / (std::pow(2.0, alpha) - 1.0);

      if (bias_estimate > cfg_.epsilon / std::numbers::sqrt2) {
        if (L < cfg_.L_max) {
          L++;
          converged = false;
        } else {
          // Reached max levels, must terminate
          converged = true;
        }
      } else {
        converged = true;
      }
    }

    // 5. Final Aggregation
    ReturnStruct res;
    double df = std::exp(-r * T);
    res.price = 0.0;
    double var_total = 0.0;
    res.total_steps = 0;

    res.alpha_estimated = alpha;
    res.beta_estimated = beta;

    for (size_t l = 0; l <= L; ++l) {
      res.paths_per_level.push_back(N_l[l]);
      // Cost for L0 is base_steps. For L>0 it is fine_steps + coarse_steps (1.5
      // * fine)
      size_t steps_per_path =
          (l == 0) ? cfg_.base_steps : (cfg_.base_steps * (1 << l) * 3 / 2);
      res.total_steps += N_l[l] * steps_per_path;
      double mean_dP = sum_dP[l] / N_l[l];
      res.mean_diffs.push_back(mean_dP * df);
      res.var_diffs.push_back(V_l[l] * df * df);

      res.price += mean_dP * df;
      var_total += (V_l[l] / N_l[l]);
    }

    res.price_std_err = std::sqrt(var_total) * df;

    return res;
  }

private:
  void evaluate_level(size_t l, size_t dN, double S0, double r, double q,
                      double T, const Instrument &instr, double &sum_dP,
                      double &sum_sq_dP, std::vector<RNG> &rng_pool) const {

    const double h = S0 * cfg_.fd_bump;
    const auto tracker_cfg = instr.template get_tracker_config<Mode>(S0, h);

    double local_sum = 0.0;
    double local_sq_sum = 0.0;

    // Extract Base Stepper from the Coupled Stepper Type
    using BaseStepper = typename CoupledStepperType::BaseStepperType;

    if (l == 0) {
      // Level 0: Uncoupled standard simulation
      size_t coarse_steps = cfg_.base_steps;
      double dt_coarse = T / static_cast<double>(coarse_steps);

      typename BaseStepper::VolGlobalWorkspace shared_vol_wksp =
          BaseStepper::build_global_workspace(model_, dt_coarse);

#pragma omp parallel
      {
        BaseStepper stepper(model_, dt_coarse, r, q, T, shared_vol_wksp);
        RNG &rng = rng_pool[omp_get_thread_num()];
        typename BaseStepper::State state;

        double t_l_sum = 0.0;
        double t_l_sq_sum = 0.0;

#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < dN; ++i) {
          stepper.start_path(state, tracker_cfg, S0);
          stepper.multi_step(state, tracker_cfg, rng, coarse_steps);

          ResultType p_base, d1, d2;
          instr.template calculate_to_buffer<Mode>(state, S0, h, p_base, d1,
                                                   d2);

          // For level 0, difference is just P0
          t_l_sum += p_base;
          t_l_sq_sum += p_base * p_base;
        }

#pragma omp atomic
        local_sum += t_l_sum;
#pragma omp atomic
        local_sq_sum += t_l_sq_sum;
      }

    } else {
      // Level l > 0: Coupled simulation (evaluating P_fine - P_coarse)
      size_t fine_steps = cfg_.base_steps * (1 << l);
      size_t coarse_steps = fine_steps / 2;
      double dt_fine = T / static_cast<double>(fine_steps);

      // The Coupled Stepper expects Fine Workspace parameters
      typename CoupledStepperType::VolGlobalWorkspace shared_vol_wksp =
          CoupledStepperType::build_global_workspace(model_, dt_fine);

#pragma omp parallel
      {
        CoupledStepperType stepper(model_, dt_fine, r, q, T, shared_vol_wksp);
        RNG &rng = rng_pool[omp_get_thread_num()];
        typename CoupledStepperType::CoupledState state;

        double t_l_sum = 0.0;
        double t_l_sq_sum = 0.0;

#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < dN; ++i) {
          stepper.start_path(state, tracker_cfg, S0);
          stepper.multi_step_coupled(state, tracker_cfg, rng, coarse_steps);

          ResultType p_fine, d1_f, d2_f;
          ResultType p_coarse, d1_c, d2_c;

          instr.template calculate_to_buffer<Mode>(state.fine, S0, h, p_fine,
                                                   d1_f, d2_f);
          instr.template calculate_to_buffer<Mode>(state.coarse, S0, h,
                                                   p_coarse, d1_c, d2_c);

          // Delta P_l = P_fine(l) - P_coarse(l-1)
          double diff = p_fine - p_coarse;
          t_l_sum += diff;
          t_l_sq_sum += diff * diff;
        }

#pragma omp atomic
        local_sum += t_l_sum;
#pragma omp atomic
        local_sq_sum += t_l_sq_sum;
      }
    }

    sum_dP += local_sum;
    sum_sq_dP += local_sq_sum;
  }

private:
  const Model &model_;
  MLMCConfig cfg_;
};
