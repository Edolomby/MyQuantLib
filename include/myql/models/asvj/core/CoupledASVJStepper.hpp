#pragma once
#include <array>
#include <boost/random.hpp>
#include <cmath>
#include <numbers>
#include <tuple>

#include <myql/instruments/trackers/PathTrackers.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/models/asvj/data/ModelParams.hpp>
#include <myql/models/asvj/policies/JumpPolicies.hpp>
#include <myql/models/asvj/policies/VolSchemes.hpp>
#include <type_traits>

// =============================================================================
// THE COUPLED STEPPER CLASS FOR MLMC
// =============================================================================

template <typename VolScheme1,                      // Mandatory: 1st Vol Scheme
          typename VolScheme2 = NullVolScheme,      // Optional: 2nd Vol Scheme
          typename JumpPolicy = NoJumps,            // Optional: Jumps
          typename TrackerPolicy = TrackerEuropean> // Optional: Tracker
class CoupledASVJStepper {
  // DEDUCE Number of Cirs AUTOMATICALLY at compile time
  static constexpr int NumCirs = (std::is_same_v<VolScheme1, NullVolScheme> &&
                                  std::is_same_v<VolScheme2, NullVolScheme>)
                                     ? 0
                                 : (std::is_same_v<VolScheme2, NullVolScheme>)
                                     ? 1
                                     : 2;

  // Static Checks
  static_assert(NumCirs == 0 || NumCirs == 1 || NumCirs == 2,
                "CoupledASVJStepper only supports 0, 1 or 2 CIR processes.");
  static_assert(NumCirs == 0 || std::is_same_v<VolScheme1, SchemeNV> ||
                    std::is_same_v<VolScheme1, SchemeExact> ||
                    std::is_same_v<VolScheme1, SchemeNCI>,
                "VolScheme1 must be SchemeNV, SchemeExact, or SchemeNCI");
  static_assert(NumCirs <= 1 || std::is_same_v<VolScheme2, SchemeNV> ||
                    std::is_same_v<VolScheme2, SchemeExact> ||
                    std::is_same_v<VolScheme2, SchemeNCI>,
                "When NumCirs == 2, VolScheme2 must be SchemeNV, SchemeExact, "
                "or SchemeNCI");

public:
  using BaseStepperType =
      ASVJStepper<VolScheme1, VolScheme2, JumpPolicy, TrackerPolicy>;
  // Public Aliases for Engine
  using Policy = TrackerPolicy;
  using JumpP = typename JumpPolicy::Params;

  using ModelParams = typename std::conditional<
      NumCirs == 0, ZeroFactorModel<JumpP>,
      typename std::conditional<NumCirs == 1, SingleFactorModel<JumpP>,
                                DoubleFactorModel<JumpP>>::type>::type;

  // Resolve Workspace Size
  using VolGlobalWorkspace = typename std::conditional<
      NumCirs == 0, std::tuple<>,
      typename std::conditional<
          NumCirs == 1, std::tuple<typename VolScheme1::GlobalWorkspace>,
          std::tuple<typename VolScheme1::GlobalWorkspace,
                     typename VolScheme2::GlobalWorkspace>>::type>::type;

  struct State : public TrackerPolicy::ExtraState {
    double logS;
    std::array<double, NumCirs> V;
  };

  struct CoupledState {
    State fine;
    State coarse;
  };

  // To Build the global workspaces before parallel region
  static VolGlobalWorkspace build_global_workspace(const ModelParams &model,
                                                   double dt) {
    if constexpr (NumCirs == 0) {
      return std::make_tuple();
    } else if constexpr (NumCirs == 1) {
      return std::make_tuple(
          VolScheme1::build_global_workspace(model.heston, dt));
    } else {
      return std::make_tuple(
          VolScheme1::build_global_workspace(model.heston1, dt),
          VolScheme2::build_global_workspace(model.heston2, dt));
    }
  }

  // =========================================================================
  // CONSTRUCTOR
  // =========================================================================
  CoupledASVJStepper(const ModelParams &model, double dt_fine, double r,
                     double q, double total_time,
                     const VolGlobalWorkspace &vol_gw)
      : jump_params_(model.jump), dt_fine_(dt_fine), dt_coarse_(dt_fine * 2.0) {

    // If Path Dependent: Prepare workspace for small step 'dt_fine'.
    // If European: Prepare workspace for ONE giant step 'total_time'.
    if constexpr (TrackerPolicy::is_path_dependent) {
      JumpPolicy::prepare(jump_wksp_fine_, jump_params_, dt_fine_);
    } else {
      JumpPolicy::prepare(jump_wksp_fine_, jump_params_, total_time);
    }

    if constexpr (NumCirs == 0) {
      double var = model.vol * model.vol;
      auto &gbm_fine = std::get<0>(diffusion_data_fine_);
      gbm_fine.drift_dt = (r - q - 0.5 * var) * dt_fine_;
      gbm_fine.vol_dt = model.vol * std::sqrt(dt_fine_);

      auto &gbm_coarse = std::get<0>(diffusion_data_coarse_);
      gbm_coarse.drift_dt = (r - q - 0.5 * var) * dt_coarse_;
      gbm_coarse.vol_dt = model.vol * std::sqrt(dt_coarse_);

    } else {
      double r_part = (r - q) / static_cast<double>(NumCirs);
      if constexpr (NumCirs == 1) {
        init_cir_impl<VolScheme1>(std::get<0>(diffusion_data_fine_),
                                  std::get<0>(vol_gw), model.heston, r_part,
                                  dt_fine_);
        init_cir_impl<VolScheme1>(std::get<0>(diffusion_data_coarse_),
                                  std::get<0>(vol_gw), model.heston, r_part,
                                  dt_coarse_);
      } else {
        init_cir_impl<VolScheme1>(std::get<0>(diffusion_data_fine_),
                                  std::get<0>(vol_gw), model.heston1, r_part,
                                  dt_fine_);
        init_cir_impl<VolScheme2>(std::get<1>(diffusion_data_fine_),
                                  std::get<1>(vol_gw), model.heston2, r_part,
                                  dt_fine_);

        init_cir_impl<VolScheme1>(std::get<0>(diffusion_data_coarse_),
                                  std::get<0>(vol_gw), model.heston1, r_part,
                                  dt_coarse_);
        init_cir_impl<VolScheme2>(std::get<1>(diffusion_data_coarse_),
                                  std::get<1>(vol_gw), model.heston2, r_part,
                                  dt_coarse_);
      }
    }
  }

  // Standard start_path
  void start_path(CoupledState &state,
                  const typename TrackerPolicy::Config &cfg, double S0) const {
    // Initialize Fine
    state.fine.logS = std::log(S0);
    if constexpr (NumCirs >= 1)
      state.fine.V[0] = std::get<0>(diffusion_data_fine_).v0;
    if constexpr (NumCirs >= 2)
      state.fine.V[1] = std::get<1>(diffusion_data_fine_).v0;

    TrackerPolicy::init(state.fine, cfg, state.fine.logS, dt_fine_);

    // Initialize Coarse
    state.coarse.logS = std::log(S0);
    if constexpr (NumCirs >= 1)
      state.coarse.V[0] = std::get<0>(diffusion_data_coarse_).v0;
    if constexpr (NumCirs >= 2)
      state.coarse.V[1] = std::get<1>(diffusion_data_coarse_).v0;

    TrackerPolicy::init(state.coarse, cfg, state.coarse.logS, dt_coarse_);
  }

  // =========================================================================
  // COUPLED MULTI-STEP ENGINE
  // =========================================================================
  template <typename RNG>
  void multi_step_coupled(CoupledState &state,
                          const typename TrackerPolicy::Config &cfg, RNG &rng,
                          unsigned n_coarse_steps) {
    boost::random::uniform_real_distribution<double> dist_uni(0.0, 1.0);

    std::array<double, NumCirs> v_fine_curr = state.fine.V;
    std::array<double, NumCirs> v_coarse_curr =
        state.coarse.V; // To remember coarse state if needed

    double current_logS_fine = state.fine.logS;
    double current_logS_coarse = state.coarse.logS;

    double total_time = n_coarse_steps * dt_coarse_;

    // =======================================================================
    // OPTIMIZATION: 0-Factor (GBM) + European (Not Path Dependent)
    // =======================================================================
    if constexpr (NumCirs == 0 && !TrackerPolicy::is_path_dependent) {
      double G_fine = Numerics::normcdfinv_as241(dist_uni(rng));

      // Fine stock jump
      auto &gbm_fine = std::get<0>(diffusion_data_fine_);
      double total_drift_fine = gbm_fine.drift_dt * (n_coarse_steps * 2);
      double total_vol_fine =
          gbm_fine.vol_dt * std::sqrt(static_cast<double>(n_coarse_steps * 2));
      current_logS_fine += total_drift_fine + total_vol_fine * G_fine;

      // Exact coupling for single standard normal: Z_coarse = Z_fine trivially
      auto &gbm_coarse = std::get<0>(diffusion_data_coarse_);
      double total_drift_coarse = gbm_coarse.drift_dt * n_coarse_steps;
      double total_vol_coarse =
          gbm_coarse.vol_dt * std::sqrt(static_cast<double>(n_coarse_steps));
      current_logS_coarse += total_drift_coarse + total_vol_coarse * G_fine;

      // Jumps
      double jump =
          JumpPolicy::compute_log_jump(jump_params_, rng, jump_wksp_fine_);
      current_logS_fine += jump;
      current_logS_coarse +=
          jump; // strongly coupled exact same jump size and number

      state.fine.logS = current_logS_fine;
      state.coarse.logS = current_logS_coarse;

      TrackerPolicy::finalize(state.fine, cfg, dt_fine_, total_time);
      TrackerPolicy::finalize(state.coarse, cfg, dt_coarse_, total_time);

      return; // EXIT EARLY
    }

    // =======================================================================
    // STANDARD LOOP: Path Dependent OR Stochastic Volatility (1 or 2 CIRs)
    // =======================================================================
    // Jumps (EUROPEAN ONLY for NumCirs > 0)
    // Synchronize by adding jump BEFORE the diffusion loop
    if constexpr (!TrackerPolicy::is_path_dependent && NumCirs > 0) {
      double jump = JumpPolicy::compute_log_jump(
          jump_params_, rng, jump_wksp_fine_); // Using fine as reference
      current_logS_fine += jump;
      current_logS_coarse += jump; // Exact same
    }

    for (unsigned k = 0; k < n_coarse_steps; ++k) {

      // 1. Draw Brownian Increments for the log-stock
      double G_fine_1 = Numerics::normcdfinv_as241(dist_uni(rng));
      double G_fine_2 = Numerics::normcdfinv_as241(dist_uni(rng));
      double G_coarse = (G_fine_1 + G_fine_2) / std::numbers::sqrt2;

      // CASE 0: Path-Dependent Geometric Brownian Motion (e.g., Asian GBM)
      if constexpr (NumCirs == 0) {
        auto &gbm_fine = std::get<0>(diffusion_data_fine_);
        auto &gbm_coarse = std::get<0>(diffusion_data_coarse_);

        // One coarse step
        G_coarse = (G_fine_1 + G_fine_2) / std::numbers::sqrt2;
        current_logS_coarse +=
            gbm_coarse.drift_dt + gbm_coarse.vol_dt * G_coarse;

        double jump_fine_1 = 0.0, jump_fine_2 = 0.0;

        // Two fine steps
        current_logS_fine += gbm_fine.drift_dt + gbm_fine.vol_dt * G_fine_1;
        if constexpr (TrackerPolicy::is_path_dependent) {
          jump_fine_1 =
              JumpPolicy::compute_log_jump(jump_params_, rng, jump_wksp_fine_);
          current_logS_fine += jump_fine_1;
        }
        state.fine.logS = current_logS_fine;
        if constexpr (TrackerPolicy::is_path_dependent)
          TrackerPolicy::update(state.fine, cfg, dt_fine_);

        current_logS_fine += gbm_fine.drift_dt + gbm_fine.vol_dt * G_fine_2;
        if constexpr (TrackerPolicy::is_path_dependent) {
          jump_fine_2 =
              JumpPolicy::compute_log_jump(jump_params_, rng, jump_wksp_fine_);
          current_logS_fine += jump_fine_2;
        }
        state.fine.logS = current_logS_fine;
        if constexpr (TrackerPolicy::is_path_dependent) {
          TrackerPolicy::update(state.fine, cfg, dt_fine_);
          current_logS_coarse += jump_fine_1 + jump_fine_2;
          state.coarse.logS = current_logS_coarse;
          TrackerPolicy::update(state.coarse, cfg, dt_coarse_);
        }
      }
      // CASE 1: Single Factor
      else if constexpr (NumCirs == 1) {
        auto &data_fine = std::get<0>(diffusion_data_fine_);
        auto &data_coarse = std::get<0>(diffusion_data_coarse_);

        const DiscrParams &Cf = data_fine.C;
        const DiscrParams &Cc = data_coarse.C;

        // FINE STEP 1
        double v_fine_mid;
        double Z_v_fine_1 = 0.0;
        if constexpr (std::is_same_v<VolScheme1, SchemeNV>) {
          Z_v_fine_1 = Numerics::normcdfinv_as241(dist_uni(rng));
          v_fine_mid = VolScheme1::evolve_coupled(v_fine_curr[0], Z_v_fine_1,
                                                  data_fine.wksp);
        } else {
          v_fine_mid = VolScheme1::evolve(v_fine_curr[0], rng, data_fine.wksp);
        }

        double jump_fine_1 = 0.0, jump_fine_2 = 0.0;

        double v_avg_root_f1 = std::sqrt(Cf[3] * (v_fine_curr[0] + v_fine_mid));
        current_logS_fine += Cf[0] + Cf[1] * v_fine_curr[0] +
                             Cf[2] * v_fine_mid + v_avg_root_f1 * G_fine_1;

        if constexpr (TrackerPolicy::is_path_dependent) {
          jump_fine_1 =
              JumpPolicy::compute_log_jump(jump_params_, rng, jump_wksp_fine_);
          current_logS_fine += jump_fine_1;
        }
        state.fine.logS = current_logS_fine;
        if constexpr (TrackerPolicy::is_path_dependent)
          TrackerPolicy::update(state.fine, cfg, dt_fine_);

        // FINE STEP 2
        double v_fine_next;
        double Z_v_fine_2 = 0.0;
        if constexpr (std::is_same_v<VolScheme1, SchemeNV>) {
          Z_v_fine_2 = Numerics::normcdfinv_as241(dist_uni(rng));
          v_fine_next = VolScheme1::evolve_coupled(v_fine_mid, Z_v_fine_2,
                                                   data_fine.wksp);
        } else {
          v_fine_next = VolScheme1::evolve(v_fine_mid, rng, data_fine.wksp);
        }

        double v_avg_root_f2 = std::sqrt(Cf[3] * (v_fine_mid + v_fine_next));
        current_logS_fine += Cf[0] + Cf[1] * v_fine_mid + Cf[2] * v_fine_next +
                             v_avg_root_f2 * G_fine_2;

        if constexpr (TrackerPolicy::is_path_dependent) {
          jump_fine_2 =
              JumpPolicy::compute_log_jump(jump_params_, rng, jump_wksp_fine_);
          current_logS_fine += jump_fine_2;
        }
        state.fine.logS = current_logS_fine;
        if constexpr (TrackerPolicy::is_path_dependent)
          TrackerPolicy::update(state.fine, cfg, dt_fine_);

        // COARSE STEP EXACT COUPLING:
        double v_coarse_next;
        if constexpr (std::is_same_v<VolScheme1, SchemeNCI> ||
                      std::is_same_v<VolScheme1, SchemeExact>) {
          // Exact sampling - simply copy the sampled trajectory
          v_coarse_next = v_fine_next;
        } else { // SchemeNV standard strong coupling via Gaussian path
          double Z_v_coarse = (Z_v_fine_1 + Z_v_fine_2) / std::numbers::sqrt2;
          v_coarse_next = VolScheme1::evolve_coupled(
              v_coarse_curr[0], Z_v_coarse, data_coarse.wksp);
        }

        if (v_avg_root_f1 + v_avg_root_f2 > 1e-15) {
          G_coarse = (v_avg_root_f1 * G_fine_1 + v_avg_root_f2 * G_fine_2) /
                     std::hypot(v_avg_root_f1, v_avg_root_f2);
        }
        double v_avg_root_c =
            std::sqrt(Cc[3] * (v_coarse_curr[0] + v_coarse_next));
        current_logS_coarse += Cc[0] + Cc[1] * v_coarse_curr[0] +
                               Cc[2] * v_coarse_next + v_avg_root_c * G_coarse;

        v_fine_curr[0] = v_fine_next;
        v_coarse_curr[0] = v_coarse_next;

        if constexpr (TrackerPolicy::is_path_dependent) {
          current_logS_coarse += jump_fine_1 + jump_fine_2;
          state.coarse.logS = current_logS_coarse;
          TrackerPolicy::update(state.coarse, cfg, dt_coarse_);
        }
      }
      // CASE 2: Double Factor (Optimization in Law)
      else {
        auto &data0_fine = std::get<0>(diffusion_data_fine_);
        auto &data1_fine = std::get<1>(diffusion_data_fine_);
        auto &data0_coarse = std::get<0>(diffusion_data_coarse_);
        auto &data1_coarse = std::get<1>(diffusion_data_coarse_);

        const DiscrParams &Cf0 = data0_fine.C, &Cf1 = data1_fine.C;
        const DiscrParams &Cc0 = data0_coarse.C, &Cc1 = data1_coarse.C;

        // FINE STEP 1
        double v0_fine_mid, v1_fine_mid;
        double Z_v0_fine_1 = 0.0, Z_v1_fine_1 = 0.0;

        if constexpr (std::is_same_v<VolScheme1, SchemeNV>) {
          Z_v0_fine_1 = Numerics::normcdfinv_as241(dist_uni(rng));
          v0_fine_mid = VolScheme1::evolve_coupled(v_fine_curr[0], Z_v0_fine_1,
                                                   data0_fine.wksp);
        } else {
          v0_fine_mid =
              VolScheme1::evolve(v_fine_curr[0], rng, data0_fine.wksp);
        }

        if constexpr (std::is_same_v<VolScheme2, SchemeNV>) {
          Z_v1_fine_1 = Numerics::normcdfinv_as241(dist_uni(rng));
          v1_fine_mid = VolScheme2::evolve_coupled(v_fine_curr[1], Z_v1_fine_1,
                                                   data1_fine.wksp);
        } else {
          v1_fine_mid =
              VolScheme2::evolve(v_fine_curr[1], rng, data1_fine.wksp);
        }

        double jump_fine_1 = 0.0, jump_fine_2 = 0.0;

        double v_avg_root_f1 =
            std::sqrt(Cf0[3] * (v_fine_curr[0] + v0_fine_mid) +
                      Cf1[3] * (v_fine_curr[1] + v1_fine_mid));

        current_logS_fine += Cf0[0] + Cf0[1] * v_fine_curr[0] +
                             Cf0[2] * v0_fine_mid + Cf1[0] +
                             Cf1[1] * v_fine_curr[1] + Cf1[2] * v1_fine_mid +
                             v_avg_root_f1 * G_fine_1;

        if constexpr (TrackerPolicy::is_path_dependent) {
          jump_fine_1 =
              JumpPolicy::compute_log_jump(jump_params_, rng, jump_wksp_fine_);
          current_logS_fine += jump_fine_1;
        }
        state.fine.logS = current_logS_fine;
        if constexpr (TrackerPolicy::is_path_dependent)
          TrackerPolicy::update(state.fine, cfg, dt_fine_);

        // FINE STEP 2
        double v0_fine_next, v1_fine_next;
        double Z_v0_fine_2 = 0.0, Z_v1_fine_2 = 0.0;

        if constexpr (std::is_same_v<VolScheme1, SchemeNV>) {
          Z_v0_fine_2 = Numerics::normcdfinv_as241(dist_uni(rng));
          v0_fine_next = VolScheme1::evolve_coupled(v0_fine_mid, Z_v0_fine_2,
                                                    data0_fine.wksp);
        } else {
          v0_fine_next = VolScheme1::evolve(v0_fine_mid, rng, data0_fine.wksp);
        }

        if constexpr (std::is_same_v<VolScheme2, SchemeNV>) {
          Z_v1_fine_2 = Numerics::normcdfinv_as241(dist_uni(rng));
          v1_fine_next = VolScheme2::evolve_coupled(v1_fine_mid, Z_v1_fine_2,
                                                    data1_fine.wksp);
        } else {
          v1_fine_next = VolScheme2::evolve(v1_fine_mid, rng, data1_fine.wksp);
        }

        double v_avg_root_f2 = std::sqrt(Cf0[3] * (v0_fine_mid + v0_fine_next) +
                                         Cf1[3] * (v1_fine_mid + v1_fine_next));

        current_logS_fine += Cf0[0] + Cf0[1] * v0_fine_mid +
                             Cf0[2] * v0_fine_next + Cf1[0] +
                             Cf1[1] * v1_fine_mid + Cf1[2] * v1_fine_next +
                             v_avg_root_f2 * G_fine_2;

        if constexpr (TrackerPolicy::is_path_dependent) {
          jump_fine_2 =
              JumpPolicy::compute_log_jump(jump_params_, rng, jump_wksp_fine_);
          current_logS_fine += jump_fine_2;
        }
        state.fine.logS = current_logS_fine;
        if constexpr (TrackerPolicy::is_path_dependent)
          TrackerPolicy::update(state.fine, cfg, dt_fine_);

        // COARSE STEP
        double v0_coarse_next, v1_coarse_next;

        // Note: For Double factor, handling is symmetrical
        if constexpr (std::is_same_v<VolScheme1, SchemeNCI> ||
                      std::is_same_v<VolScheme1, SchemeExact>) {
          v0_coarse_next = v0_fine_next;
        } else {
          double Z_v0_coarse =
              (Z_v0_fine_1 + Z_v0_fine_2) / std::numbers::sqrt2;
          v0_coarse_next = VolScheme1::evolve_coupled(
              v_coarse_curr[0], Z_v0_coarse, data0_coarse.wksp);
        }

        if constexpr (std::is_same_v<VolScheme2, SchemeNCI> ||
                      std::is_same_v<VolScheme2, SchemeExact>) {
          v1_coarse_next = v1_fine_next;
        } else {
          double Z_v1_coarse =
              (Z_v1_fine_1 + Z_v1_fine_2) / std::numbers::sqrt2;
          v1_coarse_next = VolScheme2::evolve_coupled(
              v_coarse_curr[1], Z_v1_coarse, data1_coarse.wksp);
        }

        double v_avg_root_c =
            std::sqrt(Cc0[3] * (v_coarse_curr[0] + v0_coarse_next) +
                      Cc1[3] * (v_coarse_curr[1] + v1_coarse_next));

        if (v_avg_root_f1 + v_avg_root_f2 > 1e-15) {
          G_coarse = (v_avg_root_f1 * G_fine_1 + v_avg_root_f2 * G_fine_2) /
                     std::hypot(v_avg_root_f1, v_avg_root_f2);
        }

        current_logS_coarse +=
            Cc0[0] + Cc0[1] * v_coarse_curr[0] + Cc0[2] * v0_coarse_next +
            Cc1[0] + Cc1[1] * v_coarse_curr[1] + Cc1[2] * v1_coarse_next +
            v_avg_root_c * G_coarse;

        v_fine_curr[0] = v0_fine_next;
        v_fine_curr[1] = v1_fine_next;
        v_coarse_curr[0] = v0_coarse_next;
        v_coarse_curr[1] = v1_coarse_next;

        if constexpr (TrackerPolicy::is_path_dependent) {
          current_logS_coarse += jump_fine_1 + jump_fine_2;
          state.coarse.logS = current_logS_coarse;
          TrackerPolicy::update(state.coarse, cfg, dt_coarse_);
        }
      }

      // Final Path state resolution
      if constexpr (!TrackerPolicy::is_path_dependent) {
        state.coarse.logS = current_logS_coarse;
      }
    }

    state.fine.logS = current_logS_fine;
    state.fine.V = v_fine_curr;

    state.coarse.logS = current_logS_coarse;
    state.coarse.V = v_coarse_curr;

    TrackerPolicy::finalize(state.fine, cfg, dt_fine_, total_time);
    TrackerPolicy::finalize(state.coarse, cfg, dt_coarse_, total_time);
  }

private:
  using DiscrParams = std::array<double, 4>;

  struct GbmData {
    double drift_dt;
    double vol_dt;
  };

  template <typename Scheme> struct CirData {
    DiscrParams C;
    typename Scheme::Workspace wksp;
    double v0;
  };

  using DiffusionTuple = typename std::conditional<
      NumCirs == 0, std::tuple<GbmData>,
      typename std::conditional<
          NumCirs == 1, std::tuple<CirData<VolScheme1>>,
          std::tuple<CirData<VolScheme1>, CirData<VolScheme2>>>::type>::type;

  DiffusionTuple diffusion_data_fine_;
  DiffusionTuple diffusion_data_coarse_;

  typename JumpPolicy::Workspace jump_wksp_fine_;
  JumpP jump_params_;
  double dt_fine_;
  double dt_coarse_;

  // Private Helper Methods
  template <typename Scheme>
  void init_cir_impl(CirData<Scheme> &data,
                     const typename Scheme::GlobalWorkspace &vol_gw,
                     const HestonParams &p, double r_part, double time_step) {
    DiscrParams &C = data.C;
    data.v0 = p.v0;

    double kappa = p.kappa;
    double theta = p.theta;
    double sigma = p.sigma;
    double rho = p.rho;
    double h = time_step;

    // Log-Stock discretization parameters
    C[0] = (r_part - rho * kappa * theta / sigma) * h;
    double term = (kappa * rho / sigma - 0.5) * h * 0.5;
    C[1] = term - (rho / sigma);
    C[2] = term + (rho / sigma);
    C[3] = h * (1.0 - rho * rho) * 0.5;

    Scheme::prepare(data.wksp, vol_gw, p, h);
  }
};

// =============================================================================
// CONVENIENCE ALIASES FOR COUPLED STEPPERS
// =============================================================================

// --- ZERO FACTOR STEPPERS ---
template <typename Tracker = TrackerEuropean>
using CoupledBlackScholesStepper =
    CoupledASVJStepper<NullVolScheme, NullVolScheme, NoJumps, Tracker>;

template <typename Tracker = TrackerEuropean>
using CoupledMertonJDStepper =
    CoupledASVJStepper<NullVolScheme, NullVolScheme, MertonJump, Tracker>;

template <typename Tracker = TrackerEuropean>
using CoupledKouJDStepper =
    CoupledASVJStepper<NullVolScheme, NullVolScheme, KouJump, Tracker>;

// --- SINGLE FACTOR STEPPERS ---
template <typename VolScheme, typename Tracker = TrackerEuropean>
using CoupledHestonStepper =
    CoupledASVJStepper<VolScheme, NullVolScheme, NoJumps, Tracker>;

template <typename VolScheme, typename Tracker = TrackerEuropean>
using CoupledBatesStepper =
    CoupledASVJStepper<VolScheme, NullVolScheme, MertonJump, Tracker>;

template <typename VolScheme, typename Tracker = TrackerEuropean>
using CoupledBatesKouStepper =
    CoupledASVJStepper<VolScheme, NullVolScheme, KouJump, Tracker>;

// --- DOUBLE FACTOR STEPPERS ---
template <typename VolScheme1, typename VolScheme2,
          typename Tracker = TrackerEuropean>
using CoupledDoubleHestonStepper =
    CoupledASVJStepper<VolScheme1, VolScheme2, NoJumps, Tracker>;

template <typename VolScheme1, typename VolScheme2,
          typename Tracker = TrackerEuropean>
using CoupledDoubleBatesStepper =
    CoupledASVJStepper<VolScheme1, VolScheme2, MertonJump, Tracker>;

template <typename VolScheme1, typename VolScheme2,
          typename Tracker = TrackerEuropean>
using CoupledDoubleBatesKouStepper =
    CoupledASVJStepper<VolScheme1, VolScheme2, KouJump, Tracker>;
