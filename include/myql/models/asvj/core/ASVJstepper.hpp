#pragma once
#include <array>
#include <boost/random.hpp>
#include <cmath>
#include <tuple>

#include <myql/instruments/trackers/PathTrackers.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/data/ModelParams.hpp>
#include <myql/models/asvj/policies/JumpPolicies.hpp>
#include <myql/models/asvj/policies/VolSchemes.hpp>
#include <type_traits>

// =============================================================================
// THE STEPPER CLASS
// =============================================================================

template <typename VolScheme1,                      // Mandatory: 1st Vol Scheme
          typename VolScheme2 = NullVolScheme,      // Optional: 2nd Vol Scheme
          typename JumpPolicy = NoJumps,            // Optional: Jumps
          typename TrackerPolicy = TrackerEuropean> // Optional: Tracker
class ASVJStepper {
  // DEDUCE Number of Cirs AUTOMATICALLY at compile time
  static constexpr int NumCirs = (std::is_same_v<VolScheme1, NullVolScheme> &&
                                  std::is_same_v<VolScheme2, NullVolScheme>)
                                     ? 0
                                 : (std::is_same_v<VolScheme2, NullVolScheme>)
                                     ? 1
                                     : 2;

  // Static Checks
  static_assert(NumCirs == 0 || NumCirs == 1 || NumCirs == 2,
                "ASVJStepper only supports 0, 1 or 2 CIR processes.");
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
  ASVJStepper(const ModelParams &model, double time_step, double r, double q,
              double total_time, const VolGlobalWorkspace &vol_gw)
      : jump_params_(model.jump), dt_(time_step) {

    // If Path Dependent: Prepare workspace for small step 'dt'.
    // If European: Prepare workspace for ONE giant step 'total_time'.
    if constexpr (TrackerPolicy::is_path_dependent) {
      JumpPolicy::prepare(jump_wksp_, jump_params_, dt_);
    } else {
      JumpPolicy::prepare(jump_wksp_, jump_params_, total_time);
    }

    if constexpr (NumCirs == 0) {
      double var = model.vol * model.vol;
      auto &gbm = std::get<0>(diffusion_data_);
      gbm.drift_dt = (r - q - 0.5 * var) * dt_;
      gbm.vol_dt = model.vol * std::sqrt(dt_);
    } else {
      double r_part = (r - q) / static_cast<double>(NumCirs);
      if constexpr (NumCirs == 1) {
        init_cir_impl<VolScheme1>(std::get<0>(diffusion_data_),
                                  std::get<0>(vol_gw), model.heston, r_part,
                                  dt_);
      } else {
        init_cir_impl<VolScheme1>(std::get<0>(diffusion_data_),
                                  std::get<0>(vol_gw), model.heston1, r_part,
                                  dt_);
        init_cir_impl<VolScheme2>(std::get<1>(diffusion_data_),
                                  std::get<1>(vol_gw), model.heston2, r_part,
                                  dt_);
      }
    }
  }

  void start_path(State &state, const typename TrackerPolicy::Config &cfg,
                  double S0) const {
    // Reset Physics (The Stepper's job)
    state.logS = std::log(S0);
    // Use the NumCirs template parameter to decide how much to initialize
    if constexpr (NumCirs >= 1)
      state.V[0] = std::get<0>(diffusion_data_).v0;
    if constexpr (NumCirs >= 2)
      state.V[1] = std::get<1>(diffusion_data_).v0;

    TrackerPolicy::init(state, cfg, state.logS, dt_);
  }

  // =========================================================================
  // MULTI-STEP ENGINE
  // =========================================================================
  template <typename RNG>
  void multi_step(State &state, const typename TrackerPolicy::Config &cfg,
                  RNG &rng, unsigned n_steps) {
    boost::random::uniform_real_distribution<double> dist_uni(0.0, 1.0);
    std::array<double, NumCirs> v_curr = state.V;
    double current_logS = state.logS;
    double total_time = n_steps * dt_;

    // =======================================================================
    // OPTIMIZATION: 0-Factor (GBM) + European (Not Path Dependent)
    // =======================================================================
    if constexpr (NumCirs == 0 && !TrackerPolicy::is_path_dependent) {
      double G_stock = Numerics::normcdfinv_as241(dist_uni(rng));
      auto &gbm = std::get<0>(diffusion_data_);
      // Scale dt-drift by N, and dt-vol by sqrt(N) to reach T
      double total_drift = gbm.drift_dt * n_steps;
      double total_vol = gbm.vol_dt * std::sqrt(static_cast<double>(n_steps));

      current_logS += total_drift + total_vol * G_stock;

      // Add Giant Jump for total_time (already pre-prepared in constructor)
      double jump = JumpPolicy::compute_log_jump(jump_params_, rng, jump_wksp_);
      current_logS += jump;

      state.logS = current_logS;
      TrackerPolicy::finalize(state, cfg, dt_, total_time); // eu=exp the logS

      return; // EXIT EARLY
    }

    // =======================================================================
    // STANDARD LOOP: Path Dependent OR Stochastic Volatility (1 or 2 CIRs)
    // =======================================================================
    for (unsigned k = 0; k < n_steps; ++k) {
      // CASE 0: Path-Dependent Geometric Brownian Motion (e.g., Asian GBM)
      if constexpr (NumCirs == 0) {
        double G_stock = Numerics::normcdfinv_as241(dist_uni(rng));
        auto &gbm = std::get<0>(diffusion_data_);
        current_logS += gbm.drift_dt * dt_ + gbm.vol_dt * G_stock;
      }
      // CASE 1: Single Factor
      else if constexpr (NumCirs == 1) {
        auto &data = std::get<0>(diffusion_data_);
        const DiscrParams &C = data.C;
        double G_stock = Numerics::normcdfinv_as241(dist_uni(rng));

        double v_next = VolScheme1::evolve(v_curr[0], rng, data.wksp);
        double v_avg_root = std::sqrt(C[3] * (v_curr[0] + v_next));

        current_logS +=
            C[0] + C[1] * v_curr[0] + C[2] * v_next + v_avg_root * G_stock;

        v_curr[0] = v_next;
      }
      // CASE 2: Double Factor (Optimization in Law)
      else {
        auto &data0 = std::get<0>(diffusion_data_);
        auto &data1 = std::get<1>(diffusion_data_);
        const DiscrParams &C0 = data0.C, &C1 = data1.C;
        double G_stock = Numerics::normcdfinv_as241(dist_uni(rng));

        double v0_next = VolScheme1::evolve(v_curr[0], rng, data0.wksp);
        double v1_next = VolScheme2::evolve(v_curr[1], rng, data1.wksp);
        double v_avg_root = std::sqrt(C0[3] * (v_curr[0] + v0_next) +
                                      C1[3] * (v_curr[1] + v1_next));

        current_logS += C0[0] + C0[1] * v_curr[0] + C0[2] * v0_next + C1[0] +
                        C1[1] * v_curr[1] + C1[2] * v1_next +
                        v_avg_root * G_stock;

        v_curr[0] = v0_next;
        v_curr[1] = v1_next;
      }
      // Jumps (PATH DEPENDENT ONLY)
      if constexpr (TrackerPolicy::is_path_dependent) {
        double jump =
            JumpPolicy::compute_log_jump(jump_params_, rng, jump_wksp_);
        current_logS += jump;

        state.logS = current_logS;
        TrackerPolicy::update(state, cfg, dt_);
      }
    }

    // Jumps (EUROPEAN ONLY for NumCirs > 0)
    // Optimization: One giant jump at the end
    if constexpr (!TrackerPolicy::is_path_dependent && NumCirs > 0) {
      double jump = JumpPolicy::compute_log_jump(jump_params_, rng, jump_wksp_);
      current_logS += jump;
    }

    state.logS = current_logS;
    state.V = v_curr;

    // to renormalize and exponentiate
    TrackerPolicy::finalize(state, cfg, dt_, total_time);
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

  // Data members
  DiffusionTuple diffusion_data_;
  typename JumpPolicy::Workspace jump_wksp_;
  JumpP jump_params_;
  double dt_;

  // Precomputed variables for 0-Factor models
  double gbm_drift_dt_ = 0.0;
  double gbm_vol_dt_ = 0.0;

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
// CONVENIENCE ALIASES FOR STEPPERS
// =============================================================================

// --- ZERO FACTOR STEPPERS ---

// Pure Black-Scholes (Requires only the Tracker)
template <typename Tracker = TrackerEuropean>
using BlackScholesStepper =
    ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, Tracker>;

// Merton Jump-Diffusion
template <typename Tracker = TrackerEuropean>
using MertonJDStepper =
    ASVJStepper<NullVolScheme, NullVolScheme, MertonJump, Tracker>;

// Kou Jump-Diffusion
template <typename Tracker = TrackerEuropean>
using KouJDStepper =
    ASVJStepper<NullVolScheme, NullVolScheme, KouJump, Tracker>;

// --- SINGLE FACTOR STEPPERS ---

// Standard Heston (Requires VolScheme, Tracker is optional)
template <typename VolScheme, typename Tracker = TrackerEuropean>
using HestonStepper = ASVJStepper<VolScheme, NullVolScheme, NoJumps, Tracker>;

// Bates Model
template <typename VolScheme, typename Tracker = TrackerEuropean>
using BatesStepper = ASVJStepper<VolScheme, NullVolScheme, MertonJump, Tracker>;

// --- DOUBLE FACTOR STEPPERS ---

// Double Heston (Requires both VolSchemes, Tracker is optional)
template <typename VolScheme1, typename VolScheme2,
          typename Tracker = TrackerEuropean>
using DoubleHestonStepper =
    ASVJStepper<VolScheme1, VolScheme2, NoJumps, Tracker>;

// Double Bates Model
template <typename VolScheme1, typename VolScheme2,
          typename Tracker = TrackerEuropean>
using DoubleBatesStepper =
    ASVJStepper<VolScheme1, VolScheme2, MertonJump, Tracker>;

// `namespace myql {`) ? SHould I use it?