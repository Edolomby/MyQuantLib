#pragma once
#include <array>
#include <boost/random.hpp>
#include <cmath>
#include <myql/instruments/trackers/PathTrackers.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/data/ModelParams.hpp>
#include <myql/models/asvj/policies/JumpPolicies.hpp>
#include <myql/models/asvj/policies/VolSchemes.hpp>
#include <type_traits>

// =============================================================================
// THE STEPPER CLASS
// =============================================================================

template <int NumCirs,           // 1 or 2
          typename VolScheme,    // Restricted or General
          typename JumpPolicy,   // Merton, Kou, or NoJumps
          typename TrackerPolicy // European, GeoAsian, ArithAsian
          >
class ASVJStepper {
  // Static Checks
  static_assert(NumCirs == 1 || NumCirs == 2,
                "ASVJStepper only supports 1 or 2 CIR processes.");
  static_assert(std::is_same_v<VolScheme, SchemeLowVol> ||
                    std::is_same_v<VolScheme, SchemeHighVol>,
                "VolScheme must be SchemeLowVol or SchemeHighVol");

public:
  // Public Aliases for Engine
  using Policy = TrackerPolicy;
  using JumpP = typename JumpPolicy::Params;

  using ModelParams =
      typename std::conditional<NumCirs == 1, SingleFactorModel<JumpP>,
                                DoubleFactorModel<JumpP>>::type;

  struct State : public TrackerPolicy::ExtraState {
    double logS;
    std::array<double, NumCirs> V;
  };

private:
  using DiscrParams = std::array<double, 8>;

  struct CirData {
    DiscrParams C;
    typename VolScheme::Workspace wksp;
    double v0;
  };

  std::array<CirData, NumCirs> cir_data_;
  typename JumpPolicy::Workspace jump_wksp_; // Persistent Workspace
  JumpP jump_params_;
  double dt_;

public:
  // =========================================================================
  // CONSTRUCTOR
  // =========================================================================
  ASVJStepper(const ModelParams &model, double time_step, double r, double q,
              double total_time)
      : jump_params_(model.jump), dt_(time_step) {

    // [OPTIMIZATION]:
    // If Path Dependent: Prepare workspace for small step 'dt'.
    // If European: Prepare workspace for ONE giant step 'total_time'.
    if constexpr (TrackerPolicy::is_path_dependent) {
      JumpPolicy::prepare(jump_wksp_, jump_params_, dt_);
    } else {
      JumpPolicy::prepare(jump_wksp_, jump_params_, total_time);
    }

    double r_part = (r - q) / static_cast<double>(NumCirs);

    auto init_cir = [&](CirData &data, const HestonParams &p) {
      DiscrParams &C = data.C;
      data.v0 = p.v0;

      double kappa = p.kappa;
      double theta = p.theta;
      double sigma = p.sigma;
      double rho = p.rho;
      double h = time_step;

      bool is_high_vol_regime = (sigma * sigma) > (4.0 * kappa * theta);

      if constexpr (std::is_same_v<VolScheme, SchemeLowVol>) {
        if (is_high_vol_regime)
          throw std::runtime_error("SchemeLowVol violation.");
        double ex = std::exp(-kappa * h * 0.5);
        double psi = (std::abs(kappa) < 1e-10) ? h * 0.5 : (1.0 - ex) / kappa;
        double D = (kappa * theta) - (sigma * sigma * 0.25);
        C[0] = ex;
        C[1] = D * psi;
        C[2] = sigma * std::sqrt(h) * 0.5;
      } else {
        double ex = std::exp(-kappa * h);
        double psi = (std::abs(kappa) < 1e-8) ? h : (1.0 - ex) / kappa;
        C[0] = (2.0 / (sigma * sigma * psi)) * ex;
        C[1] = (2.0 * kappa * theta) / (sigma * sigma);
        C[2] = 0.5 * (sigma * sigma * psi);
      }

      C[3] = (r_part - rho * kappa * theta / sigma) * h;
      double term = (kappa * rho / sigma - 0.5) * h * 0.5;
      C[4] = term - (rho / sigma);
      C[5] = term + (rho / sigma);
      C[6] = std::sqrt(h * (1.0 - rho * rho) * 0.5);
      C[7] = std::sqrt(time_step);

      VolScheme::prepare(data.wksp, p);
    };

    if constexpr (NumCirs == 1) {
      init_cir(cir_data_[0], model.heston);
    } else {
      init_cir(cir_data_[0], model.heston1);
      init_cir(cir_data_[1], model.heston2);
    }
  }

  void start_path(State &state, double S0) const {
    // 1. Reset Physics (The Stepper's job)
    state.logS = std::log(S0);
    // Use the NumCirs template parameter to decide how much to initialize
    if constexpr (NumCirs >= 1) {
      state.V[0] = cir_data_[0].v0;
    }
    if constexpr (NumCirs >= 2) {
      state.V[1] = cir_data_[1].v0;
    }

    // 2. Reset Trajectory/History (The Tracker's job)
    // We delegate this because the Stepper knows 'dt_' but the Engine doesn't
    // need to know it.
    TrackerPolicy::init(state, state.logS, dt_);
  }

  // =========================================================================
  // MULTI-STEP ENGINE
  // =========================================================================
  template <typename RNG>
  void multi_step(State &state, RNG &rng, unsigned n_steps) {
    boost::random::uniform_real_distribution<double> dist_uni(0.0, 1.0);
    std::array<double, NumCirs> v_curr = state.V;
    double current_logS = state.logS;

    for (unsigned k = 0; k < n_steps; ++k) {
      // 1. Diffusion Steps (Heston)
      for (unsigned i = 0; i < NumCirs; ++i) {
        CirData &data = cir_data_[i];
        const DiscrParams &C = data.C;

        double G_stock = Numerics::normcdfinv_as241(dist_uni(rng));
        double Z_V = 0.0;
        if constexpr (std::is_same_v<VolScheme, SchemeLowVol>)
          Z_V = Numerics::normcdfinv_as241(dist_uni(rng));

        double v_next = VolScheme::evolve(v_curr[i], C, Z_V, rng, data.wksp);

        double v_safe_curr = std::max(0.0, v_curr[i]);
        double v_safe_next = std::max(0.0, v_next);
        double v_avg_root = std::sqrt(v_safe_curr + v_safe_next);

        current_logS += C[3] + C[4] * v_curr[i] + C[5] * v_next +
                        C[6] * v_avg_root * G_stock;

        v_curr[i] = v_next;
      }

      // 2. Jumps (PATH DEPENDENT ONLY)
      // Jump every step because we need accurate path averages
      if constexpr (TrackerPolicy::is_path_dependent) {
        double jump =
            JumpPolicy::compute_log_jump(jump_params_, rng, jump_wksp_);
        current_logS += jump;

        state.logS = current_logS;
        TrackerPolicy::update(state, dt_);
      }
    }

    // 3. Jumps (EUROPEAN ONLY)
    // Optimization: One giant jump at the end
    if constexpr (!TrackerPolicy::is_path_dependent) {
      // The workspace was already prepared with 'total_time' in the
      // constructor!
      double jump = JumpPolicy::compute_log_jump(jump_params_, rng, jump_wksp_);
      current_logS += jump;
    }

    state.logS = current_logS;
    state.V = v_curr;

    double total_time = n_steps * dt_;
    TrackerPolicy::finalize(state, dt_, total_time);
  }
};