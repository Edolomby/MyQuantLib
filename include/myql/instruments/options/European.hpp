#pragma once
#include <myql/core/PricingTypes.hpp>
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/trackers/PathTrackers.hpp>
#include <type_traits>

// =============================================================================
// UNIFIED EUROPEAN INSTRUMENT (Scalar & Vectorized)
// =============================================================================
template <typename PayoffPolicy, typename StrikeContainer = double>
class EuropeanOption {
  StrikeContainer strikes_;
  double T_;
  PayoffPolicy payoff_func_;

  // Helper trait to know if we are in 1D or N-D mode at compile time
  static constexpr bool is_scalar = std::is_floating_point_v<StrikeContainer>;

public:
  using Tracker = TrackerEuropean;
  using ResultType = StrikeContainer; // double OR std::vector<double>
  using PayoffType = PayoffPolicy;

  // Constructor handles both double and std::vector<double> automatically
  EuropeanOption(const StrikeContainer &K, double T,
                 const PayoffPolicy &payoff = PayoffPolicy())
      : strikes_(K), T_(T), payoff_func_(payoff) {}

  template <GreekMode Mode = GreekMode::None>
  typename Tracker::Config
  get_tracker_config([[maybe_unused]] double S0 = 100.0,
                     [[maybe_unused]] double h = 0.0,
                     [[maybe_unused]] double T_bumped = 0.0) const {
    return {};
  }

  // -------------------------------------------------------------------------
  // GREEK-AWARE UNIFIED BUFFER
  // -------------------------------------------------------------------------
  template <GreekMode Mode, typename State>
  void calculate_to_buffer(const State &state, double S0, double h,
                           ResultType &base, ResultType &up,
                           ResultType &dn) const {

    double S_T = state.logS;

    double mult_up = 1.0;
    double mult_dn = 1.0;
    if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
      mult_up = (S0 + h) / S0;
      mult_dn = (S0 - h) / S0;
    }

    // --- COMPILE-TIME BRANCH: SCALAR (1D) ---
    if constexpr (is_scalar) {
      base = payoff_func_.template operator()<Mode>(S_T, strikes_);
      if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
        up = payoff_func_.template operator()<Mode>(S_T * mult_up, strikes_);
        dn = payoff_func_.template operator()<Mode>(S_T * mult_dn, strikes_);
      }
    }
    // --- COMPILE-TIME BRANCH: VECTORIZED (N-D) ---
    else {
      for (size_t i = 0; i < strikes_.size(); ++i) {
        base[i] = payoff_func_.template operator()<Mode>(S_T, strikes_[i]);
        if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
          up[i] = payoff_func_.template operator()<Mode>(S_T * mult_up,
                                                         strikes_[i]);
          dn[i] = payoff_func_.template operator()<Mode>(S_T * mult_dn,
                                                         strikes_[i]);
        }
      }
    }
  }

  size_t size() const {
    if constexpr (is_scalar)
      return 1;
    else
      return strikes_.size();
  }

  double get_maturity() const { return T_; }
  const StrikeContainer &get_strikes() const { return strikes_; }
  PayoffPolicy &get_payoff_mut() { return payoff_func_; }
};