#pragma once
#include <myql/core/PricingTypes.hpp>
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/trackers/PathTrackers.hpp>
#include <type_traits>

template <typename PayoffT, bool FixedStrike, typename StrikeContainer = double>
class LookbackOption {
  StrikeContainer strikes_;
  double T_;
  PayoffT payoff_func_;

  static constexpr bool is_scalar = std::is_floating_point_v<StrikeContainer>;

  // Helper to compute payoff based on strike type //prevent to many ifs
  inline double compute_payoff(double spot_scaled, double extreme_scaled,
                               double K) const {
    if constexpr (FixedStrike) {
      return payoff_func_(extreme_scaled, K);
    } else {
      return payoff_func_(spot_scaled, extreme_scaled);
    }
  }

public:
  using Tracker = TrackerLookback;
  using ResultType = StrikeContainer;
  using PayoffType = PayoffT;

  LookbackOption(const StrikeContainer &K, double T) : strikes_(K), T_(T) {}

  template <GreekMode Mode = GreekMode::None>
  typename Tracker::Config
  get_tracker_config([[maybe_unused]] double S0 = 100.0,
                     [[maybe_unused]] double h = 0.0) const {
    return {};
  }

  template <GreekMode Mode, typename State>
  void calculate_to_buffer(const State &state, double S0, double h,
                           ResultType &base, ResultType &up,
                           ResultType &dn) const {

    constexpr bool is_call = (PayoffT::Type == OptionType::Call);
    double S_T = state.logS;

    double extreme = 0.0;
    if constexpr (FixedStrike) {
      extreme = is_call ? state.maxLogS : state.minLogS;
    } else {
      extreme = is_call ? state.minLogS : state.maxLogS;
    }

    double mult_up = 1.0;
    double mult_dn = 1.0;
    if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
      mult_up = (S0 + h) / S0;
      mult_dn = (S0 - h) / S0;
    }

    if constexpr (is_scalar) {
      base = compute_payoff(S_T, extreme, strikes_);
      if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
        up = compute_payoff(S_T * mult_up, extreme * mult_up, strikes_);
        dn = compute_payoff(S_T * mult_dn, extreme * mult_dn, strikes_);
      }
    } else {
      for (size_t i = 0; i < strikes_.size(); ++i) {
        base[i] = compute_payoff(S_T, extreme, strikes_[i]);
        if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
          up[i] = compute_payoff(S_T * mult_up, extreme * mult_up, strikes_[i]);
          dn[i] = compute_payoff(S_T * mult_dn, extreme * mult_dn, strikes_[i]);
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
};