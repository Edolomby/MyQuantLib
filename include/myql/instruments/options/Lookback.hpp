#pragma once
#include <cmath>
#include <myql/core/PricingTypes.hpp>
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/trackers/PathTrackers.hpp>
#include <type_traits>

template <typename PayoffT, bool FixedStrike, typename StrikeContainer = double>
class LookbackOption {
  StrikeContainer strikes_;
  double T_;
  PayoffT payoff_func_;

  // Historical extremes observed before the pricing date.
  // 0.0 means "not set" — the tracker will default to S0.
  double hist_min_S_ = 0.0;
  double hist_max_S_ = 0.0;

  static constexpr bool is_scalar = std::is_floating_point_v<StrikeContainer>;

  // Helper to compute payoff based on strike type (Mode forwarded to payoff)
  template <GreekMode Mode = GreekMode::None>
  inline double compute_payoff(double spot_scaled, double extreme_scaled,
                               double K) const {
    if constexpr (FixedStrike) {
      return payoff_func_.template operator()<Mode>(extreme_scaled, K);
    } else {
      // Floating call: max(spot - extreme, 0) -> payoff_func_Call(spot,
      // extreme) Floating put:  max(extreme - spot, 0) -> payoff_func_Put(spot,
      // extreme)
      return payoff_func_.template operator()<Mode>(spot_scaled,
                                                    extreme_scaled);
    }
  }

public:
  using Tracker = TrackerLookback;
  using ResultType = StrikeContainer;
  using PayoffType = PayoffT;

  // hist_min_S: observed minimum price over the elapsed period (0 = no history)
  // hist_max_S: observed maximum price over the elapsed period (0 = no history)
  LookbackOption(const StrikeContainer &K, double T, double hist_min_S = 0.0,
                 double hist_max_S = 0.0, const PayoffT &payoff = PayoffT())
      : strikes_(K), T_(T), payoff_func_(payoff), hist_min_S_(hist_min_S),
        hist_max_S_(hist_max_S) {}

  template <GreekMode Mode = GreekMode::None>
  typename Tracker::Config
  get_tracker_config([[maybe_unused]] double S0 = 100.0,
                     [[maybe_unused]] double h = 0.0,
                     [[maybe_unused]] double T_bumped = 0.0) const {
    Tracker::Config cfg;
    if (hist_min_S_ > 0.0)
      cfg.hist_min_logS = std::log(hist_min_S_);
    if (hist_max_S_ > 0.0)
      cfg.hist_max_logS = std::log(hist_max_S_);
    return cfg;
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
      base = compute_payoff<Mode>(S_T, extreme, strikes_);
      if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
        up = compute_payoff<Mode>(S_T * mult_up, extreme * mult_up, strikes_);
        dn = compute_payoff<Mode>(S_T * mult_dn, extreme * mult_dn, strikes_);
      }
    } else {
      for (size_t i = 0; i < strikes_.size(); ++i) {
        base[i] = compute_payoff<Mode>(S_T, extreme, strikes_[i]);
        if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
          up[i] = compute_payoff<Mode>(S_T * mult_up, extreme * mult_up,
                                       strikes_[i]);
          dn[i] = compute_payoff<Mode>(S_T * mult_dn, extreme * mult_dn,
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
  PayoffT &get_payoff_mut() { return payoff_func_; }
};