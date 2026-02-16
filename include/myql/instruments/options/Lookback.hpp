#pragma once
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/trackers/PathTrackers.hpp>
#include <type_traits>
#include <vector>

// =============================================================================
// SINGLE LOOKBACK OPTION
// =============================================================================
template <typename PayoffT, bool FixedStrike> class LookbackOption {
  double strike_;
  double T_;
  PayoffT payoff_func_;

public:
  using Tracker = TrackerLookback;
  using ResultType = double;

  LookbackOption(double K, double T) : strike_(K), T_(T) {}

  template <typename State> double calculate(const State &state) const {
    // Determine Call/Put to select Min or Max
    constexpr bool is_call =
        std::is_same_v<PayoffT, PayoffVanilla<OptionType::Call>>;

    if constexpr (FixedStrike) {
      // Fixed Strike
      // Call: max(Max - K, 0) -> Use maxLogS (which is Max Price)
      // Put:  max(K - Min, 0) -> Use minLogS (which is Min Price)
      double extreme = is_call ? state.maxLogS : state.minLogS;
      return payoff_func_(extreme, strike_);
    } else {
      // Floating Strike
      // Call: max(S_T - Min, 0) -> Strike is Min
      // Put:  max(Max - S_T, 0) -> Strike is Max
      double floating_strike = is_call ? state.minLogS : state.maxLogS;
      return payoff_func_(state.logS, floating_strike);
    }
  }

  double get_maturity() const { return T_; }
};

// =============================================================================
// LOOKBACK FIXED STRIP (Smile/Vector)
// =============================================================================
template <typename PayoffT> class LookbackFixedStrip {
  std::vector<double> strikes_;
  double T_;
  PayoffT payoff_func_;

public:
  using Tracker = TrackerLookback;
  using ResultType = std::vector<double>;

  LookbackFixedStrip(const std::vector<double> &strikes, double T)
      : strikes_(strikes), T_(T) {}

  template <typename State>
  std::vector<double> calculate(const State &state) const {
    std::vector<double> results;
    results.reserve(strikes_.size());

    constexpr bool is_call =
        std::is_same_v<PayoffT, PayoffVanilla<OptionType::Call>>;

    // Optimization: Fetch extreme value ONCE
    double extreme = is_call ? state.maxLogS : state.minLogS;

    for (double K : strikes_) {
      results.push_back(payoff_func_(extreme, K));
    }
    return results;
  }

  double get_maturity() const { return T_; }
};