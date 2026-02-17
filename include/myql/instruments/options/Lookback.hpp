#pragma once
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/trackers/PathTrackers.hpp>
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
  using PayoffType = PayoffT;

  LookbackOption(double K, double T) : strike_(K), T_(T) {}

  template <typename State> double calculate(const State &state) const {
    double res;
    calculate_to_buffer(state, res);
    return res;
  }

  template <typename State>
  void calculate_to_buffer(const State &state, double &buffer) const {
    // We use the static constexpr Type we added to the Payoff structs
    constexpr bool is_call = (PayoffT::Type == OptionType::Call);

    if constexpr (FixedStrike) {
      double extreme = is_call ? state.maxLogS : state.minLogS;
      buffer = payoff_func_(extreme, strike_);
    } else {
      double floating_strike = is_call ? state.minLogS : state.maxLogS;
      buffer = payoff_func_(state.logS, floating_strike);
    }
  }

  size_t size() const { return 1; }
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
  using PayoffType = PayoffT;

  LookbackFixedStrip(const std::vector<double> &strikes, double T)
      : strikes_(strikes), T_(T) {}

  template <typename State>
  void calculate_to_buffer(const State &state,
                           std::vector<double> &buffer) const {
    constexpr bool is_call = (PayoffT::Type == OptionType::Call);
    double extreme = is_call ? state.maxLogS : state.minLogS;

    for (size_t i = 0; i < strikes_.size(); ++i) {
      buffer[i] = payoff_func_(extreme, strikes_[i]);
    }
  }

  size_t size() const { return strikes_.size(); }
  double get_maturity() const { return T_; }
  const std::vector<double> &get_strikes() const { return strikes_; }
};