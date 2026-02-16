#pragma once
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/trackers/PathTrackers.hpp>
#include <type_traits>
#include <vector>

// =============================================================================
// SINGLE ASIAN OPTION
// =============================================================================
// TrackerT: TrackerGeoAsian or TrackerArithAsian
// PayoffT: PayoffVanilla or PayoffDigital
// FixedStrike: true = Fixed K, false = Floating (Strike is Average)
template <typename TrackerT, typename PayoffT, bool FixedStrike>
class AsianOption {
  double strike_;
  double T_;
  PayoffT payoff_func_;

public:
  using Tracker = TrackerT;
  using ResultType = double;

  AsianOption(double K, double T) : strike_(K), T_(T) {}

  template <typename State> double calculate(const State &state) const {
    // 1. Get the Average Price from the Tracker state
    double average_price = 0.0;

    if constexpr (std::is_same_v<Tracker, TrackerGeoAsian>) {
      // TrackerGeoAsian::finalize converted sumLogS -> Geometric Average Price
      average_price = state.sumLogS;
    } else {
      // TrackerArithAsian::finalize converted sumS -> Arithmetic Average Price
      average_price = state.sumS;
    }

    // 2. Payoff
    if constexpr (FixedStrike) {
      // Fixed: Payoff(Average, K)
      return payoff_func_(average_price, strike_);
    } else {
      // Floating: Payoff(Spot, Average)
      // Tracker finalized logS -> Price S_T
      return payoff_func_(state.logS, average_price);
    }
  }

  double get_maturity() const { return T_; }
};

// =============================================================================
// ASIAN FIXED STRIP (Smile/Vector)
// =============================================================================
template <typename TrackerT, typename PayoffT> class AsianFixedStrip {
  std::vector<double> strikes_;
  double T_;
  PayoffT payoff_func_;

public:
  using Tracker = TrackerT;
  using ResultType = std::vector<double>;

  AsianFixedStrip(const std::vector<double> &strikes, double T)
      : strikes_(strikes), T_(T) {}

  template <typename State>
  std::vector<double> calculate(const State &state) const {
    std::vector<double> results;
    results.reserve(strikes_.size());

    // 1. Get Average Price ONCE
    double average_price = 0.0;
    if constexpr (std::is_same_v<Tracker, TrackerGeoAsian>) {
      average_price = state.sumLogS;
    } else {
      average_price = state.sumS;
    }

    // 2. Vectorized Payoff
    for (double K : strikes_) {
      results.push_back(payoff_func_(average_price, K));
    }
    return results;
  }

  double get_maturity() const { return T_; }
};