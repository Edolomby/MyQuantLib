#pragma once
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/trackers/PathTrackers.hpp>
#include <vector>

// =============================================================================
// SINGLE EUROPEAN OPTION
// =============================================================================
template <typename PayoffPolicy> class EuropeanOption {
  double strike_;
  double T_;
  PayoffPolicy payoff_func_;

public:
  using Tracker = TrackerEuropean;
  using ResultType = double;

  EuropeanOption(double K, double T) : strike_(K), T_(T) {}

  template <typename State> double calculate(const State &state) const {
    // TrackerEuropean::finalize has already converted logS -> Price
    return payoff_func_(state.logS, strike_);
  }

  double get_maturity() const { return T_; }
};

// =============================================================================
// EUROPEAN STRIP (Smile/Vector)
// =============================================================================
template <typename PayoffPolicy> class EuropeanStrip {
  std::vector<double> strikes_;
  double T_;
  PayoffPolicy payoff_func_;

public:
  using Tracker = TrackerEuropean;
  using ResultType = std::vector<double>;

  EuropeanStrip(const std::vector<double> &strikes, double T)
      : strikes_(strikes), T_(T) {}

  template <typename State>
  std::vector<double> calculate(const State &state) const {
    std::vector<double> results;
    results.reserve(strikes_.size());

    // Optimization: Access memory once
    double S_T = state.logS; // the state has already been exponentiated

    for (double K : strikes_) {
      results.push_back(payoff_func_(S_T, K));
    }
    return results;
  }

  double get_maturity() const { return T_; }
};