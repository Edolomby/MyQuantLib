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
  using PayoffType = PayoffPolicy;

  EuropeanOption(double K, double T) : strike_(K), T_(T) {}

  typename Tracker::Config get_tracker_config() const { return {}; }

  template <typename State> double calculate(const State &state) const {
    // TrackerEuropean::finalize has already converted logS -> Price
    return payoff_func_(state.logS, strike_);
  }

  // Buffer-based calculate (scalar version)
  template <typename State>
  void calculate_to_buffer(const State &state, double &buffer) const {
    buffer = payoff_func_(state.logS, strike_);
  }

  size_t size() const { return 1; }
  double get_maturity() const { return T_; }
  double get_strike() const { return strike_; }
};

// =============================================================================
// EUROPEAN STRIP (Vector - Buffer Support)
// =============================================================================
template <typename PayoffPolicy> class EuropeanStrip {
  std::vector<double> strikes_;
  double T_;
  PayoffPolicy payoff_func_;

public:
  using Tracker = TrackerEuropean;
  using ResultType = std::vector<double>;
  using PayoffType = PayoffPolicy;

  EuropeanStrip(const std::vector<double> &strikes, double T)
      : strikes_(strikes), T_(T) {}

  typename Tracker::Config get_tracker_config() const { return {}; }

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

  // Buffer-base calculate
  template <typename State>
  void calculate_to_buffer(const State &state,
                           std::vector<double> &buffer) const {
    double S_T = state.logS;
    for (size_t i = 0; i < strikes_.size(); ++i) {
      buffer[i] = payoff_func_(S_T, strikes_[i]);
    }
  }

  size_t size() const { return strikes_.size(); }
  double get_maturity() const { return T_; }
  const std::vector<double> &get_strikes() const { return strikes_; }
};