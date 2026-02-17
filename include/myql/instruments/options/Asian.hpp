#pragma once
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/trackers/PathTrackers.hpp>
#include <vector>

template <typename TrackerT, typename PayoffT, bool FixedStrike>
class AsianOption {
  double strike_;
  double T_;
  PayoffT payoff_func_;

public:
  using Tracker = TrackerT;
  using ResultType = double;
  using PayoffType = PayoffT;

  AsianOption(double K, double T) : strike_(K), T_(T) {}

  template <typename State>
  void calculate_to_buffer(const State &state, double &buffer) const {
    double average_price;
    if constexpr (std::is_same_v<Tracker, TrackerGeoAsian>) {
      average_price = state.sumLogS;
    } else {
      average_price = state.sumS;
    }

    if constexpr (FixedStrike) {
      buffer = payoff_func_(average_price, strike_);
    } else {
      buffer = payoff_func_(state.logS, average_price);
    }
  }

  size_t size() const { return 1; }
  double get_maturity() const { return T_; }
};

template <typename TrackerT, typename PayoffT> class AsianFixedStrip {
  std::vector<double> strikes_;
  double T_;
  PayoffT payoff_func_;

public:
  using Tracker = TrackerT;
  using ResultType = std::vector<double>;
  using PayoffType = PayoffT;

  AsianFixedStrip(const std::vector<double> &strikes, double T)
      : strikes_(strikes), T_(T) {}

  template <typename State>
  void calculate_to_buffer(const State &state,
                           std::vector<double> &buffer) const {
    double average_price;
    if constexpr (std::is_same_v<Tracker, TrackerGeoAsian>) {
      average_price = state.sumLogS;
    } else {
      average_price = state.sumS;
    }

    for (size_t i = 0; i < strikes_.size(); ++i) {
      buffer[i] = payoff_func_(average_price, strikes_[i]);
    }
  }

  size_t size() const { return strikes_.size(); }
  double get_maturity() const { return T_; }
};