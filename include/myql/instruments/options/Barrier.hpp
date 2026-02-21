#pragma once
#include <cmath>
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/trackers/PathTrackers.hpp>
#include <vector>

// =============================================================================
// ENUMS
// =============================================================================
enum class BarrierDirection { Up, Down };
enum class BarrierAction { In, Out };
enum class RebateTiming { None, AtHit, AtMaturity };

// =============================================================================
// SINGLE BARRIER OPTION
// =============================================================================
template <typename PayoffT, BarrierDirection Dir, BarrierAction Action,
          RebateTiming Timing>
class BarrierOption {
  double strike_;
  double barrier_;
  double T_;
  double rate_;
  double rebate_;
  PayoffT payoff_func_;

public:
  // Map the direction to the correct tracker at compile-time!
  using Tracker = TrackerBarrier<(Dir == BarrierDirection::Up)>;
  using ResultType = double;
  using PayoffType = PayoffT;

  BarrierOption(double K, double B, double T, double r = 0.0,
                double rebate = 0.0)
      : strike_(K), barrier_(B), T_(T), rate_(r), rebate_(rebate) {}

  typename Tracker::Config get_tracker_config() const {
    return {std::log(barrier_)};
  }

  template <typename State> double calculate(const State &state) const {
    double res;
    calculate_to_buffer(state, res);
    return res;
  }

  template <typename State>
  void calculate_to_buffer(const State &state, double &buffer) const {

    // Compile-time Rebate Logic
    double active_rebate = 0.0;
    if constexpr (Timing != RebateTiming::None) {
      active_rebate = rebate_;
      if constexpr (Timing == RebateTiming::AtHit) {
        if (state.is_hit) {
          // Forward compound to maturity T
          active_rebate = rebate_ * std::exp(rate_ * (T_ - state.hit_time));
        }
      }
    }

    // Payoff Logic (Tracker already converted state.logS to Price)
    double S_T = state.logS;

    if constexpr (Action == BarrierAction::In) {
      buffer = state.is_hit ? payoff_func_(S_T, strike_) : active_rebate;
    } else { // BarrierAction::Out
      buffer = state.is_hit ? active_rebate : payoff_func_(S_T, strike_);
    }
  }

  size_t size() const { return 1; }
  double get_maturity() const { return T_; }
  double get_strike() const { return strike_; }
  double get_barrier() const { return barrier_; } // Critical for the Engine
};

// =============================================================================
// BARRIER FIXED STRIP (Vectorized)
// =============================================================================
template <typename PayoffT, BarrierDirection Dir, BarrierAction Action,
          RebateTiming Timing>
class BarrierFixedStrip {
  std::vector<double> strikes_;
  double barrier_;
  double T_;
  double rate_;
  double rebate_;
  PayoffT payoff_func_;

public:
  using Tracker = TrackerBarrier<(Dir == BarrierDirection::Up)>;
  using ResultType = std::vector<double>;
  using PayoffType = PayoffT;

  BarrierFixedStrip(const std::vector<double> &strikes, double B, double T,
                    double r = 0.0, double rebate = 0.0)
      : strikes_(strikes), barrier_(B), T_(T), rate_(r), rebate_(rebate) {}

  typename Tracker::Config get_tracker_config() const {
    return {std::log(barrier_)};
  }

  template <typename State>
  void calculate_to_buffer(const State &state,
                           std::vector<double> &buffer) const {

    double active_rebate = 0.0;
    if constexpr (Timing != RebateTiming::None) {
      active_rebate = rebate_;
      if constexpr (Timing == RebateTiming::AtHit) {
        if (state.is_hit) {
          active_rebate = rebate_ * std::exp(rate_ * (T_ - state.hit_time));
        }
      }
    }

    double S_T = state.logS;

    for (size_t i = 0; i < strikes_.size(); ++i) {
      if constexpr (Action == BarrierAction::In) {
        buffer[i] =
            state.is_hit ? payoff_func_(S_T, strikes_[i]) : active_rebate;
      } else {
        buffer[i] =
            state.is_hit ? active_rebate : payoff_func_(S_T, strikes_[i]);
      }
    }
  }

  size_t size() const { return strikes_.size(); }
  double get_maturity() const { return T_; }
  double get_barrier() const { return barrier_; }
  const std::vector<double> &get_strikes() const { return strikes_; }
};

// =============================================================================
// Template Aliases
// =============================================================================

// UP-AND-OUT OPTIONS
template <RebateTiming Timing = RebateTiming::None>
using UpAndOutCall =
    BarrierOption<PayoffVanilla<OptionType::Call>, BarrierDirection::Up,
                  BarrierAction::Out, Timing>;

template <RebateTiming Timing = RebateTiming::None>
using UpAndOutPut =
    BarrierOption<PayoffVanilla<OptionType::Put>, BarrierDirection::Up,
                  BarrierAction::Out, Timing>;

// UP-AND-IN OPTIONS
template <RebateTiming Timing = RebateTiming::None>
using UpAndInCall =
    BarrierOption<PayoffVanilla<OptionType::Call>, BarrierDirection::Up,
                  BarrierAction::In, Timing>;

template <RebateTiming Timing = RebateTiming::None>
using UpAndInPut =
    BarrierOption<PayoffVanilla<OptionType::Put>, BarrierDirection::Up,
                  BarrierAction::In, Timing>;

// DOWN-AND-OUT OPTIONS
template <RebateTiming Timing = RebateTiming::None>
using DownAndOutCall =
    BarrierOption<PayoffVanilla<OptionType::Call>, BarrierDirection::Down,
                  BarrierAction::Out, Timing>;

template <RebateTiming Timing = RebateTiming::None>
using DownAndOutPut =
    BarrierOption<PayoffVanilla<OptionType::Put>, BarrierDirection::Down,
                  BarrierAction::Out, Timing>;

// DOWN-AND-IN OPTIONS
template <RebateTiming Timing = RebateTiming::None>
using DownAndInCall =
    BarrierOption<PayoffVanilla<OptionType::Call>, BarrierDirection::Down,
                  BarrierAction::In, Timing>;

template <RebateTiming Timing = RebateTiming::None>
using DownAndInPut =
    BarrierOption<PayoffVanilla<OptionType::Put>, BarrierDirection::Down,
                  BarrierAction::In, Timing>;