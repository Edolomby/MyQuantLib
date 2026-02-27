#pragma once
#include <cmath>
#include <myql/core/PricingTypes.hpp>
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/trackers/PathTrackers.hpp>

// =============================================================================
// ENUMS
// =============================================================================
enum class BarrierDirection { Up, Down };
enum class BarrierAction { In, Out };
enum class RebateTiming { None, AtHit, AtMaturity };

// =============================================================================
// UNIFIED SINGLE BARRIER OPTION
// =============================================================================
template <typename PayoffT, BarrierDirection Dir, BarrierAction Action,
          RebateTiming Timing, typename StrikeContainer = double>
class BarrierOption {
  StrikeContainer strikes_;
  double barrier_;
  double T_;
  double rate_;
  double rebate_;
  PayoffT payoff_func_;

  static constexpr bool is_scalar = std::is_floating_point_v<StrikeContainer>;

  inline double compute_payoff(double spot, double K, bool is_hit,
                               double hit_time) const {
    double active_rebate = 0.0;
    if constexpr (Timing != RebateTiming::None) {
      active_rebate = rebate_;
      if constexpr (Timing == RebateTiming::AtHit) {
        if (is_hit)
          active_rebate = rebate_ * std::exp(rate_ * (T_ - hit_time));
      }
    }

    if constexpr (Action == BarrierAction::In) {
      return is_hit ? payoff_func_(spot, K) : active_rebate;
    } else {
      return is_hit ? active_rebate : payoff_func_(spot, K);
    }
  }

public:
  using Tracker = TrackerBarrier<(Dir == BarrierDirection::Up)>;
  using ResultType = StrikeContainer;
  using PayoffType = PayoffT;

  BarrierOption(const StrikeContainer &K, double B, double T, double r = 0.0,
                double rebate = 0.0, const PayoffT &payoff = PayoffT())
      : strikes_(K), barrier_(B), T_(T), rate_(r), rebate_(rebate),
        payoff_func_(payoff) {}

  // 1. CONFIGURATION: Pass the GreekMode, S0, and h to compute the 3 barriers
  template <GreekMode Mode>
  typename Tracker::Config get_tracker_config(double S0 = 100.0,
                                              double h = 0.0) const {
    typename Tracker::Config cfg;
    cfg.barrier_log_base = std::log(barrier_);

    if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
      // Inversely scale the barriers!
      cfg.barrier_log_up = std::log(barrier_ * S0 / (S0 + h));
      cfg.barrier_log_dn = std::log(barrier_ * S0 / (S0 - h));
    } else {
      cfg.barrier_log_up = cfg.barrier_log_base;
      cfg.barrier_log_dn = cfg.barrier_log_base;
    }
    return cfg;
  }

  // 2. BUFFER CALCULATION: Perfectly unbiased
  template <GreekMode Mode, typename State>
  void calculate_to_buffer(const State &state, double S0, double h,
                           ResultType &base, ResultType &up,
                           ResultType &dn) const {

    double S_T = state.logS;

    double mult_up = 1.0, mult_dn = 1.0;
    if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
      mult_up = (S0 + h) / S0;
      mult_dn = (S0 - h) / S0;
    }

    if constexpr (is_scalar) {
      base =
          compute_payoff(S_T, strikes_, state.is_hit_base, state.hit_time_base);
      if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
        up = compute_payoff(S_T * mult_up, strikes_, state.is_hit_up,
                            state.hit_time_up);
        dn = compute_payoff(S_T * mult_dn, strikes_, state.is_hit_dn,
                            state.hit_time_dn);
      }
    } else {
      for (size_t i = 0; i < strikes_.size(); ++i) {
        base[i] = compute_payoff(S_T, strikes_[i], state.is_hit_base,
                                 state.hit_time_base);
        if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
          up[i] = compute_payoff(S_T * mult_up, strikes_[i], state.is_hit_up,
                                 state.hit_time_up);
          dn[i] = compute_payoff(S_T * mult_dn, strikes_[i], state.is_hit_dn,
                                 state.hit_time_dn);
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
  double get_barrier() const { return barrier_; }
  PayoffT &get_payoff_mut() { return payoff_func_; }
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