#pragma once
#include <myql/core/PricingTypes.hpp>
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/trackers/PathTrackers.hpp>
#include <type_traits>

template <typename TrackerT, typename PayoffT, bool FixedStrike,
          typename StrikeContainer = double>
class AsianOption {
  StrikeContainer strikes_;
  double T_;
  PayoffT payoff_func_;

  // Historical state for in-flight options (option started before today).
  // These store the **observed averages** over [t_past, t_0]:
  //   past_avg_log_ = average of log(S)  (e.g. exp(past_avg_log_) = geo-avg)
  //   past_avg_S_   = arithmetic average of S
  //   t_elapsed_    = calendar time already elapsed
  // The tracker's init() reconstructs the partial integral as avg * t_elapsed.
  double past_avg_log_ = 0.0;
  double past_avg_S_ = 0.0;
  double t_elapsed_ = 0.0;

  static constexpr bool is_scalar = std::is_floating_point_v<StrikeContainer>;
  static constexpr bool is_arith = std::is_same_v<TrackerT, TrackerArithAsian>;

  // Helper to compute payoff based on strike type
  inline double compute_payoff(double spot_scaled, double avg_scaled,
                               double K) const {
    if constexpr (FixedStrike) {
      return payoff_func_(avg_scaled, K);
    } else {
      return payoff_func_(spot_scaled, avg_scaled);
    }
  }

public:
  using Tracker = TrackerT;
  using ResultType = StrikeContainer;
  using PayoffType = PayoffT;

  // ── Default: fresh option starting today ──────────────────────────────────
  AsianOption(const StrikeContainer &K, double T) : strikes_(K), T_(T) {}

  // ── In-flight GeoAsian ─────────────────────────────────────────────────────
  // past_avg_log: observed average of log(S) over the elapsed period
  // t_elapsed:    calendar time between option start date and today
  AsianOption(const StrikeContainer &K, double T, double past_avg_log,
              double t_elapsed)
      : strikes_(K), T_(T), past_avg_log_(past_avg_log), t_elapsed_(t_elapsed) {
  }

  // ── In-flight ArithAsian ───────────────────────────────────────────────────
  // past_avg_S:   observed arithmetic average of S  over the elapsed period
  // past_avg_log: observed average of log(S)         over the elapsed period
  // t_elapsed:    calendar time between option start date and today
  AsianOption(const StrikeContainer &K, double T, double past_avg_S,
              double past_avg_log, double t_elapsed)
      : strikes_(K), T_(T), past_avg_log_(past_avg_log),
        past_avg_S_(past_avg_S), t_elapsed_(t_elapsed) {}

  template <GreekMode Mode = GreekMode::None>
  typename Tracker::Config
  get_tracker_config([[maybe_unused]] double S0 = 100.0,
                     [[maybe_unused]] double h = 0.0) const {
    typename Tracker::Config cfg;
    cfg.t_elapsed = t_elapsed_;
    if constexpr (is_arith) {
      cfg.past_avg_log = past_avg_log_;
      cfg.past_avg_S = past_avg_S_;
    } else {
      // GeoAsian (or any single-integral tracker)
      cfg.past_avg_log = past_avg_log_;
    }
    return cfg;
  }

  template <GreekMode Mode, typename State>
  void calculate_to_buffer(const State &state, double S0, double h,
                           ResultType &base, ResultType &up,
                           ResultType &dn) const {

    double avg = 0.0;
    if constexpr (std::is_same_v<Tracker, TrackerGeoAsian>) {
      avg = state.sumLogS;
    } else {
      avg = state.sumS;
    }

    double S_T = state.logS;

    double mult_up = 1.0;
    double mult_dn = 1.0;
    if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
      mult_up = (S0 + h) / S0;
      mult_dn = (S0 - h) / S0;
    }

    if constexpr (is_scalar) {
      base = compute_payoff(S_T, avg, strikes_);
      if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
        up = compute_payoff(S_T * mult_up, avg * mult_up, strikes_);
        dn = compute_payoff(S_T * mult_dn, avg * mult_dn, strikes_);
      }
    } else {
      for (size_t i = 0; i < strikes_.size(); ++i) {
        base[i] = compute_payoff(S_T, avg, strikes_[i]);
        if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
          up[i] = compute_payoff(S_T * mult_up, avg * mult_up, strikes_[i]);
          dn[i] = compute_payoff(S_T * mult_dn, avg * mult_dn, strikes_[i]);
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