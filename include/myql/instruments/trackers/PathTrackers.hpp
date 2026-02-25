#pragma once
#include <cmath>

// =============================================================================
// 1. EUROPEAN TRACKER (Standard)
// =============================================================================
// Minimal overhead. Only converts LogS -> S at the very end.
struct TrackerEuropean {
  static constexpr bool is_path_dependent = false;

  // No extra state needed for European, just the spot in the main State
  struct ExtraState {};
  struct Config {}; // No static contract rules needed

  template <typename State>
  static inline void init(State &, const Config &, double, double) {}

  template <typename State>
  static inline void update(State &, const Config &, double) {}

  template <typename State>
  static inline void finalize(State &s, const Config &, double, double) {
    s.logS = std::exp(s.logS); // Convert Log price to Price
  }
};

// =============================================================================
// 2. GEOMETRIC ASIAN TRACKER
// =============================================================================
// Tracks integral of log(S). Used for Geometric Asian Options.
struct TrackerGeoAsian {
  static constexpr bool is_path_dependent = true;

  struct ExtraState {
    double sumLogS; // Accumulator for \int log(S_t) dt
  };

  // Config carries optional historical state for in-flight options.
  // past_avg_log  = observed average of log(S) over [t_past, t_0]  (0 if new)
  // t_elapsed     = time already elapsed since the option started   (0 if new)
  // init() reconstructs the partial integral as: past_avg_log * t_elapsed
  struct Config {
    double past_avg_log = 0.0; // observed average of log(S) over elapsed period
    double t_elapsed = 0.0;    // elapsed time before today
  };

  template <typename State>
  static inline void init(State &s, const Config &cfg, double, double dt) {
    // Reconstruct \int log(S) dt from the observed average, then open the
    // first half-step of the future simulation leg.
    s.sumLogS = cfg.past_avg_log * cfg.t_elapsed + 0.5 * s.logS * dt;
  }

  template <typename State>
  static inline void update(State &s, const Config &, double dt) {
    s.sumLogS += s.logS * dt;
  }

  template <typename State>
  static inline void finalize(State &s, const Config &cfg, double dt,
                              double T) {
    // Trapezoidal correction for better convergence
    s.sumLogS -= 0.5 * s.logS * dt;
    // Total averaging window is T_residual + t_elapsed
    const double T_total = T + cfg.t_elapsed;
    s.sumLogS = std::exp(s.sumLogS / T_total);
    s.logS =
        std::exp(s.logS); // Convert to price used in floating strike payoff
  }
};

// =============================================================================
// 3. ARITHMETIC ASIAN TRACKER
// =============================================================================
// Tracks integral of S (for payoff) AND integral of log(S) (for Control
// Variate).
struct TrackerArithAsian {
  static constexpr bool is_path_dependent = true;

  struct ExtraState {
    double sumLogS; // For Control Variate (Geometric Average)
    double sumS;    // For Payoff (Arithmetic Average)
  };

  // Config carries optional historical state for in-flight options.
  // past_avg_log = observed average of log(S)  over [t_past, t_0]  (0 if new)
  // past_avg_S   = observed arithmetic average of S over [t_past, t_0] (0 if
  // new) t_elapsed    = elapsed time before today (0 if new)
  // init() reconstructs partial integrals as: past_avg_* * t_elapsed
  struct Config {
    double past_avg_log = 0.0;
    double past_avg_S = 0.0;
    double t_elapsed = 0.0;
  };

  template <typename State>
  static inline void init(State &s, const Config &cfg, double, double dt) {
    s.sumLogS = cfg.past_avg_log * cfg.t_elapsed + 0.5 * s.logS * dt;
    s.sumS = cfg.past_avg_S * cfg.t_elapsed + 0.5 * std::exp(s.logS) * dt;
  }

  template <typename State>
  static inline void update(State &s, const Config &, double dt) {
    s.sumLogS += s.logS * dt;
    s.sumS += std::exp(s.logS) * dt;
  }

  template <typename State>
  static inline void finalize(State &s, const Config &cfg, double dt,
                              double T) {
    // Trapezoidal correction for better convergence
    s.sumLogS -= 0.5 * s.logS * dt;
    s.sumS -= 0.5 * std::exp(s.logS) * dt;
    const double T_total = T + cfg.t_elapsed;
    s.sumLogS = std::exp(s.sumLogS / T_total);
    s.sumS = s.sumS / T_total;
    s.logS =
        std::exp(s.logS); // Convert to price used in floating strike payoff
  }
};

// =============================================================================
// 4. LOOKBACK TRACKER (Fixed & Floating)
// =============================================================================
// Tracks Minimum and Maximum over the path.
struct TrackerLookback {
  static constexpr bool is_path_dependent = true;

  struct ExtraState {
    double minLogS;
    double maxLogS;
  };

  // Config carries optional historical extremes for in-flight options.
  // hist_min_logS = log(S_min) observed before today (+inf  → use log(S0))
  // hist_max_logS = log(S_max) observed before today (-inf  → use log(S0))
  struct Config {
    double hist_min_logS = +1e300; // sentinel: not set
    double hist_max_logS = -1e300; // sentinel: not set
  };

  template <typename State>
  static inline void init(State &s, const Config &cfg, double logS0, double) {
    // If history was provided, pre-seed min/max from it; otherwise use S0.
    s.minLogS = (cfg.hist_min_logS < 1e300) ? cfg.hist_min_logS : logS0;
    s.maxLogS = (cfg.hist_max_logS > -1e300) ? cfg.hist_max_logS : logS0;
  }

  template <typename State>
  static inline void update(State &s, const Config &, double) {
    s.maxLogS = (s.logS > s.maxLogS) ? s.logS : s.maxLogS;
    s.minLogS = (s.logS < s.minLogS) ? s.logS : s.minLogS;
  }

  template <typename State>
  static inline void finalize(State &s, const Config &, double, double) {
    s.minLogS = std::exp(s.minLogS); // Min Price
    s.maxLogS = std::exp(s.maxLogS); // Max Price
    s.logS = std::exp(s.logS);       // Terminal Price
  }
};

// =============================================================================
// 5. BARRIER TRACKER
// =============================================================================
template <bool IsUpBarrier> struct TrackerBarrier {
  static constexpr bool is_path_dependent = true;

  struct ExtraState {
    double current_time;

    // Track all three scenarios simultaneously!
    bool is_hit_base, is_hit_up, is_hit_dn;
    double hit_time_base, hit_time_up, hit_time_dn;
  };

  struct Config {
    double barrier_log_base;
    double barrier_log_up;
    double barrier_log_dn;
  };

  template <typename State>
  static inline void init(State &s, const Config &cfg, double logS0, double) {
    s.current_time = 0.0;

    s.is_hit_base = false;
    s.hit_time_base = 0.0;
    s.is_hit_up = false;
    s.hit_time_up = 0.0;
    s.is_hit_dn = false;
    s.hit_time_dn = 0.0;

    // Check Day 0 breaches
    if constexpr (IsUpBarrier) {
      if (logS0 >= cfg.barrier_log_base) {
        s.is_hit_base = true;
      }
      if (logS0 >= cfg.barrier_log_up) {
        s.is_hit_up = true;
      }
      if (logS0 >= cfg.barrier_log_dn) {
        s.is_hit_dn = true;
      }
    } else {
      if (logS0 <= cfg.barrier_log_base) {
        s.is_hit_base = true;
      }
      if (logS0 <= cfg.barrier_log_up) {
        s.is_hit_up = true;
      }
      if (logS0 <= cfg.barrier_log_dn) {
        s.is_hit_dn = true;
      }
    }
  }

  template <typename State>
  static inline void update(State &s, const Config &cfg, double dt) {
    s.current_time += dt;

    /* if constexpr (IsUpBarrier) {
      if (!s.is_hit_base && s.logS >= cfg.barrier_log_base) {
        s.is_hit_base = true;
        s.hit_time_base = s.current_time;
      }
      if (!s.is_hit_up && s.logS >= cfg.barrier_log_up) {
        s.is_hit_up = true;
        s.hit_time_up = s.current_time;
      }
      if (!s.is_hit_dn && s.logS >= cfg.barrier_log_dn) {
        s.is_hit_dn = true;
        s.hit_time_dn = s.current_time;
      }
    } else {
      if (!s.is_hit_base && s.logS <= cfg.barrier_log_base) {
        s.is_hit_base = true;
        s.hit_time_base = s.current_time;
      }
      if (!s.is_hit_up && s.logS <= cfg.barrier_log_up) {
        s.is_hit_up = true;
        s.hit_time_up = s.current_time;
      }
      if (!s.is_hit_dn && s.logS <= cfg.barrier_log_dn) {
        s.is_hit_dn = true;
        s.hit_time_dn = s.current_time;
      }
    } */

    if constexpr (IsUpBarrier) {
      // 1. GATEKEEPER: Easiest barrier (Lowest log-price)
      if (s.logS >= cfg.barrier_log_up) {
        if (!s.is_hit_up) {
          s.is_hit_up = true;
          s.hit_time_up = s.current_time;
        }
        // 2. MIDDLE BARRIER: Only check if it passed the first!
        if (s.logS >= cfg.barrier_log_base) {
          if (!s.is_hit_base) {
            s.is_hit_base = true;
            s.hit_time_base = s.current_time;
          }
          // 3. HARDEST BARRIER: Only check if it passed the middle!
          if (s.logS >= cfg.barrier_log_dn) {
            if (!s.is_hit_dn) {
              s.is_hit_dn = true;
              s.hit_time_dn = s.current_time;
            }
          }
        }
      }
    } else {
      // 1. GATEKEEPER: Easiest barrier (Highest log-price)
      if (s.logS <= cfg.barrier_log_dn) {
        if (!s.is_hit_dn) {
          s.is_hit_dn = true;
          s.hit_time_dn = s.current_time;
        }
        // 2. MIDDLE BARRIER: Only check if it passed the first!
        if (s.logS <= cfg.barrier_log_base) {
          if (!s.is_hit_base) {
            s.is_hit_base = true;
            s.hit_time_base = s.current_time;
          }
          // 3. HARDEST BARRIER: Only check if it passed the middle!
          if (s.logS <= cfg.barrier_log_up) {
            if (!s.is_hit_up) {
              s.is_hit_up = true;
              s.hit_time_up = s.current_time;
            }
          }
        }
      }
    }
  }

  template <typename State>
  static inline void finalize(State &s, const Config &, double, double) {
    s.logS = std::exp(s.logS);
  }
};