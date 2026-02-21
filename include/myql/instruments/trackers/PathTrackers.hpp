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
  struct Config {}; // We could add forward_start_time later

  template <typename State>
  static inline void init(State &s, const Config &, double, double dt) {
    s.sumLogS = 0.5 * s.logS * dt;
  }

  template <typename State>
  static inline void update(State &s, const Config &, double dt) {
    s.sumLogS += s.logS * dt;
  }

  template <typename State>
  static inline void finalize(State &s, const Config &, double dt, double T) {
    // Trapezoidal correction for better convergence
    s.sumLogS -= 0.5 * s.logS * dt;
    s.sumLogS = std::exp(s.sumLogS / T);
    s.logS = std::exp(s.logS); // Convert to prize used in floating strike payff
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
  struct Config {}; // We could add forward_start_time later

  template <typename State>
  static inline void init(State &s, const Config &, double, double dt) {
    s.sumLogS = 0.5 * s.logS * dt;
    s.sumS = 0.5 * std::exp(s.logS) * dt;
  }

  template <typename State>
  static inline void update(State &s, const Config &, double dt) {
    s.sumLogS += s.logS * dt;
    s.sumS += std::exp(s.logS) * dt;
  }

  template <typename State>
  static inline void finalize(State &s, const Config &, double dt, double T) {
    // Trapezoidal correction for better convergence
    s.sumLogS -= 0.5 * s.logS * dt;
    s.sumS -= 0.5 * std::exp(s.logS) * dt;
    s.sumLogS = std::exp(s.sumLogS / T);
    s.sumS = s.sumS / T;
    s.logS = std::exp(s.logS); // Convert to prize used in floating strike payff
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
  struct Config {}; // We could add forward_start_time later

  template <typename State>
  static inline void init(State &s, const Config &, double logS0, double) {
    s.minLogS = logS0;
    s.maxLogS = logS0;
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
// Tracks if a specific barrier level is breached and records the exact hit
// time.
template <bool IsUpBarrier> struct TrackerBarrier {
  static constexpr bool is_path_dependent = true;

  struct ExtraState {
    double current_time;
    double hit_time;
    bool is_hit;
  };
  struct Config {
    double barrier_log;
  };

  template <typename State>
  static inline void init(State &s, const Config &cfg, double logS0, double) {
    s.current_time = 0.0;
    s.hit_time = 0.0;
    // Check if we start already breached (unlikely, but mathematically safe)
    if constexpr (IsUpBarrier) {
      s.is_hit = (logS0 >= cfg.barrier_log);
    } else {
      s.is_hit = (logS0 <= cfg.barrier_log);
    }
  }

  template <typename State>
  static inline void update(State &s, const Config &cfg, double dt) {
    s.current_time += dt;

    // Only check if we haven't hit it yet
    if (!s.is_hit) {
      if constexpr (IsUpBarrier) {
        if (s.logS >= cfg.barrier_log) {
          s.is_hit = true;
          s.hit_time = s.current_time;
        }
      } else {
        if (s.logS <= cfg.barrier_log) {
          s.is_hit = true;
          s.hit_time = s.current_time;
        }
      }
    }
  }

  template <typename State>
  static inline void finalize(State &s, const Config &, double, double) {
    s.logS = std::exp(s.logS); // Convert terminal log spot to Price
  }
};