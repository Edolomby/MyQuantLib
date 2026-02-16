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

  template <typename State> static inline void init(State &, double, double) {}

  template <typename State> static inline void update(State &, double) {}

  template <typename State>
  static inline void finalize(State &s, double, double) {
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

  template <typename State>
  static inline void init(State &s, double, double dt) {
    s.sumLogS = 0.5 * s.logS * dt;
  }

  template <typename State> static inline void update(State &s, double dt) {
    s.sumLogS += s.logS * dt;
  }

  template <typename State>
  static inline void finalize(State &s, double dt, double T) {
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

  template <typename State>
  static inline void init(State &s, double, double dt) {
    s.sumLogS = 0.5 * s.logS * dt;
    s.sumS = 0.5 * std::exp(s.logS) * dt;
  }

  template <typename State> static inline void update(State &s, double dt) {
    s.sumLogS += s.logS * dt;
    s.sumS += std::exp(s.logS) * dt;
  }

  template <typename State>
  static inline void finalize(State &s, double dt, double T) {
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

  template <typename State>
  static inline void init(State &s, double logS0, double) {
    s.minLogS = logS0;
    s.maxLogS = logS0;
  }

  template <typename State> static inline void update(State &s, double) {
    s.maxLogS = (s.logS > s.maxLogS) ? s.logS : s.maxLogS;
    s.minLogS = (s.logS < s.minLogS) ? s.logS : s.minLogS;
  }

  template <typename State>
  static inline void finalize(State &s, double, double) {
    s.minLogS = std::exp(s.minLogS); // Min Price
    s.maxLogS = std::exp(s.maxLogS); // Max Price
    s.logS = std::exp(s.logS);       // Terminal Price
  }
};