#pragma once
#include <algorithm>
#include <cmath>
#include <myql/core/PricingTypes.hpp>
#include <vector>

enum class OptionType { Call, Put };

// =============================================================================
// VANILLA PAYOFF
// =============================================================================

// Used for: European, Asian, Lookback
// Logic: max(Spot - Strike, 0) OR max(Strike - Spot, 0)
template <OptionType OT> struct PayoffVanilla {
  static constexpr OptionType Type = OT;
  static constexpr bool needs_smoothing = false;

  // Mode is intentionally ignored: vanilla payoff is already C1-smooth
  template <GreekMode Mode = GreekMode::None>
  double operator()(double spot, double strike) const {
    if constexpr (OT == OptionType::Call)
      return std::max(spot - strike, 0.0);
    else
      return std::max(strike - spot, 0.0);
  }
};

// =============================================================================
// DIGITAL PAYOFFS
// =============================================================================

// Cash-or-Nothing
// Logic (sharp): 1.0 if ITM, else 0.0
// Logic (smooth, Mode != None): erfc approximation of Heaviside with bandwidth
// eps
//   Smooth variant activates automatically when GreekMode::Essential/Full is
//   requested to prevent variance explosions in pathwise Delta/Gamma
//   estimators. Bias is O(eps); sweet-spot: eps ~ sigma * K * sqrt(T) / 5.
template <OptionType OT> struct PayoffCashOrNothing {
  static constexpr OptionType Type = OT;
  static constexpr bool needs_smoothing = true;
  double eps = -1.0; // -1 means auto-calculate bandwidth

  template <GreekMode Mode = GreekMode::None>
  double operator()(double spot, double strike) const {
    double diff = (OT == OptionType::Call) ? (spot - strike) : (strike - spot);
    if constexpr (Mode == GreekMode::None)
      return (diff > 0.0) ? 1.0 : 0.0; // exact step, zero overhead
    else
      return 0.5 * std::erfc(-diff * M_SQRT1_2 / eps); // smooth Phi(diff/eps)
  }
};

// Asset-or-Nothing
// Logic (sharp): Spot if ITM, else 0.0
// Logic (smooth, Mode != None): spot * Phi((S-K)/eps)
template <OptionType OT> struct PayoffAssetOrNothing {
  static constexpr OptionType Type = OT;
  static constexpr bool needs_smoothing = true;
  double eps = -1.0; // -1 means auto-calculate bandwidth

  template <GreekMode Mode = GreekMode::None>
  double operator()(double spot, double strike) const {
    double diff = (OT == OptionType::Call) ? (spot - strike) : (strike - spot);
    if constexpr (Mode == GreekMode::None)
      return (diff > 0.0) ? spot : 0.0; // exact step, zero overhead
    else
      return spot * 0.5 * std::erfc(-diff * M_SQRT1_2 / eps); // smooth
  }
};