#pragma once
#include <algorithm>
#include <cmath>

enum class OptionType { Call, Put };

// -----------------------------------------------------------------------------
// VANILLA PAYOFF
// -----------------------------------------------------------------------------

// Used for: European, Asian, Lookback
// Logic: max(Spot - Strike, 0) OR max(Strike - Spot, 0)
template <OptionType OT> struct PayoffVanilla {
  static constexpr OptionType Type = OT;

  double operator()(double spot, double strike) const {
    if constexpr (OT == OptionType::Call)
      return std::max(spot - strike, 0.0);
    else
      return std::max(strike - spot, 0.0);
  }
};

// -----------------------------------------------------------------------------
// DIGITAL PAYOFFS
// -----------------------------------------------------------------------------

// Cash-or-Nothing
// Logic: 1.0 if ITM, else 0.0
template <OptionType OT> struct PayoffCashOrNothing {
  static constexpr OptionType Type = OT;

  double operator()(double spot, double strike) const {
    if constexpr (OT == OptionType::Call)
      return (spot > strike) ? 1.0 : 0.0;
    else
      return (strike > spot) ? 1.0 : 0.0;
  }
};

// Asset-or-Nothing
// Logic: Spot if ITM, else 0.0
template <OptionType OT> struct PayoffAssetOrNothing {
  static constexpr OptionType Type = OT;

  double operator()(double spot, double strike) const {
    if constexpr (OT == OptionType::Call)
      return (spot > strike) ? spot : 0.0;
    else
      return (strike > spot) ? spot : 0.0;
  }
};