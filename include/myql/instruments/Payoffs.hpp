#pragma once
#include <algorithm>
#include <cmath>

enum class OptionType { Call, Put };

// 1. VANILLA PAYOFF
// Used for: European, Asian, Lookback
// Logic: max(Spot - Strike, 0) OR max(Strike - Spot, 0)
template <OptionType Type> struct PayoffVanilla {
  double operator()(double spot, double strike) const {
    if constexpr (Type == OptionType::Call)
      return std::max(spot - strike, 0.0);
    else
      return std::max(strike - spot, 0.0);
  }
};

// 2. DIGITAL PAYOFF (Cash-or-Nothing)
// Logic: 1.0 if ITM, else 0.0
template <OptionType Type> struct PayoffDigital {
  double operator()(double spot, double strike) const {
    if constexpr (Type == OptionType::Call)
      return (spot > strike) ? 1.0 : 0.0;
    else
      return (strike > spot) ? 1.0 : 0.0;
  }
};

// 3. ASSET-OR-NOTHING
// Logic: Spot if ITM, else 0.0
template <OptionType Type> struct PayoffAssetOrNothing {
  double operator()(double spot, double strike) const {
    if constexpr (Type == OptionType::Call)
      return (spot > strike) ? spot : 0.0;
    else
      return (strike > spot) ? spot : 0.0;
  }
};