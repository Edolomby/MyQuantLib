#pragma once
#include <algorithm>
#include <cmath>
// #include <type_traits>

#include <myql/engines/fourier/FourierEngine.hpp>
#include <myql/engines/fourier/kernels/GilPeleazKernel.hpp>
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/European.hpp>

// -----------------------------------------------------------------------------
// HELPER: PAYOFF CATEGORY DETECTION
// -----------------------------------------------------------------------------
// These traits allow us to detect the payoff type at compile-time to
// dispatch the most efficient pricing logic (e.g., skip P1 for
// Cash-or-Nothing).

template <typename T> struct is_vanilla : std::false_type {};
template <OptionType OT>
struct is_vanilla<PayoffVanilla<OT>> : std::true_type {};

template <typename T> struct is_cash_or_nothing : std::false_type {};
template <OptionType OT>
struct is_cash_or_nothing<PayoffCashOrNothing<OT>> : std::true_type {};

template <typename T> struct is_asset_or_nothing : std::false_type {};
template <OptionType OT>
struct is_asset_or_nothing<PayoffAssetOrNothing<OT>> : std::true_type {};

// Detect Strip (Vector of Options)
template <typename T> struct is_strip : std::false_type {};
template <typename P> struct is_strip<EuropeanStrip<P>> : std::true_type {};

// -----------------------------------------------------------------------------
// GENERIC FOURIER PRICER
// -----------------------------------------------------------------------------
template <typename Model, typename Instrument,
          typename Traits = AffineTraits<Model>>
typename Instrument::ResultType price_fourier(
    const Model &model, double S0, double r, double q, const Instrument &instr,
    const FourierEngine::Config &engine_cfg = FourierEngine::Config()) {

  // ===========================================================================
  // STRIP HANDLING (Vector of Options)
  // ===========================================================================
  if constexpr (is_strip<Instrument>::value) {
    typename Instrument::ResultType prices;
    prices.reserve(instr.get_strikes().size());

    // Loop over K (Direct Integration for now, Carr-Madan FFT later)
    for (double K : instr.get_strikes()) {
      // 1. Create a temporary Single Option with the same Payoff Type
      using SingleOptType = EuropeanOption<typename Instrument::PayoffType>;
      SingleOptType single_opt(K, instr.get_maturity());

      // 2. Recursive Call: This reuses the logic below (Vanilla/Digital
      // dispatch)
      prices.push_back(price_fourier<Model, SingleOptType, Traits>(
          model, S0, r, q, single_opt, engine_cfg));
    }
    return prices;
  }

  // ===========================================================================
  // SINGLE OPTION LOGIC STARTS HERE
  // ===========================================================================
  else {
    // 1. EXTRACT DATA
    double K = instr.get_strike(); // Compile error if Strip reaches here (Safe)
    double T = instr.get_maturity();
    double K_norm = K / S0;

    double df_r = std::exp(-r * T);
    double df_q = std::exp(-q * T);
    constexpr double INV_PI = 0.31830988618379067154;
    FourierEngine engine(engine_cfg);

    // 2. COMPILE-TIME DISPATCH
    using PayoffT = typename Instrument::PayoffType;
    constexpr OptionType OT = PayoffT::Type;

    // --- CASE A: VANILLA ---
    if constexpr (is_vanilla<PayoffT>::value) {
      GilPelaezKernel<Model, Traits> k1(model, T, r, q, K_norm, false);
      GilPelaezKernel<Model, Traits> k2(model, T, r, q, K_norm, true);

      double P1 = 0.5 + engine.calculate_integral(k1, T) * INV_PI;
      double P2 = 0.5 + engine.calculate_integral(k2, T) * INV_PI;

      double price = S0 * df_q * P1 - K * df_r * P2;

      if constexpr (OT == OptionType::Call)
        return std::max(0.0, price);
      else
        return std::max(0.0, price - S0 * df_q + K * df_r);
    }

    // --- CASE B: CASH-OR-NOTHING ---
    else if constexpr (is_cash_or_nothing<PayoffT>::value) {
      GilPelaezKernel<Model, Traits> k2(model, T, r, q, K_norm, true);
      double P2 = 0.5 + engine.calculate_integral(k2, T) * INV_PI;

      if constexpr (OT == OptionType::Call)
        return df_r * P2;
      else
        return df_r * (1.0 - P2);
    }

    // --- CASE C: ASSET-OR-NOTHING ---
    else if constexpr (is_asset_or_nothing<PayoffT>::value) {
      GilPelaezKernel<Model, Traits> k1(model, T, r, q, K_norm, false);
      double P1 = 0.5 + engine.calculate_integral(k1, T) * INV_PI;

      if constexpr (OT == OptionType::Call)
        return S0 * df_q * P1;
      else
        return S0 * df_q * (1.0 - P1);
    }

    else {
      static_assert(is_vanilla<PayoffT>::value, "Unsupported Payoff Type");
      return 0.0;
    }
  }
}