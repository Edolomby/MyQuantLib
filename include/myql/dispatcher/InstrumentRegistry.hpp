#pragma once
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/Asian.hpp>
#include <myql/instruments/options/Barrier.hpp>
#include <myql/instruments/options/European.hpp>
#include <myql/instruments/options/Lookback.hpp>

namespace myql::dispatcher {

// =============================================================================
// RUNTIME INSTRUMENT REGISTRY
// =============================================================================
// Always use vectorized strips for European (multi-strike) dispatch.
// Exotics default to scalar double (single strike or strike-less floating).
using StandardStrip = std::vector<double>;

// -----------------------------------------------------------------------------
// EUROPEAN INSTRUMENTS (MC + Fourier)
// -----------------------------------------------------------------------------
using VanillaCall =
    EuropeanOption<PayoffVanilla<OptionType::Call>, StandardStrip>;
using VanillaPut =
    EuropeanOption<PayoffVanilla<OptionType::Put>, StandardStrip>;

using CashCall =
    EuropeanOption<PayoffCashOrNothing<OptionType::Call>, StandardStrip>;
using CashPut =
    EuropeanOption<PayoffCashOrNothing<OptionType::Put>, StandardStrip>;

using AssetCall =
    EuropeanOption<PayoffAssetOrNothing<OptionType::Call>, StandardStrip>;
using AssetPut =
    EuropeanOption<PayoffAssetOrNothing<OptionType::Put>, StandardStrip>;

// The European-only variant: safe for both price_mc and price_fourier
using AnyEuropeanInstrument = std::variant<VanillaCall, VanillaPut, CashCall,
                                           CashPut, AssetCall, AssetPut>;

// -----------------------------------------------------------------------------
// ASIAN INSTRUMENTS (MC only)
// Fixed-strike: payoff = max(Avg - K, 0) — strike is passed in
// Floating-strike: payoff = max(S_T - Avg, 0) — no external strike
// -----------------------------------------------------------------------------

// Geometric Asian — Fixed Strike
using GeoAsianFixedCall =
    AsianOption<TrackerGeoAsian, PayoffVanilla<OptionType::Call>, true>;
using GeoAsianFixedPut =
    AsianOption<TrackerGeoAsian, PayoffVanilla<OptionType::Put>, true>;

// Geometric Asian — Floating Strike (StrikeContainer = double, unused)
using GeoAsianFloatingCall =
    AsianOption<TrackerGeoAsian, PayoffVanilla<OptionType::Call>, false,
                double>;
using GeoAsianFloatingPut =
    AsianOption<TrackerGeoAsian, PayoffVanilla<OptionType::Put>, false, double>;

// Arithmetic Asian — Fixed Strike
using ArithAsianFixedCall =
    AsianOption<TrackerArithAsian, PayoffVanilla<OptionType::Call>, true>;
using ArithAsianFixedPut =
    AsianOption<TrackerArithAsian, PayoffVanilla<OptionType::Put>, true>;

// Arithmetic Asian — Floating Strike
using ArithAsianFloatingCall =
    AsianOption<TrackerArithAsian, PayoffVanilla<OptionType::Call>, false,
                double>;
using ArithAsianFloatingPut =
    AsianOption<TrackerArithAsian, PayoffVanilla<OptionType::Put>, false,
                double>;

// -----------------------------------------------------------------------------
// LOOKBACK INSTRUMENTS (MC only)
// Fixed-strike: payoff = max(Max - K, 0) or max(K - Min, 0)
// Floating-strike: payoff = max(S_T - Min, 0) or max(Max - S_T, 0)
// -----------------------------------------------------------------------------
using LookbackFixedCall = LookbackOption<PayoffVanilla<OptionType::Call>, true>;
using LookbackFixedPut = LookbackOption<PayoffVanilla<OptionType::Put>, true>;

using LookbackFloatingCall =
    LookbackOption<PayoffVanilla<OptionType::Call>, false, double>;
using LookbackFloatingPut =
    LookbackOption<PayoffVanilla<OptionType::Put>, false, double>;

// -----------------------------------------------------------------------------
// BARRIER INSTRUMENTS (MC only, RebateTiming::None)
// -----------------------------------------------------------------------------
using UpAndOutCallT = UpAndOutCall<RebateTiming::None>;
using UpAndOutPutT = UpAndOutPut<RebateTiming::None>;
using UpAndInCallT = UpAndInCall<RebateTiming::None>;
using UpAndInPutT = UpAndInPut<RebateTiming::None>;
using DownAndOutCallT = DownAndOutCall<RebateTiming::None>;
using DownAndOutPutT = DownAndOutPut<RebateTiming::None>;
using DownAndInCallT = DownAndInCall<RebateTiming::None>;
using DownAndInPutT = DownAndInPut<RebateTiming::None>;

// -----------------------------------------------------------------------------
// FULL MC VARIANT: all instruments
// -----------------------------------------------------------------------------
using AnyMCInstrument = std::variant< // European
    VanillaCall, VanillaPut, CashCall, CashPut, AssetCall, AssetPut,
    // Geometric Asian
    GeoAsianFixedCall, GeoAsianFixedPut, GeoAsianFloatingCall,
    GeoAsianFloatingPut,
    // Arithmetic Asian
    ArithAsianFixedCall, ArithAsianFixedPut, ArithAsianFloatingCall,
    ArithAsianFloatingPut,
    // Lookback
    LookbackFixedCall, LookbackFixedPut, LookbackFloatingCall,
    LookbackFloatingPut,
    // Barrier
    UpAndOutCallT, UpAndOutPutT, UpAndInCallT, UpAndInPutT, DownAndOutCallT,
    DownAndOutPutT, DownAndInCallT, DownAndInPutT>;

// =============================================================================
// RUNTIME PARAMETERS
// =============================================================================
struct RuntimeInstrumentParams {
  // European / fixed-strike Asian / fixed-strike Lookback
  std::vector<double> strikes;
  double maturity;

  // Barrier
  double barrier_level = 0.0;
  double rebate = 0.0;
  double rate = 0.0; // needed for AtHit rebate discounting
};

// =============================================================================
// FACTORY: European instruments
// =============================================================================
inline AnyEuropeanInstrument
build_european(const std::string &type, const std::string &option_type,
               const RuntimeInstrumentParams &params) {

  const auto &K = params.strikes;
  double T = params.maturity;

  if (type == "Vanilla") {
    if (option_type == "Call")
      return VanillaCall(K, T);
    if (option_type == "Put")
      return VanillaPut(K, T);
  } else if (type == "CashOrNothing") {
    if (option_type == "Call")
      return CashCall(K, T);
    if (option_type == "Put")
      return CashPut(K, T);
  } else if (type == "AssetOrNothing") {
    if (option_type == "Call")
      return AssetCall(K, T);
    if (option_type == "Put")
      return AssetPut(K, T);
  }

  throw std::invalid_argument("Unknown European instrument: " + type + " " +
                              option_type);
}

// =============================================================================
// FACTORY: All MC instruments (delegates to European for European types)
// =============================================================================
inline AnyMCInstrument build_instrument(const std::string &type,
                                        const std::string &option_type,
                                        const RuntimeInstrumentParams &params) {

  const auto &K = params.strikes;
  double T = params.maturity;
  double K0 = K.empty() ? 0.0 : K[0]; // scalar strike for fixed exotics
  double B = params.barrier_level;

  // --- European ---
  if (type == "Vanilla") {
    if (option_type == "Call")
      return VanillaCall(K, T);
    if (option_type == "Put")
      return VanillaPut(K, T);
  } else if (type == "CashOrNothing") {
    if (option_type == "Call")
      return CashCall(K, T);
    if (option_type == "Put")
      return CashPut(K, T);
  } else if (type == "AssetOrNothing") {
    if (option_type == "Call")
      return AssetCall(K, T);
    if (option_type == "Put")
      return AssetPut(K, T);

    // --- Geometric Asian ---
  } else if (type == "GeoAsian") {
    if (option_type == "Call")
      return GeoAsianFixedCall(K0, T);
    if (option_type == "Put")
      return GeoAsianFixedPut(K0, T);
  } else if (type == "GeoAsianFloating") {
    double dummy = 0.0;
    if (option_type == "Call")
      return GeoAsianFloatingCall(dummy, T);
    if (option_type == "Put")
      return GeoAsianFloatingPut(dummy, T);

    // --- Arithmetic Asian ---
  } else if (type == "ArithAsian") {
    if (option_type == "Call")
      return ArithAsianFixedCall(K0, T);
    if (option_type == "Put")
      return ArithAsianFixedPut(K0, T);
  } else if (type == "ArithAsianFloating") {
    double dummy = 0.0;
    if (option_type == "Call")
      return ArithAsianFloatingCall(dummy, T);
    if (option_type == "Put")
      return ArithAsianFloatingPut(dummy, T);

    // --- Lookback ---
  } else if (type == "LookbackFixed") {
    if (option_type == "Call")
      return LookbackFixedCall(K0, T);
    if (option_type == "Put")
      return LookbackFixedPut(K0, T);
  } else if (type == "LookbackFloating") {
    double dummy = 0.0;
    if (option_type == "Call")
      return LookbackFloatingCall(dummy, T);
    if (option_type == "Put")
      return LookbackFloatingPut(dummy, T);

    // --- Barrier ---
  } else if (type == "UpAndOut") {
    if (option_type == "Call")
      return UpAndOutCallT(K0, B, T, params.rate, params.rebate);
    if (option_type == "Put")
      return UpAndOutPutT(K0, B, T, params.rate, params.rebate);
  } else if (type == "UpAndIn") {
    if (option_type == "Call")
      return UpAndInCallT(K0, B, T, params.rate, params.rebate);
    if (option_type == "Put")
      return UpAndInPutT(K0, B, T, params.rate, params.rebate);
  } else if (type == "DownAndOut") {
    if (option_type == "Call")
      return DownAndOutCallT(K0, B, T, params.rate, params.rebate);
    if (option_type == "Put")
      return DownAndOutPutT(K0, B, T, params.rate, params.rebate);
  } else if (type == "DownAndIn") {
    if (option_type == "Call")
      return DownAndInCallT(K0, B, T, params.rate, params.rebate);
    if (option_type == "Put")
      return DownAndInPutT(K0, B, T, params.rate, params.rebate);
  }

  throw std::invalid_argument("Unknown instrument: " + type + " " +
                              option_type);
}

} // namespace myql::dispatcher
