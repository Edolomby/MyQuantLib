#pragma once
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/Barrier.hpp>
#include <myql/instruments/options/European.hpp>

namespace myql::dispatcher {

// =============================================================================
// RUNTIME INSTRUMENT REGISTRY
// =============================================================================

// Always use vectorized strips for runtime dispatch to maximize throughput
using StandardStrip = std::vector<double>;

// Supported Concrete Instrument Types
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

// TODO: Add Barrier, Asian, Lookback when needed...

// The universal variant
using AnyInstrument = std::variant<VanillaCall, VanillaPut, CashCall, CashPut,
                                   AssetCall, AssetPut>;

// Generic config passed to the factory
struct RuntimeInstrumentParams {
  std::vector<double> strikes;
  double maturity;

  // For barriers
  double barrier_level = 0.0;
  double rebate = 0.0;
};

// =============================================================================
// FACTORY
// =============================================================================
inline AnyInstrument
build_instrument(const std::string &type,
                 const std::string &option_type, // "Call" or "Put"
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

  throw std::invalid_argument("Unknown instrument combo: " + type + " " +
                              option_type);
}

} // namespace myql::dispatcher
