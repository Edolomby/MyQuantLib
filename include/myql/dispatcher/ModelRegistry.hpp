#pragma once
#include <stdexcept>
#include <string>
#include <variant>

#include <myql/models/asvj/core/ASVJmodel.hpp>

namespace myql::dispatcher {

// =============================================================================
// RUNTIME MODEL REGISTRY
// =============================================================================

// The universal variant capable of holding any of our 9 ASVJ compile-time
// models
using AnyModel =
    std::variant<BlackScholesModel, MertonModel, KouModel, HestonModel,
                 BatesModel, BatesKouModel, DoubleHestonModel, DoubleBatesModel,
                 DoubleBatesKouModel>;

// A pure virtual struct to allow passing generic parameters to the factory
struct RuntimeModelParams {
  HestonParams heston1{};
  HestonParams heston2{};
  double vol = 0.0; // for 0-factor models like Black-Scholes

  // Which type of jump params is populated depends on the model requested
  MertonParams merton{};
  KouParams kou{};
};

// =============================================================================
// FACTORY
// =============================================================================
inline AnyModel build_model(const std::string &model_type,
                            const RuntimeModelParams &params) {

  // --- 0-Factor Models ---
  if (model_type == "BlackScholes") {
    return BlackScholesModel(params.vol);
  }
  if (model_type == "Merton") {
    return MertonModel(params.vol, params.merton);
  }
  if (model_type == "Kou") {
    return KouModel(params.vol, params.kou);
  }

  // --- 1-Factor Models ---
  if (model_type == "Heston") {
    return HestonModel(params.heston1);
  }
  if (model_type == "Bates") {
    return BatesModel(params.heston1, params.merton);
  }
  if (model_type == "BatesKou") {
    return BatesKouModel(params.heston1, params.kou);
  }

  // --- 2-Factor Models ---
  if (model_type == "DoubleHeston") {
    return DoubleHestonModel(params.heston1, params.heston2);
  }
  if (model_type == "DoubleBates") {
    return DoubleBatesModel(params.heston1, params.heston2, params.merton);
  }
  if (model_type == "DoubleBatesKou") {
    return DoubleBatesKouModel(params.heston1, params.heston2, params.kou);
  }

  throw std::invalid_argument("Unknown model type: " + model_type);
}

} // namespace myql::dispatcher
