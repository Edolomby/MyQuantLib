#pragma once
#include <type_traits>

#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/models/asvj/data/ModelParams.hpp>
#include <myql/models/asvj/policies/JumpPolicies.hpp>
#include <myql/models/asvj/policies/VolSchemes.hpp>

namespace myql::dispatcher {

// =============================================================================
// JUMP TRAITS: Map Parameter structs to Policy classes
// =============================================================================
template <typename ParamT> struct JumpPolicyFor {
  // Default (should not hit)
  using type = NoJumps;
};

template <> struct JumpPolicyFor<NoJumpParams> {
  using type = NoJumps;
};

template <> struct JumpPolicyFor<MertonParams> {
  using type = MertonJump;
};

template <> struct JumpPolicyFor<KouParams> {
  using type = KouJump;
};

// =============================================================================
// MODEL TRAITS: Determine NumCirs and Jump Policy
// =============================================================================

// Base templates (should only match if an unknown model type is passed)
template <typename Model> struct ModelTraits {
  static constexpr int num_cirs = 0;
  using jump_param_type = NoJumpParams;
};

// Zero-Factor Models
template <typename JumpParamT> struct ModelTraits<ZeroFactorModel<JumpParamT>> {
  static constexpr int num_cirs = 0;
  using jump_param_type = JumpParamT;
};

// Single-Factor Models
template <typename JumpParamT>
struct ModelTraits<SingleFactorModel<JumpParamT>> {
  static constexpr int num_cirs = 1;
  using jump_param_type = JumpParamT;
};

// Double-Factor Models
template <typename JumpParamT>
struct ModelTraits<DoubleFactorModel<JumpParamT>> {
  static constexpr int num_cirs = 2;
  using jump_param_type = JumpParamT;
};

// =============================================================================
// STEPPER RESOLUTION
// =============================================================================
// Given a Model, an Instrument, and optionally a desired VolScheme, deduce the
// Stepper
template <typename Model, typename Instrument,
          typename DesiredVolScheme = SchemeNCI>
struct StepperFor {
  using Traits = ModelTraits<Model>;
  using JumpPolicy =
      typename JumpPolicyFor<typename Traits::jump_param_type>::type;
  using Tracker = typename Instrument::Tracker;

  // Resolve Vol Schemes based on Number of CIRs
  static constexpr int NC = Traits::num_cirs;
  using V1 = std::conditional_t<NC >= 1, DesiredVolScheme, NullVolScheme>;
  using V2 = std::conditional_t<NC == 2, DesiredVolScheme, NullVolScheme>;

  using type = ASVJStepper<V1, V2, JumpPolicy, Tracker>;
};

} // namespace myql::dispatcher
