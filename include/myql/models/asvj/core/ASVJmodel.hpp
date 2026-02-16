#pragma once
#include <myql/models/asvj/data/ModelParams.hpp>

// ==========================================
// 1. Model Containers (The Assemblies)
// ==========================================

/* * These structs combine volatility and jumps into a single Model.
 * The Stepper will accept these as its configuration.
 */

// A. Single Factor Models (Heston, Bates)
// ----------------------------------------
template <typename JumpParamType> struct SingleFactorModel {
  HestonParams heston;
  JumpParamType jump;

  // Constructor
  SingleFactorModel(const HestonParams &h,
                    const JumpParamType &j = JumpParamType())
      : heston(h), jump(j) {}
};

// B. Double Factor Models (Double Heston, Double Bates)
// -----------------------------------------------------
template <typename JumpParamType> struct DoubleFactorModel {
  HestonParams heston1;
  HestonParams heston2;
  JumpParamType jump;

  // Correlation between the two variance processes (often 0, but technically
  // possible)
  double rho_v1_v2;

  DoubleFactorModel(const HestonParams &h1, const HestonParams &h2,
                    const JumpParamType &j = JumpParamType(),
                    double correlation_v1_v2 = 0.0)
      : heston1(h1), heston2(h2), jump(j), rho_v1_v2(correlation_v1_v2) {}
};

// ==========================================
// 2. Convenience Aliases
// ==========================================

// Standard Heston (1 Vol, No Jumps)
using HestonModel = SingleFactorModel<NoJumpParams>;

// Bates Model (1 Vol, Merton Jumps)
using BatesModel = SingleFactorModel<MertonParams>;

// Bates-Kou Model (1 Vol, Kou Jumps)
using BatesKouModel = SingleFactorModel<KouParams>;

// Double Heston (2 Vols, No Jumps)
using DoubleHestonModel = DoubleFactorModel<NoJumpParams>;

// Double Bates (2 Vols, Merton Jumps)
using DoubleBatesModel = DoubleFactorModel<MertonParams>;

// Double Bates-Kou (2 Vols, Kou Jumps)
using DoubleBatesKouModel = DoubleFactorModel<KouParams>;
