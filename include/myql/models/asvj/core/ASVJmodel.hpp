#pragma once
#include <myql/models/asvj/data/ModelParams.hpp>

// ==========================================
// Model Data Containers
// ==========================================

/* * These structs combine volatility and jumps into a single Model.
 * The Stepper will accept these as its configuration.
 */

// A. Zero Factor Models (GBM, Merton, Kou)
// ----------------------------------------
template <typename JumpParamType = NoJumpParams> struct ZeroFactorModel {
  static constexpr int num_variance_factors = 0;
  double vol;
  JumpParamType jump;

  // Constructor
  ZeroFactorModel(double v, const JumpParamType &j = JumpParamType{})
      : vol(v), jump(j) {}
};

// B. Single Factor Models (Heston, Bates)
// ----------------------------------------
template <typename JumpParamType> struct SingleFactorModel {
  static constexpr int num_variance_factors = 1;
  HestonParams heston;
  JumpParamType jump;

  // Constructor
  SingleFactorModel(const HestonParams &h,
                    const JumpParamType &j = JumpParamType())
      : heston(h), jump(j) {}
};

// C. Double Factor Models (Double Heston, Double Bates)
// -----------------------------------------------------
template <typename JumpParamType> struct DoubleFactorModel {
  static constexpr int num_variance_factors = 2;
  HestonParams heston1;
  HestonParams heston2;
  JumpParamType jump;

  // no correlation between the two variance processes
  // Constructor
  DoubleFactorModel(const HestonParams &h1, const HestonParams &h2,
                    const JumpParamType &j = JumpParamType())
      : heston1(h1), heston2(h2), jump(j) {}
};

// ==========================================
// Convenience Aliases
// ==========================================

// --- ZERO FACTOR MODELS ---

// Standard Black-Scholes / Geometric Brownian Motion (0 Vols, No Jumps)
using BlackScholesModel = ZeroFactorModel<NoJumpParams>;

// Merton Jump-Diffusion (0 Vols, Merton Jumps)
using MertonModel = ZeroFactorModel<MertonParams>;

// Kou Jump-Diffusion (0 Vols, Kou Jumps)
using KouModel = ZeroFactorModel<KouParams>;

// --- SINGLE FACTOR MODELS ---

// Standard Heston (1 Vol, No Jumps)
using HestonModel = SingleFactorModel<NoJumpParams>;

// Bates Model (1 Vol, Merton Jumps)
using BatesModel = SingleFactorModel<MertonParams>;

// Bates-Kou Model (1 Vol, Kou Jumps)
using BatesKouModel = SingleFactorModel<KouParams>;

// --- DOUBLE FACTOR MODELS ---

// Double Heston (2 Vols, No Jumps)
using DoubleHestonModel = DoubleFactorModel<NoJumpParams>;

// Double Bates (2 Vols, Merton Jumps)
using DoubleBatesModel = DoubleFactorModel<MertonParams>;

// Double Bates-Kou (2 Vols, Kou Jumps)
using DoubleBatesKouModel = DoubleFactorModel<KouParams>;
