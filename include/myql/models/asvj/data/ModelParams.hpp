#pragma once
#include <cmath>

// =============================================================================
// 1. DIFFUSION PARAMETERS
// =============================================================================
struct HestonParams {
  double kappa; // Mean reversion
  double theta; // Long-run var
  double sigma; // Vol of vol
  double rho;   // Correlation
  double v0;    // Initial var

  // Helper: Check if safe for "Restricted" (Low Vol) approximation
  // You defined this limit as sigma^2 <= 4*kappa*theta
  bool is_low_vol_regime() const {
    return (sigma * sigma) <= (4.0 * kappa * theta);
  }

  // Helper: Check if safe Feller condition is fulfilled
  // You defined this limit as sigma^2 <= 2*kappa*theta
  bool is_Feller_regime() const {
    return (sigma * sigma) <= (2.0 * kappa * theta);
  }
};

// =============================================================================
// 2. JUMP PARAMETERS (The Atoms)
// =============================================================================

// No Jumps (Empty generic placeholder)
struct NoJumpParams {};

// Merton (Gaussian Jumps)
struct MertonParams {
  double lambda; // Jump intensity
  double mu;     // Mean jump size
  double delta;  // Jump std dev
};

// Kou (Double Exponential Jumps)
struct KouParams {
  double lambda; // Intensity
  double p_up;   // Probability of up jump
  double eta1;   // Inverse mean of up jump
  double eta2;   // Inverse mean of down jump
};