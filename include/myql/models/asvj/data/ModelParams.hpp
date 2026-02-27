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

// =============================================================================
// 3. EXPECTED AVERAGE VARIANCE HELPERS (for Payoff Smoothing)
// Provides rigorous expected average variance
// $\bar{v}_T = \frac{1}{T} \int_0^T \mathbb{E}[v_t] dt$
// =============================================================================
namespace myql::models::variance {

// --- Diffusion Variance ---
inline double expected_average_variance(double vol, double /*T*/) {
  return vol * vol;
}

inline double expected_average_variance(const HestonParams &h, double T) {
  if (std::abs(h.kappa * T) > 1e-8) {
    return h.theta +
           (h.v0 - h.theta) * (1.0 - std::exp(-h.kappa * T)) / (h.kappa * T);
  }
  return h.v0; // Taylor fallback for tiny kappa*T
}

// --- Jump Variance ---
inline double jump_variance(const NoJumpParams &) { return 0.0; }

inline double jump_variance(const MertonParams &j) {
  return j.lambda * (j.mu * j.mu + j.delta * j.delta);
}

inline double jump_variance(const KouParams &j) {
  return j.lambda *
         (j.p_up / (j.eta1 * j.eta1) + (1.0 - j.p_up) / (j.eta2 * j.eta2));
}

} // namespace myql::models::variance