#pragma once
#include <algorithm>
#include <cmath>
#include <myql/engines/fourier/FourierEngine.hpp>
#include <myql/instruments/kernels/EuropeanStrategy.hpp>

// -----------------------------------------------------------------------------
// GENERIC ASVJ FOURIER PRICER
// -----------------------------------------------------------------------------

/**
 * @brief Prices a European Option using Fourier Inversion (Gil-Pelaez).
 * Supports ANY model compatible with AffineTraits (Heston, Bates,
 * DoubleHeston...).
 */
template <typename Model>
double price_european_fourier(const Model &model, double S0, double K, double T,
                              double r, double q, bool is_call = true,
                              const typename FourierEngine::Config &engine_cfg =
                                  FourierEngine::Config()) {

  // 1. Normalization
  // We price the option for Spot=1, Strike=K/S0.
  // Result is then scaled by S0.
  double K_norm = K / S0;

  // 2. Instantiate Strategies
  // P1: "Delta" Probability (uses shifted characteristic function)
  EuropeanStrategy<Model> strat_P1(model, T, r, q, K_norm,
                                   false); // is_P2=false

  // P2: Risk-Neutral Probability
  EuropeanStrategy<Model> strat_P2(model, T, r, q, K_norm, true); // is_P2=true

  // 3. Configure Engine
  FourierEngine engine(engine_cfg);

  // 4. Compute Integrals
  // The engine handles the adaptive grid and OpenMP parallelism.
  // We integrate from 0 to infinity (handled by engine upper bound search).
  double I1 = engine.calculate_integral(strat_P1, T);
  double I2 = engine.calculate_integral(strat_P2, T);

  // 5. Apply Gil-Pelaez Formula
  // P(x>k) = 0.5 + 1/pi * Integral
  constexpr double INV_PI = 0.31830988618379067154;
  double P1 = 0.5 + I1 * INV_PI;
  double P2 = 0.5 + I2 * INV_PI;

  // 6. Calculate Option Price
  // Call = S * e^{-qT} * P1 - K * e^{-rT} * P2
  double df_q = std::exp(-q * T);
  double df_r = std::exp(-r * T);

  // spot_term - strike_term
  double call_price = S0 * df_q * P1 - K * df_r * P2;

  // 7. Output Result
  if (is_call) {
    return std::max(0.0, call_price);
  } else {
    // Put-Call Parity: P = C - S*e^-qT + K*e^-rT
    // This is often numerically more stable than pricing the Put directly via
    // Fourier for deep OTM/ITM options.
    return std::max(0.0, call_price - S0 * df_q + K * df_r);
  }
}