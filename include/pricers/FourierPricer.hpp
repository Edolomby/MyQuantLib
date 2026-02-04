#pragma once
#include "engines/FourierEngine.hpp"
#include "engines/strategies/EuropeanStrategy.hpp"
#include <algorithm>
#include <cmath>
#include <numbers> // C++20 for std::numbers::pi_v

// -----------------------------------------------------------------------------
// GENERIC FOURIER PRICER
// Works with ANY Model that satisfies the 'FourierModel' concept
// (Heston, Bates, RoughHeston, VarianceGamma, etc.)
// -----------------------------------------------------------------------------
template <typename Model>
double price_european_fourier(const Model &model, double S0, double K, double T,
                              double r, double q, bool is_call = true,
                              const typename FourierEngine::Config &engine_cfg =
                                  FourierEngine::Config()) {
  // 1. Renormalization (S0 -> 1.0)
  // We solve the problem for Spot = 1.0, Strike = K/S0
  double K_norm = K / S0;

  // 2. Instantiate Generic Strategies
  // The compiler builds a strategy SPECIFIC to the 'Model' type passed in.
  // No virtual function overhead. Direct inlining.
  EuropeanStrategy<Model> strat_P1(model, T, r, q, K_norm, false); // P1 (Delta)
  EuropeanStrategy<Model> strat_P2(model, T, r, q, K_norm,
                                   true); // P2 (Risk Neutral)

  // 3. Configure Engine
  FourierEngine engine(engine_cfg);

  // Oscillation hint: High moneyness = higher frequency oscillation =
  // DEPRECATED
  // double moneyness_freq = std::abs(std::log(1.0 / K_norm));
  // 4. Integrate
  // double I1 = engine.calculate_integral(strat_P1, moneyness_freq, T);
  // double I2 = engine.calculate_integral(strat_P2, moneyness_freq, T);
  // 4. Integrate without oscillation hint
  double I1 = engine.calculate_integral(strat_P1, T);
  double I2 = engine.calculate_integral(strat_P2, T);

  // 5. Gil-Pelaez Formula
  constexpr double INV_PI = std::numbers::inv_pi_v<double>; // 1/pi
  double P1 = 0.5 + I1 * INV_PI;
  double P2 = 0.5 + I2 * INV_PI;

  // 6. Denormalize & Discount
  double df_q = std::exp(-q * T);
  double df_r = std::exp(-r * T);

  double call_price = S0 * df_q * P1 - K * df_r * P2;

  // 7. Payoff Check
  if (is_call) {
    return std::max(0.0, call_price);
  } else {
    // Put-Call Parity: P = C - S*e^-qT + K*e^-rT
    return std::max(0.0, call_price - S0 * df_q + K * df_r);
  }
}