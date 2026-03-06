#pragma once
#include <array>

// -----------------------------------------------------------------------------
// UNIVERSAL GREEK REQUEST MODES
// -----------------------------------------------------------------------------
enum class GreekMode {
  None,      // Just Price
  Essential, // Price, Delta, Gamma
  Full,      // Price, Delta, Gamma, Vega, Theta, Rho, Vanna, Charm
};

// =============================================================================
// FOURIER RESULT STRUCTS
// =============================================================================
template <GreekMode Mode, typename T = double> struct FourierResult;

template <typename T> struct FourierResult<GreekMode::None, T> {
  T price{};
};

template <typename T> struct FourierResult<GreekMode::Essential, T> {
  T price{};
  T delta{};
  T gamma{};
};

template <typename T> struct FourierResult<GreekMode::Full, T> {
  T price{};
  T delta{};
  T gamma{};
  // vega[0] = factor 1, vega[1] = factor 2 (always 0 for non-double-factor
  // models)
  std::array<T, 2> vega{};
  T theta{};
  T rho{};
  // Cross-Greeks (∂/∂S of vol/time Greeks)
  std::array<T, 2> vanna{}; // ∂Delta/∂σᵢ — standard vol-desk Greek
  T charm{};                // ∂Delta/∂T  — standard for delta hedgers
};

// =============================================================================
// MONTE CARLO RESULT STRUCTS (T = double or std::vector)
// =============================================================================
template <GreekMode Mode, typename T> struct MonteCarloResult;

template <typename T> struct MonteCarloResult<GreekMode::None, T> {
  T price;
  T price_std_err;
};

template <typename T> struct MonteCarloResult<GreekMode::Essential, T> {
  T price;
  T price_std_err;

  T delta;
  T delta_std_err;

  T gamma;
  T gamma_std_err;
};

template <typename T> struct MonteCarloResult<GreekMode::Full, T> {
  T price;
  T price_std_err;

  T delta;
  T delta_std_err;

  T gamma;
  T gamma_std_err;

  // vega[0] = factor 1, vega[1] = factor 2 (always 0 for non-double-factor
  // models)
  std::array<T, 2> vega{};
  std::array<T, 2> vega_std_err{};

  T theta;
  T theta_std_err;

  T rho;
  T rho_std_err;

  // Cross-Greeks
  std::array<T, 2> vanna{};
  std::array<T, 2> vanna_std_err{};

  T charm;
  T charm_std_err;
};
