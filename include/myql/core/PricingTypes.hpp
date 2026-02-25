#pragma once

// -----------------------------------------------------------------------------
// UNIVERSAL GREEK REQUEST MODES
// -----------------------------------------------------------------------------
enum class GreekMode {
  None,      // Just Price
  Essential, // Price, Delta, Gamma
  Full       // Price, Delta, Gamma, Vega, Theta, Rho
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
  T vega{};
  T theta{};
  T rho{};
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

  T vega;
  T vega_std_err;

  T theta;
  T theta_std_err;

  T rho;
  T rho_std_err;
};