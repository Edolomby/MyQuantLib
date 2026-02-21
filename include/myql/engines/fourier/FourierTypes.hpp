#pragma once

// -----------------------------------------------------------------------------
// REQUEST MODES
// -----------------------------------------------------------------------------
enum class GreekMode {
  None,      // Just Price
  Essential, // Price, Delta, Gamma
  Full       // Price, Delta, Gamma, Vega, Theta, Rho, maybe VannaVolga (Future)
};

// -----------------------------------------------------------------------------
// KERNEL TARGETS (Mathematical Factorization)
// -----------------------------------------------------------------------------
enum class KernelTarget {
  Price, // W(u) = 1 / (iu)
  Dx,    // W(u) = 1
  Dxx,   // W(u) = iu
  Vega,  // Placeholder for Full Mode
  Theta, // Placeholder for Full Mode
  Rho    // Placeholder for Full Mode
};

// -----------------------------------------------------------------------------
// RESULT STRUCTS (Compile-Time Pruned)
// -----------------------------------------------------------------------------
template <GreekMode Mode> struct FourierResult;

template <> struct FourierResult<GreekMode::None> {
  double price{0.0};
};

template <> struct FourierResult<GreekMode::Essential> {
  double price{0.0};
  double delta{0.0};
  double gamma{0.0};
};

template <> struct FourierResult<GreekMode::Full> {
  double price{0.0};
  double delta{0.0};
  double gamma{0.0};
  double vega{0.0};
  double theta{0.0};
  double rho{0.0};
};