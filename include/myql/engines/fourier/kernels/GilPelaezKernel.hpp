#pragma once
#include <cmath>
#include <complex>
#include <myql/core/PricingTypes.hpp>
#include <myql/models/asvj/core/AffineTraits.hpp>

// -----------------------------------------------------------------------------
// KERNEL TARGETS (Mathematical Factorization specific to Fourier)
// -----------------------------------------------------------------------------
enum class KernelTarget {
  Price, // W(u) = 1 / (iu)
  Dx,    // W(u) = 1
  Dxx,   // W(u) = iu
  Vega,  // Placeholder for Full Mode
  Theta, // Placeholder for Full Mode
  Rho    // Placeholder for Full Mode
};

// =============================================================================
// EUROPEAN OPTION STRATEGY (Generic)
// Implements the Gil-Pelaez Integrand for P1 and P2 probabilities.
// use general KernelTarget enum to select kernel (enabling greeks)
// =============================================================================
template <KernelTarget Target, typename Model, typename Traits>
class GilPelaezKernel {
public:
  using Complex = std::complex<double>;

  GilPelaezKernel(const Model &model, double T, double r, double q,
                  double K_norm, bool is_P2)
      : model_(model), T_(T), logK_(std::log(K_norm)), is_P2_(is_P2) {
    rate_drift_ = (r - q) * T_;
  }

  // -------------------------------------------------------------------------
  // The Integrand: Re[ exp(total_exponent) * W(u) ]
  // -------------------------------------------------------------------------
  inline double operator()(double u_real) const {
    Complex u(u_real, 0.0);
    const Complex I(0.0, 1.0);

    // 1. Measure Shift: P1 uses (u - i), P2 uses u
    Complex u_shifted = is_P2_ ? u : (u - I);

    // 2. Characteristic Function
    Complex psi_martingale =
        Traits::characteristic_log_martingale(model_, u_shifted, T_);

    // 3. Base Exponent: i*u*(r-q)T + psi(u) - i*u*ln(K/S0)
    Complex final_exponent =
        (I * u * rate_drift_) + psi_martingale - (I * u * logK_);
    Complex base_term = std::exp(final_exponent);

    // 4. Compile-Time Dispatch for W(u)
    if constexpr (Target == KernelTarget::Price) {
      return std::real(base_term / (I * u));
    } else if constexpr (Target == KernelTarget::Dx) {
      return std::real(base_term);
    } else if constexpr (Target == KernelTarget::Dxx) {
      return std::real(base_term * (I * u));
    } else {
      static_assert(Target == KernelTarget::Price ||
                        Target == KernelTarget::Dx ||
                        Target == KernelTarget::Dxx,
                    "Full Greeks (Vega, Theta, Rho) are planned but not yet "
                    "implemented.");
      return 0.0;
    }
  }

  inline double magnitude_sq(double u_real) const {
    double val = this->operator()(u_real);
    return val * val;
  }

private:
  const Model &model_;
  double T_;
  double rate_drift_;
  double logK_; // log(K/S0) is the log of the normalized strike
  bool is_P2_;
};
