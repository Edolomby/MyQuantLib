#pragma once
#include <cmath>
#include <complex>
#include <myql/models/asvj/core/AffineTraits.hpp>

// =============================================================================
// EUROPEAN OPTION STRATEGY (Generic)
// Implements the Gil-Pelaez Integrand for P1 and P2 probabilities.
// =============================================================================

template <typename Model> class EuropeanStrategy {
public:
  using Complex = std::complex<double>;

  // Constructor
  EuropeanStrategy(const Model &model, double T, double r, double q,
                   double K_norm, bool is_P2)
      : model_(model), T_(T), logK_(std::log(K_norm)), is_P2_(is_P2) {
    rate_drift_ = (r - q) * T_;
  }

  // -------------------------------------------------------------------------
  // 1. The Integrand: Re[ e^{-i*u*k} * Phi(u_shifted) / (i*u) ]
  // -------------------------------------------------------------------------
  double operator()(double u_real) const {
    // Limit at u=0 is handled by the caller (Gil-Pelaez formula 0.5 + ...)
    if (std::abs(u_real) < 1e-12)
      return 0.0; // Soft fallback, though engine avoids 0.

    Complex u(u_real, 0.0);
    const Complex I(0.0, 1.0);

    // 1. Calculate Martingale CF (Heston + Compensated Jumps)
    Complex u_shifted = is_P2_ ? u : (u - I);
    Complex psi_martingale = AffineTraits<Model>::characteristic_log_martingale(
        model_, u_shifted, T_);

    // 2. Add Risk-Free Rate
    // exp( i*u * (r-q)T )
    Complex total_exponent = (I * u * rate_drift_) + psi_martingale;

    // 3. Gil-Pelaez Kernel
    // exp( total - i*u*k ) / (i*u)
    Complex final_exponent = total_exponent - (I * u * logK_);
    return std::real(std::exp(final_exponent) / (I * u));
  }

  // -------------------------------------------------------------------------
  // 2. Magnitude (For Upper Bound Search)
  // -------------------------------------------------------------------------
  // Used by the engine to decide where to cut the integral (0 to Inf).
  double magnitude_sq(double u_real) const {
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