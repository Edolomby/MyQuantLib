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
  Vega,  // d(psi)/dv0i * base / (iu)  -- Full Mode
  Theta, // (d(psi)/dT + iu*(r-q)) * base / (iu)  -- Full Mode
  Rho,   // T * Re[base]  -- Full Mode
  // Cross-Greek kernels for vanna / charm
  VegaDx,  // d(psi)/dv0i * base          -- Full (vanna for digital payoffs)
  ThetaDx, // (d(psi)/dT + iu*(r-q))*base -- Full (charm for digital payoffs)
};

// =============================================================================
// GIL-PELAEZ KERNEL
// Template parameters:
//   Target     -- selects the integrand formula (compile-time)
//   Model      -- the model type
//   Traits     -- AffineTraits specialization for Model
//   FactorIdx  -- var factor index for Vega kernel (compile-time, default 0)
// =============================================================================
template <KernelTarget Target, typename Model, typename Traits,
          int FactorIdx = 0>
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

    // 3. Base Exponent: i*u*(r-q)*T + psi(u_shifted) - i*u*ln(K/S0)
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

    } else if constexpr (Target == KernelTarget::Vega) {
      // d(psi)/dv0i evaluated at u_shifted, multiplied into the Price integrand
      // FactorIdx selects which variance factor to differentiate (compile-time)
      Complex dpsi_dv0 =
          Traits::template d_cf_dv0<FactorIdx>(model_, u_shifted, T_);
      return std::real(base_term * dpsi_dv0 / (I * u));

    } else if constexpr (Target == KernelTarget::Theta) {
      // d/dT of Re[base/(iu)] = Re[(d_psi/dT + i*u*(r-q)) * base / (iu)]
      Complex dpsi_dT = Traits::d_cf_dT(model_, u_shifted, T_);
      double rate_per_T = (T_ > 1e-12) ? (rate_drift_ / T_) : 0.0;
      Complex dT_factor = dpsi_dT + I * u * rate_per_T;
      return std::real(base_term * dT_factor / (I * u));

    } else if constexpr (Target == KernelTarget::Rho) {
      // d/dr of i*u*(r-q)*T in exponent gives i*u*T
      // so d/dr Re[base/(iu)] = T * Re[base]
      return T_ * std::real(base_term);

      // -------------------------------------------------------------------------
      // Cross-Greek kernels: same as above but with Dx / Dxx weighting
      // -------------------------------------------------------------------------
    } else if constexpr (Target == KernelTarget::VegaDx) {
      // ∂²V/∂S∂σᵢ kernel: d(psi)/dv0i * base  (Dx weighting = no 1/(iu))
      Complex dpsi_dv0 =
          Traits::template d_cf_dv0<FactorIdx>(model_, u_shifted, T_);
      return std::real(base_term * dpsi_dv0);

    } else {
      // KernelTarget::ThetaDx
      // ∂²V/∂S∂T kernel: (d(psi)/dT + iu*(r-q)) * base  (Dx weighting)
      Complex dpsi_dT = Traits::d_cf_dT(model_, u_shifted, T_);
      double rate_per_T = (T_ > 1e-12) ? (rate_drift_ / T_) : 0.0;
      Complex dT_factor = dpsi_dT + I * u * rate_per_T;
      return std::real(base_term * dT_factor);
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