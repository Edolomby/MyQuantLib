#pragma once
#include <cmath>
#include <complex>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/policies/CFPolicies.hpp>

template <typename Model> struct AffineTraits;

// =============================================================================
// Characteristic Function Helpers
// =============================================================================

namespace detail {
// Recall: The Heston formula used here (Albrecher) ALREADY contains the
// -0.5*v convexity correction in the diffusion term.
using Complex = std::complex<double>;

inline Complex heston_log_characteristic(const HestonParams &p,
                                         const Complex &u, double t) {
  if (t < 1e-8)
    return 0.0;
  Complex I(0.0, 1.0);
  double sigma2 = p.sigma * p.sigma;
  Complex d_prod = p.rho * p.sigma * I * u;
  Complex D_root = std::sqrt((d_prod - p.kappa) * (d_prod - p.kappa) +
                             sigma2 * (I * u + u * u));
  Complex g = (p.kappa - d_prod - D_root) / (p.kappa - d_prod + D_root);
  Complex term_A =
      (p.kappa * p.theta / sigma2) *
      ((p.kappa - d_prod - D_root) * t -
       2.0 * std::log((1.0 - g * std::exp(-D_root * t)) / (1.0 - g)));
  Complex term_B =
      (p.v0 / sigma2) * (p.kappa - d_prod - D_root) *
      ((1.0 - std::exp(-D_root * t)) / (1.0 - g * std::exp(-D_root * t)));
  return term_A + term_B;
}

// Derivative of Heston log-CF w.r.t. initial variance v0
// Result: A * (1 - exp(-D*t)) / (sigma^2 * (1 - g * exp(-D*t)))
// where A = kappa - rho*sigma*i*u - D
inline Complex heston_d_cf_dv0(const HestonParams &p, const Complex &u,
                               double t) {
  if (t < 1e-8)
    return 0.0;
  Complex I(0.0, 1.0);
  double sigma2 = p.sigma * p.sigma;
  Complex d_prod = p.rho * p.sigma * I * u;
  Complex D_root = std::sqrt((d_prod - p.kappa) * (d_prod - p.kappa) +
                             sigma2 * (I * u + u * u));
  Complex g = (p.kappa - d_prod - D_root) / (p.kappa - d_prod + D_root);
  Complex A = p.kappa - d_prod - D_root;
  Complex eT = std::exp(-D_root * t);
  return (A * (1.0 - eT)) / (sigma2 * (1.0 - g * eT));
}

// Derivative of Heston log-CF w.r.t. maturity T
// Differentiates both term_A and term_B of the Albrecher formula:
//   d(term_A)/dT = (kappa*theta/sigma^2) * (A - 2*g*D*eT / (1 - g*eT))
//   d(term_B)/dT = (v0/sigma^2) * A * D * eT * (1-g) / (1 - g*eT)^2
inline Complex heston_d_cf_dT(const HestonParams &p, const Complex &u,
                              double t) {
  if (t < 1e-8)
    return 0.0;
  Complex I(0.0, 1.0);
  double sigma2 = p.sigma * p.sigma;
  Complex d_prod = p.rho * p.sigma * I * u;
  Complex D_root = std::sqrt((d_prod - p.kappa) * (d_prod - p.kappa) +
                             sigma2 * (I * u + u * u));
  Complex g = (p.kappa - d_prod - D_root) / (p.kappa - d_prod + D_root);
  Complex A = p.kappa - d_prod - D_root;
  Complex eT = std::exp(-D_root * t);
  Complex denom = 1.0 - g * eT;

  Complex dA_dT =
      (p.kappa * p.theta / sigma2) * (A - 2.0 * g * D_root * eT / denom);
  Complex dB_dT =
      (p.v0 / sigma2) * A * D_root * eT * (1.0 - g) / (denom * denom);
  return dA_dT + dB_dT;
}

// Policy Mapper logic
template <typename T> struct Map;
template <> struct Map<NoJumpParams> {
  using Type = CF_Policies::NoJumps;
};
template <> struct Map<MertonParams> {
  using Type = CF_Policies::MertonJump;
};
template <> struct Map<KouParams> {
  using Type = CF_Policies::KouJump;
};
} // namespace detail

// =============================================================================
// TRAIT SPECIALIZATION: Zero Factor (Black-Scholes + Jumps)
// =============================================================================
template <typename JumpParamType>
struct AffineTraits<ZeroFactorModel<JumpParamType>> {
  using Model = ZeroFactorModel<JumpParamType>;
  using Complex = std::complex<double>;
  using JumpPolicy = typename detail::Map<JumpParamType>::Type;

  // Delegated from Model so any generic code using the traits reads this value
  static constexpr int num_variance_factors = Model::num_variance_factors;

  static Complex characteristic_log_martingale(const Model &m, const Complex &u,
                                               double t) {
    // 1. Black-Scholes Diffusion (Martingale: -0.5*vol^2*t + vol*W_t)
    // Characteristic Exponent: -0.5 * vol^2 * t * (i*u + u^2)
    double var_t = m.vol * m.vol * t;
    Complex phi_bs = -0.5 * var_t * (Complex(0.0, 1.0) * u + u * u);

    // 2. Jump (Compensated Exponent)
    Complex phi_j = JumpPolicy::compensated_exponent(m.jump, u);

    return phi_bs + (phi_j * t);
  }

  // For ZeroFactor, vol is already a direct volatility parameter.
  // FactorIdx is ignored (only one vol parameter exists).
  template <int FactorIdx>
  static Complex d_cf_dv0(const Model &m, const Complex &u, double t) {
    Complex I(0.0, 1.0);
    return -m.vol * t * (I * u + u * u);
  }

  // d(psi)/dT = -0.5*vol^2*(i*u + u^2) + jump_compensated_exponent
  static Complex d_cf_dT(const Model &m, const Complex &u, double /*t*/) {
    Complex I(0.0, 1.0);
    return -0.5 * m.vol * m.vol * (I * u + u * u) +
           JumpPolicy::compensated_exponent(m.jump, u);
  }

  // Chain-rule factor: 1.0 because vol IS sigma directly
  template <int FactorIdx>
  static double vega_chain_factor(const Model & /*m*/) {
    return 1.0;
  }

  // Returns a copy of the model with the variance factor bumped by dv
  template <int FactorIdx> static Model bump_v0(const Model &m, double dv) {
    Model bumped = m;
    bumped.vol += dv;
    return bumped;
  }
};

// =============================================================================
// TRAIT SPECIALIZATION: Single Factor (Heston + Jumps)
// =============================================================================
template <typename JumpParamType>
struct AffineTraits<SingleFactorModel<JumpParamType>> {
  using Model = SingleFactorModel<JumpParamType>;
  using Complex = std::complex<double>;
  using JumpPolicy = typename detail::Map<JumpParamType>::Type;

  static constexpr int num_variance_factors = Model::num_variance_factors;

  static Complex characteristic_log_martingale(const Model &m, const Complex &u,
                                               double t) {
    // 1. Heston (Already Martingale-Corrected for diffusion)
    Complex phi_h = detail::heston_log_characteristic(m.heston, u, t);

    // 2. Jump (returns the Compensated Exponent directly)
    // Formula: t * lambda * [ E[e^iuJ] - 1 - iu*k ]
    Complex phi_j = JumpPolicy::compensated_exponent(m.jump, u);

    return phi_h + (phi_j * t);
  }

  // FactorIdx is ignored (only one Heston factor)
  template <int FactorIdx>
  static Complex d_cf_dv0(const Model &m, const Complex &u, double t) {
    return detail::heston_d_cf_dv0(m.heston, u, t);
  }

  // d(psi)/dT = d(heston)/dT + jump_compensated_exponent
  static Complex d_cf_dT(const Model &m, const Complex &u, double t) {
    return detail::heston_d_cf_dT(m.heston, u, t) +
           JumpPolicy::compensated_exponent(m.jump, u);
  }

  // Chain-rule factor: 2*sqrt(v0) to convert d/dv0 into d/d(sigma)
  template <int FactorIdx> static double vega_chain_factor(const Model &m) {
    return 2.0 * std::sqrt(m.heston.v0);
  }

  // Returns a copy of the model with the variance factor bumped by dv
  template <int FactorIdx> static Model bump_v0(const Model &m, double dv) {
    Model bumped = m;
    bumped.heston.v0 += dv;
    return bumped;
  }
};

// =============================================================================
// TRAIT SPECIALIZATION: Double Factor (Double Heston + Jumps)
// =============================================================================
template <typename JumpParamType>
struct AffineTraits<DoubleFactorModel<JumpParamType>> {
  using Model = DoubleFactorModel<JumpParamType>;
  using Complex = std::complex<double>;
  using JumpPolicy = typename detail::Map<JumpParamType>::Type;

  static constexpr int num_variance_factors = Model::num_variance_factors;

  static Complex characteristic_log_martingale(const Model &m, const Complex &u,
                                               double t) {
    Complex phi_h1 = detail::heston_log_characteristic(m.heston1, u, t);
    Complex phi_h2 = detail::heston_log_characteristic(m.heston2, u, t);
    Complex phi_j = JumpPolicy::compensated_exponent(m.jump, u);
    return phi_h1 + phi_h2 + (phi_j * t);
  }

  // FactorIdx selects heston1 (0) or heston2 (1) at compile time
  template <int FactorIdx>
  static Complex d_cf_dv0(const Model &m, const Complex &u, double t) {
    if constexpr (FactorIdx == 0)
      return detail::heston_d_cf_dv0(m.heston1, u, t);
    else
      return detail::heston_d_cf_dv0(m.heston2, u, t);
  }

  // d(psi)/dT = sum of both Heston T-derivatives + jump_compensated_exponent
  static Complex d_cf_dT(const Model &m, const Complex &u, double t) {
    return detail::heston_d_cf_dT(m.heston1, u, t) +
           detail::heston_d_cf_dT(m.heston2, u, t) +
           JumpPolicy::compensated_exponent(m.jump, u);
  }

  // Chain-rule factor: 2*sqrt(v0i) for the selected factor at compile time
  template <int FactorIdx> static double vega_chain_factor(const Model &m) {
    if constexpr (FactorIdx == 0)
      return 2.0 * std::sqrt(m.heston1.v0);
    else
      return 2.0 * std::sqrt(m.heston2.v0);
  }

  // Returns a copy of the model with the variance factor bumped by dv
  template <int FactorIdx> static Model bump_v0(const Model &m, double dv) {
    Model bumped = m;
    if constexpr (FactorIdx == 0)
      bumped.heston1.v0 += dv;
    else
      bumped.heston2.v0 += dv;
    return bumped;
  }
};