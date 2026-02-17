#pragma once
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
// TRAIT SPECIALIZATION: Single Factor
// =============================================================================
template <typename JumpParamType>
struct AffineTraits<SingleFactorModel<JumpParamType>> {
  using Model = SingleFactorModel<JumpParamType>;
  using Complex = std::complex<double>;
  using JumpPolicy = typename detail::Map<JumpParamType>::Type;

  // Returns the MARTINGALE Log-Characteristic Function
  static Complex characteristic_log_martingale(const Model &m, const Complex &u,
                                               double t) {
    // 1. Heston (Already Martingale-Corrected for diffusion)
    Complex phi_h = detail::heston_log_characteristic(m.heston, u, t);

    // 2. Jump (returns the Compensated Exponent directly)
    // Formula: t * lambda * [ E[e^iuJ] - 1 - iu*k ]
    Complex phi_j = JumpPolicy::compensated_exponent(m.jump, u);

    return phi_h + (phi_j * t);
  }
};

// =============================================================================
// TRAIT SPECIALIZATION: Double Factor
// =============================================================================
template <typename JumpParamType>
struct AffineTraits<DoubleFactorModel<JumpParamType>> {
  using Model = DoubleFactorModel<JumpParamType>;
  using Complex = std::complex<double>;
  using JumpPolicy = typename detail::Map<JumpParamType>::Type;

  static Complex characteristic_log_martingale(const Model &m, const Complex &u,
                                               double t) {
    Complex phi_h1 = detail::heston_log_characteristic(m.heston1, u, t);
    Complex phi_h2 = detail::heston_log_characteristic(m.heston2, u, t);
    Complex phi_j = JumpPolicy::compensated_exponent(m.jump, u);

    return phi_h1 + phi_h2 + (phi_j * t);
  }
};