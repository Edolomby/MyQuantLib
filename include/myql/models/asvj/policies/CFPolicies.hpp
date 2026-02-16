#pragma once
#include <complex>
#include <myql/models/asvj/data/ModelParams.hpp>

namespace CF_Policies {
using Complex = std::complex<double>;

// 1. MERTON (Gaussian)
struct MertonJump {
  static inline Complex compensated_exponent(const MertonParams &p,
                                             const Complex &u) {
    // Common constants
    constexpr Complex I(0.0, 1.0);
    double half_delta2 = 0.5 * p.delta * p.delta;

    // 1. Raw Characteristic Exponent Part: exp(iu*mu - 0.5*u^2*delta^2)
    Complex exponent_phi = (I * u * p.mu) - (half_delta2 * u * u);
    Complex E_eiuJ = std::exp(exponent_phi);

    // 2. Compensator Part: exp(mu + 0.5*delta^2)
    // This is the mean jump size k = E[e^J] - 1
    double exponent_comp = p.mu + half_delta2;
    double k = std::exp(exponent_comp) - 1.0;

    // 3. Fused Result: lambda * [ (E[eiuJ] - 1) - iu * k ]
    return p.lambda * ((E_eiuJ - 1.0) - (I * u * k));
  }
};

// 2. KOU (Double Exponential)
struct KouJump {
  static inline Complex compensated_exponent(const KouParams &p,
                                             const Complex &u) {
    constexpr Complex I(0.0, 1.0);

    // 1. Raw Characteristic Exponent
    // phi(u) = p_up * eta1 / (eta1 - iu) + ...
    Complex term_up_phi = (p.p_up * p.eta1) / (p.eta1 - I * u);
    Complex term_down_phi = ((1.0 - p.p_up) * p.eta2) / (p.eta2 + I * u);
    Complex phi_part = term_up_phi + term_down_phi - 1.0;

    // 2. Compensator (Mean Jump Size k)
    // k = p_up * eta1 / (eta1 - 1) + ...
    // Only valid if eta1 > 1.0
    double term_up_k = (p.p_up * p.eta1) / (p.eta1 - 1.0);
    double term_down_k = ((1.0 - p.p_up) * p.eta2) / (p.eta2 + 1.0);
    double k = term_up_k + term_down_k - 1.0;

    // 3. Fused Result: lambda * [ phi(u) - iu * k ]
    return p.lambda * (phi_part - (I * u * k));
  }
};

// 3. NO JUMPS
struct NoJumps {
  static inline Complex compensated_exponent(const NoJumpParams &,
                                             const Complex &) {
    return 0.0;
  }
};
} // namespace CF_Policies