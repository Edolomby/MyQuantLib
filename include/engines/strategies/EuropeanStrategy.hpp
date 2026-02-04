#pragma once
#include <cmath>
#include <complex>

using Complex = std::complex<double>;
using namespace std::complex_literals;

// Template parameter 'Model' can be HestonModel, BatesModel, etc.
// Requirement: Model must have a 'get_characteristic_function(...)' method.
template <typename Model> class EuropeanStrategy {
public:
  const Model &model_; // Direct reference to the specific model type
  double T_;
  double drift_factor_; // (r - q) * T
  double log_k_norm_;   // log(K / S0)
  bool is_P2_;          // Measure: P2 vs P1 (Delta)

  EuropeanStrategy(const Model &model, double T, double r, double q,
                   double K_norm, bool is_P2)
      : model_(model), T_(T), is_P2_(is_P2) {
    drift_factor_ = (r - q) * T;
    log_k_norm_ = std::log(K_norm);
  }

  // 1. The Integrand
  // The compiler will inline this AND the model's get_characteristic_function
  double operator()(double xi) const {
    // no zero checks we do no calculate integrand in 0

    double lambda = is_P2_ ? 0.5 : -0.5;

    // CALL THE GENERIC MODEL
    // Since 'Model' is known at compile time, this is a DIRECT call (no
    // vtable). should be inlined by the compiler
    Complex cf = model_.get_characteristic_function(xi, T_, lambda, is_P2_);

    Complex drift = std::exp(1i * xi * drift_factor_);
    Complex kernel = std::exp(-1i * xi * log_k_norm_) / (1i * xi);

    return (drift * cf * kernel).real();
  }

  // 2. Magnitude_sq (for upper bound search)
  double magnitude_sq(double xi) const {
    if (std::abs(xi) < 1e-12)
      return 1.0;
    double lambda = is_P2_ ? 0.5 : -0.5;
    Complex cf = model_.get_characteristic_function(xi, T_, lambda, is_P2_);
    return std::norm(cf) / (xi * xi);
  }
};