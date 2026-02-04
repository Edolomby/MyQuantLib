#pragma once
#include <Eigen/Dense>
#include <complex>
#include <functional>

using Complex = std::complex<double>;
using State = Eigen::Array2cd;

class HestonModel {
public:
  struct Parameters {
    double kappa; // Speed of reversion
    double theta; // Long-run variance
    double sigma; // Volatility of volatility
    double rho;   // Correlation
    double v0;    // Initial volatility
  };

  // Stack-allocated struct for MC parameters
  struct DiscrParams {
    double v0, v1, v2;         // CIR drift/diffusion parts (indices 0,1,2)
    double ls0, ls1, ls2, ls3; // Log-price update parts (indices 3,4,5,6)
    double h_2;                // h/2 for integration (index 7)
  };

  HestonModel(const Parameters &params);

  // Getters
  const Parameters &get_params() const { return p_; }

  // 1. European Option Math (Characteristic Function)
  Complex get_characteristic_function(double v, double t, double lambda,
                                      bool measure_shift_b) const;

  // 2. Asian Option Math (Riccati System)
  std::function<State(double, const State &)>
  get_asian_riccati_system(const Complex &s_freq, const Complex &w_freq,
                           double T) const;

  State get_asian_initial_condition(const Complex &s_freq,
                                    const Complex &w_freq, double T) const;

  // 3. Monte Carlo Math (Discretization)
  DiscrParams get_discretization_params(double dt, double r, double q) const;

private:
  Parameters p_;

  // Internal helper structure
  struct ZCoeffs {
    Complex z1, z2, z3, z4;
  };

  // Helper method to compute coefficients for the Asian Riccati system
  ZCoeffs compute_asian_z_coeffs(const Complex &s, const Complex &w,
                                 double T) const;
};