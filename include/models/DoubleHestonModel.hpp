#pragma once
#include <Eigen/Dense>
#include <complex>
#include <functional>

using Complex = std::complex<double>;
using State = Eigen::Array2cd;

class DoubleHestonModel {
public:
  struct Parameters {
    double kappa1;  // Speed of reversion v1
    double theta1;  // Long-run variance v1
    double sigma1;  // Volatility of volatility v1
    double rho1;    // Correlation WS Wv1
    double v01;     // Initial volatility v1
    double kappa2;  // Speed of reversion v2
    double theta2;  // Long-run variance v2
    double sigma2;  // Volatility of volatility v2
    double rho2;    // Correlation WS Wv2
    double v02;     // Initial volatility v2
    double r;       // Risk-free rate
    double q = 0.0; // Dividend yield
  };

  // Stack-allocated struct for MC parameters
  struct DiscrParams {
    double v10, v11, v12;      // CIR drift/diffusion parts for v1
    double v20, v21, v22;      // CIR drift/diffusion parts for v2
    double ls0, ls1, ls2, ls3; // Log-price update parts
    double h_2;                // h/2 for integration
  };

  DoubleHestonModel(const Parameters &params);

  // Getters
  const Parameters &get_params() const { return p_; }

  // 1. European Option Math (Characteristic Function)
  // Implements the "Stable" form from Albrecher et al.
  Complex get_characteristic_function(double v, double t, double lambda,
                                      bool measure_shift_b) const;

  // 2. Monte Carlo Math (Discretization)
  DiscrParams get_discretization_params(double dt) const;

private:
  Parameters p_;
};