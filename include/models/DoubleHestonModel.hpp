#pragma once
#include <Eigen/Dense>
#include <complex>
#include <functional>

using Complex = std::complex<double>;
using State = Eigen::Array2cd;

struct Parameters {
  double kappa; // Speed of reversion
  double theta; // Long-run variance
  double sigma; // Volatility of volatility
  double rho;   // Correlation
  double v0;    // Initial volatility
};

struct DoubleParameters {
  Parameters p1;
  Parameters p2;
};

class DoubleHestonModel {
public:
  DoubleHestonModel(const DoubleParameters &params);

  struct DiscrParams {
    double v10, v11, v12;      // CIR drift/diffusion parts for v1
    double v20, v21, v22;      // CIR drift/diffusion parts for v2
    double ls0, ls1, ls2, ls3; // Log-price update parts
    double h_2;                // h/2 for integration
  };

  // Getters
  const DoubleParameters &get_params() const { return p_; }

  // 1. European Option Math (Characteristic Function)
  Complex get_characteristic_function(double v, double t, double lambda,
                                      bool measure_shift_b) const;

  // 2. Monte Carlo Math (Discretization)
  DiscrParams get_discretization_params(double dt, double r, double q) const;

private:
  DoubleParameters p_;

  // Helper to calculate the component for a single variance process
  std::complex<double> compute_component(double v, double T, double kappa,
                                         double theta, double sigma, double rho,
                                         double v0) const;
};