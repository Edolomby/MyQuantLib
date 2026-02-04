#include "models/HestonModel.hpp"
#include <cmath>
#include <iostream>

using namespace std::complex_literals;

HestonModel::HestonModel(const Parameters &params) : p_(params) {}

// =========================================================
// 1. EUROPEAN OPTION IMPLEMENTATION
// =========================================================

Complex HestonModel::get_characteristic_function(double v, double T,
                                                 double lambda,
                                                 bool measure_shift_b) const {
  // Mapping parameters to BmHeston.cpp notation:
  // a = kappa * theta
  // b = kappa (or kappa - rho*sigma if shifted)
  // s = sigma (vol of vol)

  double a = p_.kappa * p_.theta;
  double b = p_.kappa;
  double s = p_.sigma;
  double rho = p_.rho;

  // Adjust 'b' for measure change (P1 vs P2)
  if (!measure_shift_b) {
    b -= rho * s;
  }

  double s2 = s * s;

  // --- Albrecher et al. (2006) Stable Formulation ---
  Complex b_rhosv1i = b - rho * s * v * 1i;
  Complex Delta_root =
      std::sqrt(b_rhosv1i * b_rhosv1i + s2 * v * (v + 2.0 * lambda * 1i));

  // Stable 'g' (Minus over Plus)
  Complex g = (b_rhosv1i - Delta_root) / (b_rhosv1i + Delta_root);

  // Pre-calculate Exponential
  Complex exp_Dt = std::exp(-Delta_root * T);

  // Calculate term_t
  Complex term_t = (b_rhosv1i - Delta_root) / s2;

  Complex phi =
      a * (term_t * T - 2.0 / s2 * std::log((1.0 - g * exp_Dt) / (1.0 - g)));

  // Calculate psi (Variance component)
  Complex psi = p_.v0 * term_t * ((1.0 - exp_Dt) / (1.0 - g * exp_Dt));

  return std::exp(phi + psi);
}

// =========================================================
// 2. ASIAN OPTION IMPLEMENTATION
// =========================================================

// Note: Return type must be fully qualified (HestonModel::ZCoeffs)
HestonModel::ZCoeffs HestonModel::compute_asian_z_coeffs(const Complex &s,
                                                         const Complex &w,
                                                         const double T) const {
  ZCoeffs z;

  double rho = p_.rho;
  double kap = p_.kappa;
  double omr2 = 1.0 - rho * rho;
  double fact = 2.0 * rho * kap - p_.sigma;

  // Formula matching BmHeston.cpp (Eq 13)
  z.z1 = (s * s * omr2) / (2.0 * T * T);
  z.z2 = (s * fact) / (2.0 * p_.sigma * T) + (s * w * omr2) / T;
  z.z3 = (s * rho) / (p_.sigma * T) + (w * fact) / (2.0 * p_.sigma) +
         (w * w * omr2) / 2.0;
  z.z4 = (w * rho) / p_.sigma;

  return z;
}

State HestonModel::get_asian_initial_condition(const Complex &s_freq,
                                               const Complex &w_freq,
                                               const double T) const {
  ZCoeffs z = compute_asian_z_coeffs(s_freq, w_freq, T);
  State y;
  // Boundary condition at tau=0: C(0)=z4, D(0)=0
  y << z.z4, Complex(0.0, 0.0);
  return y;
}

std::function<State(double, const State &)>
HestonModel::get_asian_riccati_system(const Complex &s_freq,
                                      const Complex &w_freq,
                                      const double T) const {

  ZCoeffs z = compute_asian_z_coeffs(s_freq, w_freq, T);
  Parameters p = p_; // Capture by value

  // ODE: dy/dt = f(t, y)
  // Note: 't' here represents the integration variable (time to maturity)
  return [z, p](double t, const State &y) -> State {
    Complex C = y(0);

    // BmHeston Eq A12: B(t) = z1*t^2 + z2*t + z3
    Complex B_val = z.z1 * (t * t) + z.z2 * t + z.z3;

    // dC/dt
    Complex dC = p.kappa * C - 0.5 * p.sigma * p.sigma * (C * C) - B_val;

    // dD/dt
    Complex dD = -p.kappa * p.theta * C;

    return State(dC, dD);
  };
}

// =========================================================
// 3. MONTE CARLO IMPLEMENTATION
// =========================================================

// Note: Return type must be fully qualified (HestonModel::DiscrParams)
HestonModel::DiscrParams
HestonModel::get_discretization_params(double dt, double r, double q) const {
  DiscrParams C;
  // dt = time step
  // r = risk free rate
  // q = dividend yield

  double a = p_.kappa * p_.theta;
  double b = p_.kappa;
  double sigma = p_.sigma;
  double net_r = r - q;
  double rho = p_.rho;

  // CIR PART (Exact simulation / Alfonsi)
  if (sigma * sigma > 4.0 * a) {
    // High volatility case
    double ex = std::exp(-b * dt);
    double psi = (b == 0.0) ? dt : (1.0 - ex) / b;

    C.v0 = 2.0 / (sigma * sigma * psi) * ex;
    C.v1 = 2.0 * a / (sigma * sigma);
    C.v2 = 2.0 / (sigma * sigma * psi);
  } else {
    // Low volatility case
    double ex = std::exp(-b * dt / 2.0);
    double psi = (b == 0.0) ? dt / 2.0 : (1.0 - ex) / b;
    double D = a - sigma * sigma / 4.0;
    double s_star = sigma * std::sqrt(dt) / 2.0;

    C.v0 = ex;
    C.v1 = D * psi;
    C.v2 = s_star;
  }

  // HESTON PART
  C.ls0 = (net_r - rho * a / sigma) * dt;
  C.ls1 = (b * rho / sigma - 0.5) * dt / 2.0 - rho / sigma;
  C.ls2 = (b * rho / sigma - 0.5) * dt / 2.0 + rho / sigma;
  C.ls3 = std::sqrt(dt * (1.0 - rho * rho) / 2.0);

  // INTEGRATION PART
  C.h_2 = dt / 2.0;

  return C;
}