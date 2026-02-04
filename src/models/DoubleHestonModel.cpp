#include "models/DoubleHestonModel.hpp"
#include <cmath>
#include <iostream>

using namespace std::complex_literals;

DoubleHestonModel::DoubleHestonModel(const Parameters &params) : p_(params) {}

// =========================================================
// 1. EUROPEAN OPTION IMPLEMENTATION
// =========================================================

Complex DoubleHestonModel::get_characteristic_function(
    double v, double t, double lambda, bool measure_shift_b) const {
  // a = kappa * theta
  // b = kappa (or kappa - rho*sigma if shifted)
  // s = sigma (vol of vol)
  // t is the time to maturity

  double a1 = p_.kappa1 * p_.theta1;
  double b1 = p_.kappa1;
  double s1 = p_.sigma1;
  double rho1 = p_.rho1;

  double a2 = p_.kappa2 * p_.theta2;
  double b2 = p_.kappa2;
  double s2 = p_.sigma2;
  double rho2 = p_.rho2;

  // Adjust 'b' for measure change (P1 vs P2)
  if (measure_shift_b) {
    b1 -= rho1 * s1;
    b2 -= rho2 * s2;
  }

  double s1_sq = s1 * s1;
  double s2_sq = s2 * s2;

  // --- Albrecher et al. (2006) Stable Formulation ---
  // TO BE WRITTEN FROM HERE!!!!!!!
  Complex b_rhosv1i = b - rho * s * v * 1i;
  Complex Delta_root =
      std::sqrt(b_rhosv1i * b_rhosv1i + s2 * v * (v + 2.0 * lambda * 1i));

  // Stable 'g' (Minus over Plus)
  Complex g = (b_rhosv1i - Delta_root) / (b_rhosv1i + Delta_root);

  // Pre-calculate Exponential
  Complex exp_Dt = std::exp(-Delta_root * t);

  // Calculate term_t
  Complex term_t = (b_rhosv1i - Delta_root) / s2;

  Complex phi = (p_.r * v * 1i + a * term_t) * t -
                2.0 * a / s2 * std::log((1.0 - g * exp_Dt) / (1.0 - g));

  // Calculate psi (Variance component)
  Complex psi = p_.v0 * term_t * ((1.0 - exp_Dt) / (1.0 - g * exp_Dt));

  return std::exp(phi + psi);
}

// =========================================================
// 2. MONTE CARLO IMPLEMENTATION
// =========================================================

DoubleHestonModel::DiscrParams
DoubleHestonModel::get_discretization_params(const double dt) const {
  DiscrParams C;

  // COMPLETELY CHECK AND REWRITE THE HESTON PART ACCORDINGLY

  double a1 = p_.kappa1 * p_.theta1;
  double b1 = p_.kappa1;
  double sigma1 = p_.sigma1;
  double r = p_.r - p_.q;
  double rho1 = p_.rho1;

  double a2 = p_.kappa2 * p_.theta2;
  double b2 = p_.kappa2;
  double sigma2 = p_.sigma2;
  double rho2 = p_.rho2;

  // CIR PART (Exact simulation / Alfonsi)
  if (sigma1 * sigma1 > 4.0 * a1) {
    double ex = std::exp(-b1 * dt);
    double psi = (b1 == 0.0) ? dt : (1.0 - ex) / b1;
    C.v10 = 2.0 / (sigma1 * sigma1 * psi) * ex;
    C.v11 = 2.0 * a1 / (sigma1 * sigma1);
    C.v12 = 2.0 / (sigma1 * sigma1 * psi);
  } else {
    // Low volatility fix
    double ex = std::exp(-b1 * dt / 2.0);
    double psi = (b1 == 0.0) ? dt / 2.0 : (1.0 - ex) / b1;
    double D = a1 - sigma1 * sigma1 / 4.0;
    double s_star = sigma1 * std::sqrt(dt) / 2.0;

    C.v10 = ex;
    C.v11 = D * psi;
    C.v12 = s_star;
  }

  if (sigma2 * sigma2 > 4.0 * a2) {
    double ex = std::exp(-b2 * dt);
    double psi = (b2 == 0.0) ? dt : (1.0 - ex) / b2;
    C.v20 = 2.0 / (sigma2 * sigma2 * psi) * ex;
    C.v21 = 2.0 * a2 / (sigma2 * sigma2);
    C.v22 = 2.0 / (sigma2 * sigma2 * psi);
  } else {
    // Low volatility fix
    double ex = std::exp(-b2 * dt / 2.0);
    double psi = (b2 == 0.0) ? dt / 2.0 : (1.0 - ex) / b2;
    double D = a2 - sigma2 * sigma2 / 4.0;
    double s_star = sigma2 * std::sqrt(dt) / 2.0;

    C.v20 = ex;
    C.v21 = D * psi;
    C.v22 = s_star;
  }

  // HESTON PART
  C.ls0 = (r - rho1 * a1 / sigma1) * dt;
  C.ls1 = (b1 * rho1 / sigma1 - 0.5) * dt / 2.0 - rho1 / sigma1;
  C.ls2 = (b1 * rho1 / sigma1 - 0.5) * dt / 2.0 + rho1 / sigma1;
  C.ls3 = std::sqrt(dt * (1.0 - rho1 * rho1) / 2.0);

  // INTEGRATION PART
  C.h_2 = dt / 2.0;

  return C;
}