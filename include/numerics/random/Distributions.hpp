#pragma once
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <cmath>

namespace numerics {
namespace random {

// Fast Normal Inverse (Acklam's algorithm)
// -----------------------------------------------------------------------------
// Inverse Normal Cumulative Distribution Function (AS241 / Acklam)
// Precision: ~1e-16 (Double Precision)
// -----------------------------------------------------------------------------
inline double normcdfinv_as241(double p) {
  // Coefficients for the rational approximation
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                             -2.759285104469687e+02, 1.383577518672690e+02,
                             -3.066479806614716e+01, 2.506628277459239e+00};

  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                             -1.556989798598866e+02, 6.680131188771972e+01,
                             -1.328068155288572e+01};

  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                             -2.400758277161838e+00, -2.549732539343734e+00,
                             4.374664141464968e+00,  2.938163982698783e+00};

  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                             2.445134137142996e+00, 3.754408661907416e+00};

  // Define break-points
  const double p_low = 0.02425;
  const double p_high = 1.0 - p_low;

  if (p <= 0.0)
    return -std::numeric_limits<double>::infinity();
  if (p >= 1.0)
    return std::numeric_limits<double>::infinity();

  // Rational approximation for central region
  if (p > p_low && p < p_high) {
    double q = p - 0.5;
    double r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r +
            a[5]) *
           q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
  }

  // Rational approximation for tails
  double q = (p <= p_low) ? p : 1.0 - p;
  double r = std::sqrt(-2.0 * std::log(q));

  double x =
      (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) /
      ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0);

  return (p <= p_low) ? x : -x;
}

} // namespace random
} // namespace numerics