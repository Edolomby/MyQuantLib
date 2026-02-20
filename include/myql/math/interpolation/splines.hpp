#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

namespace myql {
namespace math {
namespace interpolation {

// Fritsch-Carlson Monotone Cubic Hermite Spline
// Implements the algorithm described in Haastrecht & Pelsser (2008), Appendix
// A.1. Optimized for EQUIDISTANT grids to allow O(1) lookups.
class MonotoneCubicSpline {
public:
  MonotoneCubicSpline() = default;

  /**
   * @param x_min The starting value of the x-grid (usually 0.0 for inverse
   * CDFs)
   * @param x_max The ending value of the x-grid (usually 1.0 - epsilon)
   * @param y     The y-values (e.g., the inverse CDF values) corresponding to
   * the equidistant x-grid
   */
  MonotoneCubicSpline(double x_min, double x_max, const std::vector<double> &y)
      : x_min_(x_min), x_max_(x_max), y_(y), n_(y.size()) {

    if (n_ < 2)
      throw std::invalid_argument("Spline requires at least 2 points.");

    dx_ = (x_max_ - x_min_) / (n_ - 1);
    inv_dx_ = 1.0 / dx_; // Precompute inverse for speed

    calculate_slopes();
  }

  /**
   * @brief Interpolate at point x
   * Corresponds to equation (51) in the paper.
   */
  double operator()(double x) const {
    // 1. O(1) Lookup Strategy
    // Clamp x to grid bounds to handle floating point noise
    if (x <= x_min_)
      return y_[0];
    if (x >= x_max_)
      return y_[n_ - 1];

    // Direct index calculation
    double pos = (x - x_min_) * inv_dx_;
    size_t i = static_cast<size_t>(pos);

    // Safety check for right boundary case
    if (i >= n_ - 1)
      i = n_ - 2;

    // 2. Relative position t in [0, 1]
    double t = pos - static_cast<double>(i);

    double t2 = t * t;
    double t3 = t2 * t;

    double h00 = 2 * t3 - 3 * t2 + 1.0;
    double h10 = t3 - 2 * t2 + t;
    double h01 = -2 * t3 + 3 * t2;
    double h11 = t3 - t2;

    // Delta_i = (y_{i+1} - y_i) / dx [cite: 818]
    // We factor out dx to match the paper's formula structure
    // J(U) = h00*y_i + h01*y_{i+1} + Delta_i * (m_i*h10 + m_{i+1}*h11) * dx?
    // Wait, standard Hermite formula uses derivatives m_i directly.
    // The paper defines m_i as the derivative.
    // Eq (51) says: J(U) = h00*y_i + h01*y_{i+1} + Delta_i * (m_i*h10 +
    // m_{i+1}*h11) BUT careful: In A.1, m_i are 'weights' derived from slopes,
    // not raw derivatives. Let's stick to the standard Hermite form: y(t) =
    // h00*p0 + h10*m0*dx + h01*p1 + h11*m1*dx where m0, m1 are the slopes
    // stored in m_.

    return h00 * y_[i] + h01 * y_[i + 1] +
           (m_[i] * h10 + m_[i + 1] * h11) *
               dx_; // Scale derivative by interval length
  }

private:
  double x_min_, x_max_, dx_, inv_dx_;
  std::vector<double> y_;
  std::vector<double> m_; // The calculated slopes (derivatives)
  size_t n_;

  /**
   * @brief Fritsch-Carlson Algorithm implementation [cite: 817-828]
   */
  void calculate_slopes() {
    m_.resize(n_);
    std::vector<double> secants(n_ - 1);

    // 1. Calculate secant slopes (Delta_k)
    for (size_t k = 0; k < n_ - 1; ++k) {
      secants[k] = (y_[k + 1] - y_[k]) * inv_dx_;
    }

    // 2. Initialize tangents (m_k) as average of secants [cite: 820]
    m_[0] = secants[0];
    m_[n_ - 1] = secants[n_ - 2];

    for (size_t k = 1; k < n_ - 1; ++k) {
      m_[k] = 0.5 * (secants[k - 1] + secants[k]);
    }

    // 3. Monotonicity enforcement [cite: 823-828]
    for (size_t k = 0; k < n_ - 1; ++k) {
      double delta = secants[k];

      if (std::abs(delta) < 1e-12) {
        // If the secant is zero (flat), slopes must be zero [cite: 824]
        m_[k] = 0.0;
        m_[k + 1] = 0.0;
      } else {
        double alpha = m_[k] / delta;
        double beta = m_[k + 1] / delta;
        double sum_sq = alpha * alpha + beta * beta;

        // Constraint region check [cite: 827]
        if (sum_sq > 9.0) {
          double tau = 3.0 / std::sqrt(sum_sq);
          m_[k] = tau * alpha * delta;
          m_[k + 1] = tau * beta * delta;
        }
      }
    }
  }
};

} // namespace interpolation
} // namespace math
} // namespace myql
