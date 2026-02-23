#pragma once
#include <cassert>
#include <cmath>
#include <vector>

namespace myql::utils {

// =============================================================================
// VECTOR ARITHMETIC OPERATORS (Element-wise)
// =============================================================================

// Vector += Vector (Accumulation)
template <typename T>
inline std::vector<T> &operator+=(std::vector<T> &lhs,
                                  const std::vector<T> &rhs) {
  assert(lhs.size() == rhs.size() &&
         "VectorOps: Size mismatch in += accumulation");
  for (size_t i = 0; i < lhs.size(); ++i) {
    lhs[i] += rhs[i];
  }
  return lhs;
}

// Vector + Vector (Element-wise Addition)
// Used in: (p_up - p_base*2.0 + p_dn) for gamma finite differences
template <typename T>
inline std::vector<T> operator+(const std::vector<T> &lhs,
                                const std::vector<T> &rhs) {
  assert(lhs.size() == rhs.size() && "VectorOps: Size mismatch in + addition");
  std::vector<T> result = lhs;
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] += rhs[i];
  }
  return result;
}

// Vector * Vector (Element-wise Multiplication for Variance)
// Used in: sum_sq += payoff * payoff
template <typename T>
inline std::vector<T> operator*(const std::vector<T> &lhs,
                                const std::vector<T> &rhs) {
  assert(lhs.size() == rhs.size() &&
         "VectorOps: Size mismatch in * multiplication");
  std::vector<T> result = lhs;
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] *= rhs[i];
  }
  return result;
}

// 3. Vector - Vector (Element-wise Subtraction)
// Used in: variance = sum_sq - mean^2
template <typename T>
inline std::vector<T> operator-(const std::vector<T> &lhs,
                                const std::vector<T> &rhs) {
  assert(lhs.size() == rhs.size() &&
         "VectorOps: Size mismatch in - subtraction");
  std::vector<T> result = lhs;
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] -= rhs[i];
  }
  return result;
}

// 4. Vector / Scalar (Averaging)
// Used in: mean = total_sum / N
template <typename T>
inline std::vector<T> operator/(const std::vector<T> &lhs, double scalar) {
  std::vector<T> result = lhs;
  double inv_scalar = 1.0 / scalar; // Optimization: Multiplication is faster
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] *= inv_scalar;
  }
  return result;
}

// 5. Vector * Scalar (Scaling)
// Used in: result = vector * discount_factor
template <typename T>
inline std::vector<T> operator*(const std::vector<T> &lhs, double scalar) {
  std::vector<T> result = lhs;
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] *= scalar;
  }
  return result;
}

// Order commutativity: Scalar * Vector
template <typename T>
inline std::vector<T> operator*(double scalar, const std::vector<T> &rhs) {
  return rhs * scalar;
}

// =============================================================================
// HELPER FUNCTIONS (For Standard Error)
// =============================================================================

// Compute Sqrt of every element
template <typename T>
inline std::vector<T> element_wise_sqrt(const std::vector<T> &vec) {
  std::vector<T> result = vec;
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = std::sqrt(result[i]);
  }
  return result;
}

} // namespace myql::utils