#pragma once
#include <cmath>
#include <limits>
#include <type_traits> // Required for std::invoke_result_t, std::is_convertible_v

namespace numerics {

// -------------------------------------------------------------------------
// Implementation Detail (Hidden Logic)
// -------------------------------------------------------------------------
namespace detail {

// We use a template for 'Func' so the compiler can INLINE the call.
// Avoid using std::function<double(double)>, we would pay a virtual call
// overhead at every single step (millions of times), killing performance.
template <typename Func>
inline double adaptive_simpson_impl(const Func &f, double a, double b,
                                    double eps, double S, double fa, double fb,
                                    double fc, int depth) {
  double h = b - a;
  double c = 0.5 * (a + b);
  double d = 0.5 * (a + c);
  double e = 0.5 * (c + b);

  // Evaluate function at new midpoints
  double fd = f(d);
  double fe = f(e);

  // Simpson's 1/3 rule for each half
  // h/12 comes from (h/2) / 6 where h/2 is the sub-interval width
  double h_12 = h * (1.0 / 12.0);
  double Sleft = h_12 * (fa + 4.0 * fd + fc);
  double Sright = h_12 * (fc + 4.0 * fe + fb);
  double S2 = Sleft + Sright;

  // Safety Check: Propagate NaNs instantly
  // If the math blows up (e.g. log(-1)), stop calculating to save cycles.
  // commented because of enabled option clang(-Wnan-infinity-disabled)
  // if (std::isnan(S2) || std::isinf(S2))
  //   return std::numeric_limits<double>::quiet_NaN();

  // Convergence Check
  // Richardson Extrapolation: Error is approx (S2 - S) / 15
  if (depth <= 0 || std::abs(S2 - S) <= 15.0 * eps) {
    // Return the refined value + Richardson correction
    return S2 + (S2 - S) * (1.0 / 15.0);
  }

  // Recurse
  // Split error budget between children (eps / 2)
  return adaptive_simpson_impl(f, a, c, 0.5 * eps, Sleft, fa, fc, fd,
                               depth - 1) +
         adaptive_simpson_impl(f, c, b, 0.5 * eps, Sright, fc, fb, fe,
                               depth - 1);
}
} // namespace detail

// -------------------------------------------------------------------------
// User-Facing API
// -------------------------------------------------------------------------

/**
 * @brief Adaptive Simpson's Integration Rule (Optimized for HPC)
 * * @tparam Func A callable type (lambda, functor, or function pointer).
 * MUST return double and accept double.
 * @param f The function to integrate.
 * @param a Lower bound.
 * @param b Upper bound.
 * @param eps Desired absolute error tolerance.
 * @param max_depth Maximum recursion depth (stack safety). Default 25.
 * @return double The approximate integral.
 */
template <typename Func>
inline double adaptive_simpson(const Func &f, double a, double b, double eps,
                               unsigned int max_depth = 25) {

  // ---------------------------------------------------------------------
  // TYPE SAFETY CHECK (Zero Runtime Cost)
  // ---------------------------------------------------------------------
  // "Check if calling 'f' with a 'double' returns something convertible to
  // 'double'"
  static_assert(
      std::is_convertible_v<std::invoke_result_t<Func, double>, double>,
      "Error: The integrand function must return a double (or convertible "
      "type).");

  double c = (a + b) * 0.5;
  double h = b - a;

  // Initial evaluations
  double fa = f(a);
  double fb = f(b);
  double fc = f(c);

  // Initial coarse estimate
  double S = (1.0 / 6.0) * h * (fa + 4.0 * fc + fb);

  return detail::adaptive_simpson_impl(f, a, b, eps, S, fa, fb, fc, max_depth);
}

} // namespace numerics
