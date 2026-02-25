#pragma once
#include <vector>

#include <myql/dispatcher/InstrumentRegistry.hpp>
#include <myql/dispatcher/ModelRegistry.hpp>
#include <myql/dispatcher/StepperTraits.hpp>

#include <myql/pricers/fourier/FourierPricer.hpp>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>

namespace myql::dispatcher {

// =============================================================================
// RUNTIME RESULT DTO
// =============================================================================
// A unified output structure for both MC and Fourier that encapsulates
// vectorized results safely for the boundary layer.
struct DispatchResult {
  std::vector<double> prices;
  // Essential Greeks
  std::vector<double> deltas; // Empty if GreekMode == None
  std::vector<double> gammas; // Empty if GreekMode == None
  // Full Greeks (planned for future release / MC extension)
  std::vector<double> vegas;
  std::vector<double> thetas;
  std::vector<double> rhos;

  // MC Error Metrics
  std::vector<double> prices_std_err; // MC only, empty for Fourier
  std::vector<double> deltas_std_err; // MC only
  std::vector<double> gammas_std_err; // MC only
  std::vector<double> vegas_std_err;  // MC only
  std::vector<double> thetas_std_err; // MC only
  std::vector<double> rhos_std_err;   // MC only

  double time_ms = 0.0;
};

// Internal Helper to convert MonteCarloResult/FourierResult to DispatchResult
template <typename EngineResult>
DispatchResult format_dispatch_result(const EngineResult &res) {
  DispatchResult out;

  // ResultType can be double (scalar exotic) or vector<double> (European/Asian
  // strip). DispatchResult always stores vector<double>, so wrap scalars in {}.
  auto to_vec = [](const auto &v) -> std::vector<double> {
    if constexpr (std::is_same_v<std::decay_t<decltype(v)>, double>)
      return {v};
    else
      return v;
  };

  out.prices = to_vec(res.price);

  if constexpr (requires { res.delta; }) {
    out.deltas = to_vec(res.delta);
    out.gammas = to_vec(res.gamma);

    if constexpr (requires { res.vega; }) {
      out.vegas = to_vec(res.vega);
      out.thetas = to_vec(res.theta);
      out.rhos = to_vec(res.rho);
    }
  }

  if constexpr (requires { res.price_std_err; }) {
    out.prices_std_err = to_vec(res.price_std_err);
    if constexpr (requires { res.delta_std_err; }) {
      out.deltas_std_err = to_vec(res.delta_std_err);
      out.gammas_std_err = to_vec(res.gamma_std_err);

      if constexpr (requires { res.vega_std_err; }) {
        out.vegas_std_err = to_vec(res.vega_std_err);
        out.thetas_std_err = to_vec(res.theta_std_err);
        out.rhos_std_err = to_vec(res.rho_std_err);
      }
    }
  }
  return out;
}

// =============================================================================
// DISPATCH ENGINE: MONTE CARLO
// =============================================================================
template <GreekMode Mode = GreekMode::None,
          typename ExplicitVolScheme = SchemeNCI>
DispatchResult price_mc(const AnyModel &m_var, const AnyMCInstrument &i_var,
                        const MonteCarloConfig &cfg, double S0, double r,
                        double q) {

  // Nested std::visit! Double runtime dispatch resolving to a single compiled
  // type.
  return std::visit(
      [&](auto &m) -> DispatchResult {
        return std::visit(
            [&](auto &i) -> DispatchResult {
              using ConcreteModel = std::decay_t<decltype(m)>;
              using ConcreteInstrument = std::decay_t<decltype(i)>;

              // Deduce the exact Stepper needed via the trait
              using Stepper =
                  typename StepperFor<ConcreteModel, ConcreteInstrument,
                                      ExplicitVolScheme>::type;

              // Instantiate the zero-overhead generic pricer
              MonteCarloPricer<ConcreteModel, Stepper, ConcreteInstrument, Mode>
                  pricer(m, cfg);

              // Price it
              auto result = pricer.calculate(S0, r, q, i);

              // Return unified bundle
              return format_dispatch_result(result);
            },
            i_var);
      },
      m_var);
}

// =============================================================================
// DISPATCH ENGINE: FOURIER
// =============================================================================
template <GreekMode Mode = GreekMode::None>
DispatchResult
price_fourier(const AnyModel &m_var, const AnyEuropeanInstrument &i_var,
              const FourierEngine::Config &cfg, double S0, double r, double q) {

  return std::visit(
      [&](auto &m) -> DispatchResult {
        return std::visit(
            [&](auto &i) -> DispatchResult {
              using ConcreteModel = std::decay_t<decltype(m)>;
              using ConcreteInstrument = std::decay_t<decltype(i)>;

              // Instantiate the Fourier Pricer directly
              FourierPricer<ConcreteModel, ConcreteInstrument, Mode> pricer(
                  m, cfg);

              // Price it
              auto result = pricer.calculate(S0, r, q, i);

              // Return unified bundle
              return format_dispatch_result(result);
            },
            i_var);
      },
      m_var);
}

} // namespace myql::dispatcher
