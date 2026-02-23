#pragma once
#include <algorithm>
#include <cmath>
#include <type_traits>

#include <myql/core/PricingTypes.hpp>
#include <myql/engines/fourier/FourierEngine.hpp>
#include <myql/engines/fourier/kernels/GilPelaezKernel.hpp>
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/European.hpp>

// =============================================================================
// PAYOFF CATEGORY DETECTION TRAITS
// =============================================================================
template <typename T> struct is_vanilla : std::false_type {};
template <OptionType OT>
struct is_vanilla<PayoffVanilla<OT>> : std::true_type {};

template <typename T> struct is_cash_or_nothing : std::false_type {};
template <OptionType OT>
struct is_cash_or_nothing<PayoffCashOrNothing<OT>> : std::true_type {};

template <typename T> struct is_asset_or_nothing : std::false_type {};
template <OptionType OT>
struct is_asset_or_nothing<PayoffAssetOrNothing<OT>> : std::true_type {};

// =============================================================================
// FOURIER PRICER
// Implemented with Symmetric API with MonteCarloPricer:
//   FourierPricer<Model, Instrument, Mode> pricer(model, cfg);
//   auto res = pricer.calculate(S0, r, q, instr);
//   res.price, res.delta, ...
// =============================================================================
template <typename Model, typename Instrument, GreekMode Mode = GreekMode::None,
          typename Traits = AffineTraits<Model>>
class FourierPricer {
public:
  using ResultType =
      typename Instrument::ResultType; // double or vector<double>
  using ReturnStruct = FourierResult<Mode, ResultType>;

  FourierPricer(const Model &model,
                const FourierEngine::Config &cfg = FourierEngine::Config{})
      : model_(model), cfg_(cfg) {}

  // -------------------------------------------------------------------------
  // CALCULATE — scalar or vectorized, dispatched at compile time
  // -------------------------------------------------------------------------
  ReturnStruct calculate(double S0, double r, double q,
                         const Instrument &instr) const {
    constexpr bool is_vectorized = !std::is_floating_point_v<ResultType>;

    if constexpr (is_vectorized) {
      ReturnStruct results;
      results.price.reserve(instr.size());
      if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
        results.delta.reserve(instr.size());
        results.gamma.reserve(instr.size());
      }
      if constexpr (Mode == GreekMode::Full) {
        results.vega.reserve(instr.size());
        results.theta.reserve(instr.size());
        results.rho.reserve(instr.size());
      }

      for (double K : instr.get_strikes()) {
        using ScalarT = EuropeanOption<typename Instrument::PayoffType>;
        ScalarT scalar_opt(K, instr.get_maturity());
        auto sr = price_scalar(model_, S0, r, q, scalar_opt, cfg_);

        results.price.push_back(sr.price);
        if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
          results.delta.push_back(sr.delta);
          results.gamma.push_back(sr.gamma);
        }
        if constexpr (Mode == GreekMode::Full) {
          results.vega.push_back(sr.vega);
          results.theta.push_back(sr.theta);
          results.rho.push_back(sr.rho);
        }
      }
      return results;

    } else {
      return price_scalar(model_, S0, r, q, instr, cfg_);
    }
  }

private:
  const Model &model_;
  FourierEngine::Config cfg_;

  // -------------------------------------------------------------------------
  // PRICE_SCALAR — all Gil-Pelaez math for a single-strike instrument
  // -------------------------------------------------------------------------
  template <typename ScalarInstrument>
  static FourierResult<Mode, double>
  price_scalar(const Model &model, double S0, double r, double q,
               const ScalarInstrument &instr,
               const FourierEngine::Config &engine_cfg) {
    double K = instr.get_strikes();
    double T = instr.get_maturity();
    double K_norm = K / S0;
    double df_r = std::exp(-r * T);
    double df_q = std::exp(-q * T);
    constexpr double INV_PI = 0.31830988618379067154;

    FourierEngine engine(engine_cfg);
    FourierResult<Mode, double> result;

    using PayoffT = typename ScalarInstrument::PayoffType;
    constexpr OptionType OT = PayoffT::Type;
    double sign = (OT == OptionType::Call) ? 1.0 : -1.0;

    // --- VANILLA ---
    if constexpr (is_vanilla<PayoffT>::value) {
      GilPelaezKernel<KernelTarget::Price, Model, Traits> k1(model, T, r, q,
                                                             K_norm, false);
      GilPelaezKernel<KernelTarget::Price, Model, Traits> k2(model, T, r, q,
                                                             K_norm, true);
      double P1 = 0.5 + engine.calculate_integral(k1, T) * INV_PI;
      double P2 = 0.5 + engine.calculate_integral(k2, T) * INV_PI;

      result.price =
          (OT == OptionType::Call)
              ? std::max(0.0, S0 * df_q * P1 - K * df_r * P2)
              : std::max(0.0, K * df_r * (1.0 - P2) - S0 * df_q * (1.0 - P1));

      if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
        result.delta =
            (OT == OptionType::Call) ? (df_q * P1) : (df_q * (P1 - 1.0));
        GilPelaezKernel<KernelTarget::Dx, Model, Traits> k_dP1(model, T, r, q,
                                                               K_norm, false);
        double I_x = engine.calculate_integral(k_dP1, T) * INV_PI;
        result.gamma = (df_q * I_x) / S0;
      }

      // --- CASH-OR-NOTHING ---
    } else if constexpr (is_cash_or_nothing<PayoffT>::value) {
      GilPelaezKernel<KernelTarget::Price, Model, Traits> k2(model, T, r, q,
                                                             K_norm, true);
      double P2 = 0.5 + engine.calculate_integral(k2, T) * INV_PI;
      result.price = df_r * ((OT == OptionType::Call) ? P2 : (1.0 - P2));

      if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
        GilPelaezKernel<KernelTarget::Dx, Model, Traits> k_dP2(model, T, r, q,
                                                               K_norm, true);
        GilPelaezKernel<KernelTarget::Dxx, Model, Traits> k_d2P2(model, T, r, q,
                                                                 K_norm, true);
        double I_x = engine.calculate_integral(k_dP2, T) * INV_PI;
        double I_xx = engine.calculate_integral(k_d2P2, T) * INV_PI;
        result.delta = sign * df_r * I_x / S0;
        result.gamma = sign * df_r * (I_xx - I_x) / (S0 * S0);
      }

      // --- ASSET-OR-NOTHING ---
    } else if constexpr (is_asset_or_nothing<PayoffT>::value) {
      GilPelaezKernel<KernelTarget::Price, Model, Traits> k1(model, T, r, q,
                                                             K_norm, false);
      double P1 = 0.5 + engine.calculate_integral(k1, T) * INV_PI;
      result.price = S0 * df_q * ((OT == OptionType::Call) ? P1 : (1.0 - P1));

      if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
        GilPelaezKernel<KernelTarget::Dx, Model, Traits> k_dP1(model, T, r, q,
                                                               K_norm, false);
        GilPelaezKernel<KernelTarget::Dxx, Model, Traits> k_d2P1(model, T, r, q,
                                                                 K_norm, false);
        double I_x = engine.calculate_integral(k_dP1, T) * INV_PI;
        double I_xx = engine.calculate_integral(k_d2P1, T) * INV_PI;
        double base_delta = (OT == OptionType::Call) ? P1 : (1.0 - P1);
        result.delta = df_q * (base_delta + sign * I_x);
        result.gamma = sign * df_q * (I_x + I_xx) / S0;
      }

    } else {
      static_assert(is_vanilla<PayoffT>::value ||
                        is_cash_or_nothing<PayoffT>::value ||
                        is_asset_or_nothing<PayoffT>::value,
                    "Unsupported Payoff Type for FourierPricer");
    }

    return result;
  }
};