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
// Symmetric API with MonteCarloPricer:
//   FourierPricer<Model, Instrument, Mode> pricer(model, cfg);
//   auto res = pricer.calculate(S0, r, q, instr);
//   res.price, res.delta, res.vega[0], res.vega[1], ...
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

  // =========================================================================
  // CALCULATE -- scalar or vectorized, dispatched at compile time
  // =========================================================================
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
        results.vega[0].reserve(instr.size());
        results.vega[1].reserve(instr.size());
        results.theta.reserve(instr.size());
        results.rho.reserve(instr.size());
        results.vanna[0].reserve(instr.size());
        results.vanna[1].reserve(instr.size());
        results.charm.reserve(instr.size());
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
          results.vega[0].push_back(sr.vega[0]);
          results.vega[1].push_back(sr.vega[1]);
          results.theta.push_back(sr.theta);
          results.rho.push_back(sr.rho);
          results.vanna[0].push_back(sr.vanna[0]);
          results.vanna[1].push_back(sr.vanna[1]);
          results.charm.push_back(sr.charm);
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

  // =========================================================================
  // PRICE_SCALAR -- all Gil-Pelaez math for a single-strike instrument
  // =========================================================================
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
    constexpr int nf = Model::num_variance_factors;

    // -------------------------------------------------------------------------
    // VANILLA
    // -------------------------------------------------------------------------
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

      if constexpr (Mode == GreekMode::Full) {
        // --- Vega factor 0 ---
        GilPelaezKernel<KernelTarget::Vega, Model, Traits, 0> kv0_P1(
            model, T, r, q, K_norm, false);
        GilPelaezKernel<KernelTarget::Vega, Model, Traits, 0> kv0_P2(
            model, T, r, q, K_norm, true);
        double dP1_dv0 = engine.calculate_integral(kv0_P1, T) * INV_PI;
        double dP2_dv0 = engine.calculate_integral(kv0_P2, T) * INV_PI;
        result.vega[0] = (S0 * df_q * dP1_dv0 - K * df_r * dP2_dv0) *
                         Traits::template vega_chain_factor<0>(model);

        // --- Vega factor 1 ---
        if constexpr (nf >= 2) {
          GilPelaezKernel<KernelTarget::Vega, Model, Traits, 1> kv1_P1(
              model, T, r, q, K_norm, false);
          GilPelaezKernel<KernelTarget::Vega, Model, Traits, 1> kv1_P2(
              model, T, r, q, K_norm, true);
          double dP1_dv1 = engine.calculate_integral(kv1_P1, T) * INV_PI;
          double dP2_dv1 = engine.calculate_integral(kv1_P2, T) * INV_PI;
          result.vega[1] = (S0 * df_q * dP1_dv1 - K * df_r * dP2_dv1) *
                           Traits::template vega_chain_factor<1>(model);
        }

        // --- Theta: d(price)/dT ---
        GilPelaezKernel<KernelTarget::Theta, Model, Traits> kt_P1(
            model, T, r, q, K_norm, false);
        GilPelaezKernel<KernelTarget::Theta, Model, Traits> kt_P2(
            model, T, r, q, K_norm, true);
        double dP1_dT = engine.calculate_integral(kt_P1, T) * INV_PI;
        double dP2_dT = engine.calculate_integral(kt_P2, T) * INV_PI;
        if (OT == OptionType::Call) {
          result.theta =
              S0 * df_q * (-q * P1 + dP1_dT) + K * df_r * (r * P2 - dP2_dT);
        } else {
          result.theta = K * df_r * (r * (1.0 - P2) - dP2_dT) +
                         S0 * df_q * (q * (1.0 - P1) + dP1_dT);
        }

        // --- Rho: d(price)/dr ---
        GilPelaezKernel<KernelTarget::Rho, Model, Traits> kr_P1(model, T, r, q,
                                                                K_norm, false);
        GilPelaezKernel<KernelTarget::Rho, Model, Traits> kr_P2(model, T, r, q,
                                                                K_norm, true);
        double dP1_dr = engine.calculate_integral(kr_P1, T) * INV_PI;
        double dP2_dr = engine.calculate_integral(kr_P2, T) * INV_PI;
        double disc_term = (OT == OptionType::Call)
                               ? (T * K * df_r * P2)
                               : (-T * K * df_r * (1.0 - P2));
        result.rho = S0 * df_q * dP1_dr + disc_term - K * df_r * dP2_dr;

        // --- Vanna: ∂Delta/∂σᵢ ---
        // Delta_call = df_q * P1  =>  vanna_call = df_q * ∂P1/∂v0
        // dP1_dv0 is already computed in the Vega block above.
        result.vanna[0] =
            df_q * dP1_dv0 * Traits::template vega_chain_factor<0>(model);
        if constexpr (nf >= 2) {
          // Re-evaluate dP1/dv1 (was scoped inside the vega nf>=2 block)
          GilPelaezKernel<KernelTarget::Vega, Model, Traits, 1> kv1_vanna(
              model, T, r, q, K_norm, false);
          double dP1_dv1_local =
              engine.calculate_integral(kv1_vanna, T) * INV_PI;
          result.vanna[1] = df_q * dP1_dv1_local *
                            Traits::template vega_chain_factor<1>(model);
        }

        // --- Charm: ∂Delta/∂T ---
        // Delta_call = df_q * P1  =>  ∂/∂T[df_q*P1] = df_q*(-q*P1 + dP1/dT)
        // dP1_dT is already computed in the Theta block above.
        if (OT == OptionType::Call) {
          result.charm = df_q * (-q * P1 + dP1_dT);
        } else {
          // Delta_put = df_q * (P1 - 1)  =>  charm_put = df_q*(-q*(P1-1) +
          // dP1/dT)
          result.charm = df_q * (-q * (P1 - 1.0) + dP1_dT);
        }
      }

      // -------------------------------------------------------------------------
      // CASH-OR-NOTHING
      // -------------------------------------------------------------------------
    } else if constexpr (is_cash_or_nothing<PayoffT>::value) {
      GilPelaezKernel<KernelTarget::Price, Model, Traits> k2(model, T, r, q,
                                                             K_norm, true);
      double P2 = 0.5 + engine.calculate_integral(k2, T) * INV_PI;
      double base_prob = (OT == OptionType::Call) ? P2 : (1.0 - P2);
      result.price = df_r * base_prob;

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

      if constexpr (Mode == GreekMode::Full) {
        // --- Vega factor 0 ---
        GilPelaezKernel<KernelTarget::Vega, Model, Traits, 0> kv0(
            model, T, r, q, K_norm, true);
        double dP2_dv0 = engine.calculate_integral(kv0, T) * INV_PI;
        result.vega[0] = sign * df_r * dP2_dv0 *
                         Traits::template vega_chain_factor<0>(model);

        // --- Vega factor 1 ---
        if constexpr (nf >= 2) {
          GilPelaezKernel<KernelTarget::Vega, Model, Traits, 1> kv1(
              model, T, r, q, K_norm, true);
          double dP2_dv1 = engine.calculate_integral(kv1, T) * INV_PI;
          result.vega[1] = sign * df_r * dP2_dv1 *
                           Traits::template vega_chain_factor<1>(model);
        }

        // --- Theta ---
        GilPelaezKernel<KernelTarget::Theta, Model, Traits> kt(model, T, r, q,
                                                               K_norm, true);
        double dP2_dT = engine.calculate_integral(kt, T) * INV_PI;
        result.theta = df_r * (sign * dP2_dT - r * base_prob);

        // --- Rho ---
        GilPelaezKernel<KernelTarget::Rho, Model, Traits> kr(model, T, r, q,
                                                             K_norm, true);
        double dP2_dr = engine.calculate_integral(kr, T) * INV_PI;
        result.rho = df_r * (sign * dP2_dr - T * base_prob);

        // --- Vanna: dDelta/dsigma_i ---
        // Delta = sign * df_r * I_x / S0, d/dv0 = sign * df_r * VDx0 / S0
        GilPelaezKernel<KernelTarget::VegaDx, Model, Traits, 0> kvdx0(
            model, T, r, q, K_norm, true);
        double VDx0 = engine.calculate_integral(kvdx0, T) * INV_PI;
        result.vanna[0] = sign * df_r * VDx0 / S0 *
                          Traits::template vega_chain_factor<0>(model);
        if constexpr (nf >= 2) {
          GilPelaezKernel<KernelTarget::VegaDx, Model, Traits, 1> kvdx1(
              model, T, r, q, K_norm, true);
          double VDx1 = engine.calculate_integral(kvdx1, T) * INV_PI;
          result.vanna[1] = sign * df_r * VDx1 / S0 *
                            Traits::template vega_chain_factor<1>(model);
        }

        // --- Charm: dDelta/dT ---
        // Delta = sign * df_r * I_x / S0
        // d/dT = sign * df_r * (TDx / S0 - r * I_x / S0)
        //       + sign * (-r*df_r) * I_x / S0  ... combined:
        // = sign * df_r * (TDx/S0 - r*I_x/S0) + sign*(-r*df_r)*I_x/S0
        // Simplify: d/dT[df_r * I_x] = df_r*(TDx - r*I_x), /S0
        GilPelaezKernel<KernelTarget::ThetaDx, Model, Traits> ktdx(
            model, T, r, q, K_norm, true);
        double TDx = engine.calculate_integral(ktdx, T) * INV_PI;
        // Recompute I_x for P2 (already have it above)
        double I_x_P2 = engine.calculate_integral(
                            GilPelaezKernel<KernelTarget::Dx, Model, Traits>(
                                model, T, r, q, K_norm, true),
                            T) *
                        INV_PI;
        result.charm = sign * df_r * (TDx - r * I_x_P2) / S0;
      }

      // -------------------------------------------------------------------------
      // ASSET-OR-NOTHING
      // -------------------------------------------------------------------------
    } else if constexpr (is_asset_or_nothing<PayoffT>::value) {
      GilPelaezKernel<KernelTarget::Price, Model, Traits> k1(model, T, r, q,
                                                             K_norm, false);
      double P1 = 0.5 + engine.calculate_integral(k1, T) * INV_PI;
      double base_prob = (OT == OptionType::Call) ? P1 : (1.0 - P1);
      result.price = S0 * df_q * base_prob;

      if constexpr (Mode == GreekMode::Essential || Mode == GreekMode::Full) {
        GilPelaezKernel<KernelTarget::Dx, Model, Traits> k_dP1(model, T, r, q,
                                                               K_norm, false);
        GilPelaezKernel<KernelTarget::Dxx, Model, Traits> k_d2P1(model, T, r, q,
                                                                 K_norm, false);
        double I_x = engine.calculate_integral(k_dP1, T) * INV_PI;
        double I_xx = engine.calculate_integral(k_d2P1, T) * INV_PI;
        result.delta = df_q * (base_prob + sign * I_x);
        result.gamma = sign * df_q * (I_x + I_xx) / S0;
      }

      if constexpr (Mode == GreekMode::Full) {
        // --- Vega factor 0 ---
        GilPelaezKernel<KernelTarget::Vega, Model, Traits, 0> kv0(
            model, T, r, q, K_norm, false);
        double dP1_dv0 = engine.calculate_integral(kv0, T) * INV_PI;
        result.vega[0] = S0 * df_q * sign * dP1_dv0 *
                         Traits::template vega_chain_factor<0>(model);

        // --- Vega factor 1 ---
        if constexpr (nf >= 2) {
          GilPelaezKernel<KernelTarget::Vega, Model, Traits, 1> kv1(
              model, T, r, q, K_norm, false);
          double dP1_dv1 = engine.calculate_integral(kv1, T) * INV_PI;
          result.vega[1] = S0 * df_q * sign * dP1_dv1 *
                           Traits::template vega_chain_factor<1>(model);
        }

        // --- Theta ---
        GilPelaezKernel<KernelTarget::Theta, Model, Traits> kt(model, T, r, q,
                                                               K_norm, false);
        double dP1_dT = engine.calculate_integral(kt, T) * INV_PI;
        result.theta = S0 * df_q * (-q * base_prob + sign * dP1_dT);

        // --- Rho ---
        GilPelaezKernel<KernelTarget::Rho, Model, Traits> kr(model, T, r, q,
                                                             K_norm, false);
        double dP1_dr = engine.calculate_integral(kr, T) * INV_PI;
        result.rho = S0 * df_q * sign * dP1_dr;

        // --- Vanna: dDelta/dsigma_i ---
        // Delta = df_q*(base_prob + sign*I_x)
        // where base_prob = sign*P1 (call) -> dbase_prob/dv0 = sign*dP1/dv0
        // dDelta/dv0 = df_q*(sign*dP1/dv0 + sign*VDx0/S0... )
        // More precisely, I_x = integral of Dx kernel on P1, so:
        // dI_x/dv0 = VDx0 integral. But dbase_prob/dv0 = sign * dP1/dv0.
        // So dDelta/dv0 = df_q*(sign*dP1_dv0 + sign*VDx0)
        GilPelaezKernel<KernelTarget::VegaDx, Model, Traits, 0> kvdx0(
            model, T, r, q, K_norm, false);
        double VDx0 = engine.calculate_integral(kvdx0, T) * INV_PI;
        result.vanna[0] = df_q * sign * (dP1_dv0 + VDx0) *
                          Traits::template vega_chain_factor<0>(model);
        if constexpr (nf >= 2) {
          // Re-evaluate dP1/dv1 (scoped inside vega factor 1 block above)
          GilPelaezKernel<KernelTarget::Vega, Model, Traits, 1> kv1_vanna(
              model, T, r, q, K_norm, false);
          double dP1_dv1_vanna =
              engine.calculate_integral(kv1_vanna, T) * INV_PI;
          GilPelaezKernel<KernelTarget::VegaDx, Model, Traits, 1> kvdx1(
              model, T, r, q, K_norm, false);
          double VDx1 = engine.calculate_integral(kvdx1, T) * INV_PI;
          result.vanna[1] = df_q * sign * (dP1_dv1_vanna + VDx1) *
                            Traits::template vega_chain_factor<1>(model);
        }

        // --- Charm: dDelta/dT ---
        // Delta = df_q * (base_prob + sign*I_x)
        // d/dT: df_q*(-q*(base_prob + sign*I_x) + sign*dP1/dT + sign*TDx)
        GilPelaezKernel<KernelTarget::ThetaDx, Model, Traits> ktdx(
            model, T, r, q, K_norm, false);
        double TDx = engine.calculate_integral(ktdx, T) * INV_PI;
        double I_x_P1 = engine.calculate_integral(
                            GilPelaezKernel<KernelTarget::Dx, Model, Traits>(
                                model, T, r, q, K_norm, false),
                            T) *
                        INV_PI;
        result.charm = df_q * (-q * (base_prob + sign * I_x_P1) +
                               sign * dP1_dT + sign * TDx);
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