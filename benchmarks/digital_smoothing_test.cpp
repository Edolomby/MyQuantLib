// =============================================================================
// DIGITAL SMOOTHING TEST
// =============================================================================
// Validates the natural-scale formula ns = sigma_eff * S0 * sqrt(T) across 5
// maturities (T = 0.08, 0.2, 0.5, 1.0, 3.0) and a strip of strikes.
//
// For each model (Heston, Bates) and each maturity we compute:
//   1. Fourier reference price, delta, gamma (central FD on S0)
//   2. MC with eps = ns/10 and ns/5 (GreekMode::Essential, smoothing
//      auto-activates via the Option-C template operator()<Mode>)
//
// NOTE: For Bates, sigma_eff incorporates jump contribution. For Double Heston
// it incorporates both factors. The Pricer auto-injects the scaling factor for
// the smoothed payoff dynamically utilizing these formulas!
// =============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <myql/instruments/options/European.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/models/asvj/policies/JumpPolicies.hpp>
#include <myql/models/asvj/policies/VolSchemes.hpp>
#include <myql/pricers/fourier/FourierPricer.hpp>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>
#include <myql/utils/TablePrinter.hpp>

using namespace myql;
// =============================================================================
// Generic run function: works for any model + stepper pair
// =============================================================================
template <typename ModelT, typename StepperT>
void run_table(const std::string &model_name, const ModelT &model, double S0,
               const std::vector<double> &K_vec, double r, double q,
               const MonteCarloConfig &mc_cfg_in,
               const FourierEngine::Config &f_cfg,
               const std::vector<double> &maturities,
               const std::vector<double> &fractions) {

  using SharpInstrT =
      EuropeanOption<PayoffCashOrNothing<OptionType::Call>, double>;

  const double fd_h = S0 * 1e-3;

  auto fourier_at = [&](double s, double T, double k) -> double {
    SharpInstrT instr(k, T);
    FourierPricer<ModelT, SharpInstrT> eng(model, f_cfg);
    return eng.calculate(s, r, q, instr).price;
  };

  // ── Sharp baseline accumulators (no smoothing, eps → 0) ───────────────────
  std::vector<std::string> sh_lbl_v;
  std::vector<double> sh_k_v;
  std::vector<double> sh_ref_price_v, sh_mc_price_v, sh_price_bias_v,
      sh_price_se_v;
  std::vector<double> sh_ref_delta_v, sh_delta_v, sh_delta_bias_v,
      sh_delta_se_v;
  std::vector<double> sh_ref_gamma_v, sh_gamma_v, sh_gamma_bias_v,
      sh_gamma_se_v;

  // ── Smoothed accumulators ─────────────────────────────────────────────────
  std::vector<std::string> lbl_v;
  std::vector<double> k_v, eps_v;
  std::vector<double> ref_price_v, mc_price_v, price_bias_v, price_se_v;
  std::vector<double> ref_delta_v, delta_v, delta_bias_v, delta_se_v;
  std::vector<double> ref_gamma_v, gamma_v, gamma_bias_v, gamma_se_v;

  MonteCarloConfig mc_cfg = mc_cfg_in;

  // ── Cached Fourier references per maturity and strike ─────────────────────
  std::vector<std::vector<double>> f0_cache(maturities.size(),
                                            std::vector<double>(K_vec.size()));
  std::vector<std::vector<double>> ref_delta_cache(
      maturities.size(), std::vector<double>(K_vec.size()));
  std::vector<std::vector<double>> ref_gamma_cache(
      maturities.size(), std::vector<double>(K_vec.size()));

  for (size_t i = 0; i < maturities.size(); ++i) {
    const double T = maturities[i];
    for (size_t k_idx = 0; k_idx < K_vec.size(); ++k_idx) {
      const double k = K_vec[k_idx];
      const double f0 = fourier_at(S0, T, k);
      const double f_u = fourier_at(S0 + fd_h, T, k);
      const double f_d = fourier_at(S0 - fd_h, T, k);
      f0_cache[i][k_idx] = f0;
      ref_delta_cache[i][k_idx] = (f_u - f_d) / (2.0 * fd_h);
      ref_gamma_cache[i][k_idx] = (f_u - 2.0 * f0 + f_d) / (fd_h * fd_h);
    }
  }

  // ── Pass 1: sharp baseline ────────────────────────────────────────────────
  {
    constexpr double SHARP_EPS = 1e-10;
    for (size_t i = 0; i < maturities.size(); ++i) {
      const double T = maturities[i];
      mc_cfg.time_steps =
          std::max(20ul, static_cast<size_t>(std::round(100.0 * T)));

      std::ostringstream lbl;
      lbl << "T=" << std::fixed << std::setprecision(2) << T;

      using PayoffT = PayoffCashOrNothing<OptionType::Call>;
      PayoffT sharp_payoff;
      sharp_payoff.eps = SHARP_EPS;
      EuropeanOption<PayoffT, std::vector<double>> instr(K_vec, T,
                                                         sharp_payoff);

      MonteCarloPricer<ModelT, StepperT,
                       EuropeanOption<PayoffT, std::vector<double>>,
                       GreekMode::Essential>
          pricer(model, mc_cfg);
      auto res = pricer.calculate(S0, r, q, instr);

      for (size_t k_idx = 0; k_idx < K_vec.size(); ++k_idx) {
        sh_lbl_v.push_back(lbl.str());
        sh_k_v.push_back(K_vec[k_idx]);
        sh_ref_price_v.push_back(f0_cache[i][k_idx]);
        sh_mc_price_v.push_back(res.price[k_idx]);
        sh_price_bias_v.push_back(res.price[k_idx] - f0_cache[i][k_idx]);
        sh_price_se_v.push_back(res.price_std_err[k_idx]);
        sh_ref_delta_v.push_back(ref_delta_cache[i][k_idx]);
        sh_delta_v.push_back(res.delta[k_idx]);
        sh_delta_bias_v.push_back(res.delta[k_idx] - ref_delta_cache[i][k_idx]);
        sh_delta_se_v.push_back(res.delta_std_err[k_idx]);
        sh_ref_gamma_v.push_back(ref_gamma_cache[i][k_idx]);
        sh_gamma_v.push_back(res.gamma[k_idx]);
        sh_gamma_bias_v.push_back(res.gamma[k_idx] - ref_gamma_cache[i][k_idx]);
        sh_gamma_se_v.push_back(res.gamma_std_err[k_idx]);
      }
    }
  }

  // ── Pass 2: smoothed (original loop) ─────────────────────────────────────
  for (size_t i = 0; i < maturities.size(); ++i) {
    const double T = maturities[i];
    mc_cfg.time_steps =
        std::max(10ul, static_cast<size_t>(std::round(100.0 * T)));

    const double sigma_eff =
        std::sqrt(detail::compute_expected_average_variance(model, T));
    // Vectorized payoffs use S0 as the optimal standard deviation proxy
    // bandwidth scaling over all strikes
    const double ns = sigma_eff * S0 * std::sqrt(T);

    for (double frac : fractions) {
      const double eps = ns / frac;

      std::ostringstream lbl;
      lbl << "T=" << std::fixed << std::setprecision(2) << T << " eps=ns/"
          << static_cast<int>(frac);

      using PayoffT = PayoffCashOrNothing<OptionType::Call>;
      PayoffT smooth_payoff;
      smooth_payoff.eps = eps;
      EuropeanOption<PayoffT, std::vector<double>> instr(K_vec, T,
                                                         smooth_payoff);

      MonteCarloPricer<ModelT, StepperT,
                       EuropeanOption<PayoffT, std::vector<double>>,
                       GreekMode::Essential>
          pricer(model, mc_cfg);
      auto res = pricer.calculate(S0, r, q, instr);

      for (size_t k_idx = 0; k_idx < K_vec.size(); ++k_idx) {
        lbl_v.push_back(lbl.str());
        k_v.push_back(K_vec[k_idx]);
        eps_v.push_back(eps);
        ref_price_v.push_back(f0_cache[i][k_idx]);
        mc_price_v.push_back(res.price[k_idx]);
        price_bias_v.push_back(res.price[k_idx] - f0_cache[i][k_idx]);
        price_se_v.push_back(res.price_std_err[k_idx]);
        ref_delta_v.push_back(ref_delta_cache[i][k_idx]);
        delta_v.push_back(res.delta[k_idx]);
        delta_bias_v.push_back(res.delta[k_idx] - ref_delta_cache[i][k_idx]);
        delta_se_v.push_back(res.delta_std_err[k_idx]);
        ref_gamma_v.push_back(ref_gamma_cache[i][k_idx]);
        gamma_v.push_back(res.gamma[k_idx]);
        gamma_bias_v.push_back(res.gamma[k_idx] - ref_gamma_cache[i][k_idx]);
        gamma_se_v.push_back(res.gamma_std_err[k_idx]);
      }
    }

    // --- NEW: Test Auto-Injection architecture ---
    {
      std::ostringstream auto_lbl;
      auto_lbl << "T=" << std::fixed << std::setprecision(2) << T << " [Auto]";

      using PayoffT = PayoffCashOrNothing<OptionType::Call>;
      EuropeanOption<PayoffT, std::vector<double>> auto_instr(K_vec, T);

      MonteCarloPricer<ModelT, StepperT,
                       EuropeanOption<PayoffT, std::vector<double>>,
                       GreekMode::Essential>
          pricer(model, mc_cfg);

      auto auto_res = pricer.calculate(S0, r, q, auto_instr);

      for (size_t k_idx = 0; k_idx < K_vec.size(); ++k_idx) {
        lbl_v.push_back(auto_lbl.str());
        k_v.push_back(K_vec[k_idx]);
        eps_v.push_back(ns / 10.0);
        ref_price_v.push_back(f0_cache[i][k_idx]);
        mc_price_v.push_back(auto_res.price[k_idx]);
        price_bias_v.push_back(auto_res.price[k_idx] - f0_cache[i][k_idx]);
        price_se_v.push_back(auto_res.price_std_err[k_idx]);
        ref_delta_v.push_back(ref_delta_cache[i][k_idx]);
        delta_v.push_back(auto_res.delta[k_idx]);
        delta_bias_v.push_back(auto_res.delta[k_idx] -
                               ref_delta_cache[i][k_idx]);
        delta_se_v.push_back(auto_res.delta_std_err[k_idx]);
        ref_gamma_v.push_back(ref_gamma_cache[i][k_idx]);
        gamma_v.push_back(auto_res.gamma[k_idx]);
        gamma_bias_v.push_back(auto_res.gamma[k_idx] -
                               ref_gamma_cache[i][k_idx]);
        gamma_se_v.push_back(auto_res.gamma_std_err[k_idx]);
      }
    }
  }

  // ── Print ──────────────────────────────────────────────────────────────────
  std::cout
      << "\n============================================================\n";
  std::cout << "  MODEL: " << model_name << "\n";
  std::cout << "  S0=" << S0 << "\n";
  std::cout
      << "============================================================\n\n";

  // ── Sharp baseline table ───────────────────────────────────────────────────
  std::cout << "  [Sharp Baseline — No Smoothing, eps=1e-10]\n"
            << "  Price is still accurate; pathwise Delta & Gamma → 0, and "
               "become noisy due to the discontinuous payoff.\n\n";
  printVectors(
      utils::TableConfig{utils::FloatFormat::Scientific, 4},
      {"Maturity", "Strike", "Ref Price", "MC Price", "Price Bias",
       "Price StdErr", "Ref Delta", "Sharp Delta", "Delta Bias", "Delta StdErr",
       "Ref Gamma", "Sharp Gamma", "Gamma Bias", "Gamma StdErr"},
      sh_lbl_v, sh_k_v, sh_ref_price_v, sh_mc_price_v, sh_price_bias_v,
      sh_price_se_v, sh_ref_delta_v, sh_delta_v, sh_delta_bias_v, sh_delta_se_v,
      sh_ref_gamma_v, sh_gamma_v, sh_gamma_bias_v, sh_gamma_se_v);

  // ── Smoothed table ─────────────────────────────────────────────────────────
  std::cout << "\n  [Smoothed — eps = ns/frac]\n\n";
  printVectors(utils::TableConfig{utils::FloatFormat::Scientific, 4},
               {"Config", "Strike", "eps", "Ref Price", "MC Price",
                "Price Bias", "Price StdErr", "Ref Delta", "Delta",
                "Delta Bias", "Delta StdErr", "Ref Gamma", "Gamma",
                "Gamma Bias", "Gamma StdErr"},
               lbl_v, k_v, eps_v, ref_price_v, mc_price_v, price_bias_v,
               price_se_v, ref_delta_v, delta_v, delta_bias_v, delta_se_v,
               ref_gamma_v, gamma_v, gamma_bias_v, gamma_se_v);
}

// =============================================================================
// MAIN
// =============================================================================
int main() {
  const double S0 = 100.0;
  const std::vector<double> K_vec = {90.0, 100.0, 110.0};
  const double r = 0.03;
  const double q = 0.00;

  // Shared Heston parameters
  HestonParams h_params = {2.5, 0.06, 0.4, -0.7, 0.06};
  // Merton jump parameters
  MertonParams m_params = {1.2, -0.12, 0.15}; // lambda, mu, delta

  MonteCarloConfig mc_cfg;
  mc_cfg.num_paths = 500'000;
  mc_cfg.seed = 42;
  mc_cfg.fd_bump = 1e-4;

  FourierEngine::Config f_cfg;
  f_cfg.tolerance = 1e-9;

  const std::vector<double> maturities = {0.08, 0.2, 0.5, 1.0, 3.0};
  const std::vector<double> fractions = {5.0, 15.0};

  std::cout << "\n";
  std::cout
      << "=================================================================\n";
  std::cout << "  DIGITAL OPTION SCALING VALIDATION (Vectorized Strip)\n";
  std::cout << "  Cash-or-Nothing Call | ITM, ATM, OTM | M=" << mc_cfg.num_paths
            << " paths\n";
  std::cout
      << "=================================================================\n";

  // ── 1. HESTON ─────────────────────────────────────────────────────────────
  {
    HestonModel model(h_params);
    using StepperT =
        ASVJStepper<SchemeNCI, NullVolScheme, NoJumps, TrackerEuropean>;
    run_table<HestonModel, StepperT>("Heston", model, S0, K_vec, r, q, mc_cfg,
                                     f_cfg, maturities, fractions);
  }

  // ── 2. BATES (Merton jumps) ────────────────────────────────────────────────
  {
    BatesModel model(h_params, m_params);
    using StepperT =
        ASVJStepper<SchemeNCI, NullVolScheme, MertonJump, TrackerEuropean>;
    run_table<BatesModel, StepperT>("Bates (Heston + Merton Jumps)", model, S0,
                                    K_vec, r, q, mc_cfg, f_cfg, maturities,
                                    fractions);
  }

  // ── 3. DOUBLE HESTON ───────────────────────────────────────────────────────
  {
    HestonParams h2_params = {1.5, 0.04, 0.3, -0.5, 0.16};
    DoubleHestonModel model(h_params, h2_params);
    using StepperT =
        ASVJStepper<SchemeNCI, SchemeNCI, NoJumps, TrackerEuropean>;
    run_table<DoubleHestonModel, StepperT>(
        "Double Heston (2 Volatility Factors)", model, S0, K_vec, r, q, mc_cfg,
        f_cfg, maturities, fractions);
  }

  // ── 4. EXTREME HESTON (Feller violated) ────────────────────────────────────
  {
    HestonParams extreme_h = {1.2, 0.08, 0.8, -0.9, 0.04};
    HestonModel model(extreme_h);
    using StepperT =
        ASVJStepper<SchemeNCI, NullVolScheme, NoJumps, TrackerEuropean>;
    run_table<HestonModel, StepperT>("Extreme Heston (Feller Violated)", model,
                                     S0, K_vec, r, q, mc_cfg, f_cfg, maturities,
                                     fractions);
  }

  // ── 5. BATES-KOU (High Jump Intensity) ─────────────────────────────────────
  {
    KouParams extreme_k = {3.0, 0.3, 10.0, 8.0}; // lambda=3.0 (many jumps)
    BatesKouModel model(h_params, extreme_k);
    using StepperT =
        ASVJStepper<SchemeNCI, NullVolScheme, KouJump, TrackerEuropean>;
    run_table<BatesKouModel, StepperT>("Bates-Kou (High Frequency Jumps)",
                                       model, S0, K_vec, r, q, mc_cfg, f_cfg,
                                       maturities, fractions);
  }

  return 0;
}
