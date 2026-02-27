#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// PRICERS
#include <myql/pricers/fourier/FourierPricer.hpp>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>

// MODELS & STEPPERS
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/models/asvj/policies/JumpPolicies.hpp>
#include <myql/models/asvj/policies/VolSchemes.hpp>

// INSTRUMENTS
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/European.hpp>

// UTILS
#include <myql/utils/TablePrinter.hpp>

// =============================================================================
// STORAGE (shared structure, one instance per payoff family)
// =============================================================================
struct GreekStorage {
  std::vector<std::string> model;
  std::vector<std::string> payoff;
  std::vector<std::string> greek;
  std::vector<double> T;
  std::vector<double> K;
  std::vector<double> fourier;
  std::vector<double> mc;
  std::vector<double> mc_err;
  std::vector<double> z_score;
  std::vector<std::string> status;
};

// =============================================================================
// HELPER: push one row
// =============================================================================
void log_greek(GreekStorage &store, const std::string &model_name,
               const std::string &payoff_name, const std::string &greek_name,
               double T, double K, double fourier_val, double mc_val,
               double mc_err) {
  double z = (mc_err > 1e-14) ? (mc_val - fourier_val) / mc_err : 0.0;

  std::string stat = "OK";
  if (std::abs(z) > 4.0)
    stat = "[ERROR]";
  else if (std::abs(z) > 2.5)
    stat = "[WARN]";

  store.model.push_back(model_name);
  store.payoff.push_back(payoff_name);
  store.greek.push_back(greek_name);
  store.T.push_back(T);
  store.K.push_back(K);
  store.fourier.push_back(fourier_val);
  store.mc.push_back(mc_val);
  store.mc_err.push_back(mc_err);
  store.z_score.push_back(z);
  store.status.push_back(stat);
}

// =============================================================================
// GENERIC GREEK TEST RUNNER
//   PayoffT  — any payoff supported by FourierPricer (Vanilla, CashOrNothing,
//               AssetOrNothing)
//   ModelType, StepperType — model/stepper pair
// =============================================================================
template <typename PayoffT, typename ModelType, typename StepperType>
void run_greek_test(const std::string &model_name,
                    const std::string &payoff_name, const ModelType &model,
                    const MonteCarloConfig &mc_cfg,
                    const FourierEngine::Config &f_cfg, GreekStorage &store) {
  constexpr double S0 = 100.0;
  constexpr double r = 0.05;
  constexpr double q = 0.02;

  const std::vector<double> maturities = {0.25, 1.0, 2.0};
  const std::vector<double> strike_pcts = {0.8, 0.9, 1.0, 1.1, 1.2};

  using StripT = EuropeanOption<PayoffT, std::vector<double>>;

  for (double T : maturities) {
    std::vector<double> strikes;
    for (double pct : strike_pcts)
      strikes.push_back(S0 * pct);

    StripT strip(strikes, T);

    // -------------------------------------------------------------------------
    // Fourier: analytical delta & gamma
    // -------------------------------------------------------------------------
    FourierPricer<ModelType, StripT, GreekMode::Essential> f_pricer(model,
                                                                    f_cfg);
    auto f_res = f_pricer.calculate(S0, r, q, strip);

    // -------------------------------------------------------------------------
    // Monte Carlo: pathwise FD delta & gamma
    // -------------------------------------------------------------------------
    MonteCarloPricer<ModelType, StepperType, StripT, GreekMode::Essential>
        mc_pricer(model, mc_cfg);
    auto mc_res = mc_pricer.calculate(S0, r, q, strip);

    // -------------------------------------------------------------------------
    // Log Price, Delta, Gamma per strike
    // -------------------------------------------------------------------------
    for (size_t i = 0; i < strikes.size(); ++i) {
      log_greek(store, model_name, payoff_name, "Price", T, strikes[i],
                f_res.price[i], mc_res.price[i], mc_res.price_std_err[i]);

      log_greek(store, model_name, payoff_name, "Delta", T, strikes[i],
                f_res.delta[i], mc_res.delta[i], mc_res.delta_std_err[i]);

      log_greek(store, model_name, payoff_name, "Gamma", T, strikes[i],
                f_res.gamma[i], mc_res.gamma[i], mc_res.gamma_std_err[i]);
    }
  }
}

// =============================================================================
// MAIN
// =============================================================================
int main() {
  std::cout
      << "=================================================================\n";
  std::cout
      << "  ESSENTIAL GREEK VERIFICATION: Delta & Gamma (Fourier vs MC)\n";
  std::cout << "==============================================================="
               "==\n\n";

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------
  MonteCarloConfig mc_cfg;
  mc_cfg.num_paths = 4000000;
  mc_cfg.time_steps = 50;
  mc_cfg.seed = 42;
  mc_cfg.fd_bump = 1e-3;

  FourierEngine::Config f_cfg;
  f_cfg.tolerance = 1e-9;

  // Model parameters
  HestonParams h1 = {2.5, 0.06, 0.4, -0.7, 0.05};
  HestonParams h2 = {1.5, 0.04, 0.3, -0.2, 0.04};
  MertonParams m_jmp = {1.2, -0.12, 0.15};

  // ---------------------------------------------------------------------------
  // Helper: run all models for a given payoff type and storage
  // ---------------------------------------------------------------------------
  auto run_all_models = [&]<typename PayoffT>(const std::string &payoff_name,
                                              GreekStorage &store) {
    // Heston
    {
      HestonModel m(h1);
      using S =
          ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerEuropean>;
      run_greek_test<PayoffT, HestonModel, S>("Heston", payoff_name, m, mc_cfg,
                                              f_cfg, store);
    }
    // Bates
    {
      BatesModel m(h1, m_jmp);
      using S =
          ASVJStepper<SchemeExact, NullVolScheme, MertonJump, TrackerEuropean>;
      run_greek_test<PayoffT, BatesModel, S>("Bates", payoff_name, m, mc_cfg,
                                             f_cfg, store);
    }
    // Double Heston
    {
      DoubleHestonModel m(h1, h2);
      using S = ASVJStepper<SchemeExact, SchemeNV, NoJumps, TrackerEuropean>;
      run_greek_test<PayoffT, DoubleHestonModel, S>("D-Heston", payoff_name, m,
                                                    mc_cfg, f_cfg, store);
    }
  };

  // ===========================================================================
  // TABLE 1: VANILLA  (Call & Put)
  // ===========================================================================
  std::cout << "\n--- TABLE 1: VANILLA OPTIONS ---\n";
  GreekStorage vanilla_store;
  run_all_models.template operator()<PayoffVanilla<OptionType::Call>>(
      "VanillaCall", vanilla_store);
  run_all_models.template operator()<PayoffVanilla<OptionType::Put>>(
      "VanillaPut", vanilla_store);

  myql::utils::printVectors(
      {"Model", "Payoff", "Greek", "T", "Strike", "Fourier", "MC", "MC StdErr",
       "Z-Score", "Status"},
      vanilla_store.model, vanilla_store.payoff, vanilla_store.greek,
      vanilla_store.T, vanilla_store.K, vanilla_store.fourier, vanilla_store.mc,
      vanilla_store.mc_err, vanilla_store.z_score, vanilla_store.status);

  // ===========================================================================
  // TABLE 2: DIGITAL OPTIONS (Cash-or-Nothing & Asset-or-Nothing, Call & Put)
  // ===========================================================================
  std::cout << "\n--- TABLE 2: DIGITAL OPTIONS ---\n";
  GreekStorage digital_store;
  run_all_models.template operator()<PayoffCashOrNothing<OptionType::Call>>(
      "CashCall", digital_store);
  run_all_models.template operator()<PayoffCashOrNothing<OptionType::Put>>(
      "CashPut", digital_store);
  run_all_models.template operator()<PayoffAssetOrNothing<OptionType::Call>>(
      "AssetCall", digital_store);
  run_all_models.template operator()<PayoffAssetOrNothing<OptionType::Put>>(
      "AssetPut", digital_store);

  myql::utils::printVectors(
      {"Model", "Payoff", "Greek", "T", "Strike", "Fourier", "MC", "MC StdErr",
       "Z-Score", "Status"},
      digital_store.model, digital_store.payoff, digital_store.greek,
      digital_store.T, digital_store.K, digital_store.fourier, digital_store.mc,
      digital_store.mc_err, digital_store.z_score, digital_store.status);

  return 0;
}
