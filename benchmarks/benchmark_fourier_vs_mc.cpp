#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// =============================================================================
// MYQUANTLIB INCLUDES
// =============================================================================
#include <myql/engines/fourier/FourierPricer.hpp>
#include <myql/engines/montecarlo/MonteCarloEngine.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/models/asvj/data/ModelParams.hpp>
#include <myql/models/asvj/policies/JumpPolicies.hpp>
#include <myql/models/asvj/policies/VolSchemes.hpp>
#include <myql/utils/TablePrinter.hpp>

// NEW: Include the Instruments
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/European.hpp>

// RNG
#include <boost/random/xoshiro.hpp>

// =============================================================================
// GLOBAL DATA STORAGE
// =============================================================================
struct BenchmarkResults {
  std::vector<std::string> scenarios;
  std::vector<std::string> details;
  std::vector<double> fourier_prices;
  std::vector<double> mc_prices;
  std::vector<double> diffs;
  std::vector<double> mc_stderrs;
  std::vector<double> z_scores;
  std::vector<double> time_fourier;
  std::vector<double> time_mc;
};

BenchmarkResults results;

// =============================================================================
// TEST RUNNER
// =============================================================================
template <typename ModelType, typename StepperType, typename RNG>
void run_single_test(const std::string &model_name, const ModelType &model,
                     double S0, double K, double T, double r, double q,
                     bool is_call, const std::string &moneyness_label,
                     const MonteCarloConfig &mc_cfg,
                     const FourierEngine::Config &four_cfg) {

  // A. Fourier Pricing (Reference)
  double price_fourier = 0.0;
  double t_four = 0.0;
  {
    auto start = std::chrono::high_resolution_clock::now();
    price_fourier =
        price_european_fourier(model, S0, K, T, r, q, is_call, four_cfg);
    auto end = std::chrono::high_resolution_clock::now();
    t_four = std::chrono::duration<double>(end - start).count() * 1000.0; // ms
  }

  // B. Monte Carlo Pricing
  std::pair<double, double> res_mc = {0.0, 0.0};
  double t_mc = 0.0;

  auto start = std::chrono::high_resolution_clock::now();

  // We must handle the type dispatch here because OptionType::Call is a
  // template arg
  if (is_call) {
    // 1. Define the specific Instrument Type
    using InstrumentT = EuropeanOption<PayoffVanilla<OptionType::Call>>;

    // 2. Instantiate Instrument
    InstrumentT option(K, T);

    // 3. Configure Engine with the specific Instrument Type
    MonteCarloEngine<ModelType, StepperType, InstrumentT, RNG> engine(model,
                                                                      mc_cfg);

    // 4. Calculate
    // Note: We don't pass K or T here anymore; the instrument has them!
    res_mc = engine.calculate(S0, r, q, option);

  } else {
    // Same logic for Put
    using InstrumentT = EuropeanOption<PayoffVanilla<OptionType::Put>>;

    InstrumentT option(K, T);
    MonteCarloEngine<ModelType, StepperType, InstrumentT, RNG> engine(model,
                                                                      mc_cfg);
    res_mc = engine.calculate(S0, r, q, option);
  }

  auto end = std::chrono::high_resolution_clock::now();
  t_mc = std::chrono::duration<double>(end - start).count(); // s

  // C. Record Results
  double diff = res_mc.first - price_fourier;
  double z = (res_mc.second > 1e-14) ? diff / res_mc.second : 0.0;

  std::stringstream ss;
  ss << "T=" << T << " " << moneyness_label << " "
     << (is_call ? "Call" : "Put");

  results.scenarios.push_back(model_name);
  results.details.push_back(ss.str());
  results.fourier_prices.push_back(price_fourier);
  results.mc_prices.push_back(res_mc.first);
  results.diffs.push_back(diff);
  results.mc_stderrs.push_back(res_mc.second);
  results.z_scores.push_back(z);
  results.time_fourier.push_back(t_four);
  results.time_mc.push_back(t_mc);
}

// =============================================================================
// SUITE GENERATOR & MAIN (UNCHANGED)
// =============================================================================
template <typename ModelType, typename StepperType>
void run_full_suite(const std::string &model_name, const ModelType &model,
                    const MonteCarloConfig &mc_cfg,
                    const FourierEngine::Config &four_cfg) {

  // RNG SELECTION:
  using RNG = boost::random::xoshiro256pp;
  double S0 = 100.0;
  double r = 0.03;
  double q = 0.0;

  std::vector<double> maturities = {0.08, 0.5, 1.0, 3.0};

  struct StrikeCase {
    double pct;
    std::string label;
  };
  std::vector<StrikeCase> strikes = {{0.9, "ITM"}, {1.0, "ATM"}, {1.1, "OTM"}};

  for (double T : maturities) {
    for (const auto &sk : strikes) {
      double K = S0 * sk.pct;
      run_single_test<ModelType, StepperType, RNG>(
          model_name, model, S0, K, T, r, q, true, sk.label, mc_cfg, four_cfg);
      run_single_test<ModelType, StepperType, RNG>(
          model_name, model, S0, K, T, r, q, false, sk.label, mc_cfg, four_cfg);
    }
  }
}

int main() {
  std::cout << "==============================================================="
               "=====\n";
  std::cout << "  ASVJ FRAMEWORK: BENCHMARK WITH NEW INSTRUMENTS\n";
  std::cout << "==============================================================="
               "=====\n";

  // --- CONFIG ---
  MonteCarloConfig mc_cfg;
  mc_cfg.num_paths = 200000;
  mc_cfg.time_steps = 100;
  mc_cfg.seed = 423;

  FourierEngine::Config fourier_cfg;
  fourier_cfg.tolerance = 1e-9;

  // --- PARAMETERS ---
  HestonParams h1 = {2.0, 0.04, 0.4, -0.7, 0.04};
  HestonParams h2 = {3.0, 0.02, 0.2, -0.3, 0.02};
  MertonParams j_merton = {1.0, -0.1, 0.1};
  KouParams j_kou = {3.0, 0.3, 25.0, 15.0};

  // 1. Heston
  {
    HestonModel model(h1);
    // Notice: We still manually select TrackerEuropean for the Stepper here.
    // In a fully automated Engine, the Engine could deduce this, but explicit
    // is fine.
    using Stepper = ASVJStepper<1, SchemeHighVol, NoJumps, TrackerEuropean>;
    run_full_suite<HestonModel, Stepper>("Heston", model, mc_cfg, fourier_cfg);
  }

  // 2. Bates
  {
    BatesModel model(h1, j_merton);
    using Stepper = ASVJStepper<1, SchemeHighVol, MertonJump, TrackerEuropean>;
    run_full_suite<BatesModel, Stepper>("Bates", model, mc_cfg, fourier_cfg);
  }

  // 3. Bates-Kou
  {
    BatesKouModel model(h1, j_kou);
    using Stepper = ASVJStepper<1, SchemeHighVol, KouJump, TrackerEuropean>;
    run_full_suite<BatesKouModel, Stepper>("Bates-Kou", model, mc_cfg,
                                           fourier_cfg);
  }

  // 4. Double Heston
  {
    DoubleHestonModel model(h1, h2);
    using Stepper = ASVJStepper<2, SchemeHighVol, NoJumps, TrackerEuropean>;
    run_full_suite<DoubleHestonModel, Stepper>("D-Heston", model, mc_cfg,
                                               fourier_cfg);
  }

  // 5. Double Bates
  {
    DoubleBatesModel model(h1, h2, j_merton);
    using Stepper = ASVJStepper<2, SchemeHighVol, MertonJump, TrackerEuropean>;
    run_full_suite<DoubleBatesModel, Stepper>("D-Bates", model, mc_cfg,
                                              fourier_cfg);
  }

  // 6. Double Kou
  {
    DoubleFactorModel<KouParams> model(h1, h2, j_kou);
    using Stepper = ASVJStepper<2, SchemeHighVol, KouJump, TrackerEuropean>;
    run_full_suite<DoubleFactorModel<KouParams>, Stepper>("D-Bates-Kou", model,
                                                          mc_cfg, fourier_cfg);
  }

  // PRINT RESULTS
  std::cout << "\n";
  utils::printVectors({"Model", "Details", "Fourier", "MC Price", "Diff",
                       "MC Err", "Z-Score", "Four(ms)", "MC(s)"},
                      results.scenarios, results.details,
                      results.fourier_prices, results.mc_prices, results.diffs,
                      results.mc_stderrs, results.z_scores,
                      results.time_fourier, results.time_mc);

  return 0;
}