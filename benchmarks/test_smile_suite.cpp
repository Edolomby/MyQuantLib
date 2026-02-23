#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// CORE ENGINES
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

// -----------------------------------------------------------------------------
// GLOBAL RESULTS STORAGE
// -----------------------------------------------------------------------------
struct ResultStorage {
  std::vector<std::string> model;
  std::vector<double> T;
  std::vector<double> K;
  std::vector<double> fourier;
  std::vector<double> mc;
  std::vector<double> stderr;
  std::vector<double> z_score;
  std::vector<std::string> status;
};

ResultStorage global_storage;

// -----------------------------------------------------------------------------
// TEST RUNNER
// -----------------------------------------------------------------------------
template <typename ModelType, typename StepperType>
void run_smile_test(const std::string &name, const ModelType &model,
                    const MonteCarloConfig &mc_cfg,
                    const FourierEngine::Config &f_cfg) {

  double S0 = 100.0;
  double r = 0.10;
  double q = 0.07;

  std::vector<double> maturities = {0.25, 1.0, 2.0};
  std::vector<double> strike_pcts = {0.8, 0.9, 1.0, 1.1, 1.2};

  for (double T : maturities) {
    // 1. CALCULATE STRIKES FOR THIS MATURITY
    std::vector<double> strikes;
    for (double k_pct : strike_pcts) {
      strikes.push_back(S0 * k_pct);
    }

    // 2. DEFINE THE STRIP INSTRUMENT
    using StripT =
        EuropeanOption<PayoffVanilla<OptionType::Call>, std::vector<double>>;
    StripT smile_strip(strikes, T);

    // 3. VECTORIZED FOURIER PRICING
    FourierPricer<ModelType, StripT> f_engine(model, f_cfg);
    auto fourier_results = f_engine.calculate(S0, r, q, smile_strip);

    // 4. VECTORIZED MONTE CARLO PRICING
    // The engine now uses the Buffered Pattern: 1 path = N prices.
    MonteCarloPricer<ModelType, StepperType, StripT> engine(model, mc_cfg);
    auto mc_res = engine.calculate(S0, r, q, smile_strip);

    auto mc_prices = mc_res.price;
    auto mc_errors = mc_res.price_std_err;

    // 5. STORE RESULTS (Flatten the vectors for the TablePrinter)
    for (size_t i = 0; i < strikes.size(); ++i) {
      double diff = mc_prices[i] - fourier_results.price[i];
      double z = (mc_errors[i] > 1e-12) ? diff / mc_errors[i] : 0.0;

      std::string stat_tag = "OK";
      if (std::abs(z) > 4.0)
        stat_tag = "[ERROR]";
      else if (std::abs(z) > 2.0)
        stat_tag = "[WARN]";

      global_storage.model.push_back(name);
      global_storage.T.push_back(T);
      global_storage.K.push_back(strikes[i]);
      global_storage.fourier.push_back(fourier_results.price[i]);
      global_storage.mc.push_back(mc_prices[i]);
      global_storage.stderr.push_back(mc_errors[i]);
      global_storage.z_score.push_back(z);
      global_storage.status.push_back(stat_tag);
    }
  }
}

int main() {
  std::cout << "==============================================================="
               "=====\n";
  std::cout << "  ASVJ MC SMILE VERIFICATION SUITE\n";
  std::cout << "==============================================================="
               "=====\n";

  MonteCarloConfig mc_cfg;
  mc_cfg.num_paths = 4000000; // 4M paths
  mc_cfg.time_steps = 100;
  mc_cfg.seed = 42;

  FourierEngine::Config f_cfg;
  f_cfg.tolerance = 1e-9;

  HestonParams h1 = {2.5, 0.06, 0.4, -0.7, 0.05};
  HestonParams h2 = {1.5, 0.04, 0.3, -0.2, 0.04};
  MertonParams m_jmp = {1.2, -0.12, 0.15};
  KouParams k_jmp = {2.0, 0.3, 20.0, 12.0};

  // --- MODEL EXECUTION ---
  {
    HestonModel m(h1);
    using Stepper =
        ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerEuropean>;
    run_smile_test<HestonModel, Stepper>("Heston", m, mc_cfg, f_cfg);
  }
  {
    BatesModel m(h1, m_jmp);
    using Stepper =
        ASVJStepper<SchemeExact, NullVolScheme, MertonJump, TrackerEuropean>;
    run_smile_test<BatesModel, Stepper>("Bates", m, mc_cfg, f_cfg);
  }
  {
    BatesKouModel m(h1, k_jmp);
    using Stepper =
        ASVJStepper<SchemeExact, NullVolScheme, KouJump, TrackerEuropean>;
    run_smile_test<BatesKouModel, Stepper>("Bates-Kou", m, mc_cfg, f_cfg);
  }
  {
    DoubleHestonModel m(h1, h2);
    using Stepper =
        ASVJStepper<SchemeExact, SchemeNV, NoJumps, TrackerEuropean>;
    run_smile_test<DoubleHestonModel, Stepper>("D-Heston", m, mc_cfg, f_cfg);
  }
  {
    DoubleBatesModel m(h1, h2, m_jmp);
    using Stepper =
        ASVJStepper<SchemeExact, SchemeNV, MertonJump, TrackerEuropean>;
    run_smile_test<DoubleBatesModel, Stepper>("D-Bates", m, mc_cfg, f_cfg);
  }
  {
    DoubleBatesKouModel m(h1, h2, k_jmp);
    using Stepper =
        ASVJStepper<SchemeExact, SchemeNV, KouJump, TrackerEuropean>;
    run_smile_test<DoubleBatesKouModel, Stepper>("D-Bates-Kou", m, mc_cfg,
                                                 f_cfg);
  }

  // --- FINAL PRINTING ---
  utils::printVectors({"Model", "T", "Strike", "Fourier", "MC Price", "StdErr",
                       "Z-Score", "Status"},
                      global_storage.model, global_storage.T, global_storage.K,
                      global_storage.fourier, global_storage.mc,
                      global_storage.stderr, global_storage.z_score,
                      global_storage.status);

  return 0;
}