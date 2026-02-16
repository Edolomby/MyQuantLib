#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// CORE ENGINES
#include <myql/engines/fourier/FourierPricer.hpp>
#include <myql/engines/montecarlo/MonteCarloEngine.hpp>

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
// GLOBAL RESULTS STORAGE (DECOMPOSED FOR printVectors)
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
  double r = 0.03;
  double q = 0.00;

  std::vector<double> maturities = {0.25, 1.0, 2.0};
  std::vector<double> strike_pcts = {0.8, 0.9, 1.0, 1.1, 1.2};

  for (double T : maturities) {
    for (double k_pct : strike_pcts) {
      double K = S0 * k_pct;

      // 1. Reference Price
      double p_fourier =
          price_european_fourier(model, S0, K, T, r, q, true, f_cfg);

      // 2. MC Price
      using InstrumentT = EuropeanOption<PayoffVanilla<OptionType::Call>>;
      InstrumentT call_option(K, T);

      MonteCarloEngine<ModelType, StepperType, InstrumentT> engine(model,
                                                                   mc_cfg);
      auto res_mc = engine.calculate(S0, r, q, call_option);

      // 3. Statistical Analysis (Z-Score)
      double diff = res_mc.first - p_fourier;
      double z = (res_mc.second > 1e-12) ? diff / res_mc.second : 0.0;

      std::string stat_tag = "OK";
      if (std::abs(z) > 4.0)
        stat_tag = "[ERROR]";
      else if (std::abs(z) > 2.0)
        stat_tag = "[WARN]";

      // 4. Store for TablePrinter
      global_storage.model.push_back(name);
      global_storage.T.push_back(T);
      global_storage.K.push_back(K);
      global_storage.fourier.push_back(p_fourier);
      global_storage.mc.push_back(res_mc.first);
      global_storage.stderr.push_back(res_mc.second);
      global_storage.z_score.push_back(z);
      global_storage.status.push_back(stat_tag);
    }
  }
}

int main() {
  std::cout << "==============================================================="
               "=====\n";
  std::cout << "  ASVJ MC SEVERAL STRIKES VERIFICATION SUITE\n";
  std::cout << "==============================================================="
               "=====\n";

  MonteCarloConfig mc_cfg;
  mc_cfg.num_paths = 250000;
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
    using Stepper = ASVJStepper<1, SchemeHighVol, NoJumps, TrackerEuropean>;
    run_smile_test<HestonModel, Stepper>("Heston", m, mc_cfg, f_cfg);
  }
  {
    BatesModel m(h1, m_jmp);
    using Stepper = ASVJStepper<1, SchemeHighVol, MertonJump, TrackerEuropean>;
    run_smile_test<BatesModel, Stepper>("Bates", m, mc_cfg, f_cfg);
  }
  {
    BatesKouModel m(h1, k_jmp);
    using Stepper = ASVJStepper<1, SchemeHighVol, KouJump, TrackerEuropean>;
    run_smile_test<BatesKouModel, Stepper>("Bates-Kou", m, mc_cfg, f_cfg);
  }
  {
    DoubleHestonModel m(h1, h2);
    using Stepper = ASVJStepper<2, SchemeHighVol, NoJumps, TrackerEuropean>;
    run_smile_test<DoubleHestonModel, Stepper>("D-Heston", m, mc_cfg, f_cfg);
  }
  {
    DoubleBatesModel m(h1, h2, m_jmp);
    using Stepper = ASVJStepper<2, SchemeHighVol, MertonJump, TrackerEuropean>;
    run_smile_test<DoubleBatesModel, Stepper>("D-Bates", m, mc_cfg, f_cfg);
  }
  {
    DoubleBatesKouModel m(h1, h2, k_jmp);
    using Stepper = ASVJStepper<2, SchemeHighVol, KouJump, TrackerEuropean>;
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