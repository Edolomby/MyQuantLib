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
  std::vector<std::string> model_payoff;
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
template <typename PayoffPolicy, typename ModelType, typename StepperType>
void run_payoff_verification(const std::string &model_name,
                             const std::string &payoff_name,
                             const ModelType &model,
                             const MonteCarloConfig &mc_cfg,
                             const FourierEngine::Config &f_cfg) {

  double S0 = 100.0;
  double r = 0.03;
  double q = 0.00;

  // As requested: 0.2, 1.0, 2.0 maturities
  std::vector<double> maturities = {0.2, 1.0, 2.0};
  std::vector<double> strike_pcts = {0.8, 0.9, 1.0, 1.1, 1.2};

  for (double T : maturities) {
    std::vector<double> strikes;
    for (double k_pct : strike_pcts) {
      strikes.push_back(S0 * k_pct);
    }

    // Define the Strip for this specific Payoff Type
    using StripT = EuropeanOption<PayoffPolicy, std::vector<double>>;
    StripT test_strip(strikes, T);

    // 1. Fourier Pricing
    FourierPricer<ModelType, StripT> f_engine(model, f_cfg);
    auto fourier_results = f_engine.calculate(S0, r, q, test_strip);

    // 2. Monte Carlo Pricing
    MonteCarloPricer<ModelType, StepperType, StripT> engine(model, mc_cfg);
    auto mc_res = engine.calculate(S0, r, q, test_strip);

    auto mc_prices = mc_res.price;
    auto mc_errors = mc_res.price_std_err;

    // 3. Store and Validate
    for (size_t i = 0; i < strikes.size(); ++i) {
      double diff = mc_prices[i] - fourier_results.price[i];
      double z = (mc_errors[i] > 1e-12) ? diff / mc_errors[i] : 0.0;

      std::string stat_tag = "OK";
      if (std::abs(z) > 4.0)
        stat_tag = "[ERROR]";
      else if (std::abs(z) > 2.0)
        stat_tag = "[WARN]";

      global_storage.model_payoff.push_back(model_name + " " + payoff_name);
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

// -----------------------------------------------------------------------------
// TYPE TAG (To pass types to lambdas without constructing them)
// -----------------------------------------------------------------------------
template <typename T> struct type_tag {
  using type = T;
};

int main() {
  std::cout
      << "===============================================================\n";
  std::cout << "  FULL ASVJ VERIFICATION SUITE (DIGITAL & ASSET-OR-NOTHING)\n";
  std::cout
      << "===============================================================\n";

  // 1. CONFIGURATION
  MonteCarloConfig mc_cfg;
  mc_cfg.num_paths = 500000;
  mc_cfg.time_steps = 100;
  mc_cfg.seed = 42;

  FourierEngine::Config f_cfg;
  f_cfg.tolerance = 1e-9;

  // Parameters
  HestonParams h1 = {2.5, 0.06, 0.4, -0.7, 0.05};
  HestonParams h2 = {1.5, 0.04, 0.3, -0.2, 0.04};
  MertonParams m_jmp = {1.2, -0.12, 0.15};
  KouParams k_jmp = {2.0, 0.3, 20.0, 12.0};

  // 2. UPDATED HELPER LAMBDA
  // We pass type_tag<Stepper> instead of the Stepper itself
  auto run_all_payoffs = [&](const std::string &name, const auto &model,
                             auto tag) {
    using Stepper =
        typename decltype(tag)::type; // Extract the type from the tag
    using ModelT = std::decay_t<decltype(model)>;

    run_payoff_verification<PayoffCashOrNothing<OptionType::Call>, ModelT,
                            Stepper>(name, "Cash-Call", model, mc_cfg, f_cfg);
    run_payoff_verification<PayoffCashOrNothing<OptionType::Put>, ModelT,
                            Stepper>(name, "Cash-Put", model, mc_cfg, f_cfg);
    run_payoff_verification<PayoffAssetOrNothing<OptionType::Call>, ModelT,
                            Stepper>(name, "Asset-Call", model, mc_cfg, f_cfg);
    run_payoff_verification<PayoffAssetOrNothing<OptionType::Put>, ModelT,
                            Stepper>(name, "Asset-Put", model, mc_cfg, f_cfg);
  };

  // 3. EXECUTION: Single-Factor Models
  {
    HestonModel m(h1);
    run_all_payoffs(
        "Heston", m,
        type_tag<
            ASVJStepper<SchemeNCI, NullVolScheme, NoJumps, TrackerEuropean>>{});
  }
  {
    BatesModel m(h1, m_jmp);
    run_all_payoffs("Bates", m,
                    type_tag<ASVJStepper<SchemeNCI, NullVolScheme, MertonJump,
                                         TrackerEuropean>>{});
  }
  {
    BatesKouModel m(h1, k_jmp);
    run_all_payoffs(
        "Bates-Kou", m,
        type_tag<
            ASVJStepper<SchemeNCI, NullVolScheme, KouJump, TrackerEuropean>>{});
  }

  // 4. EXECUTION: Double-Factor Models
  {
    DoubleHestonModel m(h1, h2);
    run_all_payoffs(
        "D-Heston", m,
        type_tag<ASVJStepper<SchemeNCI, SchemeNV, NoJumps, TrackerEuropean>>{});
  }
  {
    DoubleBatesModel m(h1, h2, m_jmp);
    run_all_payoffs(
        "D-Bates", m,
        type_tag<
            ASVJStepper<SchemeNCI, SchemeNV, MertonJump, TrackerEuropean>>{});
  }
  {
    DoubleBatesKouModel m(h1, h2, k_jmp);
    run_all_payoffs(
        "D-Bates-Kou", m,
        type_tag<ASVJStepper<SchemeNCI, SchemeNV, KouJump, TrackerEuropean>>{});
  }

  // 5. FINAL PRINTING
  myql::utils::printVectors({"Model/Payoff", "T", "Strike", "Fourier",
                             "MC Price", "StdErr", "Z-Score", "Status"},
                            global_storage.model_payoff, global_storage.T,
                            global_storage.K, global_storage.fourier,
                            global_storage.mc, global_storage.stderr,
                            global_storage.z_score, global_storage.status);

  return 0;
}