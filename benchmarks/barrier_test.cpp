#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// CORE ENGINES
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>

// MODELS & STEPPERS
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/models/asvj/policies/JumpPolicies.hpp>
#include <myql/models/asvj/policies/VolSchemes.hpp>

// INSTRUMENTS
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/Barrier.hpp>
#include <myql/instruments/options/European.hpp>

// UTILS
#include <myql/utils/TablePrinter.hpp>

// -----------------------------------------------------------------------------
// GLOBAL RESULTS STORAGE
// -----------------------------------------------------------------------------
struct ResultStorage {
  std::vector<std::string> model;
  std::vector<std::string> type;
  std::vector<double> T;
  std::vector<double> K;
  std::vector<double> barrier;
  std::vector<double> eur_price;
  std::vector<double> in_price;
  std::vector<double> out_price;
  std::vector<double> in_out_sum;
  std::vector<double> diff;
  std::vector<std::string> status;
};

ResultStorage global_storage;

// -----------------------------------------------------------------------------
// TEST RUNNER
// -----------------------------------------------------------------------------
template <OptionType OptT, typename ModelType, typename EurStepper,
          typename UpBarrierStepper, typename DownBarrierStepper>
void run_parity_test(const std::string &model_name, const ModelType &model,
                     const MonteCarloConfig &mc_cfg) {

  double S0 = 100.0;
  double r = 0.05;
  double q = 0.02;
  double T = 1.0;

  std::vector<double> strikes = {90.0, 100.0, 110.0};
  double B_up = 120.0;
  double B_down = 80.0;

  std::string opt_str = (OptT == OptionType::Call) ? "Call" : "Put";

  // 1. DEFINE STRIP INSTRUMENTS
  using StripEur = EuropeanOption<PayoffVanilla<OptT>, std::vector<double>>;
  using StripUI =
      BarrierOption<PayoffVanilla<OptT>, BarrierDirection::Up,
                    BarrierAction::In, RebateTiming::None, std::vector<double>>;
  using StripUO = BarrierOption<PayoffVanilla<OptT>, BarrierDirection::Up,
                                BarrierAction::Out, RebateTiming::None,
                                std::vector<double>>;
  using StripDI =
      BarrierOption<PayoffVanilla<OptT>, BarrierDirection::Down,
                    BarrierAction::In, RebateTiming::None, std::vector<double>>;
  using StripDO = BarrierOption<PayoffVanilla<OptT>, BarrierDirection::Down,
                                BarrierAction::Out, RebateTiming::None,
                                std::vector<double>>;

  StripEur eur_strip(strikes, T);
  StripUI ui_strip(strikes, B_up, T);
  StripUO uo_strip(strikes, B_up, T);
  StripDI di_strip(strikes, B_down, T);
  StripDO do_strip(strikes, B_down, T);

  // 2. RUN EUROPEAN BASELINE
  MonteCarloPricer<ModelType, EurStepper, StripEur> eur_engine(model, mc_cfg);
  auto eur_res = eur_engine.calculate(S0, r, q, eur_strip);

  // 3. RUN UP-BARRIER PARITY
  MonteCarloPricer<ModelType, UpBarrierStepper, StripUI> ui_engine(model,
                                                                   mc_cfg);
  auto ui_res = ui_engine.calculate(S0, r, q, ui_strip);

  MonteCarloPricer<ModelType, UpBarrierStepper, StripUO> uo_engine(model,
                                                                   mc_cfg);
  auto uo_res = uo_engine.calculate(S0, r, q, uo_strip);

  for (size_t i = 0; i < strikes.size(); ++i) {
    double sum_in_out = ui_res.price[i] + uo_res.price[i];
    double diff = std::abs(sum_in_out - eur_res.price[i]);
    std::string stat_tag = (diff < 1e-10) ? "PASS" : "[FAIL]";

    global_storage.model.push_back(model_name);
    global_storage.type.push_back("Up-" + opt_str);
    global_storage.T.push_back(T);
    global_storage.K.push_back(strikes[i]);
    global_storage.barrier.push_back(B_up);
    global_storage.eur_price.push_back(eur_res.price[i]);
    global_storage.in_price.push_back(ui_res.price[i]);
    global_storage.out_price.push_back(uo_res.price[i]);
    global_storage.in_out_sum.push_back(sum_in_out);
    global_storage.diff.push_back(diff);
    global_storage.status.push_back(stat_tag);
  }

  // 4. RUN DOWN-BARRIER PARITY
  MonteCarloPricer<ModelType, DownBarrierStepper, StripDI> di_engine(model,
                                                                     mc_cfg);
  auto di_res = di_engine.calculate(S0, r, q, di_strip);

  MonteCarloPricer<ModelType, DownBarrierStepper, StripDO> do_engine(model,
                                                                     mc_cfg);
  auto do_res = do_engine.calculate(S0, r, q, do_strip);

  for (size_t i = 0; i < strikes.size(); ++i) {
    double sum_in_out = di_res.price[i] + do_res.price[i];
    double diff = std::abs(sum_in_out - eur_res.price[i]);
    std::string stat_tag = (diff < 1e-10) ? "PASS" : "[FAIL]";

    global_storage.model.push_back(model_name);
    global_storage.type.push_back("Down-" + opt_str);
    global_storage.T.push_back(T);
    global_storage.K.push_back(strikes[i]);
    global_storage.barrier.push_back(B_down);
    global_storage.eur_price.push_back(eur_res.price[i]);
    global_storage.in_price.push_back(di_res.price[i]);
    global_storage.out_price.push_back(do_res.price[i]);
    global_storage.in_out_sum.push_back(sum_in_out);
    global_storage.diff.push_back(diff);
    global_storage.status.push_back(stat_tag);
  }
}

// Dummy tracker to force the European pricer traker to become path_dependent
// so its random numbers perfectly match the Barrier pricer.
struct TrackerEuropeanStepped : public TrackerEuropean {
  static constexpr bool is_path_dependent = true; // Override to true!
};

int main() {
  std::cout << "==============================================================="
               "====================================\n";
  std::cout << "  COMPREHENSIVE BARRIER IN-OUT PARITY SUITE (Calls & Puts "
               "across Models)\n";
  std::cout << "==============================================================="
               "====================================\n";

  MonteCarloConfig mc_cfg;
  mc_cfg.num_paths = 250000; // 1M paths is plenty for exact parity
  mc_cfg.time_steps = 252;
  mc_cfg.seed = 42;

  // ---------------------------------------------------------------------------
  // 1. BLACK-SCHOLES (Constant Volatility, No Jumps)
  // ---------------------------------------------------------------------------
  ZeroFactorModel<NoJumpParams> bs_model(0.2373); // 23.73% vol
  using BSEur =
      BlackScholesStepper<TrackerEuropeanStepped>; // used the stepped mode
  using BSUp = BlackScholesStepper<TrackerBarrier<true>>;
  using BSDown = BlackScholesStepper<TrackerBarrier<false>>;

  run_parity_test<OptionType::Call, decltype(bs_model), BSEur, BSUp, BSDown>(
      "Black-Scholes", bs_model, mc_cfg);
  run_parity_test<OptionType::Put, decltype(bs_model), BSEur, BSUp, BSDown>(
      "Black-Scholes", bs_model, mc_cfg);

  // ---------------------------------------------------------------------------
  // 2. HESTON (Stochastic Volatility, No Jumps)
  // ---------------------------------------------------------------------------
  HestonParams h1 = {2.5, 0.06, 0.4, -0.7, 0.05};
  HestonModel heston_model(h1);
  using HestEur =
      ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerEuropean>;
  using HestUp =
      ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerBarrier<true>>;
  using HestDown =
      ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerBarrier<false>>;

  run_parity_test<OptionType::Call, decltype(heston_model), HestEur, HestUp,
                  HestDown>("Heston", heston_model, mc_cfg);
  run_parity_test<OptionType::Put, decltype(heston_model), HestEur, HestUp,
                  HestDown>("Heston", heston_model, mc_cfg);

  // ---------------------------------------------------------------------------
  // 3. BATES (Stochastic Volatility + Merton Jumps)
  // ---------------------------------------------------------------------------
  MertonParams m_jmp = {1.2, -0.12, 0.15}; // 1.2 jumps per year
  BatesModel bates_model(h1, m_jmp);
  using BatesEur = ASVJStepper<SchemeExact, NullVolScheme, MertonJump,
                               TrackerEuropeanStepped>; // used the stepped mode
  using BatesUp =
      ASVJStepper<SchemeExact, NullVolScheme, MertonJump, TrackerBarrier<true>>;
  using BatesDown = ASVJStepper<SchemeExact, NullVolScheme, MertonJump,
                                TrackerBarrier<false>>;

  run_parity_test<OptionType::Call, decltype(bates_model), BatesEur, BatesUp,
                  BatesDown>("Bates", bates_model, mc_cfg);
  run_parity_test<OptionType::Put, decltype(bates_model), BatesEur, BatesUp,
                  BatesDown>("Bates", bates_model, mc_cfg);

  // Print Results
  myql::utils::printVectors(
      {"Model", "Type", "T", "Strike", "Barrier", "Eur Price", "In Price",
       "Out Price", "In+Out Sum", "Abs Diff", "Status"},
      global_storage.model, global_storage.type, global_storage.T,
      global_storage.K, global_storage.barrier, global_storage.eur_price,
      global_storage.in_price, global_storage.out_price,
      global_storage.in_out_sum, global_storage.diff, global_storage.status);

  return 0;
}