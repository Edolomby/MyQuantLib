#include <chrono>
#include <iomanip>

#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/Asian.hpp>
#include <myql/instruments/options/Barrier.hpp>
#include <myql/instruments/options/European.hpp>
#include <myql/instruments/options/Lookback.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/models/asvj/data/ModelParams.hpp>
#include <myql/models/asvj/policies/VolSchemes.hpp>
#include <myql/pricers/fourier/FourierPricer.hpp>
#include <myql/pricers/mlmc/MultiLevelMonteCarloPricer.hpp>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>

template <typename Model, typename Stepper, typename CoupledStepper,
          typename Instrument>
void run_benchmark(const std::string &name, const Model &model,
                   const Instrument &opt, double S0, double r, double q,
                   const MonteCarloConfig &mc_cfg, const MLMCConfig &mlmc_cfg) {

  std::cout << "*****************************************\n";
  std::cout << "  " << name << "\n";
  std::cout << "*****************************************\n";

  // 1. Fourier (Reference)
  FourierEngine::Config f_cfg;
  FourierPricer<Model, Instrument> fourier(model, f_cfg);
  auto res_f = fourier.calculate(S0, r, q, opt);
  double price_f = res_f.price;

  std::cout << "[FOURIER (Reference)]\n";
  std::cout << "  Price: " << std::fixed << std::setprecision(6) << price_f
            << "\n\n";

  // 2. Standard Monte Carlo
  MonteCarloPricer<Model, Stepper, Instrument> mc_p(model, mc_cfg);
  auto start_mc = std::chrono::high_resolution_clock::now();
  auto res_mc = mc_p.calculate(S0, r, q, opt);
  auto end_mc = std::chrono::high_resolution_clock::now();

  double bias_mc = std::abs(res_mc.price - price_f);
  double rmse_mc = std::sqrt(bias_mc * bias_mc +
                             res_mc.price_std_err * res_mc.price_std_err);

  std::cout << "[STANDARD MC (" << mc_cfg.num_paths << " paths, "
            << mc_cfg.time_steps << " steps/path)]\n";
  std::cout << "  Price: " << res_mc.price
            << "  (StdErr: " << res_mc.price_std_err << ")\n";
  std::cout << "  Bias : " << bias_mc
            << "  (Z-Scr: " << bias_mc / res_mc.price_std_err << ")\n";
  std::cout << "  RMSE : " << rmse_mc << "\n";
  std::cout << "  Steps: " << mc_cfg.num_paths * mc_cfg.time_steps << "\n";
  std::cout << "  Time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_mc -
                                                                     start_mc)
                   .count()
            << " ms\n\n";

  // 3. Multi-Level Monte Carlo
  MultiLevelMonteCarloPricer<Model, CoupledStepper, Instrument> mlmc_p(
      model, mlmc_cfg);
  auto start_m = std::chrono::high_resolution_clock::now();
  auto res_m = mlmc_p.calculate(S0, r, q, opt);
  auto end_m = std::chrono::high_resolution_clock::now();

  double bias_m = std::abs(res_m.price - price_f);
  double rmse_m =
      std::sqrt(bias_m * bias_m + res_m.price_std_err * res_m.price_std_err);

  std::cout << "[MLMC (Target RMSE: " << mlmc_cfg.epsilon << ")]\n";
  std::cout << "  Price: " << res_m.price
            << "  (StdErr: " << res_m.price_std_err << ")\n";
  std::cout << "  Bias : " << bias_m
            << "  (Z-Scr: " << bias_m / res_m.price_std_err << ")\n";
  std::cout << "  RMSE : " << rmse_m << "\n";
  std::cout << "  Steps: " << res_m.total_steps << "\n";
  std::cout << "  Estim: Alpha=" << res_m.alpha_estimated
            << ", Beta=" << res_m.beta_estimated << "\n";
  std::cout << "  Time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_m -
                                                                     start_m)
                   .count()
            << " ms\n";

  std::cout << "  Level Paths:";
  for (auto n : res_m.paths_per_level)
    std::cout << " " << n;
  std::cout << "\n\n";
}

template <typename Model, typename Stepper, typename CoupledStepper,
          typename Instrument>
void run_benchmark_no_fourier(const std::string &name, const Model &model,
                              const Instrument &opt, double S0, double r,
                              double q, const MonteCarloConfig &mc_cfg,
                              const MLMCConfig &mlmc_cfg) {

  std::cout << "*****************************************\n";
  std::cout << "  " << name << "\n";
  std::cout << "*****************************************\n";

  // 1. Standard Monte Carlo (Used as Reference)
  MonteCarloPricer<Model, Stepper, Instrument> mc_p(model, mc_cfg);
  auto start_mc = std::chrono::high_resolution_clock::now();
  auto res_mc = mc_p.calculate(S0, r, q, opt);
  auto end_mc = std::chrono::high_resolution_clock::now();

  std::cout << "[STANDARD MC (" << mc_cfg.num_paths << " paths, "
            << mc_cfg.time_steps << " steps/path)]\n";
  std::cout << "  Price: " << std::fixed << std::setprecision(6) << res_mc.price
            << "  (StdErr: " << res_mc.price_std_err << ")\n";
  std::cout << "  Steps: " << mc_cfg.num_paths * mc_cfg.time_steps << "\n";
  std::cout << "  Time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_mc -
                                                                     start_mc)
                   .count()
            << " ms\n\n";

  // 2. Multi-Level Monte Carlo
  MultiLevelMonteCarloPricer<Model, CoupledStepper, Instrument> mlmc_p(
      model, mlmc_cfg);
  auto start_m = std::chrono::high_resolution_clock::now();
  auto res_m = mlmc_p.calculate(S0, r, q, opt);
  auto end_m = std::chrono::high_resolution_clock::now();

  double bias_m = std::abs(res_m.price - res_mc.price);
  double combined_stderr =
      std::sqrt(res_m.price_std_err * res_m.price_std_err +
                res_mc.price_std_err * res_mc.price_std_err);

  std::cout << "[MLMC (Target RMSE: " << mlmc_cfg.epsilon << ")]\n";
  std::cout << "  Price: " << res_m.price
            << "  (StdErr: " << res_m.price_std_err << ")\n";
  std::cout << "  Difference from MC: " << bias_m
            << "  (Z-Scr: " << bias_m / combined_stderr << ")\n";
  std::cout << "  Steps: " << res_m.total_steps << "\n";
  std::cout << "  Estim: Alpha=" << res_m.alpha_estimated
            << ", Beta=" << res_m.beta_estimated << "\n";
  std::cout << "  Time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_m -
                                                                     start_m)
                   .count()
            << " ms\n";

  std::cout << "  Level Paths:";
  for (auto n : res_m.paths_per_level)
    std::cout << " " << n;
  std::cout << "\n\n";
}

int main() {
  double S0 = 100.0, r = 0.05, q = 0.02, T = 1.0, K = 100.0;

  MonteCarloConfig mc_cfg;
  mc_cfg.num_paths = 20000000;
  mc_cfg.time_steps = 32;
  mc_cfg.seed = 42;

  MLMCConfig mlmc_cfg;
  mlmc_cfg.epsilon = 0.005;
  mlmc_cfg.L_min = 2;
  mlmc_cfg.L_max = 8;
  mlmc_cfg.N0 = 10000;
  mlmc_cfg.base_steps = 1;
  mlmc_cfg.seed = 421;

  using EuropeanCall = EuropeanOption<PayoffVanilla<OptionType::Call>>;
  EuropeanCall opt(K, T);

  // --- CASE 1: HESTON ---
  {
    HestonParams hp = {2.0, 0.04, 0.4, -0.7, 0.04};
    HestonModel model(hp);
    using Stepper = HestonStepper<SchemeNV>;
    using CStepper = CoupledHestonStepper<SchemeNV>;
    run_benchmark<HestonModel, Stepper, CStepper, EuropeanCall>(
        "HESTON MODEL", model, opt, S0, r, q, mc_cfg, mlmc_cfg);
  }

  // --- CASE 2: BATES ---
  {
    HestonParams hp = {2.0, 0.04, 0.4, -0.7, 0.04};
    MertonParams jp = {0.1, -0.05, 0.15}; // lambda, mu, delta
    BatesModel model(hp, jp);
    using Stepper = BatesStepper<SchemeNV>;
    using CStepper = CoupledBatesStepper<SchemeNV>;
    run_benchmark<BatesModel, Stepper, CStepper, EuropeanCall>(
        "BATES MODEL (HESTON + JUMPS)", model, opt, S0, r, q, mc_cfg, mlmc_cfg);
  }

  // --- CASE 3: DOUBLE HESTON ---
  {
    HestonParams hp1 = {2.0, 0.04, 0.3, -0.6, 0.04};
    HestonParams hp2 = {1.5, 0.04, 0.2, -0.5, 0.04};
    DoubleHestonModel model(hp1, hp2);
    using Stepper = DoubleHestonStepper<SchemeNV, SchemeNV>;
    using CStepper = CoupledDoubleHestonStepper<SchemeNV, SchemeNV>;
    run_benchmark<DoubleHestonModel, Stepper, CStepper, EuropeanCall>(
        "DOUBLE HESTON (2 FACTORS)", model, opt, S0, r, q, mc_cfg, mlmc_cfg);
  }

  std::cout << "\n"
            << "=================================================" << "\n";
  std::cout << "      PATH-DEPENDENT BENCHMARKS -- ASIAN         " << "\n";
  std::cout << "=================================================" << "\n\n";

  MonteCarloConfig mc_cfg_asian;
  mc_cfg_asian.num_paths = 20000000;
  mc_cfg_asian.time_steps = 32;
  mc_cfg_asian.seed = 42;

  MLMCConfig mlmc_cfg_asian;
  mlmc_cfg_asian.epsilon = 0.005;
  mlmc_cfg_asian.L_min = 2;
  mlmc_cfg_asian.L_max = 8;
  mlmc_cfg_asian.N0 = 10000;
  mlmc_cfg_asian.base_steps = 1;
  mlmc_cfg_asian.seed = 421;

  // --- CASE 4: ARITHMETIC ASIAN HESTON ---
  {
    HestonParams hp = {2.0, 0.04, 0.4, -0.7, 0.04};
    HestonModel model(hp);
    using AsianCall =
        AsianOption<TrackerArithAsian, PayoffVanilla<OptionType::Call>, true>;
    AsianCall asian_opt(K, T);

    using Stepper = HestonStepper<SchemeNV, TrackerArithAsian>;
    using CStepper = CoupledHestonStepper<SchemeNV, TrackerArithAsian>;
    run_benchmark_no_fourier<HestonModel, Stepper, CStepper, AsianCall>(
        "ARITHMETIC ASIAN (HESTON)", model, asian_opt, S0, r, q, mc_cfg_asian,
        mlmc_cfg_asian);
  }

  // --- CASE 5: ARITHMETIC ASIAN BATES-KOU ---
  {
    HestonParams hp = {2.0, 0.04, 0.4, -0.7, 0.04};
    KouParams kp = {0.1, 0.3, 10.0,
                    10.0}; // lambda, p_up, eta1, eta2 (eta must > 1)
    BatesKouModel model(hp, kp);

    using AsianCall =
        AsianOption<TrackerArithAsian, PayoffVanilla<OptionType::Call>, true>;
    AsianCall asian_opt(K, T);

    using Stepper = BatesKouStepper<SchemeNV, TrackerArithAsian>;
    using CStepper = CoupledBatesKouStepper<SchemeNV, TrackerArithAsian>;
    run_benchmark_no_fourier<BatesKouModel, Stepper, CStepper, AsianCall>(
        "ARITHMETIC ASIAN (BATES-KOU)", model, asian_opt, S0, r, q,
        mc_cfg_asian, mlmc_cfg_asian);
  }

  // --- CASE 6: ARITHMETIC ASIAN DOUBLE HESTON ---
  {
    HestonParams hp1 = {4.0, 0.04, 0.5, -0.6, 0.04};
    HestonParams hp2 = {1.5, 0.04, 0.2, -0.5, 0.04};
    DoubleHestonModel model(hp1, hp2);

    using AsianCall =
        AsianOption<TrackerArithAsian, PayoffVanilla<OptionType::Call>, true>;
    AsianCall asian_opt(K, T);

    using Stepper = DoubleHestonStepper<SchemeNV, SchemeNV, TrackerArithAsian>;
    using CStepper =
        CoupledDoubleHestonStepper<SchemeNV, SchemeNV, TrackerArithAsian>;
    run_benchmark_no_fourier<DoubleHestonModel, Stepper, CStepper, AsianCall>(
        "ARITHMETIC ASIAN (DOUBLE HESTON)", model, asian_opt, S0, r, q,
        mc_cfg_asian, mlmc_cfg_asian);
  }

  std::cout << "\n"
            << "=================================================" << "\n";
  std::cout << "      PATH-DEPENDENT BENCHMARKS -- LOOKBACK        " << "\n";
  std::cout << "=================================================" << "\n\n";

  MonteCarloConfig mc_cfg_lookback;
  mc_cfg_lookback.num_paths = 3000000;
  mc_cfg_lookback.time_steps = 2048;
  mc_cfg_lookback.seed = 42;

  MLMCConfig mlmc_cfg_lookback;
  mlmc_cfg_lookback.epsilon = 0.01;
  mlmc_cfg_lookback.L_min = 4;
  mlmc_cfg_lookback.L_max = 11;
  mlmc_cfg_lookback.N0 = 10000;
  mlmc_cfg_lookback.base_steps = 1;
  mlmc_cfg_lookback.seed = 42;

  // --- CASE 7: ARITHMETIC ASIAN HESTON ---
  {
    HestonParams hp = {2.0, 0.04, 0.4, -0.7, 0.04};
    HestonModel model(hp);
    using LookbackPut = LookbackOption<PayoffVanilla<OptionType::Put>, false>;
    LookbackPut lookback_opt(K, T);

    using Stepper = HestonStepper<SchemeNV, TrackerLookback>;
    using CStepper = CoupledHestonStepper<SchemeNV, TrackerLookback>;
    run_benchmark_no_fourier<HestonModel, Stepper, CStepper, LookbackPut>(
        "FLOATING LOOKBACK PUT (HESTON)", model, lookback_opt, S0, r, q,
        mc_cfg_lookback, mlmc_cfg_lookback);
  }

  // --- CASE 8: LOOKBACK BATES-KOU ---
  {
    HestonParams hp = {2.0, 0.04, 0.4, -0.7, 0.04};
    KouParams kp = {0.1, 0.3, 10.0,
                    10.0}; // lambda, p_up, eta1, eta2 (eta must > 1)
    BatesKouModel model(hp, kp);

    using LookbackPut = LookbackOption<PayoffVanilla<OptionType::Put>, false>;
    LookbackPut lookback_opt(K, T);

    using Stepper = BatesKouStepper<SchemeNV, TrackerLookback>;
    using CStepper = CoupledBatesKouStepper<SchemeNV, TrackerLookback>;
    run_benchmark_no_fourier<BatesKouModel, Stepper, CStepper, LookbackPut>(
        "FLOATING LOOKBACK PUT (BATES-KOU)", model, lookback_opt, S0, r, q,
        mc_cfg_lookback, mlmc_cfg_lookback);
  }

  // --- CASE 9: LOOKBACK DOUBLE HESTON ---
  {
    HestonParams hp1 = {4.0, 0.04, 0.5, -0.6, 0.04};
    HestonParams hp2 = {1.5, 0.04, 0.2, -0.5, 0.04};
    DoubleHestonModel model(hp1, hp2);

    using LookbackPut = LookbackOption<PayoffVanilla<OptionType::Put>, false>;
    LookbackPut lookback_opt(K, T);

    using Stepper = DoubleHestonStepper<SchemeNV, SchemeNV, TrackerLookback>;
    using CStepper =
        CoupledDoubleHestonStepper<SchemeNV, SchemeNV, TrackerLookback>;
    run_benchmark_no_fourier<DoubleHestonModel, Stepper, CStepper, LookbackPut>(
        "FLOATING LOOKBACK PUT (DOUBLE HESTON)", model, lookback_opt, S0, r, q,
        mc_cfg_lookback, mlmc_cfg_lookback);
  }

  std::cout << "\n"
            << "=================================================" << "\n";
  std::cout << "      PATH-DEPENDENT -- BARRIER        " << "\n";
  std::cout << "=================================================" << "\n\n";

  MonteCarloConfig mc_cfg_barrier;
  mc_cfg_barrier.num_paths = 5000000;
  mc_cfg_barrier.time_steps = 256;
  mc_cfg_barrier.seed = 42;

  MLMCConfig mlmc_cfg_barrier;
  mlmc_cfg_barrier.epsilon = 0.01;
  mlmc_cfg_barrier.L_min = 2;
  mlmc_cfg_barrier.L_max = 8;
  mlmc_cfg_barrier.N0 = 10000;
  mlmc_cfg_barrier.base_steps = 1;
  mlmc_cfg_barrier.seed = 421;

  double B = 110.0; // Barrier

  // --- CASE 10: UP-AND-OUT CALL (HESTON) ---
  {
    HestonParams hp = {2.0, 0.04, 0.4, -0.7, 0.04};
    HestonModel model(hp);
    using UAOCall = UpAndOutCall<>;
    UAOCall opt(K, B, T);

    using Stepper = HestonStepper<SchemeNV, UAOCall::Tracker>;
    using CStepper = CoupledHestonStepper<SchemeNV, UAOCall::Tracker>;
    run_benchmark_no_fourier<HestonModel, Stepper, CStepper, UAOCall>(
        "UP-AND-OUT CALL (HESTON)", model, opt, S0, r, q, mc_cfg_barrier,
        mlmc_cfg_barrier);
  }

  // --- CASE 11: UP-AND-OUT PUT (BATES-KOU) ---
  {
    HestonParams hp = {2.0, 0.04, 0.4, -0.7, 0.04};
    KouParams kp = {0.1, 0.3, 10.0, 10.0};
    BatesKouModel model(hp, kp);
    using UAOPut = UpAndOutPut<>;
    UAOPut opt(K, B, T);

    using Stepper = BatesKouStepper<SchemeNV, UAOPut::Tracker>;
    using CStepper = CoupledBatesKouStepper<SchemeNV, UAOPut::Tracker>;
    run_benchmark_no_fourier<BatesKouModel, Stepper, CStepper, UAOPut>(
        "UP-AND-OUT PUT (BATES-KOU)", model, opt, S0, r, q, mc_cfg_barrier,
        mlmc_cfg_barrier);
  }

  // --- CASE 12: UP-AND-IN CALL (DOUBLE HESTON) ---
  {
    HestonParams hp1 = {4.0, 0.04, 0.5, -0.6, 0.04};
    HestonParams hp2 = {1.5, 0.04, 0.2, -0.5, 0.04};
    DoubleHestonModel model(hp1, hp2);
    using UAICall = UpAndInCall<>;
    UAICall opt(K, B, T);

    using Stepper = DoubleHestonStepper<SchemeNV, SchemeNV, UAICall::Tracker>;
    using CStepper =
        CoupledDoubleHestonStepper<SchemeNV, SchemeNV, UAICall::Tracker>;
    run_benchmark_no_fourier<DoubleHestonModel, Stepper, CStepper, UAICall>(
        "UP-AND-IN CALL (DOUBLE HESTON)", model, opt, S0, r, q, mc_cfg_barrier,
        mlmc_cfg_barrier);
  }

  return 0;
}
