#include <chrono>
#include <iostream>
#include <string>
#include <vector>

// PRICERS & MODELS
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/models/asvj/policies/JumpPolicies.hpp>
#include <myql/models/asvj/policies/VolSchemes.hpp>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>

// INSTRUMENTS
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/European.hpp>

// UTILS
#include <myql/utils/TablePrinter.hpp>

// -----------------------------------------------------------------------------
// DATA STORAGE FOR TABLE PRINTER
// -----------------------------------------------------------------------------
struct PerfResults {
  std::vector<std::string> model_names;
  std::vector<double> maturities;
  std::vector<double> n_strikes;
  std::vector<double> time_std;   // Standard MC (seconds)
  std::vector<double> time_strip; // Vectorized MC (seconds)
  std::vector<double> speedup;    // Factor (x)
};

PerfResults perf_data;

// -----------------------------------------------------------------------------
// BENCHMARK LOGIC
// -----------------------------------------------------------------------------
template <typename Model, typename Stepper>
void run_benchmark(const std::string &label, const Model &model,
                   const MonteCarloConfig &cfg) {

  using namespace std::chrono;
  double S0 = 100.0, r = 0.03, q = 0.01, T = 1.0;

  // We will test with a strip of 20 strikes
  std::vector<double> strikes;
  for (int i = 0; i < 20; ++i)
    strikes.push_back(80.0 + i * 2.0);

  // --- 1. STANDARD APPROACH (K individual calls to the Engine) ---
  auto start_std = high_resolution_clock::now();
  for (double K : strikes) {
    using Inst = EuropeanOption<PayoffVanilla<OptionType::Call>>;
    Inst opt(K, T);
    MonteCarloPricer<Model, Stepper, Inst> engine(model, cfg);
    auto res = engine.calculate(S0, r, q, opt);
    [[maybe_unused]] volatile double dummy = res.price; // Prevent optimization
  }
  auto end_std = high_resolution_clock::now();
  double t_std = duration<double>(end_std - start_std).count();

  // --- 2. STRIP APPROACH (1 vectorized call using the Buffer Pattern) ---
  auto start_strip = high_resolution_clock::now();
  using Strip =
      EuropeanOption<PayoffVanilla<OptionType::Call>, std::vector<double>>;
  Strip strip(strikes, T);
  MonteCarloPricer<Model, Stepper, Strip> engine_strip(model, cfg);
  auto res_strip = engine_strip.calculate(S0, r, q, strip);
  [[maybe_unused]] volatile double dummy2 =
      res_strip.price[0]; // Prevent optimization
  auto end_strip = high_resolution_clock::now();
  double t_strip = duration<double>(end_strip - start_strip).count();

  // --- STORE DATA ---
  perf_data.model_names.push_back(label);
  perf_data.maturities.push_back(T);
  perf_data.n_strikes.push_back(static_cast<double>(strikes.size()));
  perf_data.time_std.push_back(t_std);
  perf_data.time_strip.push_back(t_strip);
  perf_data.speedup.push_back(t_std / t_strip);
}

int main() {
  std::cout
      << "==================================================================\n";
  std::cout << "  PERFORMANCE COMPARISON: STANDARD vs STRIP (VECTORIZED) MC\n";
  std::cout
      << "==================================================================\n";

  MonteCarloConfig cfg;
  cfg.num_paths = 1000000; // High paths to get stable timings
  cfg.time_steps = 10;

  // kappa,theta,sigma,rho,v0;
  HestonParams h1 = {2.5, 0.06, 0.4, -0.7, 0.05}; // kappa,theta,sigma,rho,v0
  MertonParams m_jmp = {1.2, -0.12, 0.15};

  // Run Benchmarks
  {
    HestonModel m(h1);
    using Stepper =
        ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerEuropean>;
    run_benchmark<HestonModel, Stepper>("Heston", m, cfg);
  }
  {
    BatesModel m(h1, m_jmp);
    using Stepper =
        ASVJStepper<SchemeNV, NullVolScheme, MertonJump, TrackerEuropean>;
    run_benchmark<BatesModel, Stepper>("Bates", m, cfg);
  }
  {
    DoubleBatesModel m(h1, h1, m_jmp);
    using Stepper =
        ASVJStepper<SchemeExact, SchemeExact, MertonJump, TrackerEuropean>;
    run_benchmark<DoubleBatesModel, Stepper>("DoubleBates", m, cfg);
  }

  // --- FINAL PRINTING USING TablePrinter ---
  myql::utils::printVectors({"Model", "T", "N-Strikes", "Std Time(s)",
                             "Strip Time(s)", "Speedup (x)"},
                            perf_data.model_names, perf_data.maturities,
                            perf_data.n_strikes, perf_data.time_std,
                            perf_data.time_strip, perf_data.speedup);

  return 0;
}