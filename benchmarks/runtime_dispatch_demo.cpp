#include <iomanip>
#include <iostream>

#include <myql/dispatcher/InstrumentRegistry.hpp>
#include <myql/dispatcher/ModelRegistry.hpp>
#include <myql/dispatcher/PricingDispatch.hpp>

using namespace myql;
using namespace myql::dispatcher;

// A dummy JSON-like config parsing to demonstrate the power
void run_pricing_task(const std::string &model_str,
                      const std::string &instr_str,
                      const std::string &option_type_str, double S0, double r,
                      double q, const std::vector<double> &strikes, double T) {

  std::cout << "========================================================\n";
  std::cout << "DYNAMIC PRICING TASK: \n";
  std::cout << "Model: " << model_str << " | Instrument: " << instr_str << " "
            << option_type_str << "\n";
  std::cout << "========================================================\n";

  // 1. Setup raw parameter structs (usually decoded from JSON)
  RuntimeModelParams m_params;
  m_params.heston1 = {1.5, 0.04, 0.3, -0.6,
                      0.04};          // kappa, theta, sigma, rho, v0
  m_params.merton = {0.1, -0.1, 0.2}; // lambda, mu, delta
  m_params.vol = 0.2;                 // for BlackScholes

  RuntimeInstrumentParams i_params;
  i_params.strikes = strikes;
  i_params.maturity = T;

  // 2. DISPATCH REGISTRY: Build Variants at Runtime
  AnyModel model = build_model(model_str, m_params);
  AnyMCInstrument mc_instr =
      build_instrument(instr_str, option_type_str, i_params);
  AnyEuropeanInstrument f_instr =
      build_european(instr_str, option_type_str, i_params);

  // 3. Configure Engines
  MonteCarloConfig mc_cfg;
  mc_cfg.num_paths = 100000;
  mc_cfg.time_steps = 50;
  mc_cfg.fd_bump = 1e-4;

  FourierEngine::Config f_cfg;
  f_cfg.tolerance = 1e-8;

  // 4. PRICE! (The magic double-dispatch mapping to compiled temples)
  // We explicitly ask to compute Essential Greeks via Monte Carlo
  auto mc_res = price_mc<GreekMode::Essential, SchemeNCI>(model, mc_instr,
                                                          mc_cfg, S0, r, q);
  auto f_res =
      price_fourier<GreekMode::Essential>(model, f_instr, f_cfg, S0, r, q);

  // 5. Print Results
  std::cout << "\nResults for " << strikes.size() << " strikes:\n";
  std::cout << std::setw(10) << "Strike" << " | " << std::setw(15) << "MC Price"
            << " | " << std::setw(15) << "Fourier Price" << " | "
            << std::setw(15) << "MC Delta" << " | " << std::setw(15)
            << "Fourier Delta" << "\n";
  std::cout << std::string(85, '-') << "\n";

  for (size_t i = 0; i < strikes.size(); ++i) {
    std::cout << std::setw(10) << strikes[i] << " | " << std::setw(15)
              << mc_res.prices[i] << " | " << std::setw(15) << f_res.prices[i]
              << " | " << std::setw(15) << mc_res.deltas[i] << " | "
              << std::setw(15) << f_res.deltas[i] << "\n";
  }
  std::cout << "\n";
}

#include <chrono>

int main() {
  double S0 = 100.0, r = 0.05, q = 0.0;
  std::vector<double> strikes = {80.0, 90.0, 100.0, 110.0, 120.0};
  double T = 1.0;

  // pure string-based instantiation!
  try {
    std::cout << "\n=== FUNCTIONAL TEST ===\n";
    run_pricing_task("Heston", "Vanilla", "Call", S0, r, q, strikes, T);
    run_pricing_task("Merton", "CashOrNothing", "Put", S0, r, q, strikes, T);

    std::cout << "\n=== PERFORMANCE TEST (Static vs Dynamic) ===\n";
    std::cout << "Pricing 10,000 strips of options to measure overhead...\n\n";

    // 1. Setup Static
    HestonParams hp = {1.5, 0.04, 0.3, -0.6, 0.04};
    HestonModel static_model(hp);
    EuropeanOption<PayoffVanilla<OptionType::Call>, std::vector<double>>
        static_instr(strikes, T);
    FourierEngine::Config f_cfg;
    f_cfg.tolerance = 1e-8;

    auto t1 = std::chrono::high_resolution_clock::now();
    double static_sum = 0.0;
    for (int i = 0; i < 10000; ++i) {
      FourierPricer<HestonModel, decltype(static_instr), GreekMode::Essential>
          pricer(static_model, f_cfg);
      auto res = pricer.calculate(S0, r, q, static_instr);
      static_sum += res.price[1];
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> static_ms = t2 - t1;

    // 2. Setup Dynamic
    RuntimeModelParams m_params;
    m_params.heston1 = hp;
    RuntimeInstrumentParams i_params;
    i_params.strikes = strikes;
    i_params.maturity = T;

    AnyModel dynamic_model = build_model("Heston", m_params);
    AnyEuropeanInstrument dynamic_instr =
        build_european("Vanilla", "Call", i_params);

    auto t3 = std::chrono::high_resolution_clock::now();
    double dynamic_sum = 0.0;
    for (int i = 0; i < 10000; ++i) {
      auto res = price_fourier<GreekMode::Essential>(
          dynamic_model, dynamic_instr, f_cfg, S0, r, q);
      dynamic_sum += res.prices[1];
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dynamic_ms = t4 - t3;

    std::cout << "Static Template Time: " << static_ms.count() << " ms\n";
    std::cout << "Dynamic Dispatch Time: " << dynamic_ms.count() << " ms\n";
    std::cout << "Static Sum: " << static_sum
              << " | Dynamic Sum: " << dynamic_sum << "\n";
    double diff_ms = dynamic_ms.count() - static_ms.count();
    std::cout << "Difference: " << std::abs(diff_ms) << " ms ("
              << (diff_ms > 0 ? "Dynamic is slower" : "Dynamic is faster")
              << ")\n";

  } catch (const std::exception &e) {
    std::cerr << "Fatal Error: " << e.what() << "\n";
  }

  return 0;
}
