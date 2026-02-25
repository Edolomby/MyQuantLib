#include <cmath>
#include <iostream>
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/Asian.hpp>
#include <myql/instruments/options/European.hpp>
#include <myql/instruments/options/Lookback.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/pricers/fourier/FourierPricer.hpp>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>
#include <vector>

using namespace myql;

// Type alias needed for Example 3 to mirror Example 1
using CallVanilla = EuropeanOption<PayoffVanilla<OptionType::Call>>;

void example1() {
  std::cout << "\n--- Example 1 ---\n";
  // 1. Define Model Parameters (Heston)
  HestonParams heston_params = {2.5, 0.06, 0.4, -0.7,
                                0.05}; // kappa, theta, sigma, rho, v0
  HestonModel model(heston_params);

  // 2. Define the Instrument
  double strike = 100.0, maturity = 1.0;
  CallVanilla option(strike, maturity);

  double spot = 100.0, rate = 0.05, dividend = 0.02;

  // 3. Monte Carlo Pricing
  MonteCarloConfig mc_cfg;
  mc_cfg.num_paths = 100000;
  mc_cfg.time_steps = 100;

  using Stepper =
      ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerEuropean>;
  MonteCarloPricer<HestonModel, Stepper, CallVanilla> mc_pricer(model, mc_cfg);
  auto mc_res = mc_pricer.calculate(spot, rate, dividend, option);

  // 4. Fourier Pricing (Analytical)
  FourierPricer<HestonModel, CallVanilla> fourier_pricer(model);
  auto fourier_res = fourier_pricer.calculate(spot, rate, dividend, option);

  // 5. Compare Results (Z-Score)
  double z_score =
      std::abs(mc_res.price - fourier_res.price) / mc_res.price_std_err;

  std::cout << "MC Price:      " << mc_res.price << " ± "
            << mc_res.price_std_err << "\n";
  std::cout << "Fourier Price: " << fourier_res.price << "\n";
  std::cout << "Z-Score:       " << z_score << "\n";
}

void example2() {
  std::cout << "\n--- Example 2 ---\n";
  HestonModel model({2.5, 0.06, 0.4, -0.7, 0.05});
  MonteCarloConfig mc_cfg{100000, 100, 42};
  double S0 = 100.0, r = 0.05, q = 0.0;

  // A. Strip of Geometric Asian Calls (Strikes: 90, 100, 110)
  std::vector<double> strikes = {S0 * 0.9, S0, S0 * 1.1};
  using GeoAsianStrip =
      AsianOption<TrackerGeoAsian, PayoffVanilla<OptionType::Call>, true,
                  std::vector<double>>;
  GeoAsianStrip asian_strip(strikes, 1.0);

  using AsianStepper =
      ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerGeoAsian>;
  MonteCarloPricer<HestonModel, AsianStepper, GeoAsianStrip> asian_pricer(
      model, mc_cfg);

  auto asian_res = asian_pricer.calculate(S0, r, q, asian_strip);
  std::cout << "Asian Call (K=90):  " << asian_res.price[0] << "\n"
            << "Asian Call (K=100): " << asian_res.price[1] << "\n"
            << "Asian Call (K=110): " << asian_res.price[2] << "\n";

  // B. Floating Strike Lookback Put: Payoff = max(S_max - S_T, 0)
  double dummy_strike = 0.0;
  using LookbackPut =
      LookbackOption<PayoffVanilla<OptionType::Put>, false, double>;
  LookbackPut lookback(dummy_strike, 1.0);

  using LookbackStepper =
      ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerLookback>;
  MonteCarloPricer<HestonModel, LookbackStepper, LookbackPut> lb_pricer(model,
                                                                        mc_cfg);

  auto lb_res = lb_pricer.calculate(S0, r, q, lookback);
  std::cout << "Floating Lookback Put: " << lb_res.price << "\n";
}

// Assumes 'model' and 'option' from Example 1
void calculate_greeks(const HestonModel &model, const CallVanilla &option) {
  std::cout << "\n--- Example 3 ---\n";
  double S0 = 100.0, r = 0.05, q = 0.02;
  MonteCarloConfig mc_cfg{200000, 100, 42};

  // Calculate MC Greeks
  using Stepper =
      ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerEuropean>;
  MonteCarloPricer<HestonModel, Stepper, CallVanilla, GreekMode::Essential>
      mc_pricer(model, mc_cfg);
  auto res_mc = mc_pricer.calculate(S0, r, q, option);

  std::cout << "MC Delta: " << res_mc.delta << " ± " << res_mc.delta_std_err
            << "\n";
  std::cout << "MC Gamma: " << res_mc.gamma << " ± " << res_mc.gamma_std_err
            << "\n";

  // Calculate Analytical Fourier Greeks
  FourierPricer<HestonModel, CallVanilla, GreekMode::Essential> fourier_pricer(
      model);
  auto res_fourier = fourier_pricer.calculate(S0, r, q, option);

  std::cout << "Analytical Delta: " << res_fourier.delta << "\n";
  std::cout << "Analytical Gamma: " << res_fourier.gamma << "\n";
}

int main() {
  example1();
  example2();

  HestonModel model({2.5, 0.06, 0.4, -0.7, 0.05});
  CallVanilla option(100.0, 1.0);
  calculate_greeks(model, option);

  return 0;
}
