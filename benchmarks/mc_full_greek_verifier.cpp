// Verification benchmark: Compare Monte Carlo Full Greeks to Fourier Full
// Greeks
#include <cmath>
#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/European.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/models/asvj/data/ModelParams.hpp>
#include <myql/pricers/fourier/FourierPricer.hpp>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>
#include <myql/utils/TablePrinter.hpp>
#include <string>
#include <vector>

static constexpr double S0 = 100.0;
static constexpr double K = 100.0;
static constexpr double T = 1.0;
static constexpr double r = 0.05;
static constexpr double q = 0.02;

double rel_err(double mc, double ana) {
  return std::abs(mc - ana) / (std::abs(ana) + 1e-10);
}

template <typename ModelT, typename StepperT>
void compare_mc_vs_fourier(const char *label, const ModelT &model) {
  using InstrT = EuropeanOption<PayoffVanilla<OptionType::Call>>;
  InstrT opt(K, T);

  // 1. Fourier Baseline
  FourierEngine::Config f_cfg;
  f_cfg.tolerance = 1e-10;
  FourierPricer<ModelT, InstrT, GreekMode::Full> f_pricer(model, f_cfg);
  auto f_full = f_pricer.calculate(S0, r, q, opt);

  // 2. Monte Carlo Setup
  MonteCarloConfig mc_cfg;
  mc_cfg.num_paths = 500000;
  mc_cfg.time_steps = 100;
  mc_cfg.seed = 42;
  mc_cfg.fd_bump = 1e-4;
  mc_cfg.vol_bump = 1e-4;
  mc_cfg.t_bump = 1e-4;
  mc_cfg.r_bump = 1e-4;

  MonteCarloPricer<ModelT, StepperT, InstrT, GreekMode::Full> mc_pricer(model,
                                                                        mc_cfg);
  auto m_full = mc_pricer.calculate(S0, r, q, opt);

  std::vector<std::string> metrics;
  std::vector<double> fourier_vals;
  std::vector<double> mc_vals;
  std::vector<double> rel_errs;
  std::vector<double> stderrs;
  std::vector<double> biases;
  std::vector<double> zscores;
  std::vector<std::string> statuses;

  auto add_row = [&](const std::string &name, double fval, double mval,
                     double err) {
    metrics.push_back(name);
    fourier_vals.push_back(fval);
    mc_vals.push_back(mval);
    stderrs.push_back(err);

    double rel = 100.0 * rel_err(mval, fval);
    rel_errs.push_back(rel);

    double bias = mval - fval;
    biases.push_back(bias);

    double z = (err > 1e-12) ? (bias / err) : 0.0;
    zscores.push_back(z);

    if (std::abs(z) > 4.0)
      statuses.push_back("[ERROR]");
    else if (std::abs(z) > 2.0)
      statuses.push_back("[WARN]");
    else
      statuses.push_back("OK");
  };

  add_row("Price", f_full.price, m_full.price, m_full.price_std_err);
  add_row("Delta", f_full.delta, m_full.delta, m_full.delta_std_err);
  add_row("Gamma", f_full.gamma, m_full.gamma, m_full.gamma_std_err);
  add_row("Vega[0]", f_full.vega[0], m_full.vega[0], m_full.vega_std_err[0]);
  if constexpr (ModelT::num_variance_factors >= 2) {
    add_row("Vega[1]", f_full.vega[1], m_full.vega[1], m_full.vega_std_err[1]);
  }
  add_row("Theta", f_full.theta, m_full.theta, m_full.theta_std_err);
  add_row("Rho", f_full.rho, m_full.rho, m_full.rho_std_err);

  std::printf("\n=============================================\n");
  std::printf("  %s - MC (500k paths) vs Fourier\n", label);
  std::printf("=============================================\n");

  myql::utils::printVectors(
      myql::utils::TableConfig{myql::utils::FloatFormat::Fixed, 6},
      {"Metric", "Fourier", "Monte Carlo", "Rel Err %", "Bias", "MC StdErr",
       "Z-Score", "Status"},
      metrics, fourier_vals, mc_vals, rel_errs, biases, stderrs, zscores,
      statuses);
}

int main() {
  HestonParams h1 = {2.0, 0.04, 0.4, -0.7, 0.04};
  HestonParams h2 = {3.0, 0.02, 0.2, -0.3, 0.02};
  KouParams jk = {3.0, 0.3, 25.0, 15.0}; // lambda, p, eta1, eta2

  // 1. Single Factor (Heston)
  HestonModel heston(h1);
  using HestonStep = HestonStepper<SchemeNV>;
  compare_mc_vs_fourier<HestonModel, HestonStep>("Heston", heston);

  // 2. Double Factor (Double Heston)
  DoubleHestonModel dheston(h1, h2);
  using DoubleHestonStep = DoubleHestonStepper<SchemeNV, SchemeNV>;
  compare_mc_vs_fourier<DoubleHestonModel, DoubleHestonStep>("Double Heston",
                                                             dheston);

  // 3. Single Factor with Jumps (Bates-Kou)
  BatesKouModel bateskou(h1, jk);
  using BatesKouStep =
      ASVJStepper<SchemeNV, NullVolScheme, KouJump, TrackerEuropean>;
  compare_mc_vs_fourier<BatesKouModel, BatesKouStep>("Bates-Kou", bateskou);

  // 4. Double Factor with Jumps (Double Bates-Kou)
  DoubleFactorModel<KouParams> dbateskou(h1, h2, jk);
  using DBatesKouStep =
      ASVJStepper<SchemeNV, SchemeNV, KouJump, TrackerEuropean>;
  compare_mc_vs_fourier<DoubleFactorModel<KouParams>, DBatesKouStep>(
      "Double Bates-Kou", dbateskou);

  return 0;
}
