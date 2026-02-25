// =============================================================================
// LOOKBACK OPTION TESTS — Black-Scholes Verification
// =============================================================================
// Tests Monte Carlo pricing of Lookback options against:
//   (1) Goldman-Sosin-Gatto (1979) closed-form formulas (floating-strike)
//   (2) Model-independent bounds for fixed-strike variants
//
// Reference:
//   Goldman, M.B., Sosin, H.B., Gatto, M.A. (1979).
//   "Path Dependent Options: 'Buy at the Low, Sell at the High'"
//   Journal of Finance, 34(5), 1111-1127.
//
//   Haug, E.G. (2007). "The Complete Guide to Option Pricing Formulas",
//   2nd ed., McGraw-Hill, Section 2.9.
//
// Models tested: Black-Scholes (ZeroFactorModel / SchemeNV — no vol process)
// Schemes:       SchemeNCI (via StepperTraits automatic deduction)
// Steps:         252 (daily) and 2520 (10× finer) to verify convergence
// Tolerance:     3σ of the MC standard error
// =============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>

#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/European.hpp>
#include <myql/instruments/options/Lookback.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>

// =============================================================================
// GAUSSIAN HELPERS
// =============================================================================
static double N(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
[[maybe_unused]] static double n(double x) {
  return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

// =============================================================================
// GOLDMAN-SOSIN-GATTO (1979) CLOSED FORMS — FLOATING STRIKE, q = 0
// =============================================================================
// At inception the observed minimum (for call) and maximum (for put) equal S0.
// Formulae specialised to S_min = S_max = S0, with r > 0.
//
//   d1 = (r + sigma^2/2)·sqrt(T) / sigma
//   d2 = d1 - sigma·sqrt(T) = (r - sigma^2/2)·sqrt(T) / sigma
//   lambda  = sigma^2 / (2r)   (lookback correction factor)
//
// Floating Call  (payoff = S_T - S_min) at inception S_min = S0:
//   C = S0·N(d1) - S0·exp(-rT)·N(d2) + lambda·S0·(-N(-d1) +
//   exp(-rT)·N(d2))
//
// Floating Put   (payoff = S_max - S_T) at inception S_max = S0:
//   P = S0·exp(-rT)·N(-d2) - S0·N(-d1) + lambda·S0·(N(-d1) -
//   exp(-rT)·N(d2)) [by call-put symmetry of the maximum/minimum under
//   reflection]
// =============================================================================

struct LookbackBSParams {
  double S0, r, sigma, T;
};

static double gsg_float_call(const LookbackBSParams &p) {
  const double sig = p.sigma, r = p.r, T = p.T, S = p.S0;
  const double sqT = std::sqrt(T);
  const double d1 = (r + 0.5 * sig * sig) * sqT / sig;
  const double d2 = d1 - sig * sqT;
  const double df = std::exp(-r * T);
  const double lam = sig * sig / (2.0 * r);

  return S * N(d1) - S * df * N(d2) + lam * S * (-N(-d1) + df * N(d2));
}

static double gsg_float_put(const LookbackBSParams &p) {
  const double sig = p.sigma, r = p.r, T = p.T, S = p.S0;
  const double sqT = std::sqrt(T);
  const double d1 = (r + 0.5 * sig * sig) * sqT / sig;
  const double d2 = d1 - sig * sqT;
  const double df = std::exp(-r * T);
  const double lam = sig * sig / (2.0 * r);

  // Exact formula from Haug (2007) / Conze & Viswanathan (1991)
  return S * df * N(-d2) * (1.0 - lam) - S * N(-d1) * (1.0 + lam) + lam * S;
}

// Black-Scholes European call/put (for model-independent bound tests)
static double bs_call(double S, double K, double T, double r, double sig) {
  const double sqT = std::sqrt(T);
  const double d1 = (std::log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqT);
  const double d2 = d1 - sig * sqT;
  return S * N(d1) - K * std::exp(-r * T) * N(d2);
}

static double bs_put(double S, double K, double T, double r, double sig) {
  const double sqT = std::sqrt(T);
  const double d1 = (std::log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqT);
  const double d2 = d1 - sig * sqT;
  return K * std::exp(-r * T) * N(-d2) - S * N(-d1);
}

// =============================================================================
// HELPERS
// =============================================================================
static void print_header(const std::string &title) {
  std::cout << "\n" << std::string(70, '=') << "\n";
  std::cout << "  " << title << "\n";
  std::cout << std::string(70, '=') << "\n";
}

static bool check(const std::string &label, double mc, double mc_err,
                  double ref, double n_sigma = 3.0) {
  const double diff = std::abs(mc - ref);
  const bool ok = diff <= n_sigma * mc_err;
  std::cout << std::fixed << std::setprecision(5);
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << "\n"
            << "         MC=" << mc << " ± " << mc_err << "  Ref=" << ref
            << "  |diff|=" << diff << "  tol=" << n_sigma * mc_err << "\n";
  return ok;
}

static bool check_ge(const std::string &label, double mc_a, double mc_b) {
  const bool ok = mc_a >= mc_b - 1e-4; // small numerical slack
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << "\n"
            << "         " << mc_a << " >= " << mc_b << "\n";
  return ok;
}

// =============================================================================
// MAIN
// =============================================================================
int main() {

  // -------------------------------------------------------------------------
  // Common market parameters
  // -------------------------------------------------------------------------
  const double S0 = 100.0;
  const double r = 0.05;
  const double q = 0.0;
  const double sigma = 0.20;
  const double T = 0.5;
  const double K = 100.0; // ATM for fixed-strike tests

  // Black-Scholes (zero-factor model)
  BlackScholesModel bs_model(sigma);

  // MonteCarloConfig — generous paths for tight CI
  MonteCarloConfig cfg;
  cfg.num_paths = 250000;
  cfg.time_steps = 8192;
  cfg.seed = 42;

  // Closed-form reference values
  const LookbackBSParams p{S0, r, sigma, T};
  const double ref_fc = gsg_float_call(p);
  const double ref_fp = gsg_float_put(p);

  // European vanilla references (for bound tests)
  const double ref_ec = bs_call(S0, K, T, r, sigma);
  const double ref_ep = bs_put(S0, K, T, r, sigma);

  std::cout << "\nBS closed forms:\n";
  std::cout << "  Float Call  = " << ref_fc << "\n";
  std::cout << "  Float Put   = " << ref_fp << "\n";
  std::cout << "  European Call (K=S0) = " << ref_ec << "\n";
  std::cout << "  European Put  (K=S0) = " << ref_ep << "\n";

  int n_fail = 0;

  // =========================================================================
  // TEST 1: Floating-Strike Call  vs GSG formula
  // =========================================================================
  print_header("TEST 1 — Floating-Strike Lookback Call (payoff = S_T - S_min)");
  {
    // Stepper: no CIR factors for BS → TrackerLookback automatically picked
    using Stepper =
        ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, TrackerLookback>;
    using Instr = LookbackOption<PayoffVanilla<OptionType::Call>,
                                 /*FixedStrike=*/false, double>;

    double dummy = 0.0; // floating strike — no external strike
    Instr instr(dummy, T);

    MonteCarloPricer<BlackScholesModel, Stepper, Instr> pricer(bs_model, cfg);
    auto res = pricer.calculate(S0, r, q, instr);

    std::string label = "Float Call MC (" + std::to_string(cfg.time_steps) +
                        " steps) vs GSG Price";

    n_fail += !check(label, res.price, res.price_std_err, ref_fc);
  }

  // =========================================================================
  // TEST 2: Floating-Strike Call — convergence check (10× finer grid)
  // =========================================================================
  {
    MonteCarloConfig cfg_fine = cfg;
    cfg_fine.time_steps *= 10;

    using Stepper =
        ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, TrackerLookback>;
    using Instr =
        LookbackOption<PayoffVanilla<OptionType::Call>, false, double>;
    double dummy = 0.0;
    Instr instr(dummy, T);

    MonteCarloPricer<BlackScholesModel, Stepper, Instr> pricer(bs_model,
                                                               cfg_fine);
    auto res = pricer.calculate(S0, r, q, instr);

    std::string label = "Float Call MC (" +
                        std::to_string(cfg_fine.time_steps) + " steps) vs GSG";

    n_fail += !check(label, res.price, res.price_std_err, ref_fc);
  }

  // =========================================================================
  // TEST 3: Floating-Strike Put  vs GSG formula
  // =========================================================================
  print_header("TEST 2 — Floating-Strike Lookback Put (payoff = S_max - S_T)");

  {
    using Stepper =
        ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, TrackerLookback>;
    using Instr = LookbackOption<PayoffVanilla<OptionType::Put>, false, double>;
    double dummy = 0.0;
    Instr instr(dummy, T);

    MonteCarloPricer<BlackScholesModel, Stepper, Instr> pricer(bs_model, cfg);
    auto res = pricer.calculate(S0, r, q, instr);

    std::string label = "Float Put MC (" + std::to_string(cfg.time_steps) +
                        " steps) vs GSG Price";
    n_fail += !check(label, res.price, res.price_std_err, ref_fp);
  }

  // =========================================================================
  // TEST 4: Floating-Strike Put — convergence check (10× finer grid)
  // =========================================================================
  {
    MonteCarloConfig cfg_fine = cfg;
    cfg_fine.time_steps *= 10;

    using Stepper =
        ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, TrackerLookback>;
    using Instr = LookbackOption<PayoffVanilla<OptionType::Put>, false, double>;
    double dummy = 0.0;
    Instr instr(dummy, T);

    MonteCarloPricer<BlackScholesModel, Stepper, Instr> pricer(bs_model,
                                                               cfg_fine);
    auto res = pricer.calculate(S0, r, q, instr);

    std::string label = "Float Put MC (" + std::to_string(cfg_fine.time_steps) +
                        " steps) vs GSG";
    n_fail += !check(label, res.price, res.price_std_err, ref_fp);
  }

  // =========================================================================
  // TEST 4: Fixed-Strike Call  >=  Vanilla Call  (model-independent bound)
  //   payoff(S_max - K) >= payoff(S_T - K) path-by-path since S_max >= S_T
  // =========================================================================
  print_header("TEST 3 — Fixed-Strike Lookback Call >= Vanilla Call (bound)");
  {
    using Stepper =
        ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, TrackerLookback>;
    using LBInstr = LookbackOption<PayoffVanilla<OptionType::Call>,
                                   /*FixedStrike=*/true, double>;
    using EuInstr = EuropeanOption<PayoffVanilla<OptionType::Call>, double>;

    LBInstr lb_instr(K, T);
    EuInstr eu_instr(K, T);

    using EuStepper =
        ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, TrackerEuropean>;

    MonteCarloPricer<BlackScholesModel, Stepper, LBInstr> lb_pricer(bs_model,
                                                                    cfg);
    MonteCarloPricer<BlackScholesModel, EuStepper, EuInstr> eu_pricer(bs_model,
                                                                      cfg);

    auto lb_res = lb_pricer.calculate(S0, r, q, lb_instr);
    auto eu_res = eu_pricer.calculate(S0, r, q, eu_instr);

    std::cout << "  Fixed-Strike LB Call = " << lb_res.price << " ± "
              << lb_res.price_std_err << "\n";
    std::cout << "  Vanilla Call (K=S0)  = " << eu_res.price << " ± "
              << eu_res.price_std_err << "\n";
    std::cout << "  BS closed form       = " << ref_ec << "\n";

    n_fail +=
        !check_ge("Fixed LB Call >= Vanilla Call", lb_res.price, eu_res.price);
    n_fail += !check("Vanilla Call matches BS formula", eu_res.price,
                     eu_res.price_std_err, ref_ec);
  }

  // =========================================================================
  // TEST 5: Fixed-Strike Put  ≥  Vanilla Put  (model-independent bound)
  //   payoff(K - S_min) ≥ payoff(K - S_T) since S_min ≤ S_T
  // =========================================================================
  print_header("TEST 4 — Fixed-Strike Lookback Put >= Vanilla Put (bound)");
  {
    using Stepper =
        ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, TrackerLookback>;
    using LBInstr =
        LookbackOption<PayoffVanilla<OptionType::Put>, true, double>;
    using EuInstr = EuropeanOption<PayoffVanilla<OptionType::Put>, double>;

    LBInstr lb_instr(K, T);
    EuInstr eu_instr(K, T);

    using EuStepper =
        ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, TrackerEuropean>;

    MonteCarloPricer<BlackScholesModel, Stepper, LBInstr> lb_pricer(bs_model,
                                                                    cfg);
    MonteCarloPricer<BlackScholesModel, EuStepper, EuInstr> eu_pricer(bs_model,
                                                                      cfg);

    auto lb_res = lb_pricer.calculate(S0, r, q, lb_instr);
    auto eu_res = eu_pricer.calculate(S0, r, q, eu_instr);

    std::cout << "  Fixed-Strike LB Put = " << lb_res.price << " ± "
              << lb_res.price_std_err << "\n";
    std::cout << "  Vanilla Put (K=S0)  = " << eu_res.price << " ± "
              << eu_res.price_std_err << "\n";
    std::cout << "  BS closed form      = " << ref_ep << "\n";

    n_fail +=
        !check_ge("Fixed LB Put >= Vanilla Put", lb_res.price, eu_res.price);
    n_fail += !check("Vanilla Put matches BS formula", eu_res.price,
                     eu_res.price_std_err, ref_ep);
  }

  // =========================================================================
  // TEST 7: C_float + P_float = discounted expected range (consistency)
  //   C_float + P_float = exp(-rT) * E[S_max - S_min]
  //   Testing with the 81920-step runs to ensure convergence bounds.
  // =========================================================================
  print_header(
      "TEST 5 — C_float + P_float = Range Price (internal consistency)");
  {
    MonteCarloConfig cfg_fine = cfg;
    cfg_fine.time_steps *= 10;

    using Stepper =
        ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, TrackerLookback>;
    using CallInstr =
        LookbackOption<PayoffVanilla<OptionType::Call>, false, double>;
    using PutInstr =
        LookbackOption<PayoffVanilla<OptionType::Put>, false, double>;

    double dummy = 0.0;
    CallInstr c_instr(dummy, T);
    PutInstr p_instr(dummy, T);

    MonteCarloPricer<BlackScholesModel, Stepper, CallInstr> c_pricer(bs_model,
                                                                     cfg_fine);
    MonteCarloPricer<BlackScholesModel, Stepper, PutInstr> p_pricer(bs_model,
                                                                    cfg_fine);

    auto c_res = c_pricer.calculate(S0, r, q, c_instr);
    auto p_res = p_pricer.calculate(S0, r, q, p_instr);

    const double sum_mc = c_res.price + p_res.price;
    const double sum_ref = ref_fc + ref_fp;

    // Generous tolerance since we're summing two discretely biased values
    const double sum_err = std::sqrt(c_res.price_std_err * c_res.price_std_err +
                                     p_res.price_std_err * p_res.price_std_err);

    std::cout << "  C_float (fine) = " << c_res.price
              << ", P_float (fine) = " << p_res.price << "\n";
    std::cout << "  MC sum  = " << sum_mc << "  GSG sum = " << sum_ref << "\n";

    // We allow up to 4 std errs here + a little slack for remaining
    // discretisation bias
    const double bias_allowance = 0.10;
    const double diff = std::abs(sum_mc - sum_ref);
    const bool ok = diff <= (4.0 * sum_err + bias_allowance);

    std::cout << (ok ? "[PASS] " : "[FAIL] ")
              << "C_float + P_float vs GSG sum\n"
              << "         MC=" << sum_mc << " ± " << sum_err
              << "  Ref=" << sum_ref << "  |diff|=" << diff << "\n";
    n_fail += !ok;
  }

  // =========================================================================
  // SUMMARY
  // =========================================================================
  std::cout << "\n" << std::string(70, '=') << "\n";
  if (n_fail == 0)
    std::cout << "  ALL TESTS PASSED\n";
  else
    std::cout << "  " << n_fail << " TEST(S) FAILED\n";
  std::cout << std::string(70, '=') << "\n\n";

  return n_fail;
}
