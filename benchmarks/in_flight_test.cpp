// =============================================================================
// IN-FLIGHT OPTION TESTS (Historical State Verification)
// =============================================================================
// Verifies the behaviour of Asian and Lookback options that started in the
// past.
//
// 1. Continuity Test:
//    An option with "history" equal exactly to S0 and t_elapsed = 0 should
//    price identically to a fresh starting option.
//
// 2. Monotonicity Test:
//    An Asian call with a high historical average should price higher than
//    one with a low historical average.
//    A Lookback floating call (payoff S_T - S_min) with a very low historical
//    S_min should price higher than one with a high historical S_min.
// =============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>

#include <myql/instruments/Payoffs.hpp>
#include <myql/instruments/options/Asian.hpp>
#include <myql/instruments/options/Lookback.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>

// =============================================================================
// HELPERS
// =============================================================================
static void print_header(const std::string &title) {
  std::cout << "\n" << std::string(70, '=') << "\n";
  std::cout << "  " << title << "\n";
  std::cout << std::string(70, '=') << "\n";
}

static bool check_close(const std::string &label, double val_a, double val_b,
                        double tol = 1e-4) {
  const double diff = std::abs(val_a - val_b);
  const bool ok = diff <= tol;
  std::cout << std::fixed << std::setprecision(5);
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << "\n"
            << "         " << val_a << " vs " << val_b << "  |diff|=" << diff
            << "\n";
  return ok;
}

static bool check_ge(const std::string &label, double val_higher,
                     double val_lower) {
  const bool ok = val_higher > val_lower; // Strict inequality expected
  std::cout << std::fixed << std::setprecision(5);
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << "\n"
            << "         " << val_higher << " > " << val_lower << "\n";
  return ok;
}

// =============================================================================
// MAIN
// =============================================================================
int main() {
  const double S0 = 100.0;
  const double r = 0.05;
  const double q = 0.0;
  const double sigma = 0.20;
  const double T_residual = 0.5;
  const double K = 100.0;

  BlackScholesModel bs_model(sigma);

  MonteCarloConfig cfg;
  cfg.num_paths = 50000;
  cfg.time_steps = 100;
  cfg.seed = 42;

  int n_fail = 0;

  // ===========================================================================
  print_header("1. GEOMETRIC ASIAN IN-FLIGHT TESTS");
  // ===========================================================================
  {
    using Stepper =
        ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, TrackerGeoAsian>;
    using InstrGeo =
        AsianOption<TrackerGeoAsian, PayoffVanilla<OptionType::Call>,
                    /*FixedStrike=*/true, double>;

    // 1(a) Continuity
    InstrGeo fresh_asian(K, T_residual);

    // In-flight Asian pretending it just started today at S0
    // => average of log(S) over 0 elapsed time = log(S0), t_elapsed = 0
    InstrGeo inflight_zero_asian(K, T_residual, std::log(S0), 0.0);

    MonteCarloPricer<BlackScholesModel, Stepper, InstrGeo> pricer(bs_model,
                                                                  cfg);
    double price_fresh = pricer.calculate(S0, r, q, fresh_asian).price;
    double price_zero = pricer.calculate(S0, r, q, inflight_zero_asian).price;

    n_fail += !check_close("Geo Asian Call: Fresh vs In-flight(avg=S0, t=0)",
                           price_fresh, price_zero);

    // 1(b) Monotonicity
    // An option that has been running for 0.5 years already, with a very high
    // running average (avg = 150) should be deep in the money and worth more
    // than an option with a low running average (avg = 80).
    const double t_elapsed = 0.5;
    InstrGeo inflight_high_asian(K, T_residual, std::log(150.0), t_elapsed);
    InstrGeo inflight_low_asian(K, T_residual, std::log(80.0), t_elapsed);

    double price_high = pricer.calculate(S0, r, q, inflight_high_asian).price;
    double price_low = pricer.calculate(S0, r, q, inflight_low_asian).price;

    n_fail += !check_ge("Geo Asian Call (Fixed): High History > Low History",
                        price_high, price_low);
  }

  // ===========================================================================
  print_header("2. ARITHMETIC ASIAN IN-FLIGHT TESTS");
  // ===========================================================================
  {
    using Stepper =
        ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, TrackerArithAsian>;
    using InstrArith =
        AsianOption<TrackerArithAsian, PayoffVanilla<OptionType::Put>,
                    /*FixedStrike=*/true, double>;

    // 2(a) Continuity
    InstrArith fresh_asian(K, T_residual);

    // In-flight ArithAsian pretending it just started today at S0
    InstrArith inflight_zero_asian(K, T_residual, S0, std::log(S0), 0.0);

    MonteCarloPricer<BlackScholesModel, Stepper, InstrArith> pricer(bs_model,
                                                                    cfg);
    double price_fresh = pricer.calculate(S0, r, q, fresh_asian).price;
    double price_zero = pricer.calculate(S0, r, q, inflight_zero_asian).price;

    n_fail += !check_close("Arith Asian Put: Fresh vs In-flight(avg=S0, t=0)",
                           price_fresh, price_zero);

    // 2(b) Monotonicity
    // A Fixed Put pays max(K - Avg, 0).
    // A low historical average makes the put more valuable (in the money).
    const double t_elapsed = 1.0;
    InstrArith inflight_low_asian(K, T_residual, 50.0, std::log(50.0),
                                  t_elapsed);
    InstrArith inflight_high_asian(K, T_residual, 150.0, std::log(150.0),
                                   t_elapsed);

    double price_low = pricer.calculate(S0, r, q, inflight_low_asian).price;
    double price_high = pricer.calculate(S0, r, q, inflight_high_asian).price;

    n_fail +=
        !check_ge("Arith Asian Put (Fixed): Low avg History > High avg History",
                  price_low, price_high);
  }

  // ===========================================================================
  print_header("3. LOOKBACK IN-FLIGHT TESTS");
  // ===========================================================================
  {
    using Stepper =
        ASVJStepper<NullVolScheme, NullVolScheme, NoJumps, TrackerLookback>;
    // Floating Strike Call pays max(S_T - S_min, 0)
    using InstrFloatCall = LookbackOption<PayoffVanilla<OptionType::Call>,
                                          /*FixedStrike=*/false, double>;

    double dummy_K = 0.0;

    // 3(a) Continuity
    InstrFloatCall fresh_lookback(dummy_K, T_residual);
    // In-flight pretending it just started (min=S0, max=S0)
    InstrFloatCall inflight_zero_lookback(dummy_K, T_residual, S0, S0);

    MonteCarloPricer<BlackScholesModel, Stepper, InstrFloatCall> pricer(
        bs_model, cfg);
    double price_fresh = pricer.calculate(S0, r, q, fresh_lookback).price;
    double price_zero =
        pricer.calculate(S0, r, q, inflight_zero_lookback).price;

    n_fail +=
        !check_close("Lookback Float Call: Fresh vs In-flight(min/max=S0)",
                     price_fresh, price_zero);

    // 3(b) Monotonicity (Floating Call)
    // S_min = 50 vs S_min = 100. Lower S_min = higher payoff.
    InstrFloatCall inflight_low_min(dummy_K, T_residual, 50.0, 100.0);
    InstrFloatCall inflight_high_min(dummy_K, T_residual, 100.0, 100.0);

    double price_low = pricer.calculate(S0, r, q, inflight_low_min).price;
    double price_high = pricer.calculate(S0, r, q, inflight_high_min).price;

    n_fail +=
        !check_ge("Lookback Float Call: Low S_min History > High S_min History",
                  price_low, price_high);

    // 3(c) Monotonicity (Floating Put)
    // Pays max(S_max - S_T, 0). Higher S_max = higher payoff.
    using InstrFloatPut = LookbackOption<PayoffVanilla<OptionType::Put>,
                                         /*FixedStrike=*/false, double>;

    InstrFloatPut inflight_high_max(dummy_K, T_residual, 100.0,
                                    150.0); // S_max = 150
    InstrFloatPut inflight_low_max(dummy_K, T_residual, 100.0,
                                   100.0); // S_max = 100

    MonteCarloPricer<BlackScholesModel, Stepper, InstrFloatPut> pricer_put(
        bs_model, cfg);

    double p_high_max = pricer_put.calculate(S0, r, q, inflight_high_max).price;
    double p_low_max = pricer_put.calculate(S0, r, q, inflight_low_max).price;

    n_fail +=
        !check_ge("Lookback Float Put: High S_max History > Low S_max History",
                  p_high_max, p_low_max);
  }

  std::cout << "\nFailed tests: " << n_fail << "\n";
  return n_fail == 0 ? 0 : 1;
}
