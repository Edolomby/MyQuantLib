# MyQuantLib

**MyQuantLib** is a high-performance, C++20 header-only quantitative finance library focused on Option Pricing. Built from the ground up to achieve **zero-runtime-overhead** via static polymorphism (policy-based templates and compile-time specialization), the library offers incredibly fast evaluation of options under advanced stochastic volatility and jump-diffusion models.

## 🌟 Key Features

- **C++20 Header-Only**: Easy to integrate. No separate compilation of the library unit itself is required.
- **Zero-Runtime-Overhead Architecture**: Every model, instrument, and pricing scheme is resolved at compile time, eliminating expensive virtual function calls in hot loops.
- **Symmetric Pricing API**: Both the Fourier and Monte Carlo engines expose an identical, easy-to-use API (`.calculate(S0, r, q, instrument)`).
- **Comprehensive Greeks Support**:
  - *Analytical Greeks* via Fourier pricing (`GreekMode::Essential` for Delta/Gamma, `GreekMode::Full` for Vega/Theta/Rho).
  - *Pathwise Finite Differences* for Monte Carlo (`GreekMode::Essential` for Delta/Gamma).
  - *Exact Common Random Numbers (CRN)* for Monte Carlo Full Greeks, enabling mathematically exact variance reduction for Vega, Theta, and Rho via a synchronized multi-stepper architecture.
- **Runtime Dispatch Layer**: Includes a `std::variant`-based boundary layer allowing model and instrument selection purely from runtime strings (e.g. JSON/CSV parsing) while retaining the zero-overhead engine core.
- **Vectorized Evaluation**: Support for pricing "strips" of options across multiple strikes/maturities simultaneously (Struct-of-Arrays pattern).
- **Parallel Computing**: Fully utilizes OpenMP for robust Monte Carlo path simulation.

---

## 🧩 Supported Models (ASVJ Family)

MyQuantLib implements the **Affine Stochastic Volatility and Jumps (ASVJ)** framework. By mixing and matching volatility schemes and jump policies at compile time, the library naturally supports a wide range of dynamics. In this context, "Factors" refers to the number of independent stochastic diffusion processes driving the volatility:

- **0-Factor Models (Constant Volatility)**: Black-Scholes, Merton Jump-Diffusion, Kou Jump-Diffusion.
- **1-Factor Models (Single Stochastic Volatility)**: Heston, Bates (Heston + Merton Jumps), Bates-Kou (Heston + Kou Jumps).
- **2-Factor Models (Double Stochastic Volatility)**: Double Heston, Double Bates, Double Bates-Kou.

## 📈 Supported Instruments

- **Path-Independent (European)**: Vanilla (Call/Put), Cash-or-Nothing (Digital), Asset-or-Nothing (Digital).
- **Path-Dependent**:
  - **Asian**: Geometric, Arithmetic *(supports in-flight valuation via historical state)*.
  - **Lookback**: Continuous Fixed, Continuous Floating *(supports in-flight valuation via historical state)*.
  - **Barrier**: Up-and-Out, Up-and-In, Down-and-Out, Down-and-In.

## 🧮 Pricing Engines & Numerical Schemes

- **Monte Carlo Pricer**: Provides flexible `Stepper` configurations for the underlying stochastic differential equations.
  - The integration scheme for the log-asset process is always fixed, while the user has the flexibility to choose high-order schemes for the underlying CIR volatility process, including **Exact**, **NV**, and **NCI**.
  - Both the exact scheme (`SchemeExact`) and quasi-exact NCI scheme (`SchemeNCI`) rely on the property that a non-central chi-squared distribution can be represented as a mixture of Gamma and Chi-squared distributions weighted by a Poisson distribution. However, the parameters of this mixture become arbitrarily large if the volatility of variance ($\sigma$) and the time step ($\Delta t$) are both very small.
  - Therefore, the **NCI** scheme is generally superior (being faster and quasi-exact) to the **Exact** simulation unless the time step $\Delta t = T/N << 10^{-2}$.
  - Crucially, whenever the "weak" Feller condition is satisfied ($\sigma^2 \le 4\kappa\theta$), the Ninomiya-Victoir scheme (`SchemeNV`) should be strictly preferred. Not only does it avoid the aforementioned mixture degradation, but it is also the **fastest** exact scheme available, requiring only a single Gaussian draw per time step.
  - *Note: These advanced simulation schemes for the Log-Heston process have been developed and analyzed in the joint work by A. Alfonsi and E. Lombardo. For theoretical details and convergence proofs, please refer to:*
    > **High Order Approximations and Simulation Schemes for the Log-Heston Process**  
    > *SIAM Journal on Financial Mathematics* ([DOI: 10.1137/24M1679720](https://epubs.siam.org/doi/10.1137/24M1679720)) | [arXiv Preprint](https://arxiv.org/abs/2407.17151)
- **Fourier Pricer**: Employs Gil-Pelaez quadrature for very fast analytical pricing and Greeks computation. The highly oscillatory integrals are resolved using a robust **adaptive Simpson's rule with Richardson extrapolation**, achieving high precision even under extreme parameters.

---

## 🚀 Quick Start

### Requirements
- **C++20** compatible compiler (GCC 10+, Clang 11+, MSVC 19.30+)
- **CMake** 3.15+
- **OpenMP** (for multi-threading)
- **Boost 1.88+** (introduces `xoshiro256++` fast RNG via `<boost/random/xoshiro.hpp>`)
- **Eigen3** 3.3+ (for linear algebra operations)

> **⚠️ *Important Note for Windows / Visual Studio Users (OpenMP)** > Microsoft Visual Studio's default OpenMP implementation is currently limited to an older standard (OpenMP 2.0). Because MyQuantLib uses modern parallel features for its Monte Carlo engine, compiling with the default MSVC toolset may cause issues.  
> **To compile successfully on Windows, please use one of the following:**
> 1. **Clang-cl (Recommended):** Inside Visual Studio, change your Platform Toolset to `LLVM (clang-cl)`. This provides full modern C++20 and OpenMP support natively within the IDE.
> 2. **New MSVC LLVM Flag:** If using a recent MSVC version, pass the `/openmp:llvm` compiler flag instead of the standard `/openmp` flag.
> 3. **GCC / Clang:** Compile using WSL (Windows Subsystem for Linux) or MSYS2/MinGW.

### Building the Benchmarks

MyQuantLib is header-only, so to use it in your code, simply include the `include/myql` directory. To explore the codebase organization, please see the [Project Structure Overview](project_structure.md).

However, you can compile the comprehensive benchmark suite:

```bash
git clone https://github.com/Edolomby/MyQuantLib.git
cd MyQuantLib
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Run a benchmark, for example:
```bash
./EssentialGreekTest
```

### Example 1: Vanilla Pricing (Symmetric API)

This example demonstrates how to price a Vanilla Call using both the Monte Carlo and Fourier engines, comparing their results.

```cpp
#include <iostream>
#include <cmath>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>
#include <myql/pricers/fourier/FourierPricer.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/instruments/options/European.hpp>
#include <myql/instruments/Payoffs.hpp>

int main() {
    // 1. Define Model Parameters (Heston)
    HestonParams heston_params = {2.5, 0.06, 0.4, -0.7, 0.05}; // kappa, theta, sigma, rho, v0
    HestonModel model(heston_params);

    // 2. Define the Instrument
    double strike = 100.0, maturity = 1.0;
    using CallVanilla = EuropeanOption<PayoffVanilla<OptionType::Call>>;
    CallVanilla option(strike, maturity);

    double spot = 100.0, rate = 0.05, dividend = 0.02;

    // 3. Monte Carlo Pricing
    MonteCarloConfig mc_cfg;
    mc_cfg.num_paths = 100000;
    mc_cfg.time_steps = 100;
    
    using Stepper = ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerEuropean>;
    MonteCarloPricer<HestonModel, Stepper, CallVanilla> mc_pricer(model, mc_cfg);
    auto mc_res = mc_pricer.calculate(spot, rate, dividend, option);

    // 4. Fourier Pricing (Analytical)
    FourierPricer<HestonModel, CallVanilla> fourier_pricer(model);
    auto fourier_res = fourier_pricer.calculate(spot, rate, dividend, option);

    // 5. Compare Results (Z-Score)
    double z_score = std::abs(mc_res.price - fourier_res.price) / mc_res.price_std_err;

    std::cout << "MC Price:      " << mc_res.price << " +- " << mc_res.price_std_err << "\n";
    std::cout << "Fourier Price: " << fourier_res.price << "\n";
    std::cout << "Z-Score:       " << z_score << "\n";
    
    return 0;
}
```

### Example 2: Exotic Options (Vectorized Asian & Lookback)

This example demonstrates how to evaluate a **strip** of fixed-strike Asian options simultaneously, alongside a floating-strike Lookback option.

```cpp
#include <iostream>
#include <vector>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>
#include <myql/models/asvj/core/ASVJmodel.hpp>
#include <myql/models/asvj/core/ASVJstepper.hpp>
#include <myql/instruments/options/Asian.hpp>
#include <myql/instruments/options/Lookback.hpp>

int main() {
    HestonModel model({2.5, 0.06, 0.4, -0.7, 0.05});
    MonteCarloConfig mc_cfg{100000, 100, 42}; 
    double S0 = 100.0, r = 0.05, q = 0.0;

    // A. Strip of Geometric Asian Calls (Strikes: 90, 100, 110)
    std::vector<double> strikes = {S0 * 0.9, S0, S0 * 1.1};
    using GeoAsianStrip = AsianOption<TrackerGeoAsian, PayoffVanilla<OptionType::Call>, true, std::vector<double>>;
    GeoAsianStrip asian_strip(strikes, 1.0);

    using AsianStepper = ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerGeoAsian>;
    MonteCarloPricer<HestonModel, AsianStepper, GeoAsianStrip> asian_pricer(model, mc_cfg);
    
    auto asian_res = asian_pricer.calculate(S0, r, q, asian_strip);
    std::cout << "Asian Call (K=90):  " << asian_res.price[0] << "\n"
              << "Asian Call (K=100): " << asian_res.price[1] << "\n"
              << "Asian Call (K=110): " << asian_res.price[2] << "\n";

    // B. Floating Strike Lookback Put: Payoff = max(S_max - S_T, 0)
    double dummy_strike = 0.0;
    using LookbackPut = LookbackOption<PayoffVanilla<OptionType::Put>, false, double>;
    LookbackPut lookback(dummy_strike, 1.0);

    using LookbackStepper = ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerLookback>;
    MonteCarloPricer<HestonModel, LookbackStepper, LookbackPut> lb_pricer(model, mc_cfg);
    
    auto lb_res = lb_pricer.calculate(S0, r, q, lookback);
    std::cout << "Floating Lookback Put: " << lb_res.price << "\n";

    return 0;
}
```

### Example 3: Essential and Full Greeks Calculation

By upgrading the `GreekMode` template parameter, the engines automatically accumulate and return derivatives. In `GreekMode::Full`, the Monte Carlo engine uses an advanced single-loop architecture to compute Vega, Theta, and Rho using exact Common Random Numbers (CRN), minimizing standard errors.

```cpp
#include <iostream>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>
#include <myql/pricers/fourier/FourierPricer.hpp>

using namespace myql;

// Assumes 'model' and 'option' from Example 1
void calculate_greeks(const HestonModel& model, const CallVanilla& option) {
    double S0 = 100.0, r = 0.05, q = 0.02;
    MonteCarloConfig mc_cfg{200000, 100, 42};

    // Calculate MC Full Greeks
    using Stepper = ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerEuropean>;
    MonteCarloPricer<HestonModel, Stepper, CallVanilla, GreekMode::Full> mc_pricer(model, mc_cfg);
    auto res_mc = mc_pricer.calculate(S0, r, q, option);

    std::cout << "MC Delta: " << res_mc.delta << " +- " << res_mc.delta_std_err << "\n";
    std::cout << "MC Gamma: " << res_mc.gamma << " +- " << res_mc.gamma_std_err << "\n";
    std::cout << "MC Vega:  " << res_mc.vega[0] << " +- " << res_mc.vega_std_err[0] << "\n";
    std::cout << "MC Theta: " << res_mc.theta << " +- " << res_mc.theta_std_err << "\n";

    // Calculate Analytical Fourier Full Greeks
    FourierPricer<HestonModel, CallVanilla, GreekMode::Full> fourier_pricer(model);
    auto res_fourier = fourier_pricer.calculate(S0, r, q, option);

    std::cout << "\nAnalytical Delta: " << res_fourier.delta << "\n";
    std::cout << "Analytical Gamma: " << res_fourier.gamma << "\n";
    std::cout << "Analytical Vega:  " << res_fourier.vega[0] << "\n";
    std::cout << "Analytical Theta: " << res_fourier.theta << "\n";
}
```

---

## 🏗️ Architecture Notes

At its core, the library extensively relies on template meta-programming to bypass runtime branch checking. When you instantiate a `MonteCarloPricer<Model, Stepper, Instrument, Mode>`, the compiler weaves together only the relevant sub-routines:
- It excludes jump calculations if `JumpPolicy == NoJumps`.
- It unrolls vectorized arrays if evaluating a strip of options.
- It completely bypasses expensive pathwise difference accumulations if `GreekMode == None`.

This architecture translates a potentially very complex web of `if-else` rules and generic virtual polymorphism into specialized, contiguous machine code tailored precisely for the specific pricing task at hand.

---

## 🔌 Runtime Dispatch Layer

The library ships a thin **boundary layer** (`include/myql/dispatcher/`) that lets you select model, instrument, engine, and Greeks entirely at **runtime** — from strings, config files, or REST parameters — while the inner engine still executes the same **zero-overhead compiled templates**.

A nested `std::visit` resolves the `AnyModel` and `AnyInstrument` variants at the call site and routes into exactly one compiled `MonteCarloPricer<Model, Stepper, Instrument>` or `FourierPricer<Model, Instrument>` — **no virtual calls, no heap allocation**.

### Step-by-Step Usage

**1 — One header:**
```cpp
#include <myql/dispatcher/PricingDispatch.hpp>
using namespace myql::dispatcher;
```

**2 — Build model from a string:**
```cpp
RuntimeModelParams p;
p.heston1 = {2.5, 0.06, 0.4, -0.7, 0.05};  // kappa, theta, sigma, rho, v0
p.merton  = {1.2, -0.12, 0.15};             // lambda, mu_J, sigma_J (for Bates)
p.vol     = 0.20;                            // flat vol (0-factor models only)

AnyModel model = build_model("Heston", p);
```

| String | Model |
|---|---|
| `BlackScholes` | GBM, flat vol |
| `Merton` | GBM + Merton jumps |
| `Kou` | GBM + Kou double-exponential jumps |
| `Heston` | Heston stochastic vol |
| `Bates` | Heston + Merton jumps |
| `BatesKou` | Heston + Kou jumps |
| `DoubleHeston` | 2-factor Heston |
| `DoubleBates` | 2-factor Heston + Merton |
| `DoubleBatesKou` | 2-factor Heston + Kou |

**3 — Build instrument from strings:**
```cpp
RuntimeInstrumentParams ip;
ip.strikes  = {90.0, 100.0, 110.0};
ip.maturity = 1.0;

AnyInstrument instr = build_instrument("Vanilla", "Call", ip);
```

| `type` | `option_type` | Payoff |
|---|---|---|
| `Vanilla` | `Call` / `Put` | max(S−K, 0) / max(K−S, 0) |
| `CashOrNothing` | `Call` / `Put` | $1 if ITM, else $0 |
| `AssetOrNothing` | `Call` / `Put` | S if ITM, else $0 |
| `GeoAsian` / `ArithAsian` | `Call` / `Put` | max(Avg−K, 0) / max(K−Avg, 0) |
| `GeoAsianFloating` / `ArithAsianFloating` | `Call` / `Put` | max(S−Avg, 0) / max(Avg−S, 0) |
| `LookbackFixed` | `Call` / `Put` | max(Extrema−K, 0) / max(K−Extrema, 0) |
| `LookbackFloating` | `Call` / `Put` | max(S−Extrema, 0) / max(Extrema−S, 0) |
| `UpAndOut` / `DownAndIn`... | `Call` / `Put` | Standard Barrier Payoffs |

**4 — Price:**
```cpp
// Monte Carlo — price only (default)
MonteCarloConfig mc; mc.num_paths = 500000; mc.time_steps = 50;
auto res = price_mc(model, instr, mc, S0, r, q);

// Monte Carlo — price + Delta + Gamma, choosing the CIR vol scheme
auto res = price_mc<GreekMode::Essential, SchemeExact>(model, instr, mc, S0, r, q);

// Fourier — price + analytical Greeks (Delta, Gamma)
FourierEngine::Config fc; fc.tolerance = 1e-9;
auto res = price_fourier<GreekMode::Essential>(model, instr, fc, S0, r, q);
```

| `ExplicitVolScheme` (MC only) | When to prefer |
|---|---|
| `SchemeNCI` *(default)* | Safe general-purpose choice |
| `SchemeExact` | Standard Heston params, moderate dt |
| `SchemeNV` | Valid only when σ² ≤ 4κθ (weak Feller) |

**5 — Read results (unified `DispatchResult`):**
```cpp
res.prices[i];           // always present
res.deltas[i];           // GreekMode::Essential or ::Full
res.gammas[i];           // GreekMode::Essential or ::Full
res.prices_std_err[i];   // MC only
```

> **Stepper deduction is automatic** — `StepperTraits.hpp` inspects the model type to determine the number of CIR factors and jump policy, and wires the correct `ASVJStepper` template. You never specify it manually.

> **Current scope:** `AnyInstrument` covers both European-style payoffs (Vanilla, Cash-or-Nothing, Asset-or-Nothing) and Path-dependent Exotics (Asian, Barrier, Lookback). Note that Fourier pricing is strictly bound to European payoffs; routing an Exotic to `price_fourier` throws a runtime mismatch exception.

---

## 🔮 Next Steps / Future Work
- Integration of Forward starting options.
- Integration of geometric Asian options in the Fourier engine.
- Implementation of Control Variates for Monte Carlo and quasi-Monte Carlo.
- Implementation of a generalized **Calibration Module** to fit ASVJ models to market data.
- Integration of advanced acceleration techniques: **Richardson-Romberg extrapolation**, **Random grids**, **Multilevel Monte Carlo (MLMC)**, and **MultiLevel Richardson-Romberg extrapolation  (ML2R)** methods.
