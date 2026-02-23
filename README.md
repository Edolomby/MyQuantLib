# MyQuantLib

**MyQuantLib** is a high-performance, C++20 header-only quantitative finance library focused on Option Pricing. Built from the ground up to achieve **zero-runtime-overhead** via static polymorphism (templates and CRTP), the library offers incredibly fast evaluation of options under advanced stochastic volatility and jump-diffusion models.

## 🌟 Key Features

- **C++20 Header-Only**: Easy to integrate. No separate compilation of the library unit itself is required.
- **Zero-Runtime-Overhead Architecture**: Every model, instrument, and pricing scheme is resolved at compile time, eliminating expensive virtual function calls in hot loops.
- **Symmetric Pricing API**: Both the Fourier and Monte Carlo engines expose an identical, easy-to-use API (`.calculate(S0, r, q, instrument)`).
- **Comprehensive Greeks Support**:
  - *Analytical Greeks* via Fourier pricing (Delta, Gamma).
  - *Pathwise Finite Differences* for Monte Carlo (Delta, Gamma).
  - *(Full Greek mode including Vega, Theta, and Rho is planned for a future release).*
- **Vectorized Evaluation**: First-class support for pricing "strips" of options across multiple strikes/maturities simultaneously (Struct-of-Arrays pattern) for maximum CPU throughput.
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
  - **Asian**: Geometric, Arithmetic.
  - **Lookback**: Continuous Fixed, Continuous Floating.
  - **Barrier**: Up-and-Out, Up-and-In, Down-and-Out, Down-and-In (with Equivalent Barrier Method enhancements for discrete monitoring limits).

## 🧮 Pricing Engines & Numerical Schemes

- **Monte Carlo Pricer**: Provides flexible `Stepper` configurations for the underlying stochastic differential equations.
  - Supported high-order schemes for the volatility process include **Exact**, **NV**, and **NCI**.
  - The **NCI** scheme is generally superior (being faster and quasi-exact) to the **Exact** simulation unless the time step $\Delta t = T/N << 10^{-2}$. The **NV** scheme is specifically recommended whenever the "weak" Feller condition is satisfied ($\sigma^2 \le 4\kappa\theta$).
  - *Note: These advanced simulation schemes for the Log-Heston process have been developed and analyzed in the joint work by A. Alfonsi and E. Lombardo. For theoretical details and convergence proofs, please refer to:*
    > **High Order Approximations and Simulation Schemes for the Log-Heston Process**  
    > *SIAM Journal on Financial Mathematics* ([DOI: 10.1137/24M1679720](https://epubs.siam.org/doi/10.1137/24M1679720)) | [arXiv Preprint](https://arxiv.org/abs/2407.17151)
- **Fourier Pricer**: Employs Gil-Pelaez quadrature for lighting-fast analytical pricing and Greeks computation.

---

## 🚀 Quick Start

### Requirements
- **C++20** compatible compiler (GCC 10+, Clang 11+, MSVC 19.30+)
- **CMake** 3.15+
- **OpenMP** (for multi-threading)
- **Boost 1.80+** (for fast `xoshiro256pp` random number generation)
- **Eigen3** 3.3+ (for linear algebra operations)

### Building the Benchmarks

MyQuantLib is header-only, so to use it in your code, simply include the `include/myql` directory. However, you can compile the comprehensive benchmark suite:

```bash
git clone https://github.com/yourusername/MyQuantLib.git
cd MyQuantLib
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Run a benchmark, for example:
```bash
./EssentialGreekTest
```

### Usage Example

```cpp
#include <iostream>
#include <myql/pricers/montecarlo/MonteCarloPricer.hpp>
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

    // 3. Configure Monte Carlo limits
    MonteCarloConfig mc_cfg;
    mc_cfg.num_paths = 100000;
    mc_cfg.time_steps = 100;
    
    // 4. Setup the Pricer
    using Stepper = ASVJStepper<SchemeExact, NullVolScheme, NoJumps, TrackerEuropean>;
    MonteCarloPricer<HestonModel, Stepper, CallVanilla> pricer(model, mc_cfg);

    // 5. Calculate!
    double spot = 100.0, rate = 0.05, dividend = 0.02;
    auto result = pricer.calculate(spot, rate, dividend, option);

    std::cout << "Price: " << result.price << std::endl;
    return 0;
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

## 🔮 Next Steps / Future Work
- Implementation of a **Runtime Dispatch Layer** (via `std::variant`) to allow parsing model and instrument configurations from CSV/JSON/DB at runtime while retaining the zero-overhead engine core.
- Integration of the **Likelihood Ratio Method (LRM)** or Score Function Method to accurately compute MC Greeks on discontinuous digital payoffs without finite-difference boundary issues.
- Integration of full Volatility Surfaces and Yield Curves.
