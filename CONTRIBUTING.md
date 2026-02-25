# Contributing to MyQuantLib

First off, thank you for taking the time to contribute! 🎉  
MyQuantLib is a research-grade library, so contributions that improve correctness, performance, or coverage of models and instruments are especially welcome.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Guidelines](#coding-guidelines)
- [Adding a New Model](#adding-a-new-model)
- [Adding a New Instrument](#adding-a-new-instrument)
- [Writing Benchmarks / Tests](#writing-benchmarks--tests)
- [Pull Request Checklist](#pull-request-checklist)

---

## Code of Conduct

Please be respectful and constructive in all interactions. This is a research project and disagreements on numerical methods are expected — keep discussions technical and collegial.

---

## How Can I Contribute?

| Type | Examples |
|---|---|
| 🐛 Bug fix | Incorrect payoff, wrong characteristic function coefficient |
| ⚡ Performance | Faster quadrature, better SIMD usage, reduced allocations |
| 📐 New model | Add a new vol process or jump policy to the ASVJ family |
| 🧩 New instrument | Add a new path-dependent or path-independent payoff |
| 📖 Documentation | Fix typos, improve examples, add math references |
| 🔬 Benchmarks | New numerical tests, convergence studies |

Please **open an issue first** for anything beyond a trivial fix so we can discuss the design before you invest time writing code.

---

## Development Setup

### Requirements

- **C++20** compiler — GCC 10+, Clang 11+, or MSVC 19.30+
- **CMake** 3.15+
- **OpenMP** (for Monte Carlo parallelism)
- **Boost 1.84+** (introduces `xoshiro256++` via `<boost/random/xoshiro.hpp>`)
- **Eigen3 3.3+** (linear algebra)

### Build

```bash
git clone https://github.com/Edolomby/MyQuantLib.git
cd MyQuantLib
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

> **Note:** The library itself is header-only. Only the benchmark executables need to be compiled.

---

## Project Structure

```
include/myql/
├── core/                  # Shared primitive types (PricingResult, GreekMode, MonteCarloConfig, ...)
├── dispatcher/            # Runtime dispatch layer: string → variant → compiled template
├── engines/               # Low-level engine kernels (Fourier quadrature integration, ...)
├── instruments/           # Payoff definitions and path trackers
│   ├── options/           # European, Asian, Barrier, Lookback payoff headers
│   └── trackers/          # Path statistic accumulators used by the MC stepper
├── math/                  # Numerical utilities: quadrature, splines, general helpers
├── models/                # ASVJ model hierarchy
│   └── asvj/
│       ├── core/          # ASVJModel CRTP base, ASVJStepper
│       ├── data/          # Derived per-step data structures
│       └── policies/      # Vol schemes (Exact, NV, NCI) and Jump policies (Merton, Kou, None)
├── pricers/               # High-level pricer facades
│   ├── montecarlo/        # MonteCarloPricer
│   └── fourier/           # FourierPricer
└── utils/                 # General-purpose helpers: VectorOps, TablePrinter

benchmarks/                # Standalone verification programs (one per topic)
```

---

## Coding Guidelines

- **C++20** throughout. Use concepts, `if constexpr`, and structured bindings where they improve clarity.
- **No runtime overhead in the hot path.** All model/instrument/scheme branching must be resolved at compile time via templates or `if constexpr`. No virtual calls inside pricing loops.
- **Header-only.** All new code goes under `include/myql/`. No `.cpp` files for the library itself.
- **No naked `new`/`delete`.** Prefer stack allocation; use smart pointers only when heap allocation is genuinely necessary.
- **CRTP for static polymorphism.** Follow the existing `ASVJModel<Derived>` pattern when adding new models.
- **Keep template parameters explicit** in public APIs — prefer `MonteCarloPricer<Model, Stepper, Instrument>` over auto-deduction magic that hides the instantiation.
- **Formatting:** 4-space indentation, `snake_case` for variables and functions, `PascalCase` for types and template parameters.
- Add a brief Doxygen-style comment block (`/** ... */`) to every new public class and non-trivial function.

---

## Adding a New Model

1. Create `include/myql/models/asvj/params/MyModelParams.hpp` with the parameter struct.
2. Create `include/myql/models/asvj/core/MyModel.hpp` inheriting from `ASVJModel<MyModel>` and implement:
   - `characteristic_function(u, t, params)` (required for Fourier pricing)
   - `drift_v(...)` and `diffusion_v(...)` (required for Monte Carlo stepping)
3. Add a type alias to `include/myql/models/asvj/ASVJModels.hpp` (or equivalent aggregator header).
4. Register the model string in `include/myql/dispatcher/ModelRegistry.hpp` so it is accessible from the runtime dispatch layer.
5. Add a benchmark in `benchmarks/` that validates the model against a known closed-form or published result.
6. Add the benchmark target to `CMakeLists.txt`.

---

## Adding a New Instrument

1. Create `include/myql/instruments/options/MyInstrument.hpp`.
2. Implement the `Tracker` concept that accumulates path statistics and returns a payoff given final spot / path history.
3. Add the new instrument to the appropriate aggregator header.
4. If it is a European-style payoff, register it in `InstrumentRegistry.hpp` for runtime dispatch.
5. Add a benchmark verifying the payoff against a known analytic formula (e.g., put-call parity, Jensen's inequality) across at least two different models.

---

## Writing Benchmarks / Tests

MyQuantLib uses standalone **benchmark executables** (not a unit-test framework) to keep the dependency footprint small.

Guidelines for benchmarks:

- Each file should test **one focused topic** (e.g., `asian_test.cpp`, `barrier_test.cpp`).
- Every numerical check must have a **model-independent or closed-form reference** (Black-Scholes formula, put-call parity, Jensen's inequality, etc.). Avoid hard-coded "expected" numbers without justification.
- Use `std::cout` with clear pass/fail messages. A non-zero exit code (`return 1;`) signals failure.
- Document at the top of the file: which models are tested, which schemes, and what tolerance is used.

---

## Pull Request Checklist

Before opening a PR, please make sure:

- [ ] The code compiles cleanly with `-Wall -Wextra` and no new warnings.
- [ ] All existing benchmarks still pass.
- [ ] New functionality has its own benchmark with a rigorous numerical reference.
- [ ] Mathematical claims are accompanied by a citation (paper, textbook, or arXiv link).
- [ ] No hardcoded absolute paths.
- [ ] The PR description explains **what** changed and **why**, citing any relevant academic references.

---

## Questions?

Open an issue or start a discussion on GitHub. For theoretical questions about the ASVJ framework or simulation schemes, the foundational reference is:

> **High Order Approximations and Simulation Schemes for the Log-Heston Process**  
> A. Alfonsi, E. Lombardo — *SIAM Journal on Financial Mathematics* ([DOI: 10.1137/24M1679720](https://epubs.siam.org/doi/10.1137/24M1679720))

Thank you for contributing! 🚀
