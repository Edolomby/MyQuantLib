cmake_minimum_required(VERSION 3.15)
project(MyQuantLib VERSION 1.0 LANGUAGES CXX)

# ==============================================================================
# 1. COMPILER CONFIGURATION
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization Flags (Auto-detected)
if(MSVC)
    add_compile_options(/O2 /fp:fast /arch:AVX2)
else()
    # -O3: Max optimization
    # -march=native: Use instructions specific to YOUR cpu (AVX2/AVX512)
    # -ffast-math: Allow aggressive float optimizations (crucial for CF)
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)
endif()

# ==============================================================================
# 2. DEPENDENCIES
# ==============================================================================

# A. OpenMP (Parallelism)
find_package(OpenMP REQUIRED)

# B. Boost (Random Numbers)
# boost/random/xoshiro.hpp was introduced in Boost 1.88.0
# If a local Boost bundle exists at the repo root, prefer it;
# otherwise CMake will search system paths or a user-supplied -DBOOST_ROOT=...
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/boost_1_90_0")
    set(BOOST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/boost_1_90_0")
endif()
find_package(Boost 1.88.0 REQUIRED)

# C. Eigen (Linear Algebra)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ==============================================================================
# 3. LIBRARY TARGET (Header-Only)
# ==============================================================================
# Define MyQuantLib as an Interface (Header-Only) Library
add_library(MyQuantLib INTERFACE)

# Where are the headers?
target_include_directories(MyQuantLib INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link Dependencies
target_link_libraries(MyQuantLib INTERFACE 
    OpenMP::OpenMP_CXX
    Boost::boost
    Eigen3::Eigen
)

# ==============================================================================
# 4. BENCHMARKS
# ==============================================================================
macro(add_benchmark NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} PRIVATE MyQuantLib)
endmacro()

# Existing benchmark
add_benchmark(BenchFourierMC benchmarks/benchmark_fourier_vs_mc.cpp)
add_benchmark(TestSmile benchmarks/test_smile_suite.cpp)
add_benchmark(TimerMCstrip benchmarks/timer_MC_strip_vs_std.cpp)
add_benchmark(DigitalTest benchmarks/digital_verification.cpp)
add_benchmark(SplineTest benchmarks/spline_test.cpp)
add_benchmark(CirTest benchmarks/cir_test.cpp)
add_benchmark(BarrierTest benchmarks/barrier_test.cpp)
add_benchmark(EssentialGreekTest benchmarks/EssentialGreekTest.cpp)
add_benchmark(RuntimeDispatchDemo benchmarks/runtime_dispatch_demo.cpp)
add_benchmark(AsianTest benchmarks/asian_test.cpp)
add_benchmark(LookbackTest benchmarks/lookback_test.cpp)
add_benchmark(InFlightTest benchmarks/in_flight_test.cpp)
add_benchmark(DigitalSmoothingTest benchmarks/digital_smoothing_test.cpp)

# ==============================================================================
# 5. STATUS REPORT
# ==============================================================================
message(STATUS "---------------------------------------------------")
message(STATUS "  MyQuantLib Configured")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Boost Version: ${Boost_VERSION}")
message(STATUS "  Eigen Found: ${Eigen3_FOUND}")
message(STATUS "---------------------------------------------------")
