cmake_minimum_required(VERSION 3.14)

# 1. Project Name and Version
project(MyQuantLib VERSION 0.1.0 LANGUAGES CXX)

# 2. Standard C++ Settings
# We use C++20 for features like designated initializers and concepts
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 3. Compiler Options (Optimization is key for Quant code)
if(MSVC)
    add_compile_options(/W4 /O2 /fp:fast)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native -ffast-math)
endif()

# 4. Find Dependencies
# --- OpenMP (Critical for Monte Carlo parallelization) ---
find_package(OpenMP REQUIRED)

# --- Eigen3 (Linear Algebra) ---
# If Eigen is not found, you might need to set Eigen3_DIR or just include it manually
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# 5. Define Your Library (The "Core")
# We explicitly list sources here. As you add engines, add them to this list.
add_library(MyQuantLib
    src/models/HestonModel.cpp
    # Future additions:
    # src/numerics/Integrator.cpp
    # src/engines/HestonFourierEngine.cpp
)

# Link headers and dependencies to the library
target_include_directories(MyQuantLib PUBLIC 
    "${PROJECT_SOURCE_DIR}/include"
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(MyQuantLib PUBLIC 
    OpenMP::OpenMP_CXX 
    Eigen3::Eigen
)

# 6. Define Your Executable (The "Runner")
# This replaces your old single-file main()
add_executable(RunBenchmarks
    examples/main.cpp
)

# Link the runner to your library
target_link_libraries(RunBenchmarks PRIVATE MyQuantLib)

# 7. (Future) Testing Section
# enable_testing()
# add_subdirectory(tests)
