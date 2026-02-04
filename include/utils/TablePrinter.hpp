#pragma once
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace utils {

// -------------------------------------------------------------------------
// Helper: Convert any type to string with smart formatting
// -------------------------------------------------------------------------
template <typename T> std::string toString(const T &value) {
  std::ostringstream oss;
  if constexpr (std::is_floating_point_v<T>) {
    // 'defaultfloat' chooses fixed or scientific based on magnitude
    // Adjust precision if you need more digits for benchmark comparisons
    oss << std::defaultfloat << std::setprecision(8) << value;
  } else {
    oss << value;
  }
  return oss.str();
}

// -------------------------------------------------------------------------
// Generic Table Printer for Vectors
// Can accept any number of vectors of different types
// Usage: printVectors({"Col1", "Col2"}, vec1, vec2);
// -------------------------------------------------------------------------
template <typename... Ts>
void printVectors(const std::vector<std::string> &headers,
                  const std::vector<Ts> &...vecs) {

  // 1. Validation
  if (headers.size() != sizeof...(vecs)) {
    throw std::invalid_argument(
        "Error: Headers count must match the number of vectors.");
  }

  // Helper to ensure we use the exact same string for width calc and printing
  auto get_str_val = [](const auto &val) { return toString(val); };

  // 2. Calculate Column Widths
  std::vector<size_t> widths;
  for (const auto &header : headers) {
    widths.push_back(header.length());
  }

  size_t i = 0;
  auto calculate_widths = [&](const auto &vec) {
    for (const auto &item : vec) {
      widths[i] = std::max(widths[i], get_str_val(item).length());
    }
    i++;
  };
  // Fold expression to apply lambda to all vectors
  (..., calculate_widths(vecs));

  // 3. Print Table
  auto print_separator = [&]() {
    std::cout << "+";
    for (const auto &w : widths)
      std::cout << std::string(w + 2, '-') << "+";
    std::cout << "\n";
  };

  // Top Border
  print_separator();

  // Headers
  std::cout << "|";
  for (size_t j = 0; j < headers.size(); ++j) {
    std::cout << " " << std::left << std::setw(widths[j]) << headers[j] << " |";
  }
  std::cout << "\n";

  // Separator
  print_separator();

  // Data Rows
  const size_t max_rows = std::max({vecs.size()...});
  for (size_t row = 0; row < max_rows; ++row) {
    std::cout << "|";
    size_t current_col = 0;

    auto print_cell = [&](const auto &vec) {
      std::cout << " " << std::left << std::setw(widths[current_col]);
      if (row < vec.size()) {
        std::cout << get_str_val(vec[row]);
      } else {
        std::cout << ""; // Empty cell if vector is shorter
      }
      std::cout << " |";
      current_col++;
    };
    (..., print_cell(vecs));
    std::cout << "\n";
  }

  // Bottom Border
  print_separator();
}

} // namespace utils