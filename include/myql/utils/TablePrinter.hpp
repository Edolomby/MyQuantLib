#pragma once
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace myql::utils {

// -------------------------------------------------------------------------
// Float formatting options
// -------------------------------------------------------------------------
enum class FloatFormat { Default, Scientific };

struct TableConfig {
  FloatFormat format = FloatFormat::Default;
  int precision = 8; // significant digits (Default) or mantissa digits (Sci)
};

// -------------------------------------------------------------------------
// Helper: Convert any type to string, respecting TableConfig
// -------------------------------------------------------------------------
template <typename T>
std::string toString(const T &value, const TableConfig &cfg = {}) {
  std::ostringstream oss;
  if constexpr (std::is_floating_point_v<T>) {
    if (cfg.format == FloatFormat::Scientific)
      oss << std::scientific << std::setprecision(cfg.precision) << value;
    else
      oss << std::defaultfloat << std::setprecision(cfg.precision) << value;
  } else {
    oss << value;
  }
  return oss.str();
}

// -------------------------------------------------------------------------
// Generic Table Printer for Vectors
//
// TableConfig is the FIRST argument (before headers) so that the variadic
// pack deduction is unambiguous. A convenience overload without config is
// provided so existing call sites continue to compile unchanged.
//
// Usage:
//   // default formatting (backward-compatible)
//   printVectors({"Col1","Col2"}, vec1, vec2);
//
//   // scientific notation with 4 mantissa digits
//   printVectors(TableConfig{FloatFormat::Scientific, 4},
//                {"Col1","Col2"}, vec1, vec2);
// -------------------------------------------------------------------------
template <typename... Ts>
void printVectors(const TableConfig &cfg,
                  const std::vector<std::string> &headers,
                  const std::vector<Ts> &...vecs) {

  // 1. Validation
  if (headers.size() != sizeof...(vecs)) {
    throw std::invalid_argument(
        "Error: Headers count must match the number of vectors.");
  }

  auto get_str_val = [&](const auto &val) { return toString(val, cfg); };

  // 2. Calculate Column Widths
  std::vector<size_t> widths;
  for (const auto &header : headers)
    widths.push_back(header.length());

  size_t i = 0;
  auto calculate_widths = [&](const auto &vec) {
    for (const auto &item : vec)
      widths[i] = std::max(widths[i], get_str_val(item).length());
    i++;
  };
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
  for (size_t j = 0; j < headers.size(); ++j)
    std::cout << " " << std::left << std::setw(widths[j]) << headers[j] << " |";
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
      if (row < vec.size())
        std::cout << get_str_val(vec[row]);
      else
        std::cout << "";
      std::cout << " |";
      current_col++;
    };
    (..., print_cell(vecs));
    std::cout << "\n";
  }

  // Bottom Border
  print_separator();
}

// Backward-compatible overload: no config → default formatting
template <typename... Ts>
void printVectors(const std::vector<std::string> &headers,
                  const std::vector<Ts> &...vecs) {
  printVectors(TableConfig{}, headers, vecs...);
}

} // namespace myql::utils