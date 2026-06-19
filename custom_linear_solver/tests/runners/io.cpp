#include "io.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace custom_linear_solver::scripts {
namespace {

struct MatrixMarketInfo {
  std::string object;
  std::string format;
  std::string field;
  std::string symmetry;
};

struct CoordinateEntry {
  int row = 0;
  int col = 0;
  double value = 0.0;
};

std::string to_lower(std::string text) {
  for (char& ch : text) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return text;
}

std::string next_data_line(std::istream& input) {
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t first = line.find_first_not_of(" \t\r\n");
    if (first != std::string::npos && line[first] != '%') {
      return line;
    }
  }
  throw std::runtime_error("unexpected end of Matrix Market file");
}

MatrixMarketInfo read_header(std::istream& input) {
  std::string line;
  if (!std::getline(input, line)) {
    throw std::runtime_error("empty Matrix Market file");
  }

  std::string banner;
  MatrixMarketInfo info;
  std::istringstream stream(line);
  stream >> banner >> info.object >> info.format >> info.field >> info.symmetry;
  if (!stream) {
    throw std::runtime_error("invalid Matrix Market header");
  }

  banner = to_lower(banner);
  info.object = to_lower(info.object);
  info.format = to_lower(info.format);
  info.field = to_lower(info.field);
  info.symmetry = to_lower(info.symmetry);

  if (banner != "%%matrixmarket" || info.object != "matrix") {
    throw std::runtime_error("unsupported Matrix Market object");
  }
  return info;
}

int checked_int(long long value, const std::string& name) {
  if (value < 0 || value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(name + " does not fit int32");
  }
  return static_cast<int>(value);
}

}  // namespace

CsrMatrix read_matrix_market_csr(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open Matrix Market matrix: " +
                             path.string());
  }

  const MatrixMarketInfo info = read_header(input);
  const bool pattern_values = info.field == "pattern";
  const bool complex_values = info.field == "complex";
  const bool mirror_triangle =
      info.symmetry == "symmetric" || info.symmetry == "hermitian";

  if (info.format != "coordinate") {
    throw std::runtime_error("expected coordinate Matrix Market matrix: " +
                             path.string());
  }
  if (info.field != "real" && info.field != "integer" &&
      info.field != "pattern" && info.field != "complex") {
    throw std::runtime_error("unsupported Matrix Market matrix field: " +
                             info.field);
  }
  if (info.symmetry != "general" && info.symmetry != "symmetric" &&
      info.symmetry != "hermitian") {
    throw std::runtime_error("unsupported Matrix Market symmetry: " +
                             info.symmetry);
  }

  long long rows64 = 0;
  long long cols64 = 0;
  long long declared64 = 0;
  std::istringstream shape(next_data_line(input));
  shape >> rows64 >> cols64 >> declared64;
  if (!shape || rows64 <= 0 || cols64 <= 0 || declared64 < 0) {
    throw std::runtime_error("invalid Matrix Market matrix shape: " +
                             path.string());
  }
  const int rows = checked_int(rows64, "row count");
  const int cols = checked_int(cols64, "column count");
  const int declared_nnz = checked_int(declared64, "nonzero count");

  // Read coordinate entries, mirroring the off-diagonal for symmetric input.
  std::vector<CoordinateEntry> entries;
  entries.reserve(static_cast<std::size_t>(declared_nnz) *
                  (mirror_triangle ? 2u : 1u));

  for (int line_index = 0; line_index < declared_nnz; ++line_index) {
    int one_based_row = 0;
    int one_based_col = 0;
    double value = 1.0;
    double ignored_imaginary = 0.0;

    std::istringstream row_stream(next_data_line(input));
    if (pattern_values) {
      row_stream >> one_based_row >> one_based_col;
    } else if (complex_values) {
      row_stream >> one_based_row >> one_based_col >> value >>
          ignored_imaginary;
    } else {
      row_stream >> one_based_row >> one_based_col >> value;
    }
    if (!row_stream) {
      throw std::runtime_error("failed to parse Matrix Market matrix entry: " +
                               path.string());
    }

    CoordinateEntry entry{one_based_row - 1, one_based_col - 1, value};
    if (entry.row < 0 || entry.row >= rows || entry.col < 0 ||
        entry.col >= cols) {
      throw std::runtime_error("Matrix Market coordinate out of range: " +
                               path.string());
    }
    entries.push_back(entry);
    if (mirror_triangle && entry.row != entry.col) {
      entries.push_back({entry.col, entry.row, entry.value});
    }
  }

  // Sort row-major, then merge duplicate coordinates by summing values.
  std::sort(entries.begin(), entries.end(),
            [](const CoordinateEntry& lhs, const CoordinateEntry& rhs) {
              if (lhs.row != rhs.row) return lhs.row < rhs.row;
              return lhs.col < rhs.col;
            });

  std::vector<CoordinateEntry> merged;
  merged.reserve(entries.size());
  for (const CoordinateEntry& entry : entries) {
    if (!merged.empty() && merged.back().row == entry.row &&
        merged.back().col == entry.col) {
      merged.back().value += entry.value;
    } else {
      merged.push_back(entry);
    }
  }

  // Build CSR: count per-row, prefix-sum into row_ptr, then fill col/values.
  CsrMatrix matrix;
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.row_ptr.assign(static_cast<std::size_t>(rows) + 1, 0);
  matrix.col_idx.resize(merged.size());
  matrix.values.resize(merged.size());

  for (const CoordinateEntry& entry : merged) {
    ++matrix.row_ptr[entry.row + 1];
  }
  for (int row = 0; row < rows; ++row) {
    matrix.row_ptr[row + 1] += matrix.row_ptr[row];
  }
  for (std::size_t index = 0; index < merged.size(); ++index) {
    matrix.col_idx[index] = merged[index].col;
    matrix.values[index] = merged[index].value;
  }

  return matrix;
}

DenseVector read_matrix_market_vector(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open Matrix Market vector: " +
                             path.string());
  }

  const MatrixMarketInfo info = read_header(input);
  if (info.format != "array") {
    throw std::runtime_error("expected array Matrix Market vector: " +
                             path.string());
  }
  if (info.field != "real" && info.field != "integer") {
    throw std::runtime_error("unsupported Matrix Market vector field: " +
                             info.field);
  }
  if (info.symmetry != "general") {
    throw std::runtime_error("Matrix Market vector must be general: " +
                             path.string());
  }

  long long rows64 = 0;
  long long cols64 = 0;
  std::istringstream shape(next_data_line(input));
  shape >> rows64 >> cols64;
  if (!shape || rows64 <= 0 || cols64 <= 0) {
    throw std::runtime_error("invalid Matrix Market vector shape: " +
                             path.string());
  }

  DenseVector vector;
  vector.rows = checked_int(rows64, "vector row count");
  vector.cols = checked_int(cols64, "vector column count");
  const std::size_t expected = static_cast<std::size_t>(vector.rows) *
                               static_cast<std::size_t>(vector.cols);
  vector.values.reserve(expected);

  while (vector.values.size() < expected) {
    std::istringstream value_stream(next_data_line(input));
    double value = 0.0;
    while (value_stream >> value) {
      vector.values.push_back(value);
      if (vector.values.size() == expected) break;
    }
    if (!value_stream.eof()) {
      throw std::runtime_error("failed to parse Matrix Market vector value: " +
                               path.string());
    }
  }

  return vector;
}

void write_matrix_market_vector(const std::filesystem::path& path,
                                const std::vector<double>& values) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error(
        "failed to open Matrix Market vector for writing: " + path.string());
  }

  output << "%%MatrixMarket matrix array real general\n"
         << "% generated by custom_linear_solver_run\n"
         << values.size() << " 1\n"
         << std::setprecision(17);
  for (double value : values) {
    output << value << '\n';
  }
}

}  // namespace custom_linear_solver::scripts
