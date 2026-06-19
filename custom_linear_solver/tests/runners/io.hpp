#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace custom_linear_solver::scripts {

// Host CSR matrix read from a Matrix Market file.
struct CsrMatrix {
  int rows = 0;
  int cols = 0;
  std::vector<int> row_ptr;     // CSR row pointers (rows+1)
  std::vector<int> col_idx;     // CSR column indices
  std::vector<double> values;   // CSR values

  int64_t nnz() const { return static_cast<int64_t>(values.size()); }
};

// Host dense vector read from a Matrix Market file.
struct DenseVector {
  int rows = 0;
  int cols = 0;
  std::vector<double> values;

  int64_t size() const { return static_cast<int64_t>(values.size()); }
};

CsrMatrix read_matrix_market_csr(const std::filesystem::path& path);
DenseVector read_matrix_market_vector(const std::filesystem::path& path);
void write_matrix_market_vector(const std::filesystem::path& path,
                                const std::vector<double>& values);

}  // namespace custom_linear_solver::scripts
