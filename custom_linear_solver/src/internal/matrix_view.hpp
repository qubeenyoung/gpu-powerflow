#pragma once

#include <cstdint>

namespace custom_linear_solver {

enum class IndexType {
  kInt32,
  kInt64,
};

enum class DataLocation {
  kHost,
  kDevice,
};

enum class ValueType {
  kFloat64,
  kFloat32,
};

// Non-owning view of a CSR matrix. Caller owns the pointed-to storage.
struct CsrMatrixView {
  int64_t nrows = 0;  // row count
  int64_t ncols = 0;  // column count
  int64_t nnz = 0;    // nonzero count
  IndexType index_type = IndexType::kInt32;
  DataLocation location = DataLocation::kDevice;
  ValueType value_type = ValueType::kFloat64;
  const void* row_offsets = nullptr;  // CSR row pointers (nrows+1)
  const void* col_indices = nullptr;  // CSR column indices (nnz)
  const void* values = nullptr;       // CSR values (nnz); may be null pre-Setup
};

// Non-owning view of a dense vector. Caller owns the pointed-to storage.
struct DenseVectorView {
  int64_t size = 0;  // element count
  DataLocation location = DataLocation::kDevice;
  ValueType value_type = ValueType::kFloat64;
  void* values = nullptr;  // element storage
};

}  // namespace custom_linear_solver
