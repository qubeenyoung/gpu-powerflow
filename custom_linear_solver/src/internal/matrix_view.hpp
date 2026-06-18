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

struct CsrMatrixView {
  int64_t nrows = 0;
  int64_t ncols = 0;
  int64_t nnz = 0;
  IndexType index_type = IndexType::kInt32;
  DataLocation location = DataLocation::kDevice;
  ValueType value_type = ValueType::kFloat64;
  const void* row_offsets = nullptr;
  const void* col_indices = nullptr;
  const void* values = nullptr;
};

struct DenseVectorView {
  int64_t size = 0;
  DataLocation location = DataLocation::kDevice;
  ValueType value_type = ValueType::kFloat64;
  void* values = nullptr;
};

}  // namespace custom_linear_solver
