#pragma once

#include <cstdint>

namespace custom_linear_solver {

enum class IndexType {
    Int32,
    Int64,
};

enum class DataLocation {
    Host,
    Device,
};

enum class ValueType {
    Float64,
    Float32,
};

struct CsrMatrixView {
    int64_t nrows = 0;
    int64_t ncols = 0;
    int64_t nnz = 0;
    IndexType index_type = IndexType::Int32;
    DataLocation location = DataLocation::Device;
    ValueType value_type = ValueType::Float64;
    const void* row_offsets = nullptr;
    const void* col_indices = nullptr;
    const void* values = nullptr;
};

struct DenseVectorView {
    int64_t size = 0;
    DataLocation location = DataLocation::Device;
    ValueType value_type = ValueType::Float64;
    void* values = nullptr;
};

}  // namespace custom_linear_solver
