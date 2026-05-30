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

struct CsrMatrixView {
    int64_t nrows = 0;
    int64_t ncols = 0;
    int64_t nnz = 0;
    IndexType index_type = IndexType::Int32;
    DataLocation location = DataLocation::Device;
    const void* row_offsets = nullptr;
    const void* col_indices = nullptr;
    const double* values = nullptr;
};

struct DenseVectorView {
    int64_t size = 0;
    DataLocation location = DataLocation::Device;
    double* values = nullptr;
};

}  // namespace custom_linear_solver
