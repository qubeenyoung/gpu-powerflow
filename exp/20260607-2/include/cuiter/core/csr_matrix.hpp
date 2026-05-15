#pragma once

#include <cstdint>
#include <vector>

namespace cuiter {

struct CsrMatrix {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<double> values;

    int32_t nnz() const
    {
        return static_cast<int32_t>(col_idx.size());
    }

    bool has_values() const
    {
        return values.size() == col_idx.size();
    }
};

struct DeviceCsrMatrixView {
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t nnz = 0;
    const int32_t* row_ptr = nullptr;
    const int32_t* col_idx = nullptr;
    const double* values = nullptr;
};

}  // namespace cuiter
