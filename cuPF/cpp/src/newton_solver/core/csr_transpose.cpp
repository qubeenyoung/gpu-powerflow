#include "newton_solver/core/csr_transpose.hpp"

#include <cstddef>

CsrTransposePattern build_transpose_pattern(const std::vector<int32_t>& row_ptr,
                                            const std::vector<int32_t>& col_idx,
                                            int32_t dim)
{
    CsrTransposePattern out;
    const int32_t nnz = static_cast<int32_t>(col_idx.size());
    out.row_ptr.assign(static_cast<std::size_t>(dim + 1), 0);
    out.col_idx.assign(static_cast<std::size_t>(nnz), 0);
    out.src_to_transpose_pos.assign(static_cast<std::size_t>(nnz), -1);

    for (int32_t k = 0; k < nnz; ++k) {
        ++out.row_ptr[static_cast<std::size_t>(col_idx[static_cast<std::size_t>(k)] + 1)];
    }
    for (int32_t row = 0; row < dim; ++row) {
        out.row_ptr[static_cast<std::size_t>(row + 1)] +=
            out.row_ptr[static_cast<std::size_t>(row)];
    }

    std::vector<int32_t> cursor = out.row_ptr;
    for (int32_t row = 0; row < dim; ++row) {
        for (int32_t k = row_ptr[static_cast<std::size_t>(row)];
             k < row_ptr[static_cast<std::size_t>(row + 1)]; ++k) {
            const int32_t col = col_idx[static_cast<std::size_t>(k)];
            const int32_t dst = cursor[static_cast<std::size_t>(col)]++;
            out.col_idx[static_cast<std::size_t>(dst)] = row;
            out.src_to_transpose_pos[static_cast<std::size_t>(k)] = dst;
        }
    }
    return out;
}
