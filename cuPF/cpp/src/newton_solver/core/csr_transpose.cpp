#include "newton_solver/core/csr_transpose.hpp"

#include <cstddef>

// Build the sparsity pattern of J^T from J's CSR pattern via counting sort.
// Returns the transposed row_ptr/col_idx plus src_to_transpose_pos, a map from
// each source nonzero index to its slot in J^T so the numeric values can later
// be scattered without recomputing the structure. (All size_t casts simply
// widen non-negative int32 indices for vector access.)
CsrTransposePattern build_transpose_pattern(const std::vector<int32_t>& row_ptr,
                                            const std::vector<int32_t>& col_idx,
                                            int32_t dim)
{
    CsrTransposePattern out;
    const int32_t nnz = static_cast<int32_t>(col_idx.size());  // narrow size_t -> int32 count
    out.row_ptr.assign(static_cast<std::size_t>(dim + 1), 0);
    out.col_idx.assign(static_cast<std::size_t>(nnz), 0);
    out.src_to_transpose_pos.assign(static_cast<std::size_t>(nnz), -1);

    // 1) Count entries per transposed row (= per original column).
    for (int32_t k = 0; k < nnz; ++k) {
        ++out.row_ptr[static_cast<std::size_t>(col_idx[static_cast<std::size_t>(k)] + 1)];
    }
    // 2) Prefix-sum the counts into row offsets for J^T.
    for (int32_t row = 0; row < dim; ++row) {
        out.row_ptr[static_cast<std::size_t>(row + 1)] +=
            out.row_ptr[static_cast<std::size_t>(row)];
    }

    // 3) Walk J row by row; each (row, col) becomes (col, row) in J^T, placed at
    //    the running cursor for that transposed row.
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
