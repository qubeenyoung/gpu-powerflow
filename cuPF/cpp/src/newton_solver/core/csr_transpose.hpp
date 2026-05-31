#pragma once

#include <cstdint>
#include <vector>

// Generic CSR -> CSC (transpose) sparsity helpers shared by the adjoint
// pipeline and the cuDSS linear-solve backend. The pattern is built once from
// the forward Jacobian structure and reused to scatter batched values into
// their transposed positions.

struct CsrTransposePattern {
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<int32_t> src_to_transpose_pos;
};

// Build the transpose (CSC-as-CSR) sparsity pattern for a square matrix given
// its CSR row_ptr/col_idx. src_to_transpose_pos[k] maps source nnz index k to
// its position in the transposed layout.
CsrTransposePattern build_transpose_pattern(const std::vector<int32_t>& row_ptr,
                                            const std::vector<int32_t>& col_idx,
                                            int32_t dim);

// Scatter a batch of CSR value arrays into their transposed positions using a
// precomputed src_to_transpose_pos map.
template <typename T>
std::vector<T> transpose_batched_values(const std::vector<T>& values,
                                        const std::vector<int32_t>& src_to_transpose_pos,
                                        int32_t batch_size,
                                        int32_t nnz)
{
    std::vector<T> transposed(static_cast<std::size_t>(batch_size) *
                              static_cast<std::size_t>(nnz), T(0));
    for (int32_t b = 0; b < batch_size; ++b) {
        const std::size_t base =
            static_cast<std::size_t>(b) * static_cast<std::size_t>(nnz);
        for (int32_t k = 0; k < nnz; ++k) {
            const int32_t dst = src_to_transpose_pos[static_cast<std::size_t>(k)];
            transposed[base + static_cast<std::size_t>(dst)] =
                values[base + static_cast<std::size_t>(k)];
        }
    }
    return transposed;
}
