#include "cuiter/kernels/block_jacobi_kernels.hpp"

#include "cuiter/common/cuda_utils.hpp"

#include <cstdint>

namespace cuiter::kernels {
namespace {

constexpr int32_t kBlockSize = 256;

int32_t grid_for(int32_t n)
{
    return (n + kBlockSize - 1) / kBlockSize;
}

__device__ double atomic_add_double(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__global__ void scatter_csr_values_kernel(int32_t nnz,
                                          const int32_t* __restrict__ perm_value_source,
                                          const double* __restrict__ original_values,
                                          double* __restrict__ permuted_values)
{
    const int32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < nnz) {
        permuted_values[pos] = original_values[perm_value_source[pos]];
    }
}

__global__ void extract_dense_blocks_f32_kernel(int32_t nnz_perm,
                                                const int32_t* __restrict__ dense_block_offsets,
                                                const int32_t* __restrict__ dense_local_rows,
                                                const int32_t* __restrict__ dense_local_cols,
                                                const double* __restrict__ permuted_values,
                                                float* __restrict__ dense_blocks)
{
    const int32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= nnz_perm) {
        return;
    }
    const int32_t block_offset = dense_block_offsets[pos];
    if (block_offset >= 0) {
        dense_blocks[block_offset + dense_local_rows[pos] + dense_local_cols[pos]] =
            static_cast<float>(permuted_values[pos]);
    }
}

__global__ void extract_dense_blocks_f64_kernel(int32_t nnz_perm,
                                                const int32_t* __restrict__ dense_block_offsets,
                                                const int32_t* __restrict__ dense_local_rows,
                                                const int32_t* __restrict__ dense_local_cols,
                                                const double* __restrict__ permuted_values,
                                                double* __restrict__ dense_blocks)
{
    const int32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= nnz_perm) {
        return;
    }
    const int32_t block_offset = dense_block_offsets[pos];
    if (block_offset >= 0) {
        dense_blocks[block_offset + dense_local_rows[pos] + dense_local_cols[pos]] =
            permuted_values[pos];
    }
}

__global__ void extract_ras_dense_blocks_f32_kernel(int32_t map_nnz,
                                                    const int32_t* __restrict__ source_positions,
                                                    const int32_t* __restrict__ dense_block_offsets,
                                                    const int32_t* __restrict__ dense_local_rows,
                                                    const int32_t* __restrict__ dense_local_cols,
                                                    const double* __restrict__ permuted_values,
                                                    float* __restrict__ dense_blocks)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= map_nnz) {
        return;
    }
    dense_blocks[dense_block_offsets[k] + dense_local_rows[k] + dense_local_cols[k]] =
        static_cast<float>(permuted_values[source_positions[k]]);
}

__global__ void extract_ras_dense_blocks_f64_kernel(int32_t map_nnz,
                                                    const int32_t* __restrict__ source_positions,
                                                    const int32_t* __restrict__ dense_block_offsets,
                                                    const int32_t* __restrict__ dense_local_rows,
                                                    const int32_t* __restrict__ dense_local_cols,
                                                    const double* __restrict__ permuted_values,
                                                    double* __restrict__ dense_blocks)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= map_nnz) {
        return;
    }
    dense_blocks[dense_block_offsets[k] + dense_local_rows[k] + dense_local_cols[k]] =
        permuted_values[source_positions[k]];
}

__global__ void add_block_diagonal_shift_f32_kernel(int32_t num_blocks,
                                                    int32_t leading_dim,
                                                    const int32_t* __restrict__ block_sizes,
                                                    float shift,
                                                    float* __restrict__ dense_blocks)
{
    const int32_t block = blockIdx.x;
    const int32_t local = threadIdx.x;
    if (block >= num_blocks || local >= leading_dim) {
        return;
    }

    const int32_t base = block * leading_dim * leading_dim;
    const int32_t size = block_sizes[block];
    if (local < size) {
        dense_blocks[base + local + local * leading_dim] += shift;
    } else {
        dense_blocks[base + local + local * leading_dim] = 1.0f;
    }
}

__global__ void add_block_diagonal_shift_f64_kernel(int32_t num_blocks,
                                                    int32_t leading_dim,
                                                    const int32_t* __restrict__ block_sizes,
                                                    double shift,
                                                    double* __restrict__ dense_blocks)
{
    const int32_t block = blockIdx.x;
    const int32_t local = threadIdx.x;
    if (block >= num_blocks || local >= leading_dim) {
        return;
    }

    const int32_t base = block * leading_dim * leading_dim;
    const int32_t size = block_sizes[block];
    if (local < size) {
        dense_blocks[base + local + local * leading_dim] += shift;
    } else {
        dense_blocks[base + local + local * leading_dim] = 1.0;
    }
}

__global__ void block_inverse_apply_f32_kernel(int32_t leading_dim,
                                               const int32_t* __restrict__ block_starts,
                                               const int32_t* __restrict__ block_sizes,
                                               const float* __restrict__ inverse_blocks,
                                               const double* __restrict__ rhs,
                                               double* __restrict__ out)
{
    const int32_t block = blockIdx.x;
    const int32_t row = threadIdx.x;
    const int32_t size = block_sizes[block];
    if (row >= size) {
        return;
    }

    const int32_t start = block_starts[block];
    const int32_t base = block * leading_dim * leading_dim;
    double sum = 0.0;
    for (int32_t col = 0; col < size; ++col) {
        sum += static_cast<double>(inverse_blocks[base + row + col * leading_dim]) *
               rhs[start + col];
    }
    out[start + row] = sum;
}

__global__ void block_inverse_apply_f64_kernel(int32_t leading_dim,
                                               const int32_t* __restrict__ block_starts,
                                               const int32_t* __restrict__ block_sizes,
                                               const double* __restrict__ inverse_blocks,
                                               const double* __restrict__ rhs,
                                               double* __restrict__ out)
{
    const int32_t block = blockIdx.x;
    const int32_t row = threadIdx.x;
    const int32_t size = block_sizes[block];
    if (row >= size) {
        return;
    }

    const int32_t start = block_starts[block];
    const int32_t base = block * leading_dim * leading_dim;
    double sum = 0.0;
    for (int32_t col = 0; col < size; ++col) {
        sum += inverse_blocks[base + row + col * leading_dim] * rhs[start + col];
    }
    out[start + row] = sum;
}

__global__ void ras_inverse_apply_f32_kernel(int32_t leading_dim,
                                             const int32_t* __restrict__ local_offsets,
                                             const int32_t* __restrict__ local_to_global,
                                             const int32_t* __restrict__ owned_sizes,
                                             const int32_t* __restrict__ local_sizes,
                                             const float* __restrict__ inverse_blocks,
                                             const double* __restrict__ rhs,
                                             double* __restrict__ out)
{
    const int32_t block = blockIdx.x;
    const int32_t row = threadIdx.x;
    const int32_t owned_size = owned_sizes[block];
    if (row >= owned_size) {
        return;
    }
    const int32_t local_size = local_sizes[block];
    const int32_t offset = local_offsets[block];
    const int32_t base = block * leading_dim * leading_dim;
    double sum = 0.0;
    for (int32_t col = 0; col < local_size; ++col) {
        sum += static_cast<double>(inverse_blocks[base + row + col * leading_dim]) *
               rhs[local_to_global[offset + col]];
    }
    out[local_to_global[offset + row]] = sum;
}

__global__ void ras_inverse_apply_f64_kernel(int32_t leading_dim,
                                             const int32_t* __restrict__ local_offsets,
                                             const int32_t* __restrict__ local_to_global,
                                             const int32_t* __restrict__ owned_sizes,
                                             const int32_t* __restrict__ local_sizes,
                                             const double* __restrict__ inverse_blocks,
                                             const double* __restrict__ rhs,
                                             double* __restrict__ out)
{
    const int32_t block = blockIdx.x;
    const int32_t row = threadIdx.x;
    const int32_t owned_size = owned_sizes[block];
    if (row >= owned_size) {
        return;
    }
    const int32_t local_size = local_sizes[block];
    const int32_t offset = local_offsets[block];
    const int32_t base = block * leading_dim * leading_dim;
    double sum = 0.0;
    for (int32_t col = 0; col < local_size; ++col) {
        sum += inverse_blocks[base + row + col * leading_dim] *
               rhs[local_to_global[offset + col]];
    }
    out[local_to_global[offset + row]] = sum;
}

__global__ void block_lu_solve_apply_f32_kernel(int32_t leading_dim,
                                                const int32_t* __restrict__ block_starts,
                                                const int32_t* __restrict__ block_sizes,
                                                const float* __restrict__ lu_blocks,
                                                const int32_t* __restrict__ pivots,
                                                const double* __restrict__ rhs,
                                                double* __restrict__ out)
{
    const int32_t block = blockIdx.x;
    if (threadIdx.x != 0) {
        return;
    }

    const int32_t size = block_sizes[block];
    const int32_t start = block_starts[block];
    const int32_t base = block * leading_dim * leading_dim;
    const int32_t pivot_base = block * leading_dim;
    double x[64];

    for (int32_t i = 0; i < size; ++i) {
        x[i] = rhs[start + i];
    }
    for (int32_t i = 0; i < size; ++i) {
        const int32_t pivot = pivots[pivot_base + i] - 1;
        if (pivot >= 0 && pivot < size && pivot != i) {
            const double tmp = x[i];
            x[i] = x[pivot];
            x[pivot] = tmp;
        }
    }
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j < i; ++j) {
            x[i] -= static_cast<double>(lu_blocks[base + i + j * leading_dim]) * x[j];
        }
    }
    for (int32_t i = size - 1; i >= 0; --i) {
        for (int32_t j = i + 1; j < size; ++j) {
            x[i] -= static_cast<double>(lu_blocks[base + i + j * leading_dim]) * x[j];
        }
        x[i] /= static_cast<double>(lu_blocks[base + i + i * leading_dim]);
    }
    for (int32_t i = 0; i < size; ++i) {
        out[start + i] = x[i];
    }
}

__global__ void block_lu_solve_apply_f64_kernel(int32_t leading_dim,
                                                const int32_t* __restrict__ block_starts,
                                                const int32_t* __restrict__ block_sizes,
                                                const double* __restrict__ lu_blocks,
                                                const int32_t* __restrict__ pivots,
                                                const double* __restrict__ rhs,
                                                double* __restrict__ out)
{
    const int32_t block = blockIdx.x;
    if (threadIdx.x != 0) {
        return;
    }

    const int32_t size = block_sizes[block];
    const int32_t start = block_starts[block];
    const int32_t base = block * leading_dim * leading_dim;
    const int32_t pivot_base = block * leading_dim;
    double x[64];

    for (int32_t i = 0; i < size; ++i) {
        x[i] = rhs[start + i];
    }
    for (int32_t i = 0; i < size; ++i) {
        const int32_t pivot = pivots[pivot_base + i] - 1;
        if (pivot >= 0 && pivot < size && pivot != i) {
            const double tmp = x[i];
            x[i] = x[pivot];
            x[pivot] = tmp;
        }
    }
    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = 0; j < i; ++j) {
            x[i] -= lu_blocks[base + i + j * leading_dim] * x[j];
        }
    }
    for (int32_t i = size - 1; i >= 0; --i) {
        for (int32_t j = i + 1; j < size; ++j) {
            x[i] -= lu_blocks[base + i + j * leading_dim] * x[j];
        }
        x[i] /= lu_blocks[base + i + i * leading_dim];
    }
    for (int32_t i = 0; i < size; ++i) {
        out[start + i] = x[i];
    }
}

__global__ void assemble_coarse_matrix_f32_kernel(int32_t rows,
                                                  int32_t coarse_dim,
                                                  const int32_t* __restrict__ row_ptr,
                                                  const int32_t* __restrict__ col_idx,
                                                  const double* __restrict__ values,
                                                  const int32_t* __restrict__ block_ids,
                                                  const float* __restrict__ weights,
                                                  float* __restrict__ coarse_matrix)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int32_t row_block = block_ids[row];
    const float row_weight = weights[row];
    for (int32_t pos = row_ptr[row]; pos < row_ptr[row + 1]; ++pos) {
        const int32_t col = col_idx[pos];
        const int32_t col_block = block_ids[col];
        const float col_weight = weights[col];
        const float contribution =
            row_weight * static_cast<float>(values[pos]) * col_weight;
        atomicAdd(coarse_matrix + row_block + col_block * coarse_dim, contribution);
    }
}

__global__ void assemble_coarse_matrix_f64_kernel(int32_t rows,
                                                  int32_t coarse_dim,
                                                  const int32_t* __restrict__ row_ptr,
                                                  const int32_t* __restrict__ col_idx,
                                                  const double* __restrict__ values,
                                                  const int32_t* __restrict__ block_ids,
                                                  const double* __restrict__ weights,
                                                  double* __restrict__ coarse_matrix)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int32_t row_block = block_ids[row];
    const double row_weight = weights[row];
    for (int32_t pos = row_ptr[row]; pos < row_ptr[row + 1]; ++pos) {
        const int32_t col = col_idx[pos];
        const int32_t col_block = block_ids[col];
        const double col_weight = weights[col];
        const double contribution = row_weight * values[pos] * col_weight;
        atomic_add_double(coarse_matrix + row_block + col_block * coarse_dim, contribution);
    }
}

__global__ void compress_coarse_rhs_f32_kernel(int32_t n,
                                               const int32_t* __restrict__ block_ids,
                                               const float* __restrict__ weights,
                                               const double* __restrict__ residual,
                                               float* __restrict__ coarse_rhs)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        atomicAdd(coarse_rhs + block_ids[row],
                  weights[row] * static_cast<float>(residual[row]));
    }
}

__global__ void compress_coarse_rhs_f64_kernel(int32_t n,
                                               const int32_t* __restrict__ block_ids,
                                               const double* __restrict__ weights,
                                               const double* __restrict__ residual,
                                               double* __restrict__ coarse_rhs)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        atomic_add_double(coarse_rhs + block_ids[row], weights[row] * residual[row]);
    }
}

__global__ void expand_add_coarse_solution_f32_kernel(int32_t n,
                                                      const int32_t* __restrict__ block_ids,
                                                      const float* __restrict__ weights,
                                                      const float* __restrict__ coarse_solution,
                                                      double* __restrict__ out)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        out[row] += static_cast<double>(weights[row] * coarse_solution[block_ids[row]]);
    }
}

__global__ void expand_add_coarse_solution_f64_kernel(int32_t n,
                                                      const int32_t* __restrict__ block_ids,
                                                      const double* __restrict__ weights,
                                                      const double* __restrict__ coarse_solution,
                                                      double* __restrict__ out)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        out[row] += weights[row] * coarse_solution[block_ids[row]];
    }
}

__global__ void check_finite_kernel(int32_t n, const double* x, int32_t* flag)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !isfinite(x[i])) {
        atomicExch(flag, 1);
    }
}

}  // namespace

void launch_scatter_csr_values(int32_t nnz,
                               const int32_t* perm_value_source,
                               const double* original_values,
                               double* permuted_values)
{
    scatter_csr_values_kernel<<<grid_for(nnz), kBlockSize>>>(
        nnz, perm_value_source, original_values, permuted_values);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_extract_dense_blocks_f32(int32_t nnz_perm,
                                     const int32_t* dense_block_offsets,
                                     const int32_t* dense_local_rows,
                                     const int32_t* dense_local_cols,
                                     const double* permuted_values,
                                     float* dense_blocks)
{
    extract_dense_blocks_f32_kernel<<<grid_for(nnz_perm), kBlockSize>>>(
        nnz_perm, dense_block_offsets, dense_local_rows, dense_local_cols, permuted_values, dense_blocks);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_extract_dense_blocks_f64(int32_t nnz_perm,
                                     const int32_t* dense_block_offsets,
                                     const int32_t* dense_local_rows,
                                     const int32_t* dense_local_cols,
                                     const double* permuted_values,
                                     double* dense_blocks)
{
    extract_dense_blocks_f64_kernel<<<grid_for(nnz_perm), kBlockSize>>>(
        nnz_perm, dense_block_offsets, dense_local_rows, dense_local_cols, permuted_values, dense_blocks);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_extract_ras_dense_blocks_f32(int32_t map_nnz,
                                         const int32_t* source_positions,
                                         const int32_t* dense_block_offsets,
                                         const int32_t* dense_local_rows,
                                         const int32_t* dense_local_cols,
                                         const double* permuted_values,
                                         float* dense_blocks)
{
    extract_ras_dense_blocks_f32_kernel<<<grid_for(map_nnz), kBlockSize>>>(map_nnz,
                                                                          source_positions,
                                                                          dense_block_offsets,
                                                                          dense_local_rows,
                                                                          dense_local_cols,
                                                                          permuted_values,
                                                                          dense_blocks);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_extract_ras_dense_blocks_f64(int32_t map_nnz,
                                         const int32_t* source_positions,
                                         const int32_t* dense_block_offsets,
                                         const int32_t* dense_local_rows,
                                         const int32_t* dense_local_cols,
                                         const double* permuted_values,
                                         double* dense_blocks)
{
    extract_ras_dense_blocks_f64_kernel<<<grid_for(map_nnz), kBlockSize>>>(map_nnz,
                                                                          source_positions,
                                                                          dense_block_offsets,
                                                                          dense_local_rows,
                                                                          dense_local_cols,
                                                                          permuted_values,
                                                                          dense_blocks);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_add_block_diagonal_shift_f32(int32_t num_blocks,
                                         int32_t leading_dim,
                                         const int32_t* block_sizes,
                                         float shift,
                                         float* dense_blocks)
{
    add_block_diagonal_shift_f32_kernel<<<num_blocks, leading_dim>>>(
        num_blocks, leading_dim, block_sizes, shift, dense_blocks);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_add_block_diagonal_shift_f64(int32_t num_blocks,
                                         int32_t leading_dim,
                                         const int32_t* block_sizes,
                                         double shift,
                                         double* dense_blocks)
{
    add_block_diagonal_shift_f64_kernel<<<num_blocks, leading_dim>>>(
        num_blocks, leading_dim, block_sizes, shift, dense_blocks);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_block_inverse_apply_f32(int32_t num_blocks,
                                    int32_t leading_dim,
                                    const int32_t* block_starts,
                                    const int32_t* block_sizes,
                                    const float* inverse_blocks,
                                    const double* rhs,
                                    double* out)
{
    block_inverse_apply_f32_kernel<<<num_blocks, leading_dim>>>(
        leading_dim, block_starts, block_sizes, inverse_blocks, rhs, out);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_block_inverse_apply_f64(int32_t num_blocks,
                                    int32_t leading_dim,
                                    const int32_t* block_starts,
                                    const int32_t* block_sizes,
                                    const double* inverse_blocks,
                                    const double* rhs,
                                    double* out)
{
    block_inverse_apply_f64_kernel<<<num_blocks, leading_dim>>>(
        leading_dim, block_starts, block_sizes, inverse_blocks, rhs, out);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_ras_inverse_apply_f32(int32_t num_blocks,
                                  int32_t leading_dim,
                                  const int32_t* local_offsets,
                                  const int32_t* local_to_global,
                                  const int32_t* owned_sizes,
                                  const int32_t* local_sizes,
                                  const float* inverse_blocks,
                                  const double* rhs,
                                  double* out)
{
    ras_inverse_apply_f32_kernel<<<num_blocks, leading_dim>>>(leading_dim,
                                                              local_offsets,
                                                              local_to_global,
                                                              owned_sizes,
                                                              local_sizes,
                                                              inverse_blocks,
                                                              rhs,
                                                              out);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_ras_inverse_apply_f64(int32_t num_blocks,
                                  int32_t leading_dim,
                                  const int32_t* local_offsets,
                                  const int32_t* local_to_global,
                                  const int32_t* owned_sizes,
                                  const int32_t* local_sizes,
                                  const double* inverse_blocks,
                                  const double* rhs,
                                  double* out)
{
    ras_inverse_apply_f64_kernel<<<num_blocks, leading_dim>>>(leading_dim,
                                                              local_offsets,
                                                              local_to_global,
                                                              owned_sizes,
                                                              local_sizes,
                                                              inverse_blocks,
                                                              rhs,
                                                              out);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_block_lu_solve_apply_f32(int32_t num_blocks,
                                     int32_t leading_dim,
                                     const int32_t* block_starts,
                                     const int32_t* block_sizes,
                                     const float* lu_blocks,
                                     const int32_t* pivots,
                                     const double* rhs,
                                     double* out)
{
    block_lu_solve_apply_f32_kernel<<<num_blocks, 1>>>(
        leading_dim, block_starts, block_sizes, lu_blocks, pivots, rhs, out);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_block_lu_solve_apply_f64(int32_t num_blocks,
                                     int32_t leading_dim,
                                     const int32_t* block_starts,
                                     const int32_t* block_sizes,
                                     const double* lu_blocks,
                                     const int32_t* pivots,
                                     const double* rhs,
                                     double* out)
{
    block_lu_solve_apply_f64_kernel<<<num_blocks, 1>>>(
        leading_dim, block_starts, block_sizes, lu_blocks, pivots, rhs, out);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_assemble_coarse_matrix_f32(int32_t rows,
                                       int32_t coarse_dim,
                                       const int32_t* row_ptr,
                                       const int32_t* col_idx,
                                       const double* values,
                                       const int32_t* block_ids,
                                       const float* weights,
                                       float* coarse_matrix)
{
    assemble_coarse_matrix_f32_kernel<<<grid_for(rows), kBlockSize>>>(
        rows, coarse_dim, row_ptr, col_idx, values, block_ids, weights, coarse_matrix);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_assemble_coarse_matrix_f64(int32_t rows,
                                       int32_t coarse_dim,
                                       const int32_t* row_ptr,
                                       const int32_t* col_idx,
                                       const double* values,
                                       const int32_t* block_ids,
                                       const double* weights,
                                       double* coarse_matrix)
{
    assemble_coarse_matrix_f64_kernel<<<grid_for(rows), kBlockSize>>>(
        rows, coarse_dim, row_ptr, col_idx, values, block_ids, weights, coarse_matrix);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_compress_coarse_rhs_f32(int32_t n,
                                    const int32_t* block_ids,
                                    const float* weights,
                                    const double* residual,
                                    float* coarse_rhs)
{
    compress_coarse_rhs_f32_kernel<<<grid_for(n), kBlockSize>>>(
        n, block_ids, weights, residual, coarse_rhs);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_compress_coarse_rhs_f64(int32_t n,
                                    const int32_t* block_ids,
                                    const double* weights,
                                    const double* residual,
                                    double* coarse_rhs)
{
    compress_coarse_rhs_f64_kernel<<<grid_for(n), kBlockSize>>>(
        n, block_ids, weights, residual, coarse_rhs);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_expand_add_coarse_solution_f32(int32_t n,
                                           const int32_t* block_ids,
                                           const float* weights,
                                           const float* coarse_solution,
                                           double* out)
{
    expand_add_coarse_solution_f32_kernel<<<grid_for(n), kBlockSize>>>(
        n, block_ids, weights, coarse_solution, out);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_expand_add_coarse_solution_f64(int32_t n,
                                           const int32_t* block_ids,
                                           const double* weights,
                                           const double* coarse_solution,
                                           double* out)
{
    expand_add_coarse_solution_f64_kernel<<<grid_for(n), kBlockSize>>>(
        n, block_ids, weights, coarse_solution, out);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_check_finite(int32_t n, const double* x, int32_t* flag)
{
    check_finite_kernel<<<grid_for(n), kBlockSize>>>(n, x, flag);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuiter::kernels
