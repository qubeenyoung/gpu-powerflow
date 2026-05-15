#pragma once

// File responsibility:
//   - Small CUDA helper kernels for the GPU block ILU(0) pilot
//   - No CLI, no file IO, no benchmark orchestration
//   - No persistent solver state

#include <cuda_runtime.h>

#include <cstdint>

namespace gpu_block_ilu0 {

constexpr int32_t kMaxBlockDim = 32;

// Scatter CSR numeric values into padded dense block storage.
// Dense ILU blocks are row-major throughout the custom kernels.
__global__ void scatter_dense_kernel(int32_t nnz,
                                     int32_t pad,
                                     const double* values,
                                     const int32_t* nnz_block,
                                     const int32_t* nnz_local_row,
                                     const int32_t* nnz_local_col,
                                     float* blocks)
{
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nnz) {
        return;
    }

    const int32_t block = nnz_block[idx];
    if (block < 0) {
        return;
    }

    const int32_t local_row = nnz_local_row[idx];
    const int32_t local_col = nnz_local_col[idx];
    atomicAdd(&blocks[block * pad * pad + local_row * pad + local_col],
              static_cast<float>(values[idx]));
}

// Compute L(i,k) = A(i,k) * inv(U(k,k)).
// inv_diag is column-major because it is produced by cuBLAS.
__global__ void right_multiply_kernel(float* blocks,
                                      const float* inv_diag,
                                      int32_t block_index,
                                      int32_t diag_block,
                                      int32_t rows,
                                      int32_t inner,
                                      int32_t pad)
{
    __shared__ float tmp[kMaxBlockDim * kMaxBlockDim];
    const int32_t row = static_cast<int32_t>(threadIdx.y);
    const int32_t col = static_cast<int32_t>(threadIdx.x);

    if (row < rows && col < inner) {
        float sum = 0.0f;
        for (int32_t k = 0; k < inner; ++k) {
            sum += blocks[block_index * pad * pad + row * pad + k] *
                   inv_diag[diag_block * pad * pad + k + col * pad];
        }
        tmp[row * pad + col] = sum;
    }

    __syncthreads();

    if (row < rows && col < inner) {
        blocks[block_index * pad * pad + row * pad + col] = tmp[row * pad + col];
    }
}

// Apply all ILU(0) updates for one lower block L(i,k).
// Each CUDA block handles one preserved target block A(i,j).
__global__ void subtract_product_batch_kernel(float* blocks,
                                              const int32_t* update_target,
                                              const int32_t* update_rhs,
                                              int32_t begin,
                                              int32_t count,
                                              int32_t lhs_index,
                                              int32_t lhs_inner,
                                              int32_t pad)
{
    const int32_t op = static_cast<int32_t>(blockIdx.x);
    if (op >= count) {
        return;
    }

    const int32_t target = update_target[begin + op];
    const int32_t rhs = update_rhs[begin + op];
    const int32_t row = static_cast<int32_t>(threadIdx.y);
    const int32_t col = static_cast<int32_t>(threadIdx.x);
    if (row >= pad || col >= pad) {
        return;
    }

    float sum = 0.0f;
    for (int32_t k = 0; k < lhs_inner; ++k) {
        sum += blocks[lhs_index * pad * pad + row * pad + k] *
               blocks[rhs * pad * pad + k * pad + col];
    }
    blocks[target * pad * pad + row * pad + col] -= sum;
}

// Copy a row-major diagonal block into a column-major cuBLAS work buffer.
// The matrix is not mathematically transposed; only the storage convention changes.
__global__ void prepare_diag_for_cublas_kernel(const float* blocks,
                                               float* diag_work,
                                               int32_t diag_block,
                                               int32_t block_index,
                                               int32_t dim,
                                               int32_t pad,
                                               float shift)
{
    const int32_t row = static_cast<int32_t>(threadIdx.y);
    const int32_t col = static_cast<int32_t>(threadIdx.x);
    if (row >= pad || col >= pad) {
        return;
    }

    float value = 0.0f;
    if (row < dim && col < dim) {
        value = blocks[block_index * pad * pad + row * pad + col];
        if (row == col) {
            value += shift;
        }
    } else if (row == col) {
        value = 1.0f;
    }

    diag_work[diag_block * pad * pad + row + col * pad] = value;
}

__global__ void copy_vector_kernel(int32_t n, const double* src, double* dst)
{
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// y_i -= B_ij x_j for one dense off-diagonal block.
__global__ void gemv_sub_kernel(const float* blocks,
                                int32_t block_index,
                                int32_t row_begin,
                                int32_t col_begin,
                                int32_t rows,
                                int32_t cols,
                                int32_t pad,
                                const double* x,
                                double* y)
{
    const int32_t row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    for (int32_t col = 0; col < cols; ++col) {
        sum += static_cast<double>(blocks[block_index * pad * pad + row * pad + col]) *
               x[col_begin + col];
    }
    y[row_begin + row] -= sum;
}

// z_i = inv(U_ii) z_i. inv_diag is column-major from cuBLAS.
__global__ void diag_apply_kernel(const float* inv_diag,
                                  int32_t diag_block,
                                  int32_t begin,
                                  int32_t dim,
                                  int32_t pad,
                                  double* z)
{
    __shared__ double rhs[kMaxBlockDim];
    const int32_t row = static_cast<int32_t>(threadIdx.x);

    if (row < dim) {
        rhs[row] = z[begin + row];
    }

    __syncthreads();

    if (row < dim) {
        double sum = 0.0;
        for (int32_t col = 0; col < dim; ++col) {
            sum += static_cast<double>(inv_diag[diag_block * pad * pad + row + col * pad]) *
                   rhs[col];
        }
        z[begin + row] = sum;
    }
}

}  // namespace gpu_block_ilu0
