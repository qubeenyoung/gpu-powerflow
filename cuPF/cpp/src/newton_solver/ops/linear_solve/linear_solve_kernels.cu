// ---------------------------------------------------------------------------
// linear_solve_kernels.cu
//
// Small CUDA support kernels for the cuDSS linear-solve path (their typed
// launch_* wrappers are declared in cuda_linear_solve_kernels.hpp):
//   - prepare_rhs          : down-cast the FP64 residual into the FP32 working
//                            RHS buffer (mixed precision only)
//   - transpose_csr_values : scatter the batched J values into J^T value order
//                            using a precomputed position map (adjoint solve)
//
// The torch-bridge I/O kernels that used to share this file live in
// torch_bridge_kernels.cu now.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_linear_solve_kernels.hpp"

#include "utils/cuda_utils.hpp"

#include <stdexcept>


namespace {

// Element-wise FP64 -> FP32 down-cast of the residual into the FP32 RHS buffer.
__global__ void prepare_rhs_kernel(
    const double* __restrict__ src,
    float* __restrict__ dst,
    int32_t count)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) {
        return;
    }

    dst[tid] = static_cast<float>(src[tid]);  // explicit narrowing double -> float
}

// Move each batched J value to its transposed slot via the precomputed map.
// One thread per (case, J nonzero); src_to_transpose_pos[k] gives the J^T value
// index of source nonzero k (the pattern is identical across the batch).
template <typename T>
__global__ void transpose_csr_values_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int32_t* __restrict__ src_to_transpose_pos,
    int32_t nnz,
    int32_t total)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }
    const int32_t batch = tid / nnz;
    const int32_t k = tid - batch * nnz;
    const int32_t dst_pos = src_to_transpose_pos[k];
    dst[batch * nnz + dst_pos] = src[tid];
}

}  // namespace


// ===========================================================================
// Host launch wrappers (validate args, size the grid, launch on the cuPF
// stream, then check for errors).
// ===========================================================================

void launch_prepare_rhs(const double* src, float* dst, int32_t count)
{
    if (src == nullptr || dst == nullptr || count < 0) {
        throw std::runtime_error("launch_prepare_rhs: invalid arguments");
    }
    if (count == 0) {
        return;
    }

    constexpr int32_t block = 256;
    const int32_t grid = (count + block - 1) / block;
    prepare_rhs_kernel<<<grid, block, 0, cupf_current_cuda_stream()>>>(src, dst, count);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

template <typename T>
void launch_transpose_csr_values_impl(const T* src,
                                      T* dst,
                                      const int32_t* src_to_transpose_pos,
                                      int32_t nnz,
                                      int32_t batch_size)
{
    if (src == nullptr || dst == nullptr || src_to_transpose_pos == nullptr ||
        nnz <= 0 || batch_size <= 0) {
        throw std::runtime_error("launch_transpose_csr_values: invalid arguments");
    }
    constexpr int32_t block = 256;
    const int32_t total = nnz * batch_size;
    const int32_t grid = (total + block - 1) / block;
    transpose_csr_values_kernel<T><<<grid, block, 0, cupf_current_cuda_stream()>>>(
        src, dst, src_to_transpose_pos, nnz, total);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

void launch_transpose_csr_values(const double* src,
                                 double* dst,
                                 const int32_t* src_to_transpose_pos,
                                 int32_t nnz,
                                 int32_t batch_size)
{
    launch_transpose_csr_values_impl(src, dst, src_to_transpose_pos, nnz, batch_size);
}

void launch_transpose_csr_values(const float* src,
                                 float* dst,
                                 const int32_t* src_to_transpose_pos,
                                 int32_t nnz,
                                 int32_t batch_size)
{
    launch_transpose_csr_values_impl(src, dst, src_to_transpose_pos, nnz, batch_size);
}

#endif  // CUPF_WITH_CUDA
