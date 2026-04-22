// ---------------------------------------------------------------------------
// prepare_rhs.cu
//
// CUDA RHS preparation kernel for solver value buffers.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_linear_solve_kernels.hpp"

#include "utils/cuda_utils.hpp"

#include <stdexcept>


namespace {

__global__ void prepare_rhs_kernel(
    const double* __restrict__ src,
    float* __restrict__ dst,
    int32_t count)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) {
        return;
    }

    dst[tid] = static_cast<float>(src[tid]);
}

}  // namespace


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
    prepare_rhs_kernel<<<grid, block>>>(src, dst, count);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
