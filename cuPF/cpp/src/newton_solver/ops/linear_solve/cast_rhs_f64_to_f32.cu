// ---------------------------------------------------------------------------
// cast_rhs_f64_to_f32.cu
//
// Main kernel: cast_rhs_f64_to_f32_kernel.
// Converts F64 RHS to FP32 RHS on device.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_linear_solve_kernels.hpp"

#include "utils/cuda_utils.hpp"

#include <stdexcept>


namespace {

void sync_cuda_for_timing()
{
#ifdef CUPF_ENABLE_TIMING
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

__global__ void cast_rhs_f64_to_f32_kernel(
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


void launch_cast_rhs_f64_to_f32(const double* src, float* dst, int32_t count)
{
    if (src == nullptr || dst == nullptr || count < 0) {
        throw std::runtime_error("launch_cast_rhs_f64_to_f32: invalid arguments");
    }
    if (count == 0) {
        return;
    }

    constexpr int32_t block = 256;
    const int32_t grid = (count + block - 1) / block;
    cast_rhs_f64_to_f32_kernel<<<grid, block>>>(src, dst, count);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
