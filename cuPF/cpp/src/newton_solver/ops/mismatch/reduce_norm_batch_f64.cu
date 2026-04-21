// ---------------------------------------------------------------------------
// reduce_norm_batch_f64.cu
//
// Main kernel: reduce_norm_batch_f64_kernel.
// Reduces L-infinity norm of F per batch on device.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_mismatch_kernels.hpp"

#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <stdexcept>


namespace {

void sync_cuda_for_timing()
{
#ifdef CUPF_ENABLE_TIMING
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

__global__ void reduce_norm_batch_f64_kernel(
    const double* __restrict__ F,
    int32_t dimF,
    double* __restrict__ normF)
{
    extern __shared__ double sdata[];

    const int32_t batch = blockIdx.x;
    const int32_t lane = threadIdx.x;
    const int32_t base = batch * dimF;

    double local_max = 0.0;
    for (int32_t i = lane; i < dimF; i += blockDim.x) {
        const double value = fabs(F[base + i]);
        local_max = fmax(local_max, value);
    }

    sdata[lane] = local_max;
    __syncthreads();

    for (int32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            sdata[lane] = fmax(sdata[lane], sdata[lane + offset]);
        }
        __syncthreads();
    }

    if (lane == 0) {
        normF[batch] = sdata[0];
    }
}

}  // namespace


void launch_reduce_norm_batch_f64(CudaMixedStorage& storage)
{
    if (storage.dimF <= 0 || storage.batch_size <= 0) {
        throw std::runtime_error("launch_reduce_norm_batch_f64: storage is not prepared");
    }

    constexpr int32_t block = 256;
    const int32_t grid = storage.batch_size;
    const std::size_t shared_bytes = static_cast<std::size_t>(block) * sizeof(double);

    reduce_norm_batch_f64_kernel<<<grid, block, shared_bytes>>>(
        storage.d_F.data(),
        storage.dimF,
        storage.d_normF.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
