// ---------------------------------------------------------------------------
// reduce_mismatch_norm.cu
//
// Reduces the L-infinity norm of mismatch residual F per batch on device.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <stdexcept>


namespace {

__global__ void reduce_mismatch_norm_kernel(
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


void launch_reduce_mismatch_norm(CudaFp64Buffers& storage)
{
    if (storage.dimF <= 0) {
        throw std::runtime_error("launch_reduce_mismatch_norm: storage is not prepared");
    }

    constexpr int32_t block = 256;
    constexpr int32_t grid = 1;
    const auto shared_bytes = block * sizeof(double);

    reduce_mismatch_norm_kernel<<<grid, block, shared_bytes>>>(
        storage.d_F.data(),
        storage.dimF,
        storage.d_normF.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}


void launch_reduce_mismatch_norm(CudaMixedBuffers& storage)
{
    if (storage.dimF <= 0 || storage.batch_size <= 0) {
        throw std::runtime_error("launch_reduce_mismatch_norm: storage is not prepared");
    }

    constexpr int32_t block = 256;
    const int32_t grid = storage.batch_size;
    const auto shared_bytes = block * sizeof(double);

    reduce_mismatch_norm_kernel<<<grid, block, shared_bytes>>>(
        storage.d_F.data(),
        storage.dimF,
        storage.d_normF.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
