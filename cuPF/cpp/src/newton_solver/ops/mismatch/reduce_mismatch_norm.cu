// ---------------------------------------------------------------------------
// reduce_mismatch_norm.cu
//
// Reduces the L-infinity norm of mismatch residual F per batch on device.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_fp32_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>


namespace {

template <typename Scalar>
__global__ void reduce_mismatch_norm_kernel(
    const Scalar* __restrict__ F,
    int32_t dimF,
    Scalar* __restrict__ normF)
{
    extern __shared__ unsigned char shared[];
    Scalar* sdata = reinterpret_cast<Scalar*>(shared);

    const int32_t batch = blockIdx.x;
    const int32_t lane = threadIdx.x;
    const int32_t base = batch * dimF;

    Scalar local_max = Scalar(0);
    for (int32_t i = lane; i < dimF; i += blockDim.x) {
        const Scalar value = fabs(F[base + i]);
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


void launch_reduce_mismatch_norm(CudaFp64Storage& storage)
{
    if (storage.dimF <= 0) {
        throw std::runtime_error("launch_reduce_mismatch_norm: storage is not prepared");
    }

    constexpr int32_t block = 256;
    constexpr int32_t grid = 1;
    const auto shared_bytes = block * sizeof(double);

    reduce_mismatch_norm_kernel<double><<<grid, block, shared_bytes, cupf_current_cuda_stream()>>>(
        storage.d_F.data(),
        storage.dimF,
        storage.d_normF.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}


void launch_reduce_mismatch_norm(CudaFp32Storage& storage)
{
    if (storage.dimF <= 0 || storage.batch_size <= 0) {
        throw std::runtime_error("launch_reduce_mismatch_norm: storage is not prepared");
    }

    constexpr int32_t block = 256;
    const int32_t grid = storage.batch_size;
    const auto shared_bytes = block * sizeof(float);

    reduce_mismatch_norm_kernel<float><<<grid, block, shared_bytes, cupf_current_cuda_stream()>>>(
        storage.d_F.data(),
        storage.dimF,
        storage.d_normF.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}


void launch_reduce_mismatch_norm(CudaMixedStorage& storage)
{
    if (storage.dimF <= 0 || storage.batch_size <= 0) {
        throw std::runtime_error("launch_reduce_mismatch_norm: storage is not prepared");
    }

    constexpr int32_t block = 256;
    const int32_t grid = storage.batch_size;
    const auto shared_bytes = block * sizeof(double);

    reduce_mismatch_norm_kernel<double><<<grid, block, shared_bytes, cupf_current_cuda_stream()>>>(
        storage.d_F.data(),
        storage.dimF,
        storage.d_normF.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
