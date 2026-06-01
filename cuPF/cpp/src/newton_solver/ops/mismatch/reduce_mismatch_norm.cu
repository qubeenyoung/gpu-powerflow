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
#include <cstddef>
#include <cstdint>
#include <stdexcept>


namespace {

// atomicMax on a non-negative float/double via its bit pattern. For values >= 0
// the IEEE-754 bit pattern is monotonic as a signed integer, and |F| >= 0, so
// integer atomicMax computes the floating-point max. normF must be pre-zeroed.
__device__ inline void atomic_max_nonneg(float* addr, float val)
{
    atomicMax(reinterpret_cast<int*>(addr), __float_as_int(val));
}
__device__ inline void atomic_max_nonneg(double* addr, double val)
{
    atomicMax(reinterpret_cast<long long*>(addr), __double_as_longlong(val));
}

// Multi-block L-infinity reduction. gridDim.x blocks grid-stride over each
// case's residual, reduce in shared memory, then atomicMax the block partial
// into normF[case]. gridDim.y = batch_size. The launcher sizes gridDim.x so
// large/low-batch problems get many blocks (occupancy), while large batches
// (already many blocks via gridDim.y) get 1 block/case — matching the old
// single-block behavior with no contention regression.
template <typename Scalar>
__global__ void reduce_mismatch_norm_kernel(
    const Scalar* __restrict__ F,
    int32_t dimF,
    Scalar* __restrict__ normF)
{
    extern __shared__ unsigned char shared[];
    Scalar* sdata = reinterpret_cast<Scalar*>(shared);

    const int32_t batch = blockIdx.y;
    const int32_t lane = threadIdx.x;
    const int32_t base = batch * dimF;
    const int32_t stride = gridDim.x * blockDim.x;

    Scalar local_max = Scalar(0);
    for (int32_t i = blockIdx.x * blockDim.x + lane; i < dimF; i += stride) {
        local_max = fmax(local_max, fabs(F[base + i]));
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
        atomic_max_nonneg(&normF[batch], sdata[0]);
    }
}

}  // namespace


namespace {
// Shared launch: zero normF (atomicMax accumulates into it), then tile each
// case with ceil(dimF/block) blocks along grid.x and batch along grid.y.
template <typename Scalar, typename Storage>
void launch_norm(Storage& storage, int32_t batch)
{
    if (storage.dimF <= 0 || batch <= 0) {
        throw std::runtime_error("launch_reduce_mismatch_norm: storage is not prepared");
    }
    constexpr int32_t block = 256;
    // Adaptive blocks-per-case: aim for ~256 total blocks for occupancy, but
    // never more than the data needs (ceil(dimF/block)) nor fewer than 1.
    // High batch (gridDim.y large) -> 1 block/case (== old single-block path).
    constexpr int32_t kTargetTotalBlocks = 256;
    const int32_t max_blocks = (storage.dimF + block - 1) / block;
    int32_t blocks_per_case = kTargetTotalBlocks / batch;
    if (blocks_per_case < 1) blocks_per_case = 1;
    if (blocks_per_case > max_blocks) blocks_per_case = max_blocks;

    const dim3 grid(blocks_per_case, batch);
    const std::size_t shared_bytes = block * sizeof(Scalar);

    storage.d_normF.memsetZero();
    reduce_mismatch_norm_kernel<Scalar><<<grid, block, shared_bytes, cupf_current_cuda_stream()>>>(
        storage.d_F.data(), storage.dimF, storage.d_normF.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}
}  // namespace


void launch_reduce_mismatch_norm(CudaFp64Storage& storage)
{
    launch_norm<double>(storage, 1);
}


void launch_reduce_mismatch_norm(CudaFp32Storage& storage)
{
    launch_norm<float>(storage, storage.batch_size);
}


void launch_reduce_mismatch_norm(CudaMixedStorage& storage)
{
    launch_norm<double>(storage, storage.batch_size);
}

#endif  // CUPF_WITH_CUDA
