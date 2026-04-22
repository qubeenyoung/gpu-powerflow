// ---------------------------------------------------------------------------
// compute_mismatch_from_ibus.cu
//
// Computes the Newton residual F from V, Ibus, and Sbus for every batch.
// Ibus itself is computed in ops/ibus.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <stdexcept>


namespace {

__global__ void compute_mismatch_from_ibus_kernel(
    int32_t total_entries,
    int32_t dimF,
    int32_t n_bus,
    int32_t n_pv,
    int32_t n_pq,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    const double* __restrict__ ibus_re,
    const double* __restrict__ ibus_im,
    const double* __restrict__ sbus_re,
    const double* __restrict__ sbus_im,
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    double* __restrict__ F)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_entries) {
        return;
    }

    const int32_t batch = tid / dimF;
    const int32_t local = tid - batch * dimF;

    int32_t bus = 0;
    bool take_imag = false;
    if (local < n_pv) {
        bus = pv[local];
    } else if (local < n_pv + n_pq) {
        bus = pq[local - n_pv];
    } else {
        bus = pq[local - n_pv - n_pq];
        take_imag = true;
    }

    const int32_t bus_idx = batch * n_bus + bus;
    const double vr = v_re[bus_idx];
    const double vi = v_im[bus_idx];
    const double ir = ibus_re[bus_idx];
    const double ii = ibus_im[bus_idx];

    const double mis_re = vr * ir + vi * ii - sbus_re[bus_idx];
    const double mis_im = vi * ir - vr * ii - sbus_im[bus_idx];
    F[tid] = take_imag ? mis_im : mis_re;
}

}  // namespace


void launch_compute_mismatch_from_ibus(CudaFp64Buffers& storage)
{
    if (storage.n_bus <= 0 || storage.dimF <= 0) {
        throw std::runtime_error("launch_compute_mismatch_from_ibus: storage is not prepared");
    }

    constexpr int32_t block = 256;
    const int32_t total_entries = storage.dimF;
    const int32_t grid = (total_entries + block - 1) / block;

    compute_mismatch_from_ibus_kernel<<<grid, block>>>(
        total_entries,
        storage.dimF,
        storage.n_bus,
        storage.n_pvpq - storage.n_pq,
        storage.n_pq,
        storage.d_V_re.data(),
        storage.d_V_im.data(),
        storage.d_Ibus_re.data(),
        storage.d_Ibus_im.data(),
        storage.d_Sbus_re.data(),
        storage.d_Sbus_im.data(),
        storage.d_pv.data(),
        storage.d_pq.data(),
        storage.d_F.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}


void launch_compute_mismatch_from_ibus(CudaMixedBuffers& storage)
{
    if (storage.n_bus <= 0 || storage.dimF <= 0 || storage.batch_size <= 0) {
        throw std::runtime_error("launch_compute_mismatch_from_ibus: storage is not prepared");
    }

    constexpr int32_t block = 256;
    const int32_t total_entries = storage.batch_size * storage.dimF;
    const int32_t grid = (total_entries + block - 1) / block;

    compute_mismatch_from_ibus_kernel<<<grid, block>>>(
        total_entries,
        storage.dimF,
        storage.n_bus,
        storage.n_pvpq - storage.n_pq,
        storage.n_pq,
        storage.d_V_re.data(),
        storage.d_V_im.data(),
        storage.d_Ibus_re.data(),
        storage.d_Ibus_im.data(),
        storage.d_Sbus_re.data(),
        storage.d_Sbus_im.data(),
        storage.d_pv.data(),
        storage.d_pq.data(),
        storage.d_F.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
