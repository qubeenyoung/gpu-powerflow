// ---------------------------------------------------------------------------
// compute_ibus_batch_fp32.cu
//
// Main kernel: compute_ibus_batch_fp32_kernel.
// Computes Ibus64 = Ybus32 * V64 for mismatch/Jacobian inputs.
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

__global__ void compute_ibus_batch_fp32_kernel(
    int32_t total_rows,
    int32_t n_bus,
    int32_t nnz_ybus,
    int32_t ybus_values_batched,
    const int32_t* __restrict__ y_row_ptr,
    const int32_t* __restrict__ y_col,
    const float* __restrict__ y_re,
    const float* __restrict__ y_im,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    double* __restrict__ ibus_re,
    double* __restrict__ ibus_im)
{
    constexpr int32_t warp_size = 32;

    const int32_t warp_id_in_block = threadIdx.x / warp_size;
    const int32_t lane = threadIdx.x & (warp_size - 1);
    const int32_t warps_per_block = blockDim.x / warp_size;
    const int32_t row_slot = blockIdx.x * warps_per_block + warp_id_in_block;

    if (row_slot >= total_rows) {
        return;
    }

    const int32_t batch = row_slot / n_bus;
    const int32_t bus = row_slot - batch * n_bus;
    const int32_t row_begin = y_row_ptr[bus];
    const int32_t row_end = y_row_ptr[bus + 1];
    const int32_t v_base = batch * n_bus;
    const int32_t y_base = ybus_values_batched ? batch * nnz_ybus : 0;

    double acc_re = 0.0;
    double acc_im = 0.0;
    for (int32_t k = row_begin + lane; k < row_end; k += warp_size) {
        const int32_t col = y_col[k];
        const double yr = static_cast<double>(y_re[y_base + k]);
        const double yi = static_cast<double>(y_im[y_base + k]);
        const double vr = v_re[v_base + col];
        const double vi = v_im[v_base + col];
        acc_re += yr * vr - yi * vi;
        acc_im += yr * vi + yi * vr;
    }

    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        acc_re += __shfl_down_sync(0xffffffffu, acc_re, offset);
        acc_im += __shfl_down_sync(0xffffffffu, acc_im, offset);
    }

    if (lane == 0) {
        const int32_t idx = v_base + bus;
        ibus_re[idx] = acc_re;
        ibus_im[idx] = acc_im;
    }
}

}  // namespace


void launch_compute_ibus_batch_fp32(CudaMixedStorage& storage)
{
    if (storage.n_bus <= 0 || storage.nnz_ybus <= 0 || storage.batch_size <= 0) {
        throw std::runtime_error("launch_compute_ibus_batch_fp32: storage is not prepared");
    }

    constexpr int32_t block = 256;
    constexpr int32_t warp_size = 32;
    const int32_t warps_per_block = block / warp_size;
    const int32_t total_rows = storage.batch_size * storage.n_bus;
    const int32_t grid = (total_rows + warps_per_block - 1) / warps_per_block;

    compute_ibus_batch_fp32_kernel<<<grid, block>>>(
        total_rows,
        storage.n_bus,
        storage.nnz_ybus,
        storage.ybus_values_batched ? 1 : 0,
        storage.d_Ybus_indptr.data(),
        storage.d_Ybus_indices.data(),
        storage.d_Ybus_re.data(),
        storage.d_Ybus_im.data(),
        storage.d_V_re.data(),
        storage.d_V_im.data(),
        storage.d_Ibus_re.data(),
        storage.d_Ibus_im.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
