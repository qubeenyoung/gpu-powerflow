// ---------------------------------------------------------------------------
// fill_jacobian_vertex_offdiag_fp32.cu
//
// Main kernel: fill_jacobian_vertex_offdiag_fp32_kernel.
// Warp-per-(batch, active bus) direct-write offdiag/self mapped terms.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_jacobian_fp32_kernels.hpp"

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

__global__ void fill_jacobian_vertex_offdiag_fp32_kernel(
    int32_t total_active_buses,
    int32_t n_pvpq,
    int32_t n_bus,
    int32_t nnz_ybus,
    int32_t nnz_J,
    int32_t ybus_values_batched,
    const int32_t* __restrict__ pvpq,
    const int32_t* __restrict__ y_row_ptr,
    const int32_t* __restrict__ y_col,
    const float* __restrict__ y_re,
    const float* __restrict__ y_im,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    const double* __restrict__ vm,
    const int32_t* __restrict__ map11,
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map22,
    float* __restrict__ J_values)
{
    constexpr int32_t warp_size = 32;

    const int32_t warp_id_in_block = threadIdx.x / warp_size;
    const int32_t lane = threadIdx.x & (warp_size - 1);
    const int32_t warps_per_block = blockDim.x / warp_size;
    const int32_t active_slot = blockIdx.x * warps_per_block + warp_id_in_block;

    if (active_slot >= total_active_buses) {
        return;
    }

    const int32_t batch = active_slot / n_pvpq;
    const int32_t local_bus_slot = active_slot - batch * n_pvpq;
    const int32_t i = pvpq[local_bus_slot];
    const int32_t row_begin = y_row_ptr[i];
    const int32_t row_end = y_row_ptr[i + 1];
    const int32_t y_base = ybus_values_batched ? batch * nnz_ybus : 0;
    const int32_t v_base = batch * n_bus;
    const int32_t j_base = batch * nnz_J;

    const float vi_re = static_cast<float>(v_re[v_base + i]);
    const float vi_im = static_cast<float>(v_im[v_base + i]);

    for (int32_t k = row_begin + lane; k < row_end; k += warp_size) {
        const int32_t j = y_col[k];

        const float yr = y_re[y_base + k];
        const float yi = y_im[y_base + k];
        const float vj_re = static_cast<float>(v_re[v_base + j]);
        const float vj_im = static_cast<float>(v_im[v_base + j]);

        const float curr_re = yr * vj_re - yi * vj_im;
        const float curr_im = yr * vj_im + yi * vj_re;

        const float neg_j_vi_re = vi_im;
        const float neg_j_vi_im = -vi_re;
        const float term_va_re = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
        const float term_va_im = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

        const float vj_abs = static_cast<float>(vm[v_base + j]);
        float term_vm_re = 0.0f;
        float term_vm_im = 0.0f;
        if (vj_abs > 1.0e-12f) {
            const float scaled_re = curr_re / vj_abs;
            const float scaled_im = curr_im / vj_abs;
            term_vm_re = vi_re * scaled_re + vi_im * scaled_im;
            term_vm_im = vi_im * scaled_re - vi_re * scaled_im;
        }

        if (map11[k] >= 0) J_values[j_base + map11[k]] = term_va_re;
        if (map21[k] >= 0) J_values[j_base + map21[k]] = term_va_im;
        if (map12[k] >= 0) J_values[j_base + map12[k]] = term_vm_re;
        if (map22[k] >= 0) J_values[j_base + map22[k]] = term_vm_im;
    }
}

}  // namespace


void launch_fill_jacobian_vertex_offdiag_fp32(CudaMixedStorage& storage)
{
    if (storage.n_pvpq <= 0 || storage.nnz_J <= 0 || storage.batch_size <= 0) {
        throw std::runtime_error("launch_fill_jacobian_vertex_offdiag_fp32: storage is not prepared");
    }

    constexpr int32_t block = 256;
    constexpr int32_t warp_size = 32;
    const int32_t warps_per_block = block / warp_size;
    const int32_t total_active_buses = storage.batch_size * storage.n_pvpq;
    const int32_t grid = (total_active_buses + warps_per_block - 1) / warps_per_block;

    fill_jacobian_vertex_offdiag_fp32_kernel<<<grid, block>>>(
        total_active_buses,
        storage.n_pvpq,
        storage.n_bus,
        storage.nnz_ybus,
        storage.nnz_J,
        storage.ybus_values_batched ? 1 : 0,
        storage.d_pvpq.data(),
        storage.d_Ybus_indptr.data(),
        storage.d_Ybus_indices.data(),
        storage.d_Ybus_re.data(),
        storage.d_Ybus_im.data(),
        storage.d_V_re.data(),
        storage.d_V_im.data(),
        storage.d_Vm.data(),
        storage.d_mapJ11.data(),
        storage.d_mapJ21.data(),
        storage.d_mapJ12.data(),
        storage.d_mapJ22.data(),
        storage.d_J_values.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
