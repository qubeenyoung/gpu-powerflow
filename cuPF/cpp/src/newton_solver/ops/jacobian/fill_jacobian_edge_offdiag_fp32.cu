// ---------------------------------------------------------------------------
// fill_jacobian_edge_offdiag_fp32.cu
//
// Main kernel: fill_jacobian_edge_offdiag_fp32_kernel.
// Direct-writes Ybus-entry mapped Jacobian terms for every batch.
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

__global__ void fill_jacobian_edge_offdiag_fp32_kernel(
    int32_t total_edges,
    int32_t nnz_ybus,
    int32_t nnz_J,
    int32_t n_bus,
    int32_t ybus_values_batched,
    const float* __restrict__ y_re,
    const float* __restrict__ y_im,
    const int32_t* __restrict__ y_row,
    const int32_t* __restrict__ y_col,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    const double* __restrict__ vm,
    const int32_t* __restrict__ map11,
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map22,
    float* __restrict__ J_values)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_edges) {
        return;
    }

    const int32_t batch = tid / nnz_ybus;
    const int32_t k = tid - batch * nnz_ybus;
    const int32_t y_base = ybus_values_batched ? batch * nnz_ybus : 0;
    const int32_t v_base = batch * n_bus;
    const int32_t j_base = batch * nnz_J;

    const int32_t i = y_row[k];
    const int32_t j = y_col[k];

    const float yr = y_re[y_base + k];
    const float yi = y_im[y_base + k];
    const float vi_re = static_cast<float>(v_re[v_base + i]);
    const float vi_im = static_cast<float>(v_im[v_base + i]);
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

}  // namespace


void launch_fill_jacobian_edge_offdiag_fp32(CudaMixedStorage& storage)
{
    if (storage.nnz_ybus <= 0 || storage.nnz_J <= 0 || storage.batch_size <= 0) {
        throw std::runtime_error("launch_fill_jacobian_edge_offdiag_fp32: storage is not prepared");
    }

    constexpr int32_t block = 256;
    const int32_t total_edges = storage.batch_size * storage.nnz_ybus;
    const int32_t grid = (total_edges + block - 1) / block;

    fill_jacobian_edge_offdiag_fp32_kernel<<<grid, block>>>(
        total_edges,
        storage.nnz_ybus,
        storage.nnz_J,
        storage.n_bus,
        storage.ybus_values_batched ? 1 : 0,
        storage.d_Ybus_re.data(),
        storage.d_Ybus_im.data(),
        storage.d_Y_row.data(),
        storage.d_Ybus_indices.data(),
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
