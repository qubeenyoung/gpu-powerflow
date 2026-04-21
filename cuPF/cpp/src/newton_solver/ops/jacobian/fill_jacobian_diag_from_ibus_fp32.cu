// ---------------------------------------------------------------------------
// fill_jacobian_diag_from_ibus_fp32.cu
//
// Main kernel: fill_jacobian_diag_from_ibus_fp32_kernel.
// Adds Ibus-based diagonal correction after offdiag/self direct writes.
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

__global__ void fill_jacobian_diag_from_ibus_fp32_kernel(
    int32_t total_buses,
    int32_t n_bus,
    int32_t nnz_J,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    const double* __restrict__ vm,
    const double* __restrict__ ibus_re,
    const double* __restrict__ ibus_im,
    const int32_t* __restrict__ diag11,
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag22,
    float* __restrict__ J_values)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_buses) {
        return;
    }

    const int32_t batch = tid / n_bus;
    const int32_t bus = tid - batch * n_bus;
    const int32_t v_idx = batch * n_bus + bus;
    const int32_t j_base = batch * nnz_J;

    const float vr = static_cast<float>(v_re[v_idx]);
    const float vi = static_cast<float>(v_im[v_idx]);
    const float ir = static_cast<float>(ibus_re[v_idx]);
    const float ii = static_cast<float>(ibus_im[v_idx]);

    // term_va = j * V_i * conj(I_i)
    const float vi_conj_i_re = vr * ir + vi * ii;
    const float vi_conj_i_im = vi * ir - vr * ii;
    const float term_va_re = -vi_conj_i_im;
    const float term_va_im = vi_conj_i_re;

    // term_vm = conj(I_i) * Vhat_i
    const float vi_abs = static_cast<float>(vm[v_idx]);
    float term_vm_re = 0.0f;
    float term_vm_im = 0.0f;
    if (vi_abs > 1.0e-12f) {
        const float vnorm_re = vr / vi_abs;
        const float vnorm_im = vi / vi_abs;
        term_vm_re = ir * vnorm_re + ii * vnorm_im;
        term_vm_im = ir * vnorm_im - ii * vnorm_re;
    }

    if (diag11[bus] >= 0) J_values[j_base + diag11[bus]] += term_va_re;
    if (diag21[bus] >= 0) J_values[j_base + diag21[bus]] += term_va_im;
    if (diag12[bus] >= 0) J_values[j_base + diag12[bus]] += term_vm_re;
    if (diag22[bus] >= 0) J_values[j_base + diag22[bus]] += term_vm_im;
}

}  // namespace


void launch_fill_jacobian_diag_from_ibus_fp32(CudaMixedStorage& storage)
{
    if (storage.n_bus <= 0 || storage.nnz_J <= 0 || storage.batch_size <= 0) {
        throw std::runtime_error("launch_fill_jacobian_diag_from_ibus_fp32: storage is not prepared");
    }

    constexpr int32_t block = 256;
    const int32_t total_buses = storage.batch_size * storage.n_bus;
    const int32_t grid = (total_buses + block - 1) / block;

    fill_jacobian_diag_from_ibus_fp32_kernel<<<grid, block>>>(
        total_buses,
        storage.n_bus,
        storage.nnz_J,
        storage.d_V_re.data(),
        storage.d_V_im.data(),
        storage.d_Vm.data(),
        storage.d_Ibus_re.data(),
        storage.d_Ibus_im.data(),
        storage.d_diagJ11.data(),
        storage.d_diagJ21.data(),
        storage.d_diagJ12.data(),
        storage.d_diagJ22.data(),
        storage.d_J_values.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
