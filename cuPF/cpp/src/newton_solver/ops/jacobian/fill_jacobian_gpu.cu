#ifdef CUPF_WITH_CUDA

#include "fill_jacobian_gpu.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <stdexcept>


namespace {

template <typename JScalar, typename YbusScalar, typename VoltageScalar, typename IbusScalar>
__global__ void fill_jacobian_gpu_kernel(
    int32_t total_edges,
    int32_t nnz_ybus,
    int32_t nnz_J,
    int32_t n_bus,
    int32_t ybus_values_batched,
    int32_t use_cached_ibus,
    const YbusScalar* __restrict__ y_re,
    const YbusScalar* __restrict__ y_im,
    const int32_t* __restrict__ y_row,
    const int32_t* __restrict__ y_col,
    const int32_t* __restrict__ y_row_ptr,
    const VoltageScalar* __restrict__ v_re,
    const VoltageScalar* __restrict__ v_im,
    const VoltageScalar* __restrict__ vm,
    const IbusScalar* __restrict__ ibus_re,
    const IbusScalar* __restrict__ ibus_im,
    const int32_t* __restrict__ map11,
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map22,
    const int32_t* __restrict__ diag11,
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag22,
    JScalar* __restrict__ J_values)
{
    using AccumScalar = JScalar;

    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_edges) return;

    const int32_t batch  = tid / nnz_ybus;
    const int32_t k      = tid - batch * nnz_ybus;
    const int32_t y_base = ybus_values_batched ? batch * nnz_ybus : 0;
    const int32_t v_base = batch * n_bus;
    const int32_t j_base = batch * nnz_J;

    const int32_t i = y_row[k];
    const int32_t j = y_col[k];

    const AccumScalar yr     = static_cast<AccumScalar>(y_re[y_base + k]);
    const AccumScalar yi     = static_cast<AccumScalar>(y_im[y_base + k]);
    const AccumScalar vi_re  = static_cast<AccumScalar>(v_re[v_base + i]);
    const AccumScalar vi_im  = static_cast<AccumScalar>(v_im[v_base + i]);
    const AccumScalar vj_re  = static_cast<AccumScalar>(v_re[v_base + j]);
    const AccumScalar vj_im  = static_cast<AccumScalar>(v_im[v_base + j]);

    const AccumScalar curr_re = yr * vj_re - yi * vj_im;
    const AccumScalar curr_im = yr * vj_im + yi * vj_re;

    const AccumScalar neg_j_vi_re = vi_im;
    const AccumScalar neg_j_vi_im = -vi_re;
    const AccumScalar term_va_re  = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
    const AccumScalar term_va_im  = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

    const AccumScalar vj_abs = static_cast<AccumScalar>(vm[v_base + j]);
    AccumScalar term_vm_re = AccumScalar(0);
    AccumScalar term_vm_im = AccumScalar(0);
    if (vj_abs > AccumScalar(1.0e-12)) {
        const AccumScalar scaled_re = curr_re / vj_abs;
        const AccumScalar scaled_im = curr_im / vj_abs;
        term_vm_re = vi_re * scaled_re + vi_im * scaled_im;
        term_vm_im = vi_im * scaled_re - vi_re * scaled_im;
    }

    if (map11[k] >= 0) J_values[j_base + map11[k]] = static_cast<JScalar>(term_va_re);
    if (map21[k] >= 0) J_values[j_base + map21[k]] = static_cast<JScalar>(term_va_im);
    if (map12[k] >= 0) J_values[j_base + map12[k]] = static_cast<JScalar>(term_vm_re);
    if (map22[k] >= 0) J_values[j_base + map22[k]] = static_cast<JScalar>(term_vm_im);

    if (i != j) return;

    AccumScalar ir = AccumScalar(0);
    AccumScalar ii = AccumScalar(0);
    if (use_cached_ibus) {
        ir = static_cast<AccumScalar>(ibus_re[v_base + i]);
        ii = static_cast<AccumScalar>(ibus_im[v_base + i]);
    } else {
        for (int32_t row_k = y_row_ptr[i]; row_k < y_row_ptr[i + 1]; ++row_k) {
            const int32_t col    = y_col[row_k];
            const AccumScalar row_yr = static_cast<AccumScalar>(y_re[y_base + row_k]);
            const AccumScalar row_yi = static_cast<AccumScalar>(y_im[y_base + row_k]);
            const AccumScalar row_vr = static_cast<AccumScalar>(v_re[v_base + col]);
            const AccumScalar row_vi = static_cast<AccumScalar>(v_im[v_base + col]);
            ir += row_yr * row_vr - row_yi * row_vi;
            ii += row_yr * row_vi + row_yi * row_vr;
        }
    }

    const AccumScalar vi_conj_i_re = vi_re * ir + vi_im * ii;
    const AccumScalar vi_conj_i_im = vi_im * ir - vi_re * ii;
    const AccumScalar corr_va_re   = -vi_conj_i_im;
    const AccumScalar corr_va_im   =  vi_conj_i_re;

    const AccumScalar vi_abs = static_cast<AccumScalar>(vm[v_base + i]);
    AccumScalar corr_vm_re = AccumScalar(0);
    AccumScalar corr_vm_im = AccumScalar(0);
    if (vi_abs > AccumScalar(1.0e-12)) {
        const AccumScalar vnorm_re = vi_re / vi_abs;
        const AccumScalar vnorm_im = vi_im / vi_abs;
        corr_vm_re = ir * vnorm_re + ii * vnorm_im;
        corr_vm_im = ir * vnorm_im - ii * vnorm_re;
    }

    if (diag11[i] >= 0) J_values[j_base + diag11[i]] += static_cast<JScalar>(corr_va_re);
    if (diag21[i] >= 0) J_values[j_base + diag21[i]] += static_cast<JScalar>(corr_va_im);
    if (diag12[i] >= 0) J_values[j_base + diag12[i]] += static_cast<JScalar>(corr_vm_re);
    if (diag22[i] >= 0) J_values[j_base + diag22[i]] += static_cast<JScalar>(corr_vm_im);
}


template <typename JScalar, typename YbusScalar, typename VoltageScalar, typename IbusScalar>
void launch_fill_jacobian_gpu(
    int32_t nnz_ybus,
    int32_t nnz_J,
    int32_t n_bus,
    int32_t batch_size,
    bool ybus_values_batched,
    bool use_cached_ibus,
    const DeviceBuffer<YbusScalar>& y_re,
    const DeviceBuffer<YbusScalar>& y_im,
    const DeviceBuffer<int32_t>& y_row,
    const DeviceBuffer<int32_t>& y_col,
    const DeviceBuffer<int32_t>& y_row_ptr,
    const DeviceBuffer<VoltageScalar>& v_re,
    const DeviceBuffer<VoltageScalar>& v_im,
    const DeviceBuffer<VoltageScalar>& vm,
    const DeviceBuffer<IbusScalar>* ibus_re,
    const DeviceBuffer<IbusScalar>* ibus_im,
    const DeviceBuffer<int32_t>& map11,
    const DeviceBuffer<int32_t>& map21,
    const DeviceBuffer<int32_t>& map12,
    const DeviceBuffer<int32_t>& map22,
    const DeviceBuffer<int32_t>& diag11,
    const DeviceBuffer<int32_t>& diag21,
    const DeviceBuffer<int32_t>& diag12,
    const DeviceBuffer<int32_t>& diag22,
    DeviceBuffer<JScalar>& J_values)
{
    if (nnz_ybus <= 0 || nnz_J <= 0 || n_bus <= 0 || batch_size <= 0) {
        throw std::runtime_error("launch_fill_jacobian_gpu: buffers are not prepared");
    }

    J_values.memsetZero();

    constexpr int32_t block = 256;
    const int32_t total_edges = batch_size * nnz_ybus;
    const int32_t grid = (total_edges + block - 1) / block;

    fill_jacobian_gpu_kernel<JScalar, YbusScalar, VoltageScalar, IbusScalar><<<grid, block>>>(
        total_edges, nnz_ybus, nnz_J, n_bus,
        ybus_values_batched ? 1 : 0,
        use_cached_ibus ? 1 : 0,
        y_re.data(), y_im.data(), y_row.data(), y_col.data(), y_row_ptr.data(),
        v_re.data(), v_im.data(), vm.data(),
        ibus_re ? ibus_re->data() : nullptr,
        ibus_im ? ibus_im->data() : nullptr,
        map11.data(), map21.data(), map12.data(), map22.data(),
        diag11.data(), diag21.data(), diag12.data(), diag22.data(),
        J_values.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

}  // namespace


void CudaJacobianOp<double>::run(CudaFp64Buffers& buf, IterationContext& ctx)
{
    (void)ctx;
    launch_fill_jacobian_gpu<double, double, double, double>(
        static_cast<int32_t>(buf.d_Y_row.size()),
        static_cast<int32_t>(buf.d_J_values.size()),
        buf.n_bus,
        1,
        false,
        false,
        buf.d_Ybus_re, buf.d_Ybus_im,
        buf.d_Y_row, buf.d_Ybus_indices, buf.d_Ybus_indptr,
        buf.d_V_re, buf.d_V_im, buf.d_Vm,
        nullptr, nullptr,
        buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
        buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
        buf.d_J_values);
}


void CudaJacobianOp<float>::run(CudaMixedBuffers& buf, IterationContext& ctx)
{
    (void)ctx;
    launch_fill_jacobian_gpu<float, double, double, double>(
        buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
        buf.ybus_values_batched,
        true,
        buf.d_Ybus_re, buf.d_Ybus_im,
        buf.d_Y_row, buf.d_Ybus_indices, buf.d_Ybus_indptr,
        buf.d_V_re, buf.d_V_im, buf.d_Vm,
        &buf.d_Ibus_re, &buf.d_Ibus_im,
        buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
        buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
        buf.d_J_values);
}

#endif  // CUPF_WITH_CUDA
