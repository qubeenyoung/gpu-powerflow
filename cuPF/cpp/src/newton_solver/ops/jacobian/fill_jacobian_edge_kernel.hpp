// ---------------------------------------------------------------------------
// fill_jacobian_edge_kernel.hpp  —  default Jacobian variant (CudaJacobianKind::Edge)
//
// One thread per (batch case, Ybus edge). Each thread:
//   1. writes the four off-diagonal sub-block contributions of its edge by
//      direct assignment (no atomics — every J value slot is written by exactly
//      one edge), and
//   2. if the edge is a diagonal (i == j), adds the bus self term built from the
//      bus current Ibus_i (either the cached d_Ibus, or recomputed by walking
//      row i of Ybus when no cache is available).
//
// This is the production path: the cached-Ibus form reuses the result the Ibus
// stage just produced, so the diagonal costs O(1) per bus instead of a Ybus-row
// rescan. The off-diagonal math is the shared compute_edge_sensitivity().
// ---------------------------------------------------------------------------

#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/jacobian/jacobian_gpu_common.hpp"
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
    // Accumulate in the output precision; the static_cast<AccumScalar> loads
    // below just bring each (possibly higher/lower precision) input into it.
    using AccumScalar = JScalar;

    // One thread per (batch case, Ybus edge). Derive the case and the per-case
    // base offsets into the batched value/voltage/Jacobian arrays.
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_edges) return;

    const int32_t batch  = tid / nnz_ybus;
    const int32_t k      = tid - batch * nnz_ybus;
    const int32_t y_base = ybus_values_batched ? batch * nnz_ybus : 0;  // Ybus values shared unless batched
    const int32_t v_base = batch * n_bus;
    const int32_t j_base = batch * nnz_J;

    const int32_t i = y_row[k];   // edge source bus
    const int32_t j = y_col[k];   // edge target bus

    const AccumScalar yr     = static_cast<AccumScalar>(y_re[y_base + k]);
    const AccumScalar yi     = static_cast<AccumScalar>(y_im[y_base + k]);
    const AccumScalar vi_re  = static_cast<AccumScalar>(v_re[v_base + i]);
    const AccumScalar vi_im  = static_cast<AccumScalar>(v_im[v_base + i]);
    const AccumScalar vj_re  = static_cast<AccumScalar>(v_re[v_base + j]);
    const AccumScalar vj_im  = static_cast<AccumScalar>(v_im[v_base + j]);
    const AccumScalar vj_abs = static_cast<AccumScalar>(vm[v_base + j]);

    // Shared off-diagonal sensitivity (curr, angle term, magnitude term).
    const EdgeSensitivity<AccumScalar> s =
        compute_edge_sensitivity<AccumScalar>(yr, yi, vi_re, vi_im, vj_re, vj_im, vj_abs);

    // Scatter the off-diagonal contributions (one writer per slot -> plain
    // store, cast to the J storage type). A negative map entry means this edge
    // does not contribute to that sub-block (e.g. PV bus has no Q row).
    if (map11[k] >= 0) J_values[j_base + map11[k]] = static_cast<JScalar>(s.term_va_re);
    if (map21[k] >= 0) J_values[j_base + map21[k]] = static_cast<JScalar>(s.term_va_im);
    if (map12[k] >= 0) J_values[j_base + map12[k]] = static_cast<JScalar>(s.term_vm_re);
    if (map22[k] >= 0) J_values[j_base + map22[k]] = static_cast<JScalar>(s.term_vm_im);

    // Only diagonal edges (i == j) add the bus self-term below.
    if (i != j) return;

    // Bus current Ibus_i: use the cached value, or recompute from row i of Ybus.
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

    // Diagonal correction from V_i * conj(Ibus_i): adds to J11/J21 (angle)...
    const AccumScalar vi_conj_i_re = vi_re * ir + vi_im * ii;
    const AccumScalar vi_conj_i_im = vi_im * ir - vi_re * ii;
    const AccumScalar corr_va_re   = -vi_conj_i_im;
    const AccumScalar corr_va_im   =  vi_conj_i_re;

    // ...and to J12/J22 (magnitude) via the normalized voltage; skip if |V_i|~0.
    const AccumScalar vi_abs = static_cast<AccumScalar>(vm[v_base + i]);
    AccumScalar corr_vm_re = AccumScalar(0);
    AccumScalar corr_vm_im = AccumScalar(0);
    if (vi_abs > AccumScalar(1.0e-12)) {
        const AccumScalar vnorm_re = vi_re / vi_abs;
        const AccumScalar vnorm_im = vi_im / vi_abs;
        corr_vm_re = ir * vnorm_re + ii * vnorm_im;
        corr_vm_im = ir * vnorm_im - ii * vnorm_re;
    }

    // Accumulate (+=) onto the diagonal slots already holding the off-diag part.
    if (diag11[i] >= 0) J_values[j_base + diag11[i]] += static_cast<JScalar>(corr_va_re);
    if (diag21[i] >= 0) J_values[j_base + diag21[i]] += static_cast<JScalar>(corr_va_im);
    if (diag12[i] >= 0) J_values[j_base + diag12[i]] += static_cast<JScalar>(corr_vm_re);
    if (diag22[i] >= 0) J_values[j_base + diag22[i]] += static_cast<JScalar>(corr_vm_im);
}


// Launch the default edge kernel: one thread per (case, edge). J_values is
// zeroed first because the diagonal path does += onto slots the off-diagonal
// stores wrote, and unwritten slots (no contributing edge) must stay 0.
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

    // One thread per (case, edge); round the grid up to cover total_edges.
    constexpr int32_t block = 256;
    const int32_t total_edges = batch_size * nnz_ybus;
    const int32_t grid = (total_edges + block - 1) / block;

    fill_jacobian_gpu_kernel<JScalar, YbusScalar, VoltageScalar, IbusScalar><<<grid, block, 0, cupf_current_cuda_stream()>>>(
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

#endif  // CUPF_WITH_CUDA
