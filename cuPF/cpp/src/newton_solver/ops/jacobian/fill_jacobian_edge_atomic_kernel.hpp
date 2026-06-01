// ---------------------------------------------------------------------------
// fill_jacobian_edge_atomic_kernel.hpp  —  CudaJacobianKind::EdgeAtomic variant
//
// One thread per (batch case, Ybus edge), like the default kernel, but every
// write is an atomicAdd. Instead of computing the diagonal self term once from
// Ibus, each edge atomically scatters its own contribution to both the off-
// diagonal slot AND the relevant diagonal slot (the diagonal accumulates the
// negated angle term and the per-edge magnitude term). This trades the cached-
// Ibus diagonal shortcut for contention-tolerant accumulation; it is an
// alternative/experimental layout (the default Edge kernel is the production
// path). Off-diagonal math is the shared compute_edge_sensitivity().
// ---------------------------------------------------------------------------

#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/jacobian/jacobian_gpu_common.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <stdexcept>


namespace {

template <typename JScalar, typename YbusScalar, typename VoltageScalar>
__global__ void fill_jacobian_edge_atomic_kernel(
    int32_t total_edges,
    int32_t nnz_ybus,
    int32_t nnz_J,
    int32_t n_bus,
    int32_t ybus_values_batched,
    const YbusScalar* __restrict__ y_re,
    const YbusScalar* __restrict__ y_im,
    const int32_t* __restrict__ y_row,
    const int32_t* __restrict__ y_col,
    const VoltageScalar* __restrict__ v_re,
    const VoltageScalar* __restrict__ v_im,
    const VoltageScalar* __restrict__ vm,
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
    const AccumScalar vj_abs = static_cast<AccumScalar>(vm[v_base + j]);

    // Shared off-diagonal sensitivity (curr, angle term, magnitude term).
    const EdgeSensitivity<AccumScalar> s =
        compute_edge_sensitivity<AccumScalar>(yr, yi, vi_re, vi_im, vj_re, vj_im, vj_abs);

    // Off-diagonal: atomically add this edge's contribution to each sub-block.
    if (map11[k] >= 0) cupf_atomic_add(&J_values[j_base + map11[k]], static_cast<JScalar>(s.term_va_re));
    if (map21[k] >= 0) cupf_atomic_add(&J_values[j_base + map21[k]], static_cast<JScalar>(s.term_va_im));
    if (map12[k] >= 0) cupf_atomic_add(&J_values[j_base + map12[k]], static_cast<JScalar>(s.term_vm_re));
    if (map22[k] >= 0) cupf_atomic_add(&J_values[j_base + map22[k]], static_cast<JScalar>(s.term_vm_im));

    // Diagonal angle blocks: dP/dVa, dQ/dVa gather the negated angle term of
    // every incident edge (the bus row's diagonal is -sum of off-diagonals).
    if (diag11[i] >= 0) cupf_atomic_add(&J_values[j_base + diag11[i]], static_cast<JScalar>(-s.term_va_re));
    if (diag21[i] >= 0) cupf_atomic_add(&J_values[j_base + diag21[i]], static_cast<JScalar>(-s.term_va_im));

    // Diagonal magnitude blocks: dP/dVm, dQ/dVm gather V_i-normalized curr per
    // edge; skip if |V_i| ~ 0 (no magnitude unknown there).
    const AccumScalar vi_abs = static_cast<AccumScalar>(vm[v_base + i]);
    if (vi_abs > AccumScalar(1.0e-12)) {
        const AccumScalar vni_re = vi_re / vi_abs;
        const AccumScalar vni_im = vi_im / vi_abs;
        const AccumScalar diag_vm_re = vni_re * s.curr_re + vni_im * s.curr_im;
        const AccumScalar diag_vm_im = vni_im * s.curr_re - vni_re * s.curr_im;
        if (diag12[i] >= 0) cupf_atomic_add(&J_values[j_base + diag12[i]], static_cast<JScalar>(diag_vm_re));
        if (diag22[i] >= 0) cupf_atomic_add(&J_values[j_base + diag22[i]], static_cast<JScalar>(diag_vm_im));
    }
}


// Launch the edge-atomic kernel: one thread per (case, edge). J_values is
// zeroed first since every contribution is an atomicAdd onto it.
template <typename JScalar, typename YbusScalar, typename VoltageScalar>
void launch_fill_jacobian_edge_atomic(
    int32_t nnz_ybus,
    int32_t nnz_J,
    int32_t n_bus,
    int32_t batch_size,
    bool ybus_values_batched,
    const DeviceBuffer<YbusScalar>& y_re,
    const DeviceBuffer<YbusScalar>& y_im,
    const DeviceBuffer<int32_t>& y_row,
    const DeviceBuffer<int32_t>& y_col,
    const DeviceBuffer<VoltageScalar>& v_re,
    const DeviceBuffer<VoltageScalar>& v_im,
    const DeviceBuffer<VoltageScalar>& vm,
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
        throw std::runtime_error("launch_fill_jacobian_edge_atomic: buffers are not prepared");
    }

    J_values.memsetZero();
    constexpr int32_t block = 256;
    const int32_t total_edges = batch_size * nnz_ybus;
    const int32_t grid = (total_edges + block - 1) / block;

    fill_jacobian_edge_atomic_kernel<JScalar, YbusScalar, VoltageScalar><<<grid, block, 0, cupf_current_cuda_stream()>>>(
        total_edges, nnz_ybus, nnz_J, n_bus,
        ybus_values_batched ? 1 : 0,
        y_re.data(), y_im.data(), y_row.data(), y_col.data(),
        v_re.data(), v_im.data(), vm.data(),
        map11.data(), map21.data(), map12.data(), map22.data(),
        diag11.data(), diag21.data(), diag12.data(), diag22.data(),
        J_values.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

}  // namespace

#endif  // CUPF_WITH_CUDA
