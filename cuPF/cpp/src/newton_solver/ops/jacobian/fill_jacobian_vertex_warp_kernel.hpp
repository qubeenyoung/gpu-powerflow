// ---------------------------------------------------------------------------
// fill_jacobian_vertex_warp_kernel.hpp  —  CudaJacobianKind::VertexWarp variant
//
// One warp per Jacobian row (= one pvpq bus); gridDim.y indexes the batch case.
// The 32 lanes stride over that bus's Ybus row:
//   - off-diagonal edges (col != bus): each lane stores its own sub-block
//     contributions and adds the diagonal pieces into per-lane accumulators;
//   - the diagonal edge (col == bus): contributes only to the magnitude diagonal.
// After the row is consumed, the four diagonal accumulators are warp-reduced and
// lane 0 writes the single diagonal value per sub-block. No atomics: the diagonal
// is summed in registers within the warp. This is an alternative/experimental
// layout (the default Edge kernel is the production path). Off-diagonal math is
// the shared compute_edge_sensitivity().
// ---------------------------------------------------------------------------

#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/jacobian/jacobian_gpu_common.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <stdexcept>


namespace {

template <typename JScalar, typename YbusScalar, typename VoltageScalar>
__global__ void fill_jacobian_vertex_warp_kernel(
    int32_t n_rows,
    int32_t nnz_ybus,
    int32_t nnz_J,
    int32_t n_bus,
    int32_t ybus_values_batched,
    const YbusScalar* __restrict__ y_re,
    const YbusScalar* __restrict__ y_im,
    const int32_t* __restrict__ y_col,
    const int32_t* __restrict__ y_row_ptr,
    const VoltageScalar* __restrict__ v_re,
    const VoltageScalar* __restrict__ v_im,
    const VoltageScalar* __restrict__ vm,
    const int32_t* __restrict__ pvpq,
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

    // Identify this lane's warp -> Jacobian row -> bus, and the batch case.
    constexpr int32_t kWarpSize = 32;
    const int32_t lane = threadIdx.x & (kWarpSize - 1);
    const int32_t warp_in_block = threadIdx.x / kWarpSize;
    const int32_t warps_per_block = blockDim.x / kWarpSize;
    const int32_t row = blockIdx.x * warps_per_block + warp_in_block;
    const int32_t batch = blockIdx.y;
    if (row >= n_rows) return;

    const int32_t bus = pvpq[row];
    const bool has_q_row = diag21[bus] >= 0;   // PQ bus -> has a Q (magnitude) row
    const int32_t y_base = ybus_values_batched ? batch * nnz_ybus : 0;
    const int32_t v_base = batch * n_bus;
    const int32_t j_base = batch * nnz_J;

    // Per-lane partial sums of this bus's diagonal sub-block entries.
    AccumScalar diag11_acc = AccumScalar(0);
    AccumScalar diag21_acc = AccumScalar(0);
    AccumScalar diag12_acc = AccumScalar(0);
    AccumScalar diag22_acc = AccumScalar(0);

    // Lanes stride over the bus's Ybus row (i == bus, j == col).
    for (int32_t k = y_row_ptr[bus] + lane; k < y_row_ptr[bus + 1]; k += kWarpSize) {
        const int32_t col = y_col[k];
        const AccumScalar yr     = static_cast<AccumScalar>(y_re[y_base + k]);
        const AccumScalar yi     = static_cast<AccumScalar>(y_im[y_base + k]);
        const AccumScalar vi_re  = static_cast<AccumScalar>(v_re[v_base + bus]);
        const AccumScalar vi_im  = static_cast<AccumScalar>(v_im[v_base + bus]);
        const AccumScalar vj_re  = static_cast<AccumScalar>(v_re[v_base + col]);
        const AccumScalar vj_im  = static_cast<AccumScalar>(v_im[v_base + col]);
        const AccumScalar vj_abs = static_cast<AccumScalar>(vm[v_base + col]);

        // Shared off-diagonal sensitivity for this edge.
        const EdgeSensitivity<AccumScalar> s =
            compute_edge_sensitivity<AccumScalar>(yr, yi, vi_re, vi_im, vj_re, vj_im, vj_abs);

        // Magnitude diagonal piece for this edge: V_bus-normalized curr.
        const AccumScalar vi_abs = static_cast<AccumScalar>(vm[v_base + bus]);
        AccumScalar diag_vm_re = AccumScalar(0);
        AccumScalar diag_vm_im = AccumScalar(0);
        if (vi_abs > AccumScalar(1.0e-12)) {
            const AccumScalar vni_re = vi_re / vi_abs;
            const AccumScalar vni_im = vi_im / vi_abs;
            diag_vm_re = vni_re * s.curr_re + vni_im * s.curr_im;
            diag_vm_im = vni_im * s.curr_re - vni_re * s.curr_im;
        }

        if (col == bus) {
            // Self edge: contributes only to the magnitude diagonal (the angle
            // diagonal is the negated sum of the off-diagonal angle terms, which
            // the self edge does not add to).
            if (has_q_row) {
                diag12_acc += diag_vm_re + s.term_vm_re;
                diag22_acc += diag_vm_im + s.term_vm_im;
            }
            continue;
        }

        // Off-diagonal edge: angle diagonal gathers -term_va; magnitude diagonal
        // gathers diag_vm.
        diag11_acc += -s.term_va_re;
        if (has_q_row) {
            diag21_acc += -s.term_va_im;
            diag12_acc += diag_vm_re;
            diag22_acc += diag_vm_im;
        }

        // Store this edge's off-diagonal sub-block values (one writer per slot).
        if (map11[k] >= 0) J_values[j_base + map11[k]] = static_cast<JScalar>(s.term_va_re);
        if (map12[k] >= 0) J_values[j_base + map12[k]] = static_cast<JScalar>(s.term_vm_re);
        if (has_q_row) {
            if (map21[k] >= 0) J_values[j_base + map21[k]] = static_cast<JScalar>(s.term_va_im);
            if (map22[k] >= 0) J_values[j_base + map22[k]] = static_cast<JScalar>(s.term_vm_im);
        }
    }

    // Reduce the per-lane diagonal partials across the warp; lane 0 has the sum.
    diag11_acc = warp_reduce_sum(diag11_acc);
    diag21_acc = warp_reduce_sum(diag21_acc);
    diag12_acc = warp_reduce_sum(diag12_acc);
    diag22_acc = warp_reduce_sum(diag22_acc);

    if (lane == 0) {
        if (diag11[bus] >= 0) J_values[j_base + diag11[bus]] = static_cast<JScalar>(diag11_acc);
        if (has_q_row) {
            if (diag21[bus] >= 0) J_values[j_base + diag21[bus]] = static_cast<JScalar>(diag21_acc);
            if (diag12[bus] >= 0) J_values[j_base + diag12[bus]] = static_cast<JScalar>(diag12_acc);
            if (diag22[bus] >= 0) J_values[j_base + diag22[bus]] = static_cast<JScalar>(diag22_acc);
        }
    }
}


// Launch the vertex-warp kernel: grid.x covers the pvpq rows (warps_per_block
// warps each), grid.y is the batch. J_values is zeroed first (off-diagonal
// slots not touched by any edge must stay 0).
template <typename JScalar, typename YbusScalar, typename VoltageScalar>
void launch_fill_jacobian_vertex_warp(
    int32_t n_pvpq,
    int32_t nnz_ybus,
    int32_t nnz_J,
    int32_t n_bus,
    int32_t batch_size,
    bool ybus_values_batched,
    const DeviceBuffer<YbusScalar>& y_re,
    const DeviceBuffer<YbusScalar>& y_im,
    const DeviceBuffer<int32_t>& y_col,
    const DeviceBuffer<int32_t>& y_row_ptr,
    const DeviceBuffer<VoltageScalar>& v_re,
    const DeviceBuffer<VoltageScalar>& v_im,
    const DeviceBuffer<VoltageScalar>& vm,
    const DeviceBuffer<int32_t>& pvpq,
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
    if (n_pvpq <= 0 || nnz_ybus <= 0 || nnz_J <= 0 || n_bus <= 0 || batch_size <= 0) {
        throw std::runtime_error("launch_fill_jacobian_vertex_warp: buffers are not prepared");
    }

    J_values.memsetZero();
    constexpr int32_t block = 256;
    constexpr int32_t warps_per_block = block / 32;
    const dim3 grid((n_pvpq + warps_per_block - 1) / warps_per_block, batch_size);

    fill_jacobian_vertex_warp_kernel<JScalar, YbusScalar, VoltageScalar><<<grid, block, 0, cupf_current_cuda_stream()>>>(
        n_pvpq, nnz_ybus, nnz_J, n_bus,
        ybus_values_batched ? 1 : 0,
        y_re.data(), y_im.data(), y_col.data(), y_row_ptr.data(),
        v_re.data(), v_im.data(), vm.data(), pvpq.data(),
        map11.data(), map21.data(), map12.data(), map22.data(),
        diag11.data(), diag21.data(), diag12.data(), diag22.data(),
        J_values.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

}  // namespace

#endif  // CUPF_WITH_CUDA
