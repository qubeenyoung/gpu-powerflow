// ---------------------------------------------------------------------------
// fill_jacobian_gpu.cu
//
// CUDA assembly of the power-flow Jacobian — the GPU counterpart of
// fill_jacobian.cpp. The same complex sensitivities are written out in plain
// real arithmetic (no std::complex on device). One thread handles one Ybus
// edge across the batch; diagonal threads (i==j) add the Ibus self-term.
// Templated on the J / Ybus / voltage / Ibus scalar types so FP64, FP32, and
// mixed (FP64 inputs -> FP32 J) pipelines share one kernel.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "fill_jacobian_gpu.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_fp32_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/dump.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>


namespace {

__device__ __forceinline__ float cupf_atomic_add(float* address, float value)
{
    return atomicAdd(address, value);
}


__device__ __forceinline__ double cupf_atomic_add(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}


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
    // Accumulate in the output precision; static_cast<AccumScalar> below just
    // loads each input (possibly higher/lower precision) into that type.
    using AccumScalar = JScalar;

    // One thread per (batch case, Ybus edge). Derive the case and the per-case
    // base offsets into the batched value/voltage/Jacobian arrays.
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_edges) return;

    const int32_t batch  = tid / nnz_ybus;
    const int32_t k      = tid - batch * nnz_ybus;
    const int32_t y_base = ybus_values_batched ? batch * nnz_ybus : 0;
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

    // curr = Y_ij * V_j  (complex multiply in real form).
    const AccumScalar curr_re = yr * vj_re - yi * vj_im;
    const AccumScalar curr_im = yr * vj_im + yi * vj_re;

    // Angle (dVa) term: -i*V_i * conj(curr), giving the J11/J21 contributions.
    const AccumScalar neg_j_vi_re = vi_im;
    const AccumScalar neg_j_vi_im = -vi_re;
    const AccumScalar term_va_re  = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
    const AccumScalar term_va_im  = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

    // Magnitude (dVm) term: V_i * conj(curr / |V_j|); skip when |V_j| ~ 0.
    const AccumScalar vj_abs = static_cast<AccumScalar>(vm[v_base + j]);
    AccumScalar term_vm_re = AccumScalar(0);
    AccumScalar term_vm_im = AccumScalar(0);
    if (vj_abs > AccumScalar(1.0e-12)) {
        const AccumScalar scaled_re = curr_re / vj_abs;
        const AccumScalar scaled_im = curr_im / vj_abs;
        term_vm_re = vi_re * scaled_re + vi_im * scaled_im;
        term_vm_im = vi_im * scaled_re - vi_re * scaled_im;
    }

    // Scatter the off-diagonal contributions (cast to the J storage type).
    if (map11[k] >= 0) J_values[j_base + map11[k]] = static_cast<JScalar>(term_va_re);
    if (map21[k] >= 0) J_values[j_base + map21[k]] = static_cast<JScalar>(term_va_im);
    if (map12[k] >= 0) J_values[j_base + map12[k]] = static_cast<JScalar>(term_vm_re);
    if (map22[k] >= 0) J_values[j_base + map22[k]] = static_cast<JScalar>(term_vm_im);

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

    const AccumScalar yr    = static_cast<AccumScalar>(y_re[y_base + k]);
    const AccumScalar yi    = static_cast<AccumScalar>(y_im[y_base + k]);
    const AccumScalar vi_re = static_cast<AccumScalar>(v_re[v_base + i]);
    const AccumScalar vi_im = static_cast<AccumScalar>(v_im[v_base + i]);
    const AccumScalar vj_re = static_cast<AccumScalar>(v_re[v_base + j]);
    const AccumScalar vj_im = static_cast<AccumScalar>(v_im[v_base + j]);

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

    if (map11[k] >= 0) cupf_atomic_add(&J_values[j_base + map11[k]], static_cast<JScalar>(term_va_re));
    if (map21[k] >= 0) cupf_atomic_add(&J_values[j_base + map21[k]], static_cast<JScalar>(term_va_im));
    if (map12[k] >= 0) cupf_atomic_add(&J_values[j_base + map12[k]], static_cast<JScalar>(term_vm_re));
    if (map22[k] >= 0) cupf_atomic_add(&J_values[j_base + map22[k]], static_cast<JScalar>(term_vm_im));

    if (diag11[i] >= 0) cupf_atomic_add(&J_values[j_base + diag11[i]], static_cast<JScalar>(-term_va_re));
    if (diag21[i] >= 0) cupf_atomic_add(&J_values[j_base + diag21[i]], static_cast<JScalar>(-term_va_im));

    const AccumScalar vi_abs = static_cast<AccumScalar>(vm[v_base + i]);
    if (vi_abs > AccumScalar(1.0e-12)) {
        const AccumScalar vni_re = vi_re / vi_abs;
        const AccumScalar vni_im = vi_im / vi_abs;
        const AccumScalar diag_vm_re = vni_re * curr_re + vni_im * curr_im;
        const AccumScalar diag_vm_im = vni_im * curr_re - vni_re * curr_im;
        if (diag12[i] >= 0) cupf_atomic_add(&J_values[j_base + diag12[i]], static_cast<JScalar>(diag_vm_re));
        if (diag22[i] >= 0) cupf_atomic_add(&J_values[j_base + diag22[i]], static_cast<JScalar>(diag_vm_im));
    }
}


template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T value)
{
    constexpr unsigned kFullMask = 0xffffffffu;
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullMask, value, offset);
    }
    return value;
}


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

    constexpr int32_t kWarpSize = 32;
    const int32_t lane = threadIdx.x & (kWarpSize - 1);
    const int32_t warp_in_block = threadIdx.x / kWarpSize;
    const int32_t warps_per_block = blockDim.x / kWarpSize;
    const int32_t row = blockIdx.x * warps_per_block + warp_in_block;
    const int32_t batch = blockIdx.y;
    if (row >= n_rows) return;

    const int32_t bus = pvpq[row];
    const bool has_q_row = diag21[bus] >= 0;
    const int32_t y_base = ybus_values_batched ? batch * nnz_ybus : 0;
    const int32_t v_base = batch * n_bus;
    const int32_t j_base = batch * nnz_J;

    AccumScalar diag11_acc = AccumScalar(0);
    AccumScalar diag21_acc = AccumScalar(0);
    AccumScalar diag12_acc = AccumScalar(0);
    AccumScalar diag22_acc = AccumScalar(0);

    for (int32_t k = y_row_ptr[bus] + lane; k < y_row_ptr[bus + 1]; k += kWarpSize) {
        const int32_t col = y_col[k];
        const AccumScalar yr    = static_cast<AccumScalar>(y_re[y_base + k]);
        const AccumScalar yi    = static_cast<AccumScalar>(y_im[y_base + k]);
        const AccumScalar vi_re = static_cast<AccumScalar>(v_re[v_base + bus]);
        const AccumScalar vi_im = static_cast<AccumScalar>(v_im[v_base + bus]);
        const AccumScalar vj_re = static_cast<AccumScalar>(v_re[v_base + col]);
        const AccumScalar vj_im = static_cast<AccumScalar>(v_im[v_base + col]);

        const AccumScalar curr_re = yr * vj_re - yi * vj_im;
        const AccumScalar curr_im = yr * vj_im + yi * vj_re;

        const AccumScalar neg_j_vi_re = vi_im;
        const AccumScalar neg_j_vi_im = -vi_re;
        const AccumScalar term_va_re  = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
        const AccumScalar term_va_im  = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

        const AccumScalar vj_abs = static_cast<AccumScalar>(vm[v_base + col]);
        AccumScalar term_vm_re = AccumScalar(0);
        AccumScalar term_vm_im = AccumScalar(0);
        if (vj_abs > AccumScalar(1.0e-12)) {
            const AccumScalar scaled_re = curr_re / vj_abs;
            const AccumScalar scaled_im = curr_im / vj_abs;
            term_vm_re = vi_re * scaled_re + vi_im * scaled_im;
            term_vm_im = vi_im * scaled_re - vi_re * scaled_im;
        }

        const AccumScalar vi_abs = static_cast<AccumScalar>(vm[v_base + bus]);
        AccumScalar diag_vm_re = AccumScalar(0);
        AccumScalar diag_vm_im = AccumScalar(0);
        if (vi_abs > AccumScalar(1.0e-12)) {
            const AccumScalar vni_re = vi_re / vi_abs;
            const AccumScalar vni_im = vi_im / vi_abs;
            diag_vm_re = vni_re * curr_re + vni_im * curr_im;
            diag_vm_im = vni_im * curr_re - vni_re * curr_im;
        }

        if (col == bus) {
            if (has_q_row) {
                diag12_acc += diag_vm_re + term_vm_re;
                diag22_acc += diag_vm_im + term_vm_im;
            }
            continue;
        }

        diag11_acc += -term_va_re;
        if (has_q_row) {
            diag21_acc += -term_va_im;
            diag12_acc += diag_vm_re;
            diag22_acc += diag_vm_im;
        }

        if (map11[k] >= 0) J_values[j_base + map11[k]] = static_cast<JScalar>(term_va_re);
        if (map12[k] >= 0) J_values[j_base + map12[k]] = static_cast<JScalar>(term_vm_re);
        if (has_q_row) {
            if (map21[k] >= 0) J_values[j_base + map21[k]] = static_cast<JScalar>(term_va_im);
            if (map22[k] >= 0) J_values[j_base + map22[k]] = static_cast<JScalar>(term_vm_im);
        }
    }

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

// Debug-only: copy the assembled CSR Jacobian back to host and dump it.
template <typename ValueType>
void dump_cuda_jacobian_if_enabled(const char* name,
                                   int32_t iteration,
                                   int32_t dim,
                                   const DeviceBuffer<int32_t>& d_row_ptr,
                                   const DeviceBuffer<int32_t>& d_col_idx,
                                   const DeviceBuffer<ValueType>& d_values,
                                   int32_t nnz)
{
    if (!newton_solver::utils::isDumpEnabled()) {
        return;
    }
    if (dim <= 0 || nnz <= 0) {
        return;
    }

    std::vector<int32_t> row_ptr(static_cast<std::size_t>(dim + 1));
    std::vector<int32_t> col_idx(static_cast<std::size_t>(nnz));
    std::vector<ValueType> values(static_cast<std::size_t>(nnz));
    d_row_ptr.copyTo(row_ptr.data(), row_ptr.size());
    d_col_idx.copyTo(col_idx.data(), col_idx.size());
    d_values.copyTo(values.data(), values.size());
    newton_solver::utils::dumpCSR(name,
                                  iteration,
                                  row_ptr.data(),
                                  col_idx.data(),
                                  values.data(),
                                  dim,
                                  dim);
}

}  // namespace


// ===========================================================================
// Per-pipeline dispatch. Each overload picks the kernel scalar types and the
// batch / cached-Ibus / batched-Ybus flags for its storage layout.
// ===========================================================================

// FP64: all-double inputs and Jacobian, batched (recompute or cached Ibus).
void CudaJacobianOp<double>::run(CudaFp64Storage& buf, IterationContext& ctx)
{
    if (kind == CudaJacobianKind::EdgeAtomic) {
        launch_fill_jacobian_edge_atomic<double, double, double>(
            buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_row, buf.d_Ybus_indices,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    } else if (kind == CudaJacobianKind::VertexWarp) {
        launch_fill_jacobian_vertex_warp<double, double, double>(
            buf.n_pvpq, buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_indices, buf.d_Ybus_indptr,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            buf.d_pvpq,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    } else {
        launch_fill_jacobian_gpu<double, double, double, double>(
            buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            // use_cached_ibus=true: the ibus stage ran just before jacobian in the
            // NR loop (and in prepare_adjoint_cache), so d_Ibus is current. Reusing
            // it avoids recomputing the per-diagonal current injection in-kernel.
            true,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_row, buf.d_Ybus_indices, buf.d_Ybus_indptr,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            &buf.d_Ibus_re, &buf.d_Ibus_im,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    }
    dump_cuda_jacobian_if_enabled("jacobian",
                                  ctx.iter,
                                  buf.dimF,
                                  buf.d_J_row_ptr,
                                  buf.d_J_col_idx,
                                  buf.d_J_values,
                                  buf.nnz_J);
}


// Mixed: FP64 Ybus/voltage/Ibus inputs assembled into an FP32 Jacobian, batched.
void CudaJacobianOp<float>::run(CudaMixedStorage& buf, IterationContext& ctx)
{
    if (kind == CudaJacobianKind::EdgeAtomic) {
        launch_fill_jacobian_edge_atomic<float, double, double>(
            buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_row, buf.d_Ybus_indices,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    } else if (kind == CudaJacobianKind::VertexWarp) {
        launch_fill_jacobian_vertex_warp<float, double, double>(
            buf.n_pvpq, buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_indices, buf.d_Ybus_indptr,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            buf.d_pvpq,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    } else {
        launch_fill_jacobian_gpu<float, double, double, double>(
            buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            true,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_row, buf.d_Ybus_indices, buf.d_Ybus_indptr,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            &buf.d_Ibus_re, &buf.d_Ibus_im,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    }
    dump_cuda_jacobian_if_enabled("jacobian",
                                  ctx.iter,
                                  buf.dimF,
                                  buf.d_J_row_ptr,
                                  buf.d_J_col_idx,
                                  buf.d_J_values,
                                  buf.nnz_J);
}


// FP32: all-float inputs and Jacobian, batched.
void CudaJacobianOp<float>::run(CudaFp32Storage& buf, IterationContext& ctx)
{
    if (kind == CudaJacobianKind::EdgeAtomic) {
        launch_fill_jacobian_edge_atomic<float, float, float>(
            buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_row, buf.d_Ybus_indices,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    } else if (kind == CudaJacobianKind::VertexWarp) {
        launch_fill_jacobian_vertex_warp<float, float, float>(
            buf.n_pvpq, buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_indices, buf.d_Ybus_indptr,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            buf.d_pvpq,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    } else {
        launch_fill_jacobian_gpu<float, float, float, float>(
            buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            true,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_row, buf.d_Ybus_indices, buf.d_Ybus_indptr,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            &buf.d_Ibus_re, &buf.d_Ibus_im,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    }
    dump_cuda_jacobian_if_enabled("jacobian",
                                  ctx.iter,
                                  buf.dimF,
                                  buf.d_J_row_ptr,
                                  buf.d_J_col_idx,
                                  buf.d_J_values,
                                  buf.nnz_J);
}

#endif  // CUPF_WITH_CUDA
