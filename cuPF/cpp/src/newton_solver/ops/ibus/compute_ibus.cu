// ---------------------------------------------------------------------------
// compute_ibus.cu
//
// CUDA bus-current stage:  Ibus = Ybus * V   (complex sparse mat-vec).
//
// Ybus is a complex CSR matrix (n_bus x n_bus); V is the complex bus voltage.
// Ibus[i] = sum over the nonzeros (i, col) of  Ybus[i,col] * V[col].
// The result feeds the mismatch (S - V * conj(Ibus)) and Jacobian stages.
//
// Implementation: one thread per (batch case, row). Each thread walks its row's
// CSR nonzeros sequentially and accumulates one complex dot product. Power
// grids are very sparse (~2-4 nnz/row), so a scalar-per-row SpMV keeps every
// thread busy and avoids the idle warp lanes / shuffle reductions a vectorized
// (warp-per-row) kernel would incur at this nnz/row. Ybus values are shared
// across the batch (y_re[k]); only V is per-case (v at base + col).
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "compute_ibus.hpp"

#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_fp32_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <stdexcept>


namespace {

// Common precondition check for the three precision launchers below. Templated
// on the storage type so it works for FP32/FP64/Mixed (all expose the same
// batch metadata via CudaBatchedStorage).
template <typename Storage>
void require_ibus_ready(const Storage& buf)
{
    if (buf.n_bus <= 0 || buf.nnz_ybus <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("launch_compute_ibus: buffers are not prepared");
    }
    // The scalar SpMV kernel indexes Ybus values as y_re[k] (shared across the
    // batch). Per-case Ybus values would need a y_base = batch * nnz offset that
    // this kernel does not apply, so reject that configuration explicitly.
    if (buf.ybus_values_batched) {
        throw std::runtime_error(
            "launch_compute_ibus: per-case (batched) Ybus values are not supported by the Ibus kernel");
    }
}

// Scalar complex SpMV: one thread computes Ibus for a single (batch, row).
//
// Precision is split three ways so one kernel serves every profile:
//   YbusScalar  - element type of the Ybus value arrays
//   StateScalar - element type of V and the Ibus output
//   AccumScalar - type the dot product accumulates in (e.g. double even for an
//                 FP32 state, to hold the sum accurately before storing back)
template <typename YbusScalar, typename StateScalar, typename AccumScalar>
__global__ void compute_ibus_scalar_kernel(
    int32_t n_bus,
    int32_t total_entries,          // n_bus * batch_count (one thread each)
    const int32_t* __restrict__ y_row_ptr,
    const int32_t* __restrict__ y_col,
    const YbusScalar* __restrict__ y_re,
    const YbusScalar* __restrict__ y_im,
    const StateScalar* __restrict__ v_re,
    const StateScalar* __restrict__ v_im,
    StateScalar* __restrict__ ibus_re,
    StateScalar* __restrict__ ibus_im)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_entries) return;

    // Map the flat thread id to (batch case, row). Per-bus arrays are laid out
    // batch-major, so case `batch` starts at offset `base` in V/Ibus.
    const int32_t batch = tid / n_bus;
    const int32_t row   = tid - batch * n_bus;
    const int32_t base  = batch * n_bus;

    // Accumulate the complex dot product of Ybus row `row` with V (this case).
    AccumScalar acc_re = AccumScalar(0);
    AccumScalar acc_im = AccumScalar(0);
    for (int32_t k = y_row_ptr[row]; k < y_row_ptr[row + 1]; ++k) {
        const int32_t col = y_col[k];
        const AccumScalar yr = static_cast<AccumScalar>(y_re[k]);          // Ybus value (batch-shared)
        const AccumScalar yi = static_cast<AccumScalar>(y_im[k]);
        const AccumScalar vr = static_cast<AccumScalar>(v_re[base + col]); // V at this case
        const AccumScalar vi = static_cast<AccumScalar>(v_im[base + col]);
        // Complex multiply-accumulate: (yr + i*yi) * (vr + i*vi).
        acc_re += yr * vr - yi * vi;   // Re = yr*vr - yi*vi
        acc_im += yr * vi + yi * vr;   // Im = yr*vi + yi*vr
    }
    ibus_re[base + row] = static_cast<StateScalar>(acc_re);
    ibus_im[base + row] = static_cast<StateScalar>(acc_im);
}

// Configure and launch the scalar SpMV: one thread per (row, case), so the grid
// covers n_bus * batch_count entries.
template <typename YbusScalar, typename StateScalar, typename AccumScalar>
void launch_ibus_scalar(int32_t n_bus, int32_t batch_count,
                        const int32_t* y_row_ptr, const int32_t* y_col,
                        const YbusScalar* y_re, const YbusScalar* y_im,
                        const StateScalar* v_re, const StateScalar* v_im,
                        StateScalar* ibus_re, StateScalar* ibus_im)
{
    constexpr int32_t block = 256;
    const int32_t total = n_bus * batch_count;
    const int32_t grid = (total + block - 1) / block;
    compute_ibus_scalar_kernel<YbusScalar, StateScalar, AccumScalar>
        <<<grid, block, 0, cupf_current_cuda_stream()>>>(
            n_bus, total, y_row_ptr, y_col, y_re, y_im, v_re, v_im, ibus_re, ibus_im);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

}  // namespace


// FP64 profile: Ybus/V/Ibus and the accumulator are all double.
void launch_compute_ibus(CudaFp64Storage& buf)
{
    require_ibus_ready(buf);
    launch_ibus_scalar<double, double, double>(
        buf.n_bus, buf.batch_size,
        buf.d_Ybus_indptr.data(), buf.d_Ybus_indices.data(),
        buf.d_Ybus_re.data(), buf.d_Ybus_im.data(),
        buf.d_V_re.data(), buf.d_V_im.data(),
        buf.d_Ibus_re.data(), buf.d_Ibus_im.data());
}


// FP32 profile: everything (including the accumulator) is float.
void launch_compute_ibus(CudaFp32Storage& buf)
{
    require_ibus_ready(buf);
    launch_ibus_scalar<float, float, float>(
        buf.n_bus, buf.batch_size,
        buf.d_Ybus_indptr.data(), buf.d_Ybus_indices.data(),
        buf.d_Ybus_re.data(), buf.d_Ybus_im.data(),
        buf.d_V_re.data(), buf.d_V_im.data(),
        buf.d_Ibus_re.data(), buf.d_Ibus_im.data());
}


// Mixed profile: state (Ybus/V/Ibus) is double, so Ibus runs identically to
// FP64 — only the Jacobian/solve stages drop to float.
void launch_compute_ibus(CudaMixedStorage& buf)
{
    require_ibus_ready(buf);
    launch_ibus_scalar<double, double, double>(
        buf.n_bus, buf.batch_size,
        buf.d_Ybus_indptr.data(), buf.d_Ybus_indices.data(),
        buf.d_Ybus_re.data(), buf.d_Ybus_im.data(),
        buf.d_V_re.data(), buf.d_V_im.data(),
        buf.d_Ibus_re.data(), buf.d_Ibus_im.data());
}


// Op wrappers: the Ibus stage takes no per-iteration context beyond the buffers
// (the voltage it reads was written by the previous voltage_update stage).
template <>
void CudaIbusOp<CudaFp64Storage>::run(CudaFp64Storage& buf, IterationContext& ctx)
{
    (void)ctx;
    launch_compute_ibus(buf);
}

template <>
void CudaIbusOp<CudaFp32Storage>::run(CudaFp32Storage& buf, IterationContext& ctx)
{
    (void)ctx;
    launch_compute_ibus(buf);
}

template <>
void CudaIbusOp<CudaMixedStorage>::run(CudaMixedStorage& buf, IterationContext& ctx)
{
    (void)ctx;
    launch_compute_ibus(buf);
}

#endif  // CUPF_WITH_CUDA
