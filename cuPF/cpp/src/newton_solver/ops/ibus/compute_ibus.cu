// ---------------------------------------------------------------------------
// compute_ibus.cu
//
// Computes Ibus = Ybus * V.
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

// Scalar SpMV: one thread per (row, batch case), looping the row's nonzeros
// sequentially. For low average degree (~4 nnz/row in power grids) this uses
// every thread productively and avoids the warp-shuffle reduction and idle
// lanes of the vectorized kernel above. Ybus values are shared across the
// batch (y_re[k]); only V is per-case (v_re[base + col]).
template <typename YbusScalar, typename StateScalar, typename AccumScalar>
__global__ void compute_ibus_scalar_kernel(
    int32_t n_bus,
    int32_t total_entries,
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

    const int32_t batch = tid / n_bus;
    const int32_t row   = tid - batch * n_bus;
    const int32_t base  = batch * n_bus;

    AccumScalar acc_re = AccumScalar(0);
    AccumScalar acc_im = AccumScalar(0);
    for (int32_t k = y_row_ptr[row]; k < y_row_ptr[row + 1]; ++k) {
        const int32_t col = y_col[k];
        const AccumScalar yr = static_cast<AccumScalar>(y_re[k]);
        const AccumScalar yi = static_cast<AccumScalar>(y_im[k]);
        const AccumScalar vr = static_cast<AccumScalar>(v_re[base + col]);
        const AccumScalar vi = static_cast<AccumScalar>(v_im[base + col]);
        acc_re += yr * vr - yi * vi;
        acc_im += yr * vi + yi * vr;
    }
    ibus_re[base + row] = static_cast<StateScalar>(acc_re);
    ibus_im[base + row] = static_cast<StateScalar>(acc_im);
}

// Launch helper for the scalar kernel (1 thread per row*batch).
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


void launch_compute_ibus(CudaFp64Storage& buf)
{
    if (buf.n_bus <= 0 || buf.nnz_ybus <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("launch_compute_ibus: buffers are not prepared");
    }
    if (buf.ybus_values_batched) {
        throw std::runtime_error("launch_compute_ibus: batched Ybus values are not supported by the tiled Ibus kernel");
    }

    launch_ibus_scalar<double, double, double>(
        buf.n_bus, buf.batch_size,
        buf.d_Ybus_indptr.data(), buf.d_Ybus_indices.data(),
        buf.d_Ybus_re.data(), buf.d_Ybus_im.data(),
        buf.d_V_re.data(), buf.d_V_im.data(),
        buf.d_Ibus_re.data(), buf.d_Ibus_im.data());
}


void launch_compute_ibus(CudaFp32Storage& buf)
{
    if (buf.n_bus <= 0 || buf.nnz_ybus <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("launch_compute_ibus: buffers are not prepared");
    }
    if (buf.ybus_values_batched) {
        throw std::runtime_error("launch_compute_ibus: batched Ybus values are not supported by the tiled Ibus kernel");
    }

    launch_ibus_scalar<float, float, float>(
        buf.n_bus, buf.batch_size,
        buf.d_Ybus_indptr.data(), buf.d_Ybus_indices.data(),
        buf.d_Ybus_re.data(), buf.d_Ybus_im.data(),
        buf.d_V_re.data(), buf.d_V_im.data(),
        buf.d_Ibus_re.data(), buf.d_Ibus_im.data());
}


void launch_compute_ibus(CudaMixedStorage& buf)
{
    if (buf.n_bus <= 0 || buf.nnz_ybus <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("launch_compute_ibus: buffers are not prepared");
    }
    if (buf.ybus_values_batched) {
        throw std::runtime_error("launch_compute_ibus: batched Ybus values are not supported by the tiled Ibus kernel");
    }

    launch_ibus_scalar<double, double, double>(
        buf.n_bus, buf.batch_size,
        buf.d_Ybus_indptr.data(), buf.d_Ybus_indices.data(),
        buf.d_Ybus_re.data(), buf.d_Ybus_im.data(),
        buf.d_V_re.data(), buf.d_V_im.data(),
        buf.d_Ibus_re.data(), buf.d_Ibus_im.data());
}


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
