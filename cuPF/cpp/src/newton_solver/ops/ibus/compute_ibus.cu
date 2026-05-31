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

template <int32_t LANES, int32_t BTILE,
          typename YbusScalar, typename StateScalar, typename AccumScalar>
__global__ void compute_ibus_kernel(
    int32_t n_bus,
    int32_t batch_count,
    const int32_t* __restrict__ y_row_ptr,
    const int32_t* __restrict__ y_col,
    const YbusScalar* __restrict__ y_re,
    const YbusScalar* __restrict__ y_im,
    const StateScalar* __restrict__ v_re,
    const StateScalar* __restrict__ v_im,
    StateScalar* __restrict__ ibus_re,
    StateScalar* __restrict__ ibus_im)
{
    const int32_t row = blockIdx.x;
    const int32_t lane = threadIdx.x;
    const int32_t tb = threadIdx.y;
    const int32_t batch = blockIdx.y * BTILE + tb;
    if (row >= n_bus || batch >= batch_count) return;

    const int32_t base = batch * n_bus;

    AccumScalar acc_re = AccumScalar(0);
    AccumScalar acc_im = AccumScalar(0);
    for (int32_t k = y_row_ptr[row] + lane; k < y_row_ptr[row + 1]; k += LANES) {
        const int32_t col = y_col[k];
        const AccumScalar yr = static_cast<AccumScalar>(y_re[k]);
        const AccumScalar yi = static_cast<AccumScalar>(y_im[k]);
        const AccumScalar vr = static_cast<AccumScalar>(v_re[base + col]);
        const AccumScalar vi = static_cast<AccumScalar>(v_im[base + col]);
        acc_re += yr * vr - yi * vi;
        acc_im += yr * vi + yi * vr;
    }

    const int32_t linear_lane = tb * LANES + lane;
    const uint32_t mask = 0xffffffffu;
    for (int32_t offset = LANES / 2; offset > 0; offset >>= 1) {
        acc_re += __shfl_down_sync(mask, acc_re, offset, LANES);
        acc_im += __shfl_down_sync(mask, acc_im, offset, LANES);
    }

    if (lane == 0) {
        ibus_re[base + row] = static_cast<StateScalar>(acc_re);
        ibus_im[base + row] = static_cast<StateScalar>(acc_im);
    }
    (void)linear_lane;
}

}  // namespace


void launch_compute_ibus(CudaFp64Storage& buf)
{
    if (buf.n_bus <= 0 || buf.d_Ybus_re.empty()) {
        throw std::runtime_error("launch_compute_ibus: buffers are not prepared");
    }

    constexpr int32_t lanes = 32;
    constexpr int32_t btile = 1;
    const dim3 block(lanes, btile);
    const dim3 grid(buf.n_bus, 1);

    compute_ibus_kernel<lanes, btile, double, double, double><<<grid, block, 0, cupf_current_cuda_stream()>>>(
        buf.n_bus,
        1,
        buf.d_Ybus_indptr.data(),
        buf.d_Ybus_indices.data(),
        buf.d_Ybus_re.data(),
        buf.d_Ybus_im.data(),
        buf.d_V_re.data(),
        buf.d_V_im.data(),
        buf.d_Ibus_re.data(),
        buf.d_Ibus_im.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}


void launch_compute_ibus(CudaFp32Storage& buf)
{
    if (buf.n_bus <= 0 || buf.nnz_ybus <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("launch_compute_ibus: buffers are not prepared");
    }
    if (buf.ybus_values_batched) {
        throw std::runtime_error("launch_compute_ibus: batched Ybus values are not supported by the tiled Ibus kernel");
    }

    constexpr int32_t lanes = 32;
    constexpr int32_t btile = 8;
    const dim3 block(lanes, btile);
    const dim3 grid(buf.n_bus, (buf.batch_size + btile - 1) / btile);

    compute_ibus_kernel<lanes, btile, float, float, float><<<grid, block, 0, cupf_current_cuda_stream()>>>(
        buf.n_bus,
        buf.batch_size,
        buf.d_Ybus_indptr.data(),
        buf.d_Ybus_indices.data(),
        buf.d_Ybus_re.data(),
        buf.d_Ybus_im.data(),
        buf.d_V_re.data(),
        buf.d_V_im.data(),
        buf.d_Ibus_re.data(),
        buf.d_Ibus_im.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}


void launch_compute_ibus(CudaMixedStorage& buf)
{
    if (buf.n_bus <= 0 || buf.nnz_ybus <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("launch_compute_ibus: buffers are not prepared");
    }
    if (buf.ybus_values_batched) {
        throw std::runtime_error("launch_compute_ibus: batched Ybus values are not supported by the tiled Ibus kernel");
    }

    constexpr int32_t lanes = 32;
    constexpr int32_t btile = 8;
    const dim3 block(lanes, btile);
    const dim3 grid(buf.n_bus, (buf.batch_size + btile - 1) / btile);

    compute_ibus_kernel<lanes, btile, double, double, double><<<grid, block, 0, cupf_current_cuda_stream()>>>(
        buf.n_bus,
        buf.batch_size,
        buf.d_Ybus_indptr.data(),
        buf.d_Ybus_indices.data(),
        buf.d_Ybus_re.data(),
        buf.d_Ybus_im.data(),
        buf.d_V_re.data(),
        buf.d_V_im.data(),
        buf.d_Ibus_re.data(),
        buf.d_Ibus_im.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
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
