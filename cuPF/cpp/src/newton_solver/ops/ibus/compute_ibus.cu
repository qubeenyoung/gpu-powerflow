// ---------------------------------------------------------------------------
// compute_ibus.cu
//
// Computes Ibus = Ybus * V.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "compute_ibus.hpp"

#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <stdexcept>


namespace {

template <int32_t LANES, int32_t BTILE, typename YbusScalar>
__global__ void compute_ibus_kernel(
    int32_t n_bus,
    int32_t batch_count,
    const int32_t* __restrict__ y_row_ptr,
    const int32_t* __restrict__ y_col,
    const YbusScalar* __restrict__ y_re,
    const YbusScalar* __restrict__ y_im,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    double* __restrict__ ibus_re,
    double* __restrict__ ibus_im)
{
    const int32_t row = blockIdx.x;
    const int32_t lane = threadIdx.x;
    const int32_t tb = threadIdx.y;
    const int32_t batch = blockIdx.y * BTILE + tb;
    if (row >= n_bus || batch >= batch_count) return;

    const int32_t base = batch * n_bus;

    double acc_re = 0.0;
    double acc_im = 0.0;
    for (int32_t k = y_row_ptr[row] + lane; k < y_row_ptr[row + 1]; k += LANES) {
        const int32_t col = y_col[k];
        const double yr = static_cast<double>(y_re[k]);
        const double yi = static_cast<double>(y_im[k]);
        const double vr = v_re[base + col];
        const double vi = v_im[base + col];
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
        ibus_re[base + row] = acc_re;
        ibus_im[base + row] = acc_im;
    }
    (void)linear_lane;
}

}  // namespace


void launch_compute_ibus(CudaFp64Buffers& buf)
{
    if (buf.n_bus <= 0 || buf.d_Ybus_re.empty()) {
        throw std::runtime_error("launch_compute_ibus: buffers are not prepared");
    }

    constexpr int32_t lanes = 32;
    constexpr int32_t btile = 1;
    const dim3 block(lanes, btile);
    const dim3 grid(buf.n_bus, 1);

    compute_ibus_kernel<lanes, btile, double><<<grid, block>>>(
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


void launch_compute_ibus(CudaMixedBuffers& buf)
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

    compute_ibus_kernel<lanes, btile, double><<<grid, block>>>(
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
void CudaIbusOp<CudaFp64Buffers>::run(CudaFp64Buffers& buf, IterationContext& ctx)
{
    (void)ctx;
    launch_compute_ibus(buf);
}

template <>
void CudaIbusOp<CudaMixedBuffers>::run(CudaMixedBuffers& buf, IterationContext& ctx)
{
    (void)ctx;
    launch_compute_ibus(buf);
}

#endif  // CUPF_WITH_CUDA
