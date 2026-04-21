#include "mismatch.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace exp20260420::mismatch {

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err = (call);                                                          \
        if (err != cudaSuccess) {                                                          \
            throw std::runtime_error(                                                       \
                std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - " + cudaGetErrorString(err));                                         \
        }                                                                                   \
    } while (0)

namespace {

constexpr int32_t kBlock = 256;

__global__ void mismatch_pack_inline_kernel(const YbusGraph ybus,
                                            const double* __restrict__ v_re,
                                            const double* __restrict__ v_im,
                                            const float* __restrict__ sbus_re,
                                            const float* __restrict__ sbus_im,
                                            const int32_t* __restrict__ pv,
                                            const int32_t* __restrict__ pq,
                                            int32_t n_pv,
                                            int32_t n_pq,
                                            int32_t dim,
                                            double* __restrict__ F,
                                            double* __restrict__ block_norm)
{
    extern __shared__ double norm_scratch[];

    const int32_t batch = blockIdx.y;
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t n_active = n_pv + n_pq;
    const int32_t bus_base = batch * ybus.n_bus;
    const int32_t dim_base = batch * dim;

    double abs_mis = 0.0;
    if (tid < n_active) {
        int32_t bus = 0;
        if (tid < n_pv) {
            bus = pv[tid];
        } else {
            bus = pq[tid - n_pv];
        }

        double i_re = 0.0;
        double i_im = 0.0;
        for (int32_t k = ybus.row_ptr[bus]; k < ybus.row_ptr[bus + 1]; ++k) {
            const int32_t col = ybus.col[k];
            const double yr = static_cast<double>(ybus.real[k]);
            const double yi = static_cast<double>(ybus.imag[k]);
            const double vr = v_re[bus_base + col];
            const double vi = v_im[bus_base + col];
            i_re += yr * vr - yi * vi;
            i_im += yr * vi + yi * vr;
        }

        const double vr = v_re[bus_base + bus];
        const double vi = v_im[bus_base + bus];
        const double mis_re =
            vr * i_re + vi * i_im - static_cast<double>(sbus_re[bus_base + bus]);

        F[dim_base + tid] = mis_re;
        abs_mis = fabs(mis_re);

        if (tid >= n_pv) {
            const int32_t pq_slot = tid - n_pv;
            const double mis_im =
                vi * i_re - vr * i_im - static_cast<double>(sbus_im[bus_base + bus]);
            F[dim_base + n_pv + n_pq + pq_slot] = mis_im;
            abs_mis = fmax(abs_mis, fabs(mis_im));
        }
    }

    norm_scratch[threadIdx.x] = abs_mis;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            norm_scratch[threadIdx.x] =
                fmax(norm_scratch[threadIdx.x], norm_scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_norm[batch * gridDim.x + blockIdx.x] = norm_scratch[0];
    }
}

__global__ void reduce_norm_kernel(const double* __restrict__ block_norm,
                                   int32_t norm_blocks,
                                   int32_t batch_size,
                                   double* __restrict__ norm_value)
{
    extern __shared__ double norm_scratch[];

    double local = 0.0;
    for (int32_t batch = blockIdx.x; batch < batch_size; batch += gridDim.x) {
        for (int32_t i = threadIdx.x; i < norm_blocks; i += blockDim.x) {
            local = fmax(local, block_norm[batch * norm_blocks + i]);
        }
    }

    norm_scratch[threadIdx.x] = local;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            norm_scratch[threadIdx.x] =
                fmax(norm_scratch[threadIdx.x], norm_scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        norm_value[blockIdx.x] = norm_scratch[0];
    }
}

}  // namespace

void mismatchDestroy(MismatchWorkspace& ws)
{
    if (ws.block_norm != nullptr) {
        cudaFree(ws.block_norm);
    }
    if (ws.norm_value != nullptr) {
        cudaFree(ws.norm_value);
    }

    ws = MismatchWorkspace{};
}

void mismatchAnalyze(MismatchWorkspace& ws,
                     const YbusGraph& ybus,
                     const int32_t* pv,
                     int32_t n_pv,
                     const int32_t* pq,
                     int32_t n_pq,
                     int32_t batch_size,
                     cudaStream_t stream)
{
    if (ybus.n_bus <= 0 || ybus.n_edges <= 0) {
        throw std::invalid_argument("mismatchAnalyze: bad Ybus size");
    }
    if (ybus.row_ptr == nullptr || ybus.col == nullptr ||
        ybus.real == nullptr || ybus.imag == nullptr) {
        throw std::invalid_argument("mismatchAnalyze: Ybus device pointers are null");
    }
    if (n_pv < 0 || n_pq < 0 || n_pv + n_pq <= 0 || batch_size <= 0) {
        throw std::invalid_argument("mismatchAnalyze: bad dimensions");
    }
    if ((n_pv > 0 && pv == nullptr) || (n_pq > 0 && pq == nullptr)) {
        throw std::invalid_argument("mismatchAnalyze: pv/pq device pointers are null");
    }

    mismatchDestroy(ws);

    ws.n_bus = ybus.n_bus;
    ws.n_edges = ybus.n_edges;
    ws.n_pv = n_pv;
    ws.n_pq = n_pq;
    ws.dim = n_pv + 2 * n_pq;
    ws.batch_size = batch_size;
    ws.norm_blocks = (n_pv + n_pq + kBlock - 1) / kBlock;
    ws.ybus = ybus;
    ws.pv = pv;
    ws.pq = pq;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.block_norm),
                          static_cast<std::size_t>(ws.batch_size) *
                              static_cast<std::size_t>(ws.norm_blocks) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.norm_value),
                          static_cast<std::size_t>(ws.batch_size) * sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(ws.block_norm, 0,
                               static_cast<std::size_t>(ws.batch_size) *
                                   static_cast<std::size_t>(ws.norm_blocks) * sizeof(double),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(ws.norm_value, 0,
                               static_cast<std::size_t>(ws.batch_size) * sizeof(double),
                               stream));
}

double mismatchCompute(MismatchWorkspace& ws,
                       const double* v_re,
                       const double* v_im,
                       const float* sbus_re,
                       const float* sbus_im,
                       double* F,
                       cudaStream_t stream)
{
    if (ws.n_bus <= 0 || ws.dim <= 0 || ws.ybus.row_ptr == nullptr) {
        throw std::runtime_error("mismatchCompute: call mismatchAnalyze first");
    }
    if (v_re == nullptr || v_im == nullptr ||
        sbus_re == nullptr || sbus_im == nullptr || F == nullptr) {
        throw std::invalid_argument("mismatchCompute: input device pointer is null");
    }

    const dim3 grid(ws.norm_blocks, ws.batch_size);
    mismatch_pack_inline_kernel<<<grid, kBlock, kBlock * sizeof(double), stream>>>(
        ws.ybus,
        v_re,
        v_im,
        sbus_re,
        sbus_im,
        ws.pv,
        ws.pq,
        ws.n_pv,
        ws.n_pq,
        ws.dim,
        F,
        ws.block_norm);
    CUDA_CHECK(cudaGetLastError());

    reduce_norm_kernel<<<1, kBlock, kBlock * sizeof(double), stream>>>(
        ws.block_norm,
        ws.norm_blocks,
        ws.batch_size,
        ws.norm_value);
    CUDA_CHECK(cudaGetLastError());

    double host_norm = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(
        &host_norm,
        ws.norm_value,
        sizeof(double),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!std::isfinite(host_norm)) {
        throw std::runtime_error("mismatchCompute: mismatch norm is not finite");
    }
    return host_norm;
}

}  // namespace exp20260420::mismatch
