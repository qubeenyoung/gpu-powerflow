#include "mismatch.hpp"

#include <algorithm>
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

#define CUSPARSE_CHECK(call)                                                                \
    do {                                                                                    \
        cusparseStatus_t status = (call);                                                   \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                           \
            throw std::runtime_error(                                                       \
                std::string("cuSPARSE error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - status=" + std::to_string(static_cast<int>(status)));                  \
        }                                                                                   \
    } while (0)

namespace {

constexpr int32_t kBlock = 256;

__global__ void pack_complex_kernel(const float* __restrict__ re,
                                    const float* __restrict__ im,
                                    cuFloatComplex* __restrict__ z,
                                    int32_t n)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        z[tid] = make_cuFloatComplex(re[tid], im[tid]);
    }
}

__global__ void build_mismatch_kernel(const cuFloatComplex* __restrict__ v,
                                      const cuFloatComplex* __restrict__ ibus,
                                      const float* __restrict__ sbus_re,
                                      const float* __restrict__ sbus_im,
                                      const int32_t* __restrict__ pv,
                                      const int32_t* __restrict__ pq,
                                      int32_t n_pv,
                                      int32_t n_pq,
                                      float* __restrict__ F,
                                      float* __restrict__ block_norm)
{
    extern __shared__ float norm_scratch[];

    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t dim = n_pv + 2 * n_pq;

    float abs_mis = 0.0f;
    if (tid < dim) {
        int32_t bus = 0;
        bool q_part = false;

        if (tid < n_pv) {
            bus = pv[tid];
        } else if (tid < n_pv + n_pq) {
            bus = pq[tid - n_pv];
        } else {
            bus = pq[tid - n_pv - n_pq];
            q_part = true;
        }

        const cuFloatComplex vb = v[bus];
        const cuFloatComplex ib = ibus[bus];

        const float p_calc = vb.x * ib.x + vb.y * ib.y;
        const float q_calc = vb.y * ib.x - vb.x * ib.y;
        const float p_mis = p_calc - sbus_re[bus];
        const float q_mis = q_calc - sbus_im[bus];
        const float mis = q_part ? q_mis : p_mis;

        F[tid] = mis;
        abs_mis = fabsf(mis);
    }

    norm_scratch[threadIdx.x] = abs_mis;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            norm_scratch[threadIdx.x] =
                fmaxf(norm_scratch[threadIdx.x], norm_scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_norm[blockIdx.x] = norm_scratch[0];
    }
}

__global__ void reduce_norm_kernel(const float* __restrict__ block_norm,
                                   int32_t n,
                                   float* __restrict__ norm_value)
{
    extern __shared__ float norm_scratch[];

    float local = 0.0f;
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
        local = fmaxf(local, block_norm[i]);
    }

    norm_scratch[threadIdx.x] = local;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            norm_scratch[threadIdx.x] =
                fmaxf(norm_scratch[threadIdx.x], norm_scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        norm_value[0] = norm_scratch[0];
    }
}

}  // namespace

void mismatchDestroy(MismatchWorkspace& ws)
{
    if (ws.ybus_descr != nullptr) {
        cusparseDestroySpMat(ws.ybus_descr);
    }
    if (ws.v_descr != nullptr) {
        cusparseDestroyDnVec(ws.v_descr);
    }
    if (ws.ibus_descr != nullptr) {
        cusparseDestroyDnVec(ws.ibus_descr);
    }
    if (ws.cusparse != nullptr) {
        cusparseDestroy(ws.cusparse);
    }
    if (ws.ybus != nullptr) {
        cudaFree(ws.ybus);
    }
    if (ws.v != nullptr) {
        cudaFree(ws.v);
    }
    if (ws.ibus != nullptr) {
        cudaFree(ws.ibus);
    }
    if (ws.block_norm != nullptr) {
        cudaFree(ws.block_norm);
    }
    if (ws.norm_value != nullptr) {
        cudaFree(ws.norm_value);
    }
    if (ws.spmv_buffer != nullptr) {
        cudaFree(ws.spmv_buffer);
    }

    ws = MismatchWorkspace{};
}

void mismatchAnalyze(MismatchWorkspace& ws,
                     const YbusGraph& ybus,
                     const int32_t* pv,
                     int32_t n_pv,
                     const int32_t* pq,
                     int32_t n_pq,
                     cudaStream_t stream)
{
    if (ybus.n_bus <= 0 || ybus.n_edges <= 0) {
        throw std::invalid_argument("mismatchAnalyze: bad Ybus size");
    }
    if (ybus.row_ptr == nullptr || ybus.col == nullptr ||
        ybus.real == nullptr || ybus.imag == nullptr) {
        throw std::invalid_argument("mismatchAnalyze: Ybus device pointers are null");
    }
    if (n_pv < 0 || n_pq < 0 || n_pv + n_pq <= 0) {
        throw std::invalid_argument("mismatchAnalyze: bad pv/pq sizes");
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
    ws.norm_blocks = (ws.dim + kBlock - 1) / kBlock;
    ws.pv = pv;
    ws.pq = pq;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.ybus),
                          static_cast<std::size_t>(ws.n_edges) * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.v),
                          static_cast<std::size_t>(ws.n_bus) * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.ibus),
                          static_cast<std::size_t>(ws.n_bus) * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.block_norm),
                          static_cast<std::size_t>(ws.norm_blocks) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.norm_value), sizeof(float)));

    const int32_t edge_grid = (ws.n_edges + kBlock - 1) / kBlock;
    pack_complex_kernel<<<edge_grid, kBlock, 0, stream>>>(
        ybus.real, ybus.imag, ws.ybus, ws.n_edges);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUSPARSE_CHECK(cusparseCreate(&ws.cusparse));
    CUSPARSE_CHECK(cusparseSetStream(ws.cusparse, stream));
    CUSPARSE_CHECK(cusparseCreateCsr(
        &ws.ybus_descr,
        ws.n_bus,
        ws.n_bus,
        ws.n_edges,
        const_cast<int32_t*>(ybus.row_ptr),
        const_cast<int32_t*>(ybus.col),
        ws.ybus,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_C_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ws.v_descr, ws.n_bus, ws.v, CUDA_C_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ws.ibus_descr, ws.n_bus, ws.ibus, CUDA_C_32F));

    const cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
    const cuFloatComplex beta = make_cuFloatComplex(0.0f, 0.0f);
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        ws.cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        ws.ybus_descr,
        ws.v_descr,
        &beta,
        ws.ibus_descr,
        CUDA_C_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &ws.spmv_buffer_bytes));

    if (ws.spmv_buffer_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&ws.spmv_buffer, ws.spmv_buffer_bytes));
    }
}

void mismatchUpdateYbus(MismatchWorkspace& ws,
                        const YbusGraph& ybus,
                        cudaStream_t stream)
{
    if (ws.ybus == nullptr) {
        throw std::runtime_error("mismatchUpdateYbus: call mismatchAnalyze first");
    }
    if (ybus.n_bus != ws.n_bus || ybus.n_edges != ws.n_edges ||
        ybus.real == nullptr || ybus.imag == nullptr) {
        throw std::invalid_argument("mismatchUpdateYbus: bad Ybus values");
    }

    const int32_t edge_grid = (ws.n_edges + kBlock - 1) / kBlock;
    pack_complex_kernel<<<edge_grid, kBlock, 0, stream>>>(
        ybus.real, ybus.imag, ws.ybus, ws.n_edges);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

double mismatchCompute(MismatchWorkspace& ws,
                       const float* v_re,
                       const float* v_im,
                       const float* sbus_re,
                       const float* sbus_im,
                       float* F,
                       cudaStream_t stream)
{
    if (ws.cusparse == nullptr || ws.ybus_descr == nullptr) {
        throw std::runtime_error("mismatchCompute: call mismatchAnalyze first");
    }
    if (v_re == nullptr || v_im == nullptr ||
        sbus_re == nullptr || sbus_im == nullptr || F == nullptr) {
        throw std::invalid_argument("mismatchCompute: input device pointer is null");
    }

    const int32_t bus_grid = (ws.n_bus + kBlock - 1) / kBlock;
    pack_complex_kernel<<<bus_grid, kBlock, 0, stream>>>(v_re, v_im, ws.v, ws.n_bus);
    CUDA_CHECK(cudaGetLastError());

    const cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
    const cuFloatComplex beta = make_cuFloatComplex(0.0f, 0.0f);
    CUSPARSE_CHECK(cusparseSetStream(ws.cusparse, stream));
    CUSPARSE_CHECK(cusparseSpMV(
        ws.cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        ws.ybus_descr,
        ws.v_descr,
        &beta,
        ws.ibus_descr,
        CUDA_C_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        ws.spmv_buffer));

    build_mismatch_kernel<<<ws.norm_blocks, kBlock, kBlock * sizeof(float), stream>>>(
        ws.v,
        ws.ibus,
        sbus_re,
        sbus_im,
        ws.pv,
        ws.pq,
        ws.n_pv,
        ws.n_pq,
        F,
        ws.block_norm);
    CUDA_CHECK(cudaGetLastError());

    reduce_norm_kernel<<<1, kBlock, kBlock * sizeof(float), stream>>>(
        ws.block_norm,
        ws.norm_blocks,
        ws.norm_value);
    CUDA_CHECK(cudaGetLastError());

    float host_norm = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(
        &host_norm,
        ws.norm_value,
        sizeof(float),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const double norm = static_cast<double>(host_norm);
    if (!std::isfinite(norm)) {
        throw std::runtime_error("mismatchCompute: mismatch norm is not finite");
    }
    return norm;
}

}  // namespace exp20260420::mismatch
