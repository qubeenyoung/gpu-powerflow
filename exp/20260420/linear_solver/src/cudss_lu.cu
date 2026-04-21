#include "cudss_lu.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace exp20260420::linear_solver {

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err = (call);                                                          \
        if (err != cudaSuccess) {                                                          \
            throw std::runtime_error(                                                       \
                std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - " + cudaGetErrorString(err));                                         \
        }                                                                                   \
    } while (0)

#define CUDSS_CHECK(call)                                                                   \
    do {                                                                                    \
        cudssStatus_t status = (call);                                                      \
        if (status != CUDSS_STATUS_SUCCESS) {                                               \
            throw std::runtime_error(                                                       \
                std::string("cuDSS error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - status=" + std::to_string(static_cast<int>(status)));                  \
        }                                                                                   \
    } while (0)

namespace {

constexpr int32_t kBlock = 256;

__global__ void negate_rhs_kernel(const double* __restrict__ F,
                                  float* __restrict__ rhs,
                                  int32_t n)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        rhs[tid] = -static_cast<float>(F[tid]);
    }
}

}  // namespace

void cudssLuDestroy(CudssLuWorkspace& ws)
{
    if (ws.A != nullptr) {
        cudssMatrixDestroy(ws.A);
    }
    if (ws.B != nullptr) {
        cudssMatrixDestroy(ws.B);
    }
    if (ws.X != nullptr) {
        cudssMatrixDestroy(ws.X);
    }
    if (ws.data != nullptr) {
        cudssDataDestroy(ws.handle, ws.data);
    }
    if (ws.config != nullptr) {
        cudssConfigDestroy(ws.config);
    }
    if (ws.handle != nullptr) {
        cudssDestroy(ws.handle);
    }
    if (ws.rhs != nullptr) {
        cudaFree(ws.rhs);
    }

    ws = CudssLuWorkspace{};
}

void cudssLuAnalyze(CudssLuWorkspace& ws,
                    int32_t dim,
                    int32_t batch_size,
                    int64_t nnz,
                    const int32_t* row_ptr,
                    const int32_t* col_idx,
                    float* values,
                    float* dx,
                    cudaStream_t stream)
{
    if (dim <= 0 || batch_size <= 0 || nnz <= 0) {
        throw std::invalid_argument("cudssLuAnalyze: bad matrix size");
    }
    if (row_ptr == nullptr || col_idx == nullptr || values == nullptr || dx == nullptr) {
        throw std::invalid_argument("cudssLuAnalyze: device pointer is null");
    }

    cudssLuDestroy(ws);

    ws.dim = dim;
    ws.batch_size = batch_size;
    ws.nnz = nnz;
    ws.dx = dx;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.rhs),
                          static_cast<std::size_t>(dim) *
                              static_cast<std::size_t>(batch_size) * sizeof(float)));

    CUDSS_CHECK(cudssCreate(&ws.handle));
    CUDSS_CHECK(cudssSetStream(ws.handle, stream));
    CUDSS_CHECK(cudssConfigCreate(&ws.config));
    CUDSS_CHECK(cudssConfigSet(
        ws.config,
        CUDSS_CONFIG_UBATCH_SIZE,
        &ws.batch_size,
        sizeof(ws.batch_size)));
    int ubatch_index = -1;
    CUDSS_CHECK(cudssConfigSet(
        ws.config,
        CUDSS_CONFIG_UBATCH_INDEX,
        &ubatch_index,
        sizeof(ubatch_index)));
    CUDSS_CHECK(cudssDataCreate(ws.handle, &ws.data));

    CUDSS_CHECK(cudssMatrixCreateCsr(
        &ws.A,
        dim,
        dim,
        nnz,
        const_cast<int32_t*>(row_ptr),
        nullptr,
        const_cast<int32_t*>(col_idx),
        values,
        CUDA_R_32I,
        CUDA_R_32F,
        CUDSS_MTYPE_GENERAL,
        CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &ws.B,
        dim,
        1,
        dim,
        ws.rhs,
        CUDA_R_32F,
        CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &ws.X,
        dim,
        1,
        dim,
        dx,
        CUDA_R_32F,
        CUDSS_LAYOUT_COL_MAJOR));

    CUDSS_CHECK(cudssExecute(
        ws.handle,
        CUDSS_PHASE_ANALYSIS,
        ws.config,
        ws.data,
        ws.A,
        ws.X,
        ws.B));
}

void cudssLuFactorize(CudssLuWorkspace& ws, cudaStream_t stream)
{
    if (ws.handle == nullptr || ws.A == nullptr) {
        throw std::runtime_error("cudssLuFactorize: call cudssLuAnalyze first");
    }

    CUDSS_CHECK(cudssSetStream(ws.handle, stream));
    const int phase = ws.factorized
        ? CUDSS_PHASE_REFACTORIZATION
        : CUDSS_PHASE_FACTORIZATION;
    CUDSS_CHECK(cudssExecute(
        ws.handle,
        phase,
        ws.config,
        ws.data,
        ws.A,
        ws.X,
        ws.B));
    ws.factorized = true;
}

void cudssLuSolve(CudssLuWorkspace& ws,
                  const double* F,
                  cudaStream_t stream)
{
    if (ws.handle == nullptr || ws.B == nullptr || ws.X == nullptr) {
        throw std::runtime_error("cudssLuSolve: call cudssLuAnalyze first");
    }
    if (!ws.factorized) {
        throw std::runtime_error("cudssLuSolve: call cudssLuFactorize first");
    }
    if (F == nullptr) {
        throw std::invalid_argument("cudssLuSolve: F is null");
    }

    const int32_t total = ws.dim * ws.batch_size;
    const int32_t grid = (total + kBlock - 1) / kBlock;
    negate_rhs_kernel<<<grid, kBlock, 0, stream>>>(F, ws.rhs, total);
    CUDA_CHECK(cudaGetLastError());

    CUDSS_CHECK(cudssSetStream(ws.handle, stream));
    CUDSS_CHECK(cudssExecute(
        ws.handle,
        CUDSS_PHASE_SOLVE,
        ws.config,
        ws.data,
        ws.A,
        ws.X,
        ws.B));
}

}  // namespace exp20260420::linear_solver
