#pragma once

#include <cuda_runtime.h>
#include <cudss.h>

#include <cstdint>

namespace exp20260420::linear_solver {

struct CudssLuWorkspace {
    int32_t dim = 0;
    int64_t nnz = 0;

    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t B = nullptr;
    cudssMatrix_t X = nullptr;

    float* rhs = nullptr;
    float* dx = nullptr;
    bool factorized = false;
};

void cudssLuAnalyze(CudssLuWorkspace& ws,
                    int32_t dim,
                    int64_t nnz,
                    const int32_t* row_ptr,
                    const int32_t* col_idx,
                    float* values,
                    float* dx,
                    cudaStream_t stream = nullptr);

void cudssLuFactorize(CudssLuWorkspace& ws, cudaStream_t stream = nullptr);

void cudssLuSolve(CudssLuWorkspace& ws,
                  const float* F,
                  cudaStream_t stream = nullptr);

void cudssLuDestroy(CudssLuWorkspace& ws);

}  // namespace exp20260420::linear_solver
