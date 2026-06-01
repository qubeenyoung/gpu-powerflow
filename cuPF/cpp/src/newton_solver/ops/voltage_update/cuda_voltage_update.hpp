#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/solver_contexts.hpp"


struct CudaFp64Storage;
struct CudaFp32Storage;
struct CudaMixedStorage;


// ---------------------------------------------------------------------------
// CudaVoltageUpdateOp<T>: CUDA voltage update.
//
// T = double : CUDA FP64 profile (state/dx = double)
// T = float  : CUDA FP32 or Mixed profile (dx = float)
//
// CUDA FP32 keeps state in float; CUDA Mixed keeps state in double.
// ---------------------------------------------------------------------------
template <typename T>
struct CudaVoltageUpdateOp;

template <>
struct CudaVoltageUpdateOp<double> {
    void run(CudaFp64Storage& buf, IterationContext& ctx);
};

template <>
struct CudaVoltageUpdateOp<float> {
    void run(CudaFp32Storage& buf, IterationContext& ctx);
    void run(CudaMixedStorage& buf, IterationContext& ctx);
};

#endif  // CUPF_WITH_CUDA
