#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/solver_contexts.hpp"


struct CudaFp64Storage;
struct CudaFp32Storage;
struct CudaMixedStorage;


// ---------------------------------------------------------------------------
// CudaJacobianOp<T>: edge one-pass CUDA Jacobian fill.
//
// T = double : CUDA FP64 프로파일 (d_J_values = double)
// T = float  : CUDA FP32/Mixed 프로파일 (d_J_values = float)
//
// 각 특수화는 대응하는 Buffers 타입을 직접 받는다.
// ---------------------------------------------------------------------------
template <typename T>
struct CudaJacobianOp;

template <>
struct CudaJacobianOp<double> {
    void run(CudaFp64Storage& buf, IterationContext& ctx);
};

template <>
struct CudaJacobianOp<float> {
    void run(CudaFp32Storage& buf, IterationContext& ctx);
    void run(CudaMixedStorage& buf, IterationContext& ctx);
};

#endif  // CUPF_WITH_CUDA
