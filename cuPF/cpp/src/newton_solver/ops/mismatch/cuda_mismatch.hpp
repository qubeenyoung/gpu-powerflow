#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/solver_contexts.hpp"


struct CudaFp64Buffers;
struct CudaMixedBuffers;


// ---------------------------------------------------------------------------
// CudaMismatchOp<Buffers>: CUDA mismatch, FP64 and Mixed 공유.
// CudaMismatchNormOp<Buffers>: per-batch L∞ norm + convergence.
// ---------------------------------------------------------------------------
template <typename Buffers>
struct CudaMismatchOp {
    void run(Buffers& buf, IterationContext& ctx);
};

template <typename Buffers>
struct CudaMismatchNormOp {
    void run(Buffers& buf, IterationContext& ctx);
};

#endif  // CUPF_WITH_CUDA
