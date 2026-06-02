#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/solver_contexts.hpp"


struct CudaFp64Storage;
struct CudaFp32Storage;
struct CudaMixedStorage;


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

    // Graph-capture split of run(): run_device() is the device-side L∞ reduction only (capturable,
    // goes inside the iteration graph); readback() pulls the per-case norms to host, takes the
    // worst case as the batch norm, checks finiteness, and sets ctx.converged (host-side, done
    // after the graph replay). run() == run_device() + readback().
    void run_device(Buffers& buf);
    void readback(Buffers& buf, IterationContext& ctx);
};

#endif  // CUPF_WITH_CUDA
