#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/solver_contexts.hpp"


struct CudaFp64Buffers;
struct CudaMixedBuffers;


// ---------------------------------------------------------------------------
// CudaVoltageUpdateOp<T>: CUDA voltage update.
//
// T = double : CUDA FP64 profile (dx = double)
// T = float  : CUDA Mixed profile (dx = float)
//
// Va/Vm and V_re/V_im remain double in both instantiations.
// ---------------------------------------------------------------------------
template <typename T>
struct CudaVoltageUpdateOp;

template <>
struct CudaVoltageUpdateOp<double> {
    void run(CudaFp64Buffers& buf, IterationContext& ctx);
};

template <>
struct CudaVoltageUpdateOp<float> {
    void run(CudaMixedBuffers& buf, IterationContext& ctx);
};

#endif  // CUPF_WITH_CUDA
