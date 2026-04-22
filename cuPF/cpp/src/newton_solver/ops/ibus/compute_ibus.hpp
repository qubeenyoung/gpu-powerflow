#pragma once

#include "newton_solver/core/solver_contexts.hpp"

struct CpuFp64Buffers;

void compute_ibus(CpuFp64Buffers& buf);

struct CpuIbusOp {
    void run(CpuFp64Buffers& buf, IterationContext& ctx);
};

#ifdef CUPF_WITH_CUDA

struct CudaFp64Buffers;
struct CudaMixedBuffers;

void launch_compute_ibus(CudaFp64Buffers& buf);
void launch_compute_ibus(CudaMixedBuffers& buf);

template <typename Buffers>
struct CudaIbusOp {
    void run(Buffers& buf, IterationContext& ctx);
};

#endif  // CUPF_WITH_CUDA
