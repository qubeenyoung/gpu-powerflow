#pragma once

#include "newton_solver/core/solver_contexts.hpp"

struct CpuFp64Storage;

void compute_ibus(CpuFp64Storage& buf);

struct CpuIbusOp {
    void run(CpuFp64Storage& buf, IterationContext& ctx);
};

#ifdef CUPF_WITH_CUDA

struct CudaFp64Storage;
struct CudaFp32Storage;
struct CudaMixedStorage;

void launch_compute_ibus(CudaFp64Storage& buf);
void launch_compute_ibus(CudaFp32Storage& buf);
void launch_compute_ibus(CudaMixedStorage& buf);

template <typename Buffers>
struct CudaIbusOp {
    void run(Buffers& buf, IterationContext& ctx);
};

#endif  // CUPF_WITH_CUDA
