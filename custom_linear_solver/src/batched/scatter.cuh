#pragma once

// Internal — included only by batched/multifrontal_batched.cu (see lu_device.cuh for why the
// batched kernels live in headers folded into a single TU).

#include <cuda_runtime.h>

namespace custom_linear_solver::batched {
namespace {

// ---- batched numeric scatter: front_b[a_pos[q]] += values_b[o2c[q]] ----------------
// FT = front element type (double for FP64/Mixed/TC master, float for FP32). VT = input CSR
// value type (double for the FP64-input path, float for the cuPF Mixed path). The value is
// cast to FT on scatter, so any (VT -> FT) combination is valid.
template <typename FT, typename VT>
__global__ void scatter_batched(int nnz_a, long front_total, const int* __restrict__ o2c,
                                const int* __restrict__ a_pos, const VT* __restrict__ valuesB,
                                FT* __restrict__ frontB)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnz_a) return;
    const int pos = a_pos[q];
    if (pos < 0) return;
    const long b = blockIdx.y;
    atomicAdd(&frontB[b * front_total + pos], static_cast<FT>(valuesB[b * (long)nnz_a + o2c[q]]));
}

}  // namespace
}  // namespace custom_linear_solver::batched
