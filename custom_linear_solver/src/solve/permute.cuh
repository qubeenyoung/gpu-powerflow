#pragma once

// SOLVE — I/O permutation kernels.
//
// Internal — included into the factor/solve driver TUs (single TU; CUDA_SEPARABLE_COMPILATION OFF).
//
// These run at the entry and exit of the solve phase to convert between the user's RHS /
// solution layout (original column order) and the working-vector layout used by the level
// kernels (METIS nested-dissection permuted order):
//
//   gather_rhs   — at solve entry: y[k] = rhs[perm[k]]    (orig → ND order)
//   scatter_sol  — at solve exit:  sol[perm[k]] = y[k]    (ND order → orig)
//
// Templates over (RT, YT) and (YT, ST) so the I/O precision can differ from the working-vector
// precision (e.g. FP64 RHS / FP32 working vector for mixed-precision solve).
//
// Analogous to factorize/scatter.cuh — both are input-setup / output-finalize kernels distinct
// from the per-front phase pipeline.

#include <cuda_runtime.h>

namespace custom_linear_solver {
namespace {

// Gather the permuted RHS into the working vector y (per batch): y[k] = rhs[perm[k]].
//   RT = RHS element type, YT = solve working-vector type.
template <typename RT, typename YT>
__global__ void gather_rhs(int n, const RT* __restrict__ rhsB, const int* __restrict__ perm,
                           YT* __restrict__ yB)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const long b = blockIdx.y;
    yB[b * n + k] = static_cast<YT>(rhsB[b * (long)n + perm[k]]);
}

template <typename RT, typename YT>
__global__ void gather_rhs_batched(int n, int B, const RT* __restrict__ rhsB,
                                   const int* __restrict__ perm, YT* __restrict__ yB)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= n || b >= B) return;
    yB[(long)b * n + k] = static_cast<YT>(rhsB[(long)b * n + perm[k]]);
}

// Scatter the working vector y back to the solution in original order: sol[perm[k]] = y[k].
//   YT = working-vector type, ST = solution element type.
template <typename YT, typename ST>
__global__ void scatter_sol(int n, const YT* __restrict__ yB, const int* __restrict__ perm,
                            ST* __restrict__ solB)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const long b = blockIdx.y;
    solB[b * (long)n + perm[k]] = static_cast<ST>(yB[b * n + k]);
}

template <typename YT, typename ST>
__global__ void scatter_sol_batched(int n, int B, const YT* __restrict__ yB,
                                    const int* __restrict__ perm, ST* __restrict__ solB)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= n || b >= B) return;
    solB[(long)b * n + perm[k]] = static_cast<ST>(yB[(long)b * n + k]);
}

template <typename YT, typename ST>
__global__ void scatter_sol_inverse(int n, const YT* __restrict__ yB,
                                    const int* __restrict__ iperm, ST* __restrict__ solB)
{
    const int orig = blockIdx.x * blockDim.x + threadIdx.x;
    if (orig >= n) return;
    solB[orig] = static_cast<ST>(yB[iperm[orig]]);
}

template <typename YT, typename ST>
__global__ void scatter_sol_inverse_batched(int n, int B, const YT* __restrict__ yB,
                                            const int* __restrict__ iperm, ST* __restrict__ solB)
{
    const int orig = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (orig >= n || b >= B) return;
    solB[(long)b * n + orig] = static_cast<ST>(yB[(long)b * n + iperm[orig]]);
}

}  // namespace
}  // namespace custom_linear_solver
