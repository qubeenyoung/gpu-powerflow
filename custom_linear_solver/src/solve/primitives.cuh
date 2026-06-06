#pragma once

// Internal — included only by multifrontal.cu (single TU).
//
// Building blocks shared by the solve kernels (solve_fwd / solve_bwd / solve_fwd_small /
// solve_bwd_small). Each helper operates on one front and is __forceinline__.
//
// Per-front layout (same as factor) — F is row-major with leading dimension fsz, nc = pivot
// columns, uc / cb = fsz - nc = contribution-block rows.
//
// Forward solve phases (panel of size nc, fr[] = panel rows in the global y vector):
//   1. fwd_substitute  — warp-parallel L_pp * sh_piv = rhs, lane k finalizes sh_piv[k] and
//                        broadcasts so lanes j > k can update their running partial.
//   2. fwd_cb_update   — for each contribution-block row i, y[fr[nc+i]] -= sum_k L[i,k] sh_piv[k].
//
// Backward solve phases:
//   1. bwd_load_rhs    — gather y[fr[0..nc)] into rhs[] and y[fr[nc..fsz)] into x cache.
//   2. bwd_cb_subtract — rhs[k] -= sum_j U[k, nc+j] * x[j].
//   3. bwd_substitute  — warp-parallel U_pp * x = rhs, lane k (high → low) finalizes x[k].

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

constexpr int MF_MAX_NC = 64;

// Warp-parallel forward substitution L_pp * sh_piv = rhs (nc ≤ 32). One warp drives the loop;
// caller passes lane = threadIdx.x for warp-per-front kernels, or lane = threadIdx.x for block
// kernels and restricts via `if (threadIdx.x < 32)`.
template <typename T>
__device__ __forceinline__ void fwd_substitute(const T* F, int fsz, int nc, const int* fr,
                                               T* y_global, T* sh_piv, int lane)
{
    constexpr unsigned mask = 0xffffffffu;
    T part = T(0), sk = T(0);
    for (int k = 0; k < nc; ++k) {
        if (lane == k) { sk = y_global[fr[k]] + part; sh_piv[k] = sk; y_global[fr[k]] = sk; }
        sk = __shfl_sync(mask, sk, k);
        if (lane > k && lane < nc) part -= F[(long)lane * fsz + k] * sk;
    }
}

// CB row update for forward solve: y[fr[nc + i]] -= sum_k L[i, k] sh_piv[k], for i in [0, cb)
// distributed across `nt` threads starting at offset `t`.
template <typename T>
__device__ __forceinline__ void fwd_cb_update(const T* F, int fsz, int nc, int cb,
                                              const int* fr, T* y_global, const T* sh_piv,
                                              int t, int nt)
{
    for (int i = nc + t; i < fsz; i += nt) {
        T upd = T(0);
        for (int k = 0; k < nc; ++k) upd += F[(long)i * fsz + k] * sh_piv[k];
        atomicAdd(&y_global[fr[i]], -upd);
    }
}

// Backward solve — gather rhs[] and x cache from y, compute CB contribution, then warp-
// parallel substitution. rhs and xsh must point at caller-provided shared (or per-warp slab).
//
// Phase 1 — load. rhs[0..nc) ← y[fr[0..nc)], xsh[0..cb) ← y[fr[nc..fsz)].
template <typename T>
__device__ __forceinline__ void bwd_load_rhs_and_x(const T* y_global, const int* fr,
                                                   int nc, int cb, T* rhs, T* xsh,
                                                   int t, int nt)
{
    for (int k = t; k < nc; k += nt) rhs[k] = y_global[fr[k]];
    for (int j = t; j < cb; j += nt) xsh[j] = y_global[fr[nc + j]];
}

// Phase 2 — rhs[k] -= sum_j U[k, nc + j] * xsh[j], over k distributed across `width` threads.
template <typename T>
__device__ __forceinline__ void bwd_cb_subtract(const T* F, int fsz, int nc, int cb,
                                                const T* xsh, T* rhs, int t, int width)
{
    if (t < nc && t < width) {
        T pk = T(0);
        for (int j = 0; j < cb; ++j) pk += F[(long)t * fsz + (nc + j)] * xsh[j];
        rhs[t] -= pk;
    }
}

// Phase 3 — warp-parallel U_pp * x = rhs, x written to y_global[fr[0..nc)].
template <typename T>
__device__ __forceinline__ void bwd_substitute(const T* F, int fsz, int nc, const int* fr,
                                               T* y_global, const T* rhs, int lane)
{
    constexpr unsigned mask = 0xffffffffu;
    T part = T(0), xk = T(0);
    for (int k = nc - 1; k >= 0; --k) {
        if (lane == k) { xk = (rhs[k] + part) / F[(long)k * fsz + k]; y_global[fr[k]] = xk; }
        xk = __shfl_sync(mask, xk, k);
        if (lane < k) part -= F[(long)lane * fsz + k] * xk;
    }
}

}  // namespace
}  // namespace custom_linear_solver
