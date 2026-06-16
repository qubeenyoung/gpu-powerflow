#pragma once

// SOLVE — per-phase device building blocks.
//
// Internal — included into the factor/solve driver TUs (single TU; CUDA_SEPARABLE_COMPILATION OFF).
//
// Per-front phases (all run inside a __global__ solve kernel):
//
//   Forward (L y = b):
//     fwd_substitute     — warp-parallel panel substitution. nc ≤ 32. One warp drives the loop;
//                          lane k finalizes sh_piv[k] then broadcasts so lanes j > k can update
//                          their running partials.
//     fwd_cb_update      — for each contribution-block row i (i ∈ [nc, fsz)),
//                          y[fr[i]] -= sum_k L[i, k] * sh_piv[k]. atomicAdd because siblings
//                          may scatter into the same parent entries concurrently.
//
//   Backward (U x = y):
//     bwd_load_rhs_and_x — gather rhs[0..nc) ← y[fr[0..nc)] and x cache xsh[0..cb) ← y[fr[nc..fsz)].
//     bwd_cb_subtract    — rhs[k] -= sum_j U[k, nc+j] * xsh[j], over k in [0, nc).
//     bwd_substitute     — warp-parallel U_pp * x = rhs (one warp, high → low).
//                          Writes x[k] back to y[fr[k]] for k ∈ [0, nc).
//
// Per-front layout (same as factor): F is row-major with leading dimension fsz, nc = pivot
// columns, cb = fsz - nc = contribution-block rows. fr[0..fsz) is the panel-row list (front
// rows → global y indices) computed by analyze.

#include <cuda_runtime.h>

#include "internal/types.hpp"

namespace custom_linear_solver {
namespace {

__device__ __forceinline__ float solve_div(float a, float b) { return __fdividef(a, b); }
__device__ __forceinline__ double solve_div(double a, double b) { return a / b; }

// =======================================================================================
//  FORWARD SOLVE
// =======================================================================================

// Warp-parallel forward substitution L_pp * sh_piv = rhs (nc ≤ 32). Driven by one warp; for
// warp-per-front kernels `lane = threadIdx.x & 31`, for block kernels caller restricts to
// `if (threadIdx.x < 32)` and passes lane = threadIdx.x.
// SG = sub-group lane count (8 / 16 / 32). For the warp-per-front kernels SG=32 (full warp);
// the sub-group-packed tiny kernels pass SG<32, `lane` = lane within the sub-group, and `mask`
// = that sub-group's active-lane mask. The broadcast shuffle uses width=SG so each sub-group
// broadcasts row-k of its own front. SG=32 / mask=0xffffffff is the classic full-warp form
// (what every block / spine caller uses).
template <typename T, int SG = 32>
__device__ __forceinline__ void fwd_substitute(const T* F, int fsz, int nc, const int* fr,
                                               T* y_global, T* sh_piv, int lane,
                                               unsigned mask = 0xffffffffu)
{
    T part = T(0), sk = T(0);
    for (int k = 0; k < nc; ++k) {
        if (lane == k) { sk = y_global[fr[k]] + part; sh_piv[k] = sk; y_global[fr[k]] = sk; }
        sk = __shfl_sync(mask, sk, k, SG);
        if (lane > k && lane < nc) part -= F[(long)lane * fsz + k] * sk;
    }
}

template <typename T, int NC, int SG = 32>
__device__ __forceinline__ void fwd_substitute_fixed(const T* F, int fsz, const int* fr,
                                                     T* y_global, T* sh_piv, int lane,
                                                     unsigned mask = 0xffffffffu)
{
    T part = T(0), sk = T(0);
#pragma unroll
    for (int k = 0; k < NC; ++k) {
        if (lane == k) { sk = y_global[fr[k]] + part; sh_piv[k] = sk; y_global[fr[k]] = sk; }
        sk = __shfl_sync(mask, sk, k, SG);
        if (lane > k && lane < NC) part -= F[(long)lane * fsz + k] * sk;
    }
}

// ── Shared-staged (compact leading-dim) variants ──────────────────────────────────────
// P3: the regular block-per-front kernels stage the touched panel into shared with a coalesced
// load, then run the phases reading the compact panel (leading dim `ld`, not the front's `fsz`).
// ncu showed the direct global access at ~7-13% load-coalescing (bytes/sector); staging converts
// the strided gathers into one coalesced load + bank-friendly shared reads. The math is identical
// to the non-staged helpers above; only the operand source (Lsh/Ush, stride ld) changes.

// Forward panel substitution reading the compact L panel Lsh (row-major, leading dim ld = nc).
template <typename T>
__device__ __forceinline__ void fwd_substitute_sh(const T* Lsh, int ld, int nc, const int* fr,
                                                  T* y_global, T* sh_piv, int lane)
{
    T part = T(0), sk = T(0);
    for (int k = 0; k < nc; ++k) {
        if (lane == k) { sk = y_global[fr[k]] + part; sh_piv[k] = sk; y_global[fr[k]] = sk; }
        sk = __shfl_sync(0xffffffffu, sk, k);
        if (lane > k && lane < nc) part -= Lsh[(long)lane * ld + k] * sk;
    }
}

// Forward CB row update reading the compact L panel Lsh (leading dim ld = nc), rows [nc, fsz).
template <typename T>
__device__ __forceinline__ void fwd_cb_update_sh(const T* Lsh, int ld, int fsz, int nc, int cb,
                                                 const int* fr, T* y_global, const T* sh_piv,
                                                 int t, int nt)
{
    for (int i = nc + t; i < fsz; i += nt) {
        T upd = T(0);
        for (int k = 0; k < nc; ++k) upd += Lsh[(long)i * ld + k] * sh_piv[k];
        atomicAdd(&y_global[fr[i]], -upd);
    }
}

// CB row update: y[fr[nc + i]] -= sum_k L[i, k] sh_piv[k], for i in [0, cb) distributed
// across `nt` threads starting at offset `t`. atomicAdd because sibling fronts scatter
// into the same parent rows.
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

template <typename T, int NC>
__device__ __forceinline__ void fwd_cb_update_fixed(const T* F, int fsz, int cb,
                                                    const int* fr, T* y_global, const T* sh_piv,
                                                    int t, int nt)
{
    for (int i = NC + t; i < fsz; i += nt) {
        T upd = T(0);
#pragma unroll
        for (int k = 0; k < NC; ++k) upd += F[(long)i * fsz + k] * sh_piv[k];
        atomicAdd(&y_global[fr[i]], -upd);
    }
}

template <typename T>
__device__ __forceinline__ void fwd_cb_update_plain(const T* F, int fsz, int nc, int cb,
                                                    const int* fr, T* y_global, const T* sh_piv,
                                                    int t, int nt)
{
    for (int i = nc + t; i < fsz; i += nt) {
        T upd = T(0);
        for (int k = 0; k < nc; ++k) upd += F[(long)i * fsz + k] * sh_piv[k];
        y_global[fr[i]] -= upd;
    }
}

// =======================================================================================
//  BACKWARD SOLVE
// =======================================================================================

// Phase 1 — gather rhs[0..nc) ← y[fr[0..nc)] and x cache xsh[0..cb) ← y[fr[nc..fsz)].
template <typename T>
__device__ __forceinline__ void bwd_load_rhs_and_x(const T* y_global, const int* fr,
                                                   int nc, int cb, T* rhs, T* xsh,
                                                   int t, int nt)
{
    for (int k = t; k < nc; k += nt) rhs[k] = y_global[fr[k]];
    for (int j = t; j < cb; j += nt) xsh[j] = y_global[fr[nc + j]];
}

template <typename T, int NC>
__device__ __forceinline__ void bwd_load_rhs_and_x_fixed(const T* y_global, const int* fr,
                                                         int cb, T* rhs, T* xsh,
                                                         int t, int nt)
{
    if (t < NC) rhs[t] = y_global[fr[t]];
    for (int j = t; j < cb; j += nt) xsh[j] = y_global[fr[NC + j]];
}

// Phase 2 — rhs[k] -= sum_j U[k, nc + j] * xsh[j], over k in [0, nc).
//
// Two paths sharing the same math:
//   nc-parallel (original): each lane handles one row k, runs cb FMAs serial. Latency cb
//                           cycles; idles (width - nc) threads. Best when nc is close to
//                           a warp.
//   cb-parallel (new):      lanes stride over j (xsh dimension); per-row dot is a warp
//                           reduce. Uses 32 lanes regardless of nc; latency
//                           nc * (ceil(cb/32) + log2(32)). Best when cb >> nc.
//
// The reduce path only wins for very contribution-heavy small fronts. Empirically, `cb > 8*nc`
// was the best gate after switching tiny-tier packing to max-nc-based subgroups; lower gates
// overuse shuffle reduction and regress the batch solve.
// SG = sub-group lane count (8/16/32) for the warp/sub-group reduce path; `width` = the
// caller's active lane count (SG for warp/sub-group kernels, blockDim for the regular tier);
// `mask` = the sub-group's active-lane mask (0xffffffff for a full warp). SG=32 / width=32 /
// mask=0xffffffff is the classic full-warp form.
template <typename T, int SG = 32>
__device__ __forceinline__ void bwd_cb_subtract(const T* F, int fsz, int nc, int cb,
                                                const T* xsh, T* rhs, int t, int width,
                                                unsigned mask = 0xffffffffu)
{
    if (width >= 128 && cb >= 64) {
        const int lane = t & 31;
        const int warp = t >> 5;
        const int warps = width >> 5;
        for (int k = warp; k < nc; k += warps) {
            T pk = T(0);
            for (int j = lane; j < cb; j += 32)
                pk += F[(long)k * fsz + (nc + j)] * xsh[j];
            for (int off = 16; off > 0; off >>= 1)
                pk += __shfl_down_sync(0xffffffffu, pk, off);
            if (lane == 0) rhs[k] -= pk;
        }
        return;
    }

    // Warp / sub-group reduce path (width ≤ 32): lanes stride over j (xsh dim) and each row's
    // dot is a segment reduce, keeping all SG lanes busy where nc-parallel would idle cb-nc of
    // them. The regular tier (width > 32) has parallelism elsewhere and the FP64-latency-bound
    // nc-parallel path beats the reduce-bound one — empirical.
    if (width <= 32 && cb > 8 * nc) {
        for (int k = 0; k < nc; ++k) {
            T pk = T(0);
            for (int j = t; j < cb; j += SG)
                pk += F[(long)k * fsz + (nc + j)] * xsh[j];
            for (int off = SG / 2; off > 0; off >>= 1)
                pk += __shfl_down_sync(mask, pk, off, SG);
            if (t == 0) rhs[k] -= pk;
        }
    } else {
        if (t < nc && t < width) {
            T pk = T(0);
            for (int j = 0; j < cb; ++j) pk += F[(long)t * fsz + (nc + j)] * xsh[j];
            rhs[t] -= pk;
        }
    }
}

template <typename T, int NC>
__device__ __forceinline__ void bwd_cb_subtract_fixed(const T* F, int fsz, int cb,
                                                      const T* xsh, T* rhs, int t, int width)
{
    constexpr int reduce_cb_gate = (NC == 16 || NC == 14) ? 8 : 40;
    if (width >= 128 && cb >= reduce_cb_gate) {
        const int lane = t & 31;
        const int warp = t >> 5;
        const int warps = width >> 5;
        for (int k = warp; k < NC; k += warps) {
            T pk = T(0);
            for (int j = lane; j < cb; j += 32)
                pk += F[(long)k * fsz + (NC + j)] * xsh[j];
            for (int off = 16; off > 0; off >>= 1)
                pk += __shfl_down_sync(0xffffffffu, pk, off);
            if (lane == 0) rhs[k] -= pk;
        }
        return;
    }
    if (t < NC && t < width) {
        T pk = T(0);
        for (int j = 0; j < cb; ++j) pk += F[(long)t * fsz + (NC + j)] * xsh[j];
        rhs[t] -= pk;
    }
}

// Phase 3 — warp-parallel U_pp * x = rhs (high → low). x written to y_global[fr[0..nc)].
// SG / lane / mask as in fwd_substitute. SG=32 / mask=0xffffffff is the classic full-warp form.
template <typename T, int SG = 32>
__device__ __forceinline__ void bwd_substitute(const T* F, int fsz, int nc, const int* fr,
                                               T* y_global, const T* rhs, int lane,
                                               unsigned mask = 0xffffffffu)
{
    T part = T(0), xk = T(0);
    for (int k = nc - 1; k >= 0; --k) {
        if (lane == k) { xk = solve_div(rhs[k] + part, F[(long)k * fsz + k]); y_global[fr[k]] = xk; }
        xk = __shfl_sync(mask, xk, k, SG);
        if (lane < k) part -= F[(long)lane * fsz + k] * xk;
    }
}

template <typename T, int NC, int SG = 32>
__device__ __forceinline__ void bwd_substitute_fixed(const T* F, int fsz, const int* fr,
                                                     T* y_global, const T* rhs, int lane,
                                                     unsigned mask = 0xffffffffu)
{
    T part = T(0), xk = T(0);
#pragma unroll
    for (int k = NC - 1; k >= 0; --k) {
        if (lane == k) { xk = solve_div(rhs[k] + part, F[(long)k * fsz + k]); y_global[fr[k]] = xk; }
        xk = __shfl_sync(mask, xk, k, SG);
        if (lane < k) part -= F[(long)lane * fsz + k] * xk;
    }
}

}  // namespace
}  // namespace custom_linear_solver
