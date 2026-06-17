#pragma once

// FACTORIZE — shared substrate used by >=2 tiers (factorize/{small,mid,big,large}.cuh).
//
// Internal — included into the factor/solve driver TUs (single TU; CUDA_SEPARABLE_COMPILATION OFF).
//
// Dense per-front phase primitives (precision-agnostic): panel LU (lu_mid_front / lu_panel_factor),
// U-panel solve, scalar trailing, extend-add, writeback. Each tier kernel sequences these itself.
// Plus the TF32 mma macros / Ozaki helpers used by the small (blocked), big (panel), and large
// (thin-K) Tensor-Core paths, and the device-property queries (factor_num_sms / factor_warp_fill)
// used by the launch gates.

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <mma.h>
#if __CUDA_ARCH__ >= 800
#include <cuda_pipeline.h>
#endif

#include "internal/plan/front_range_caps.hpp"
#include "internal/runtime/state.hpp"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

#ifdef CLS_TF32_OZAKI_TC2
struct Tf32Pair {
    unsigned h;
    unsigned t;
};

__device__ __forceinline__ unsigned tf32_rna_bits(float x)
{
    unsigned y;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ Tf32Pair tf32_ozaki_pair(float x)
{
    const unsigned h = tf32_rna_bits(x);
    const unsigned t = tf32_rna_bits(x - __uint_as_float(h));
    return {h, t};
}
#endif

#define CLS_MMA_TF32_M16N8K8(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1) \
    asm volatile( \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 " \
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n" \
        : "+f"(C0), "+f"(C1), "+f"(C2), "+f"(C3) \
        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1))

#ifdef CLS_TF32_OZAKI_TC2
// Ozaki 2-component (head/tail) TF32 product. The first three MMA passes accumulate the
// head·head + tail·head + head·tail terms; the second-order tail·tail term is added only in the
// full variant (one extra MMA pass for a little more accuracy). CLS_TF32_OZAKI_TC2_FIRST_ORDER
// drops it. (Defined as two whole macros because a #ifdef cannot live inside a macro body.)
#ifdef CLS_TF32_OZAKI_TC2_FIRST_ORDER
#define CLS_MMA_TF32_OZAKI2(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1) \
    do { \
        CLS_MMA_TF32_M16N8K8(C0, C1, C2, C3, (A0).h, (A1).h, (A2).h, (A3).h, (B0).h, (B1).h); \
        CLS_MMA_TF32_M16N8K8(C0, C1, C2, C3, (A0).t, (A1).t, (A2).t, (A3).t, (B0).h, (B1).h); \
        CLS_MMA_TF32_M16N8K8(C0, C1, C2, C3, (A0).h, (A1).h, (A2).h, (A3).h, (B0).t, (B1).t); \
    } while (0)
#else
#define CLS_MMA_TF32_OZAKI2(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1) \
    do { \
        CLS_MMA_TF32_M16N8K8(C0, C1, C2, C3, (A0).h, (A1).h, (A2).h, (A3).h, (B0).h, (B1).h); \
        CLS_MMA_TF32_M16N8K8(C0, C1, C2, C3, (A0).t, (A1).t, (A2).t, (A3).t, (B0).h, (B1).h); \
        CLS_MMA_TF32_M16N8K8(C0, C1, C2, C3, (A0).h, (A1).h, (A2).h, (A3).h, (B0).t, (B1).t); \
        CLS_MMA_TF32_M16N8K8(C0, C1, C2, C3, (A0).t, (A1).t, (A2).t, (A3).t, (B0).t, (B1).t); \
    } while (0)
#endif
#endif

inline constexpr int kFactorDoExtend = 1;

static int factor_num_sms()
{
    static const int v = [] {
        int dev = 0; cudaGetDevice(&dev);
        int sm = 1;
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
        return sm;
    }();
    return v;
}

// ---------------------------------------------------------------------------------------
// Outer dispatcher: iterate etree levels.
//
// Single-stream path:
//   for L in 0..num_plevels:
//       issue_factor_level_range(plan, st, stream, plptr[L], plptr[L+1])
//
// Multi-stream path (fork / wait / join):
//   - st.subtree_streams[0..K-1] hold K extra streams (one per subtree, K ≤ 8).
//   - plan.h_subtree_level_off / _cnt indexes plcols[] so each subtree gets its own
//     (level → range) slice below the spine.
//
//     [main]  ----- fork_event ----┐
//                                  ├──> subtree 0 stream:  L0 ranges → L1 ranges → ...  ── join_events[0] ─┐
//                                  ├──> subtree 1 stream:  L0 ranges → L1 ranges → ...  ── join_events[1] ─┤
//                                  ├──> ...                                                                  │
//     [main]                       └──> subtree K-1 stream: ...                          ── join_events[K-1] ┤
//     [main]  --wait(join_events[0..K-1])--->  spine levels (L_spine_start..num_plevels)
//
//   The spine (`plan.spine_start_level..num_plevels`) holds the levels above the subtree
//   roots and runs on the main stream after all subtree streams have signalled completion.
//   The use_multistream gate also handles the "no spine" case (spine_start_level < 0) by
//   not enqueueing any spine levels on the main stream.
// GPU warp-slot count (SMs × warps/SM). The tier router is deterministic (front size → kernel), so
// this no longer gates split-vs-merge; it is used only by factor_small_sg to pick the small-tier
// sub-group size (pack 8/16/32 fronts per warp only while the packed grid still fills the device).
// A pure hardware quantity (device attrs), so it generalizes across matrices and GPUs.
static long factor_warp_fill()
{
    static const long v = [] {
        int dev = 0; cudaGetDevice(&dev);
        int sm = 1, tpm = 1536;
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        return (long)sm * (tpm / 32);
    }();
    return v;
}

template <typename T>
__device__ __forceinline__ T pivot_abs(T x)
{
    return x < T(0) ? -x : x;
}

template <typename T>
__device__ __forceinline__ T guarded_pivot(T piv, bool static_pivoting,
                                           double pivot_threshold, double pivot_shift,
                                           int* sing, bool leader)
{
    if (piv == T(0)) {
        if (leader) *sing = 1;
        if (!static_pivoting) return T(1);
    }
    if (static_pivoting && pivot_abs(piv) <= static_cast<T>(pivot_threshold)) {
        if (leader) *sing = 1;
        const T mag = static_cast<T>(pivot_shift);
        return (piv < T(0)) ? -mag : mag;
    }
    return piv;
}

template <typename T>
__device__ __forceinline__ void lu_mid_front(T* F, int fsz, int nc, int t, int nt,
                                               int* sing, bool static_pivoting,
                                               double pivot_threshold, double pivot_shift)
{
    // For each pivot column k:
    //   1. Read piv = F[k, k]; flag singularity.
    //   2. Divide column k below the pivot by piv.
    //   3. Rank-1 update the entire (fsz−k−1)² block below-right of the pivot.
    //   4. Block barrier between k and k+1.
    for (int k = 0; k < nc; ++k) {
        const long diag = (long)k * fsz + k;
        T piv = guarded_pivot(F[diag], static_pivoting, pivot_threshold, pivot_shift, sing, t == 0);
        if (t == 0) F[diag] = piv;
        for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
        __syncthreads();
        const int m = fsz - k - 1;
        for (int e = t; e < m * m; e += nt) {
            const int ii = k + 1 + e / m, jj = k + 1 + e % m;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        __syncthreads();
    }
}

template <typename T>
__device__ __forceinline__ void lu_panel_factor(T* F, int fsz, int nc, int t, int nt,
                                                int* sing, bool static_pivoting,
                                                double pivot_threshold, double pivot_shift)
{
    // Row-fused panel LU pays 1 block barrier/pivot (vs the two-phase form's 2). Barrier cost
    // dominates the latency-bound mid kernel (exp_260612), but the row-fused form serializes the
    // per-thread inner panel loop, which costs more on TALL fronts (more rows/thread). So take the
    // fewer-barrier path for nc<=12 always, and extend to the panel-width cap (nc<=16, n>=16K) only
    // when the front is short enough that the serial inner loop is cheap (measured: helps ACTIVSg25k
    // ~3%, avoids the SyntheticUSA regression that a blanket nc<=16 cutoff caused).
    constexpr int ROWFUSED_NC_MAX = 12;
    constexpr int ROWFUSED_WIDE_NC = 16;     // up to the panel-width cap
    constexpr int ROWFUSED_WIDE_FSZ = 96;    // only on short fronts
    const bool row_fused = (nc > 0) &&
        (nc <= ROWFUSED_NC_MAX || (nc <= ROWFUSED_WIDE_NC && fsz <= ROWFUSED_WIDE_FSZ));

    if (row_fused) {
        // Row-fused form. For each k, every thread that owns one of the rows below the pivot
        // (i.e. i = k + 1 + t (mod nt)) does the divide and the in-row panel update in a
        // single pass — `lik` is kept in a register and applied across the panel columns
        // jj ∈ (k, nc) without touching shared again. One __syncthreads per k.
        for (int k = 0; k < nc; ++k) {
            const long diag = (long)k * fsz + k;
            T piv = guarded_pivot(F[diag], static_pivoting, pivot_threshold, pivot_shift, sing, t == 0);
            if (t == 0) F[diag] = piv;
            const T inv_piv = T(1) / piv;
            for (int i = k + 1 + t; i < fsz; i += nt) {
                const T lik = F[(long)i * fsz + k] * inv_piv;
                F[(long)i * fsz + k] = lik;
                for (int jj = k + 1; jj < nc; ++jj) {
                    F[(long)i * fsz + jj] -= lik * F[(long)k * fsz + jj];
                }
            }
            __syncthreads();
        }
        return;
    }

    // Two-phase form for nc > ROWFUSED_NC_MAX. Pays 2·nc syncs (divide / rank-1 split) but
    // avoids the per-thread serial inner panel loop of the row-fused form, which is the
    // hotter cost for wide panels.
    for (int k = 0; k < nc; ++k) {
        const long diag = (long)k * fsz + k;
        T piv = guarded_pivot(F[diag], static_pivoting, pivot_threshold, pivot_shift, sing, t == 0);
        if (t == 0) F[diag] = piv;
        const T inv_piv = T(1) / piv;
        // (a) Divide column k below the pivot.
        for (int i = k + 1 + t; i < fsz; i += nt) {
            F[(long)i * fsz + k] = F[(long)i * fsz + k] * inv_piv;
        }
        __syncthreads();
        // (b) Rank-1 update the (fsz−k−1) × (nc−k−1) panel below-right of the pivot.
        const int pc = nc - 1 - k;
        for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
            const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        if (pc > 0) __syncthreads();
    }
}

// =======================================================================================
//  PHASE 2  —  U-PANEL TRIANGULAR SOLVE       (recover U from L_pp · U = U_raw)
// =======================================================================================
//
// Operates on rows [0, nc), cols [nc, fsz). Sequential in k: row k depends on the rows
// already solved [0, k), so the outer loop carries a per-row __syncthreads. The inner loop
// is parallel over the uc columns of the U panel.
//   for k in 1..nc:
//     for each j-slot of U row k in parallel:
//         U[k, j] := U[k, j] − Σ_{i<k} L_pp[k, i] · U[i, j]
//     barrier
template <typename T>
__device__ __forceinline__ void u_panel_solve(T* F, int fsz, int nc, int uc, int t, int nt)
{
    for (int k = 1; k < nc; ++k) {
        for (int e = t; e < uc; e += nt) {
            const int jj = nc + e;
            T v = F[(long)k * fsz + jj];
            for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
            F[(long)k * fsz + jj] = v;
        }
        __syncthreads();
    }
}

// Sync-free U-panel solve (exp_260612 barrier-cut): each thread owns one U column jj and forward-
// substitutes all nc rows in-thread (the per-row dependency lives inside the thread). The nc block
// barriers of the row-parallel form above collapse to ONE barrier (caller-side, before the trailing
// update reads U). Same column parallelism (uc / nt) and same arithmetic; only the barriers change.
// Wins when the front is barrier-bound (deep under-filled levels, low occupancy). Numerically
// identical to u_panel_solve (same op order per element).
template <typename T>
__device__ __forceinline__ void u_panel_solve_fewsync(T* F, int fsz, int nc, int uc, int t, int nt)
{
    for (int e = t; e < uc; e += nt) {
        const int jj = nc + e;
        for (int k = 1; k < nc; ++k) {
            T v = F[(long)k * fsz + jj];
            for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
            F[(long)k * fsz + jj] = v;
        }
    }
    __syncthreads();
}

template <typename T>
__device__ __forceinline__ void trailing_update_scalar(T* F, int fsz, int nc, int uc, int t,
                                                       int nt)
{
    // No staging: each thread owns one (ii, jj) output element of the uc × uc trailing
    // block, accumulates the dot product over the nc-wide K dimension, and subtracts from F.
    for (int e = t; e < uc * uc; e += nt) {
        const int ii = nc + e / uc, jj = nc + e % uc;
        T acc = T(0);
        for (int k = 0; k < nc; ++k) acc += F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        F[(long)ii * fsz + jj] -= acc;
    }
}

// Register-blocked trailing (exp_260612): the scalar version does 1 FMA per 2 strided shared loads
// with a per-iteration k*fsz address → ~5% FMA throughput (latency/overhead bound, not compute
// bound). Here each thread owns one row ii and a contiguous MR-wide column strip: L[ii,k] is loaded
// ONCE per k and reused across MR outputs, and U[k, jj0..) is read contiguously, so the FMA:load
// ratio rises from 1:2 to MR:(1+MR) and the strided address is amortized over MR outputs.
template <typename T, int MR>
__device__ __forceinline__ void trailing_update_rb(T* F, int fsz, int nc, int uc, int t, int nt)
{
    const int nstrip = (uc + MR - 1) / MR;
    const long work = (long)uc * nstrip;
    for (long w = t; w < work; w += nt) {
        const int r = (int)(w / nstrip);
        const int s = (int)(w % nstrip);
        const int ii = nc + r;
        const int jj0 = nc + s * MR;
        const int width = min(MR, uc - s * MR);
        const T* __restrict__ Lrow = F + (long)ii * fsz;
        T acc[MR];
        #pragma unroll
        for (int m = 0; m < MR; ++m) acc[m] = T(0);
        for (int k = 0; k < nc; ++k) {
            const T l = Lrow[k];
            const T* __restrict__ Urow = F + (long)k * fsz + jj0;
            #pragma unroll
            for (int m = 0; m < MR; ++m)
                if (m < width) acc[m] += l * Urow[m];
        }
        T* __restrict__ Orow = F + (long)ii * fsz + jj0;
        #pragma unroll
        for (int m = 0; m < MR; ++m)
            if (m < width) Orow[m] -= acc[m];
    }
}

// Dispatch: env CLS_TRAIL_RB={0(scalar),2,4,8} selects the register-block width. Default 4.
template <typename T>
__device__ __forceinline__ void trailing_update(T* F, int fsz, int nc, int uc, int t, int nt,
                                                 int rb)
{
    switch (rb) {
        case 0:  trailing_update_scalar<T>(F, fsz, nc, uc, t, nt); break;
        case 2:  trailing_update_rb<T, 2>(F, fsz, nc, uc, t, nt); break;
        case 8:  trailing_update_rb<T, 8>(F, fsz, nc, uc, t, nt); break;
        default: trailing_update_rb<T, 4>(F, fsz, nc, uc, t, nt); break;
    }
}

// =======================================================================================
//  PHASE 4  —  EXTEND-ADD       (scatter CB into parent front via atomicAdd)
// =======================================================================================
//
// The post-update contribution block C lives in this front at rows [nc, fsz), cols [nc, fsz).
// `asm_local[abase + 0..uc)` maps each of the uc CB rows / cols to its row / col index in the
// parent front (computed by analyze). atomicAdd is required because sibling fronts can scatter
// into the same parent entries concurrently. uc == 0 (no contribution block) makes the loop empty.
template <typename DstT, typename SrcT>
__device__ __forceinline__ void extend_add(DstT* Fp, int pfsz, const SrcT* Fsrc, int fsz, int nc,
                                           int uc, const int* asm_local, int abase, int t, int nt)
{
    for (int e = t; e < uc * uc; e += nt) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  static_cast<DstT>(Fsrc[(long)(nc + a) * fsz + (nc + b)]));
    }
}

// =======================================================================================
//  WRITEBACK  —  shared → global for the factored panel
// =======================================================================================
//
// Copies the factored L | U panel (rows < nc of the full row stride, plus the L strip at
// rows ≥ nc but cols < nc) back to global memory. The CB (rows ≥ nc, cols ≥ nc) stays in
// shared so Phase 4 can read it without re-touching global.
template <typename MT, typename WT>
__device__ __forceinline__ void writeback_factored(MT* M, const WT* W, int fsz, int nc, int uc,
                                                   int t, int nt)
{
    // Rows 0..nc-1 in full (the panel rows): L_pp | U.
    for (long e = t; e < (long)nc * fsz; e += nt) M[e] = static_cast<MT>(W[e]);
    // Rows nc..fsz-1, cols 0..nc-1 (the L strip below the panel).
    for (int e = t; e < uc * nc; e += nt) {
        const long id2 = (long)(nc + e / nc) * fsz + (e % nc);
        M[id2] = static_cast<MT>(W[id2]);
    }
}

// Each tier kernel sequences these phases itself (Phase 1 panel LU -> Phase 2 U-solve ->
// Phase 3 trailing; mid fronts fuse Phase 1+3 via lu_mid_front). See factorize/{mid,big}.cuh.

}  // namespace
}  // namespace custom_linear_solver
