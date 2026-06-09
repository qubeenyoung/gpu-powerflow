#pragma once

// FACTORIZE — per-phase device building blocks.
//
// Internal — included only by numeric_engine.cu (single TU; CUDA_SEPARABLE_COMPILATION OFF).
//
// A front is a square dense block with two contiguous regions (row-major, leading dim = fsz):
//
//        cols 0..nc-1 (pivot)   cols nc..fsz-1 (trailing)
//      +--------------------+---------------------+
//      |        L_pp        |         U           |   rows 0..nc-1   (panel)
//      +--------------------+---------------------+
//      |        L           |         C           |   rows nc..fsz-1 (contribution block)
//      +--------------------+---------------------+
//
//   - The pivot block L_pp is factored as L_pp · U_pp (no pivoting; the matrix is assumed
//     numerically safe under the analyze-time ordering).
//   - The U panel is recovered by solving L_pp · U = U_raw.
//   - The trailing block C is updated as C ← C − L · U.
//   - The factored panel (L | U at rows < nc, plus L below) is written back to global memory.
//   - The CB (rows ≥ nc, cols ≥ nc of the post-update front) scatter-adds into the parent
//     front (Phase 4 / extend-add) so the parent's input value at this front's contribution
//     positions accumulates the symbolic update.
//
// Phase decomposition (all run inside one __global__ factor kernel per front):
//
//   stage_in_async         — global F → shared Fs    (mid tier only; cp.async on Ampere+)
//   Phase 1: panel LU      — factor L_pp · U_pp = F_pp                          (lu_small_front / lu_panel_factor / lu_small_warp)
//   Phase 2: U-panel solve — recover U from L_pp · U = U_raw                    (u_panel_solve)
//   Phase 3: trailing GEMM — C ← C − L · U                                       (trailing_update_*)
//   writeback_factored     — shared Fs → global F     (mid tier only)
//   Phase 4: extend-add    — scatter CB into parent front via atomicAdd          (extend_add)
//
// factorize_front orchestrates Phase 1+2+3 in canonical order; each tier kernel wraps it
// with its own stage / writeback / extend-add (see factorize/kernels.cuh).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#if __CUDA_ARCH__ >= 800
#include <cuda_pipeline.h>
#endif

namespace custom_linear_solver {
namespace {

// =======================================================================================
//  STAGE-IN  —  global → shared
// =======================================================================================
//
// Copies the whole fsz×fsz front from global to dynamic shared so the four phases below
// touch only shared memory. On Ampere+ each element is issued through cp.async via
// __pipeline_memcpy_async, then committed and waited as one batch; on pre-Ampere the helper
// falls back to a synchronous strided copy. The async path lets a thread queue many loads
// before stalling, giving the SM something else to schedule while the loads are in flight.
template <typename T>
__device__ __forceinline__ void stage_in_async(T* __restrict__ Fs, const T* __restrict__ F,
                                                int fsz2, int t, int nt)
{
#if __CUDA_ARCH__ >= 800
    constexpr size_t SHAPE = sizeof(T);
    for (int e = t; e < fsz2; e += nt) {
        __pipeline_memcpy_async(&Fs[e], &F[e], SHAPE);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
#else
    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
#endif
}

// =======================================================================================
//  PHASE 1  —  PANEL LU       (factor L_pp · U_pp = F_pp in place)
// =======================================================================================
//
// Three implementations for the three tiers:
//
//   lu_small_front<T>  — fully fused Phase 1+3 (mid tier, fsz ≤ 48). Runs nc rank-1 passes
//                        over the FULL (fsz−k−1)² sub-block, redoing some trailing work
//                        on every k but saving the explicit Phase-3 trailing call. For
//                        fronts this small the redundant work is small and the savings
//                        from skipping a separate trailing kernel dominate. Caller must NOT
//                        also call trailing_update_*.
//
//   lu_panel_factor<T> — split form for mid (fsz > 48) and big tiers. Phase-3 trailing is
//                        called separately by the caller. Has two internal forms keyed off
//                        nc:
//                            nc ≤ 12 : row-fused — one __syncthreads per k.
//                            nc > 12 : two-phase — 2·nc syncs but avoids an O(nc) serial
//                                      inner update per thread that becomes expensive at
//                                      large nc.
//                        Both forms hoist `inv_piv = 1 / piv` so the column normalisation
//                        becomes a multiply (FFMA) rather than a per-thread FP divide.
//
//   lu_small_warp<FT>  — warp-fused form for the small tier. One warp owns a whole front
//                        and uses __syncwarp() so the inner barriers cost a warp shuffle,
//                        not a block-wide barrier.
//
// Singularity is communicated through *sing: any thread that encounters a zero pivot stores
// 1 there and continues with piv = 1 so the kernel does not stall; the caller decides what
// to do with the flag after the launch retires.

template <typename T>
__device__ __forceinline__ void lu_small_front(T* F, int fsz, int nc, int t, int nt, int* sing)
{
    // For each pivot column k:
    //   1. Read piv = F[k, k]; flag singularity.
    //   2. Divide column k below the pivot by piv.
    //   3. Rank-1 update the entire (fsz−k−1)² block below-right of the pivot.
    //   4. Block barrier between k and k+1.
    for (int k = 0; k < nc; ++k) {
        T piv = F[(long)k * fsz + k];
        if (piv == T(0)) { if (t == 0) *sing = 1; piv = T(1); }
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
__device__ __forceinline__ void lu_panel_factor(T* F, int fsz, int nc, int t, int nt, int* sing)
{
    constexpr int ROWFUSED_NC_MAX = 12;

    if (nc > 0 && nc <= ROWFUSED_NC_MAX) {
        // Row-fused form. For each k, every thread that owns one of the rows below the pivot
        // (i.e. i = k + 1 + t (mod nt)) does the divide and the in-row panel update in a
        // single pass — `lik` is kept in a register and applied across the panel columns
        // jj ∈ (k, nc) without touching shared again. One __syncthreads per k.
        for (int k = 0; k < nc; ++k) {
            T piv = F[(long)k * fsz + k];
            if (piv == T(0)) { if (t == 0) *sing = 1; piv = T(1); }
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
        T piv = F[(long)k * fsz + k];
        if (piv == T(0)) { if (t == 0) *sing = 1; piv = T(1); }
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

// SG = sub-group lane count (8 / 16 / 32). One sub-group of SG lanes owns one front; SG=32 is
// the classic one-warp-per-front form. `sl` is the lane within the sub-group (0..SG-1) and
// `mask` the sub-group's active-lane mask. Tiny fronts (fsz ≤ SG) keep all SG lanes busy
// instead of idling 32−fsz of a warp, and packing 32/SG fronts per warp raises memory-level
// parallelism on this latency-bound tier (see factorize/dispatch.cuh).
template <typename FT, int SG>
__device__ __forceinline__ void lu_small_warp(FT* F, int fsz, int nc, int sl, unsigned mask,
                                              int* sing)
{
    for (int k = 0; k < nc; ++k) {
        FT piv = F[(long)k * fsz + k];
        if (piv == FT(0)) {
            if (sl == 0) *sing = 1;
            piv = FT(1);
        }
        for (int i = k + 1 + sl; i < fsz; i += SG) F[(long)i * fsz + k] /= piv;
        __syncwarp(mask);
        const int m = fsz - k - 1;
        for (int e = sl; e < m * m; e += SG) {
            const int ii = k + 1 + e / m, jj = k + 1 + e % m;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        __syncwarp(mask);
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

// =======================================================================================
//  PHASE 3  —  TRAILING UPDATE       (C(uc × uc) -= L(uc × nc) · U(nc × uc))
// =======================================================================================
//
// Variant matrix (selected per Precision in factorize/dispatch.cuh):
//
//   variant                         | trailing kernel              | used by Precision
//   --------------------------------+------------------------------+------------------------
//   trailing_update_scalar<T>       | direct dot products on F     | FP64 (mid+big), scalar fallback
//   trailing_update_staged<T>       | staged-shared scalar         | FP32 (mid)
//   trailing_update_mma_fp16_ptx    | FP16 PTX mma.m16n8k8         | FP16 (mid+big)
//   trailing_update_mma_tf32_ptx    | TF32 PTX mma.m16n8k8         | TF32 (mid+big k=8)
//
// Common pattern across all variants:
//   (a) Stream the L and U panels out of F into a staging buffer (shared scratch for the
//       Tensor-Core variants; nothing for the scalar variants — they read F directly).
//   (b) Compute the dense GEMM C' = L · U into per-warp accumulators.
//   (c) Subtract C' from C in place: F[(nc + ii), (nc + jj)] -= C'[ii, jj], with
//       (ii, jj) bounds-checked against uc × uc since the staging buffer is round-up padded.
//
// All Tensor-Core trailing variants here are PTX (inline-asm `mma.sync`), not WMMA. The PTX
// form calls the instruction directly and exploits the per-lane accumulator register layout:
// each lane owns exactly four FP32 entries at known (row, col) offsets, so it subtracts
// straight from F under one bounds check — no `wmma::store_matrix_sync` → smem-scratch →
// reload round-trip, and no per-warp Csc scratch. The m16n8 accumulator (C) layout is shared
// by every shape used here (FP16 and TF32, K∈{4,8}): with groupID = lane >> 2 and
// tid = lane & 3, the four FP32 regs map to (row, col) = (groupID{,+8}, 2·tid{,+1}). The
// multiplicand (A/B) layouts differ per element type and are empirically verified — the PTX
// ISA documentation suggests one mapping that does not always match what nvcc actually emits,
// so each shape was probed at runtime before being baked in.

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

template <typename T>
__device__ __forceinline__ void trailing_update_staged(T* F, int fsz, int nc, int uc, int t,
                                                        int nt, T* sh_L, T* sh_U)
{
    // (a) Copy the L (uc × nc) and U (nc × uc) panels of F into compact shared layouts. The
    //     staged layout has unit-strided inner dimensions so the inner dot-product loop hits
    //     contiguous shared lines.
    for (int e = t; e < uc * nc; e += nt) {
        const int i = e / nc, k = e % nc;
        sh_L[e] = F[(long)(nc + i) * fsz + k];
    }
    for (int e = t; e < nc * uc; e += nt) {
        const int k = e / uc, j = e % uc;
        sh_U[e] = F[(long)k * fsz + (nc + j)];
    }
    __syncthreads();
    // (b)/(c) Each thread owns one (i, j) output, reads from the staged panels, and subtracts.
    for (int e = t; e < uc * uc; e += nt) {
        const int i = e / uc, j = e % uc;
        T acc = T(0);
        for (int k = 0; k < nc; ++k) acc += sh_L[i * nc + k] * sh_U[k * uc + j];
        F[(long)(nc + i) * fsz + (nc + j)] -= acc;
    }
}

// ---------------------------------------------------------------------------------------
//  FP16 PTX K=8       (mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32)
//
// FP16 inputs, FP32 accumulate. Per-lane register-direct trailing, structurally identical to
// the TF32 K=8 path below (same 16×8 output tile, same (ti, kc, tj8) A-reuse loop, same drain
// into F) — the only differences are that the staging panels hold __half and that FP16 packs
// two elements per 32-bit register, so m16n8k8.f16 takes 2 A-regs + 1 B-reg instead of TF32's
// 4 + 2.
//
// Per-lane multiplicand layout (probed). groupID = laneR = lane >> 2 (0..7); tid = lane & 3
// (0..3); laneC = tid * 2 ∈ {0,2,4,6}:
//   A (16×8, row-major Lh, ld = KP): two .f16 packed per reg, contiguous in K →
//       a0 = pack(A[laneR + 0, laneC], A[laneR + 0, laneC + 1])   (M_top, K pair)
//       a1 = pack(A[laneR + 8, laneC], A[laneR + 8, laneC + 1])   (M_bot, K pair)
//     loadable as a single 4-byte read of two adjacent halves (laneC even ⇒ aligned).
//   B (8×8, row-major Uh, ld = UCP): two .f16 packed per reg, contiguous in K (stride UCP in
//     memory, so packed explicitly) →
//       b0 = pack(B[laneC, laneR], B[laneC + 1, laneR])           (K pair, N = laneR)
//   C accumulator: identical to every m16n8 shape — see the section header.
__device__ __forceinline__ void trailing_update_mma_fp16_ptx(float* F, int fsz, int nc, int uc,
                                                             __half* Lh, __half* Uh,
                                                             int t, int nt)
{
    const int UCP = ((uc + 15) / 16) * 16;
    const int KP  = ((nc + 7)  / 8)  * 8;

    // (a) Stage L → Lh and U → Uh as __half, padded with zeros.
    for (int e = t; e < UCP * KP; e += nt) {
        const int i = e / KP, k = e % KP;
        Lh[e] = (i < uc && k < nc) ? __float2half(F[(long)(nc + i) * fsz + k]) : __half(0.0f);
    }
    for (int e = t; e < KP * UCP; e += nt) {
        const int k = e / UCP, j = e % UCP;
        Uh[e] = (k < nc && j < uc) ? __float2half(F[(long)k * fsz + (nc + j)]) : __half(0.0f);
    }
    __syncthreads();

    const int ntj16 = UCP / 16;            // 16-row tiles (M)
    const int ntj8  = UCP / 8;             // 8-col tiles  (N)
    const int nks   = KP  / 8;             // K-loop count (mma K=8)
    const int warp  = t >> 5;
    const int nwarp = nt >> 5;
    const int lane  = t & 31;
    const int laneR = lane >> 2;           // 0..7
    const int laneC = (lane & 3) * 2;      // 0,2,4,6

    // A-reuse hoisted path. Capped at NTJ8_MAX = 8 N-tiles (UCP ≤ 64); the unrolled tj8 loop
    // keeps the per-tile accumulators in named registers for the inline-asm "+f" binding.
    constexpr int NTJ8_MAX = 8;
    if (ntj8 <= NTJ8_MAX) {
        for (int ti = warp; ti < ntj16; ti += nwarp) {
            const int r_top = ti * 16 + laneR;
            const int r_bot = r_top + 8;
            float c[NTJ8_MAX][4];
            #pragma unroll
            for (int j = 0; j < NTJ8_MAX; ++j) { 
                c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f; 
            }
            // (b) Outer K loop: load A once per K-tile (two contiguous halves → one 4-byte
            //     read), then sweep all N tiles inside.
            for (int kc = 0; kc < nks; ++kc) {
                const unsigned a0 = *reinterpret_cast<const unsigned*>(
                    &Lh[(long)(ti * 16 + laneR + 0) * KP + kc * 8 + laneC]);
                const unsigned a1 = *reinterpret_cast<const unsigned*>(
                    &Lh[(long)(ti * 16 + laneR + 8) * KP + kc * 8 + laneC]);
                #pragma unroll
                for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                    if (tj8 >= ntj8) break;
                    // B pair is strided by UCP in memory → pack the two halves by hand.
                    const __half2 bh = __halves2half2(
                        Uh[(long)(kc * 8 + laneC + 0) * UCP + tj8 * 8 + laneR],
                        Uh[(long)(kc * 8 + laneC + 1) * UCP + tj8 * 8 + laneR]);
                    const unsigned b0 = *reinterpret_cast<const unsigned*>(&bh);
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                        : "+f"(c[tj8][0]), "+f"(c[tj8][1]), "+f"(c[tj8][2]), "+f"(c[tj8][3])
                        : "r"(a0), "r"(a1), "r"(b0));
                }
            }
            // (c) Drain accumulators straight into F with uc bounds checks.
            #pragma unroll
            for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                if (tj8 >= ntj8) break;
                const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
                if (r_top < uc) {
                    if (col0 < uc) F[(long)(nc + r_top) * fsz + (nc + col0)] -= c[tj8][0];
                    if (col1 < uc) F[(long)(nc + r_top) * fsz + (nc + col1)] -= c[tj8][1];
                }
                if (r_bot < uc) {
                    if (col0 < uc) F[(long)(nc + r_bot) * fsz + (nc + col0)] -= c[tj8][2];
                    if (col1 < uc) F[(long)(nc + r_bot) * fsz + (nc + col1)] -= c[tj8][3];
                }
            }
        }
        return;
    }

    // Fall-through for UCP > 64 (big-tier strips wider than NTJ8_MAX × 8). The (ti, tj8, kc)
    // ordering reloads A per tj8, but the absolute A-reuse saving is smaller at this size.
    for (int ti = warp; ti < ntj16; ti += nwarp) {
        const int r_top = ti * 16 + laneR;
        const int r_bot = r_top + 8;
        for (int tj8 = 0; tj8 < ntj8; ++tj8) {
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            for (int kc = 0; kc < nks; ++kc) {
                const unsigned a0 = *reinterpret_cast<const unsigned*>(
                    &Lh[(long)(ti * 16 + laneR + 0) * KP + kc * 8 + laneC]);
                const unsigned a1 = *reinterpret_cast<const unsigned*>(
                    &Lh[(long)(ti * 16 + laneR + 8) * KP + kc * 8 + laneC]);
                const __half2 bh = __halves2half2(
                    Uh[(long)(kc * 8 + laneC + 0) * UCP + tj8 * 8 + laneR],
                    Uh[(long)(kc * 8 + laneC + 1) * UCP + tj8 * 8 + laneR]);
                const unsigned b0 = *reinterpret_cast<const unsigned*>(&bh);
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(b0));
            }
            const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
            if (r_top < uc) {
                if (col0 < uc) F[(long)(nc + r_top) * fsz + (nc + col0)] -= c0;
                if (col1 < uc) F[(long)(nc + r_top) * fsz + (nc + col1)] -= c1;
            }
            if (r_bot < uc) {
                if (col0 < uc) F[(long)(nc + r_bot) * fsz + (nc + col0)] -= c2;
                if (col1 < uc) F[(long)(nc + r_bot) * fsz + (nc + col1)] -= c3;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
//  TF32 PTX K=8       (mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32)
//
// Per-lane register-direct version of the TF32 trailing. The 16×8 output of one mma.m16n8k8
// instruction places four FP32 accumulator elements in registers c0..c3 of each lane, at
// fixed (row, col) offsets relative to the warp's tile origin. Knowing those offsets at
// compile time lets each lane subtract its four entries straight from F under a single
// bounds check, eliminating the wmma::store_matrix_sync → Csc → reload round-trip used by
// the WMMA variants.
//
// One warp owns one 16-row M-strip (`ti`) of the trailing block, sweeps all N=8 column
// tiles (`tj8`), and accumulates across the K dimension (`kc`). The loop ordering is
// (ti, kc, tj8) so the A-fragments — which only depend on (ti, kc) — are loaded once per
// K-tile and reused across the inner tj8 sweep. The unrolled tj8 loop keeps the
// accumulators `c[tj8][...]` in named registers, which is required for the CUDA inline-asm
// "+f" operand binding to be stable.
//
// Per-lane A-matrix register layout (verified by probe; inner stride is M-block, NOT K):
//   a0 = A[laneR + 0, laneC + 0]   (M_top, K_even)
//   a1 = A[laneR + 8, laneC + 0]   (M_bot, K_even)
//   a2 = A[laneR + 0, laneC + 1]   (M_top, K_odd)
//   a3 = A[laneR + 8, laneC + 1]   (M_bot, K_odd)
//
// The explicit wmma::__float_to_tf32 conversion is skipped: mma's `.tf32` ABI truncates
// the low 13 bits of the FP32 input automatically, so the explicit round-to-nearest only
// changes the sign of the rounding error within TF32 precision (within the accuracy budget
// for the power-grid solve targets).
__device__ __forceinline__ void trailing_update_mma_tf32_ptx(float* F, int fsz, int nc, int uc,
                                                              float* Ltf, float* Utf,
                                                              int t, int nt)
{
    const int UCP = ((uc + 15) / 16) * 16;
    const int KP  = ((nc + 7)  / 8)  * 8;
    const int LDB = UCP + 4;          // Utf column stride padded by 4: 2·UCP is a multiple of 32,
                                      // which made the B-read a 4-way shared bank conflict; +4
                                      // spreads laneC across banks 0/8/16/24 → conflict-free.

    // (a) Stage L → Ltf and U → Utf, padded with zeros, no explicit TF32 conversion.
    for (int e = t; e < UCP * KP; e += nt) {
        const int i = e / KP, k = e % KP;
        Ltf[e] = (i < uc && k < nc) ? F[(long)(nc + i) * fsz + k] : 0.0f;
    }
    for (int e = t; e < KP * UCP; e += nt) {
        const int k = e / UCP, j = e % UCP;
        Utf[k * LDB + j] = (k < nc && j < uc) ? F[(long)k * fsz + (nc + j)] : 0.0f;
    }
    __syncthreads();

    // Tile counts and per-lane index helpers.
    const int ntj16 = UCP / 16;            // 16-row tiles (M)
    const int ntj8  = UCP / 8;             // 8-col tiles  (N)
    const int nks   = KP  / 8;             // K-loop count (mma K=8)
    const int warp  = t >> 5;
    const int nwarp = nt >> 5;
    const int lane  = t & 31;
    const int laneR = lane >> 2;           // 0..7
    const int laneC = (lane & 3) * 2;      // 0,2,4,6

    // A-reuse hoisted path. Capped at NTJ8_MAX = 8 N-tiles (UCP ≤ 64). The tj8 loop is
    // fully unrolled with an early `break` so the inline-asm operand `c[<const>][.]` binds
    // to dedicated registers; otherwise nvcc would spill the accumulator to local memory
    // and the "+f" operand binding would misbehave.
    constexpr int NTJ8_MAX = 8;
    if (ntj8 <= NTJ8_MAX) {
        for (int ti = warp; ti < ntj16; ti += nwarp) {
            const int r_top = ti * 16 + laneR;
            const int r_bot = r_top + 8;
            float c[NTJ8_MAX][4];
            #pragma unroll
            for (int j = 0; j < NTJ8_MAX; ++j) { c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f; }
            // (b) Outer K loop: load A once per K-tile, then sweep all N tiles inside.
            for (int kc = 0; kc < nks; ++kc) {
                const float* A_top = &Ltf[(ti * 16 + laneR + 0) * KP + kc * 8 + laneC];
                const float* A_bot = &Ltf[(ti * 16 + laneR + 8) * KP + kc * 8 + laneC];
                const unsigned a0 = __float_as_uint(A_top[0]);
                const unsigned a1 = __float_as_uint(A_bot[0]);
                const unsigned a2 = __float_as_uint(A_top[1]);
                const unsigned a3 = __float_as_uint(A_bot[1]);
                #pragma unroll
                for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                    if (tj8 >= ntj8) break;
                    // B-fragment per (kc, tj8): b0 = B[K_even, N=laneR], b1 = B[K_odd, N=laneR].
                    const unsigned b0 = __float_as_uint(
                        Utf[(kc * 8 + laneC + 0) * LDB + tj8 * 8 + laneR]);
                    const unsigned b1 = __float_as_uint(
                        Utf[(kc * 8 + laneC + 1) * LDB + tj8 * 8 + laneR]);
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(c[tj8][0]), "+f"(c[tj8][1]), "+f"(c[tj8][2]), "+f"(c[tj8][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
            // (c) Drain accumulators straight into F with uc bounds checks.
            //     Per-lane output positions (laneR, laneC) :
            //       c[tj8][0] → F[r_top, col0],  c[tj8][1] → F[r_top, col1]
            //       c[tj8][2] → F[r_bot, col0],  c[tj8][3] → F[r_bot, col1]
            #pragma unroll
            for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                if (tj8 >= ntj8) break;
                const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
                if (r_top < uc) {
                    if (col0 < uc) F[(long)(nc + r_top) * fsz + (nc + col0)] -= c[tj8][0];
                    if (col1 < uc) F[(long)(nc + r_top) * fsz + (nc + col1)] -= c[tj8][1];
                }
                if (r_bot < uc) {
                    if (col0 < uc) F[(long)(nc + r_bot) * fsz + (nc + col0)] -= c[tj8][2];
                    if (col1 < uc) F[(long)(nc + r_bot) * fsz + (nc + col1)] -= c[tj8][3];
                }
            }
        }
        return;
    }

    // Fall-through path for UCP > 64 (big-tier front strips wider than NTJ8_MAX × 8). The
    // (ti, tj8, kc) ordering reloads A on every tj8, but the absolute A-reuse savings are
    // smaller relative to the per-tile work at this size, so the simpler form wins.
    for (int ti = warp; ti < ntj16; ti += nwarp) {
        const int r_top = ti * 16 + laneR;
        const int r_bot = r_top + 8;
        for (int tj8 = 0; tj8 < ntj8; ++tj8) {
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            for (int kc = 0; kc < nks; ++kc) {
                const float* A_top = &Ltf[(ti * 16 + laneR + 0) * KP + kc * 8 + laneC];
                const float* A_bot = &Ltf[(ti * 16 + laneR + 8) * KP + kc * 8 + laneC];
                const unsigned a0 = __float_as_uint(A_top[0]);
                const unsigned a1 = __float_as_uint(A_bot[0]);
                const unsigned a2 = __float_as_uint(A_top[1]);
                const unsigned a3 = __float_as_uint(A_bot[1]);
                const unsigned b0 = __float_as_uint(
                    Utf[(kc * 8 + laneC + 0) * LDB + tj8 * 8 + laneR]);
                const unsigned b1 = __float_as_uint(
                    Utf[(kc * 8 + laneC + 1) * LDB + tj8 * 8 + laneR]);
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            const int col0 = tj8 * 8 + laneC;
            const int col1 = col0 + 1;
            if (r_top < uc) {
                if (col0 < uc) F[(long)(nc + r_top) * fsz + (nc + col0)] -= c0;
                if (col1 < uc) F[(long)(nc + r_top) * fsz + (nc + col1)] -= c1;
            }
            if (r_bot < uc) {
                if (col0 < uc) F[(long)(nc + r_bot) * fsz + (nc + col0)] -= c2;
                if (col1 < uc) F[(long)(nc + r_bot) * fsz + (nc + col1)] -= c3;
            }
        }
    }
}

// =======================================================================================
//  PHASE 4  —  EXTEND-ADD       (scatter CB into parent front via atomicAdd)
// =======================================================================================
//
// The post-update contribution block C lives in this front at rows [nc, fsz), cols [nc, fsz).
// `asm_local[abase + 0..uc)` maps each of the uc CB rows / cols to its row / col index in the
// parent front (computed by analyze). atomicAdd is required because sibling fronts can scatter
// into the same parent entries concurrently.
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

// =======================================================================================
//  ORCHESTRATOR  —  factorize_front runs Phase 1+2+3 in canonical order
// =======================================================================================
//
// Called by every tier kernel after the front is in place (in shared for mid, in global for
// big). For small fronts (fsz ≤ 48) the orchestrator picks the fully-fused lu_small_front
// path, which subsumes Phase 3, so the caller's `trailing()` is NEVER invoked. Otherwise
// the three phases run in order with the caller's trailing functor selecting the Phase-3
// variant.
template <typename T, typename TrailingFn>
__device__ __forceinline__ void factorize_front(T* F, int fsz, int nc, int uc, int t, int nt,
                                                int* sing, TrailingFn&& trailing)
{
    if (fsz <= 48) {
        lu_small_front<T>(F, fsz, nc, t, nt, sing);   // Phase 1 + Phase 3 fused
    } else {
        lu_panel_factor<T>(F, fsz, nc, t, nt, sing);  // Phase 1
        u_panel_solve<T>(F, fsz, nc, uc, t, nt);      // Phase 2
        trailing();                                    // Phase 3
    }
}

}  // namespace
}  // namespace custom_linear_solver
