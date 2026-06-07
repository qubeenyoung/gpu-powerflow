#pragma once

// FACTORIZE — per-phase device building blocks.
//
// Internal — included only by multifrontal.cu (single TU; CUDA_SEPARABLE_COMPILATION OFF).
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

template <typename FT>
__device__ __forceinline__ void lu_small_warp(FT* F, int fsz, int nc, int lane, int* sing)
{
    // Same algorithm as lu_small_front but the barrier is __syncwarp instead of
    // __syncthreads — exactly one warp (32 lanes) owns this front.
    for (int k = 0; k < nc; ++k) {
        FT piv = F[(long)k * fsz + k];
        if (piv == FT(0)) {
            if (lane == 0) *sing = 1;
            piv = FT(1);
        }
        for (int i = k + 1 + lane; i < fsz; i += 32) F[(long)i * fsz + k] /= piv;
        __syncwarp();
        const int m = fsz - k - 1;
        for (int e = lane; e < m * m; e += 32) {
            const int ii = k + 1 + e / m, jj = k + 1 + e % m;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        __syncwarp();
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
//   trailing_update_wmma_fp16       | FP16 WMMA m16n16k16 + Csc    | FP16 (mid+big)
//   trailing_update_wmma_tf32       | TF32 WMMA m16n16k8 + Csc     | TF32_WMMA (mid+big)
//   trailing_update_mma_tf32_ptx    | TF32 PTX mma.m16n8k8         | TF32 (mid+big k=8)
//   trailing_update_mma_tf32_k4_ptx | TF32 PTX mma.m16n8k4         | TF32 (mid k=4, hybrid)
//
// Common pattern across all variants:
//   (a) Stream the L and U panels out of F into a staging buffer (shared scratch for the
//       Tensor-Core variants; nothing for the scalar variants — they read F directly).
//   (b) Compute the dense GEMM C' = L · U into per-warp accumulators.
//   (c) Subtract C' from C in place: F[(nc + ii), (nc + jj)] -= C'[ii, jj], with
//       (ii, jj) bounds-checked against uc × uc since the staging buffer is round-up padded.
//
// Why the WMMA variants need a per-warp Csc smem scratch (steps (b) → (c)):
//
//   wmma::store_matrix_sync is the only public way to read a WMMA accumulator fragment;
//   it writes the fragment to a typed memory region. So:
//       (i)   wmma::mma_sync produces a 16×16 accumulator fragment per warp.
//       (ii)  wmma::store_matrix_sync(&Csc[warp * 256], cf, 16, row_major) spills the tile
//             into per-warp scratch.
//       (iii) The warp re-reads Csc, bounds-checks against uc × uc, and subtracts from F.
//   Cost: 1 KB / warp / 16×16 tile of smem round-trip + an extra __syncwarp.
//
//   The PTX variants skip (ii)/(iii) by calling `mma.sync` directly through inline asm and
//   exploiting the per-lane accumulator register layout: each lane owns exactly four FP32
//   entries at known (row, col) offsets, so it subtracts straight from F under one bounds
//   check. The layout is empirically verified — the PTX ISA documentation suggests one
//   mapping that does not match what nvcc actually emits, so each shape was probed at
//   runtime before being baked in.

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

// FP16 input WMMA, FP32 accumulate. Csc is the per-warp 256-float fragment-readback scratch
// described in the section header above.
__device__ __forceinline__ void trailing_update_wmma_fp16(float* F, int fsz, int nc, int uc,
                                                          __half* Lh, __half* Uh, float* Csc,
                                                          int ucp_stride, int t, int nt)
{
    namespace wmma = nvcuda::wmma;
    // (a) Stage L → Lh (UCP × KP) and U → Uh (KP × UCP), padded with zeros. WMMA m16n16k16
    //     requires both K-dimension and the panel dimensions to be multiples of 16.
    const int UCP = ((uc + 15) / 16) * 16;
    const int KP = ((nc + 15) / 16) * 16;
    for (int e = t; e < UCP * KP; e += nt) {
        const int i = e / KP, k = e % KP;
        Lh[e] = (i < uc && k < nc) ? __float2half(F[(long)(nc + i) * fsz + k]) : __float2half(0.0f);
    }
    for (int e = t; e < KP * UCP; e += nt) {
        const int k = e / UCP, j = e % UCP;
        Uh[k * ucp_stride + j] =
            (k < nc && j < uc) ? __float2half(F[(long)k * fsz + (nc + j)]) : __float2half(0.0f);
    }
    __syncthreads();
    // (b) GEMM. One warp owns one 16-row strip ti; A-fragments are loaded once per K-tile
    //     and reused across all N tiles. B-fragments are loaded per (kc, tj) pair.
    const int ntj = UCP / 16, nks = KP / 16;
    const int warp = t >> 5, nwarp = nt >> 5, lane = t & 31;
    for (int ti = warp; ti < ntj; ti += nwarp) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
        for (int kc = 0; kc < nks; ++kc)
            wmma::load_matrix_sync(af[kc], &Lh[(ti * 16) * KP + kc * 16], KP);
        for (int tj = 0; tj < ntj; ++tj) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
            wmma::fill_fragment(cf, 0.0f);
            for (int kc = 0; kc < nks; ++kc) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
                wmma::load_matrix_sync(bf, &Uh[(kc * 16) * ucp_stride + tj * 16], ucp_stride);
                wmma::mma_sync(cf, af[kc], bf, cf);
            }
            // (c) Store fragment to Csc, then re-read and subtract from F under uc bounds checks.
            wmma::store_matrix_sync(&Csc[warp * 256], cf, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int r = e >> 4, c = e & 15;
                const int ii = ti * 16 + r, jj = tj * 16 + c;
                if (ii < uc && jj < uc) F[(long)(nc + ii) * fsz + (nc + jj)] -= Csc[warp * 256 + e];
            }
            __syncwarp();
        }
    }
}

// TF32 input WMMA, FP32 accumulate. Same Csc fragment-readback pattern as the FP16 variant;
// staging panels hold float (TF32 inputs are bit-aliased FP32) instead of __half.
__device__ __forceinline__ void trailing_update_wmma_tf32(float* F, int fsz, int nc, int uc,
                                                          float* Ltf, float* Utf,
                                                          float* Csc, int t, int nt)
{
    namespace wmma = nvcuda::wmma;
    // (a) Stage L → Ltf (UCP × KP) and U → Utf (KP × UCP). TF32 m16n16k8 takes a K-tile of
    //     8 instead of 16, so KP rounds nc up to 8.
    const int UCP = ((uc + 15) / 16) * 16;
    const int KP = ((nc + 7) / 8) * 8;
    for (int e = t; e < UCP * KP; e += nt) {
        const int i = e / KP, k = e % KP;
        Ltf[e] = (i < uc && k < nc)
                     ? wmma::__float_to_tf32(F[(long)(nc + i) * fsz + k])
                     : 0.0f;
    }
    for (int e = t; e < KP * UCP; e += nt) {
        const int k = e / UCP, j = e % UCP;
        Utf[e] = (k < nc && j < uc)
                     ? wmma::__float_to_tf32(F[(long)k * fsz + (nc + j)])
                     : 0.0f;
    }
    __syncthreads();
    // (b) GEMM. Same A-load reuse pattern as the FP16 variant, scaled to the K=8 tile.
    const int ntj = UCP / 16, nks = KP / 8;
    const int warp = t >> 5, nwarp = nt >> 5, lane = t & 31;
    for (int ti = warp; ti < ntj; ti += nwarp) {
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                       wmma::row_major> af[4];
        for (int kc = 0; kc < nks; ++kc)
            wmma::load_matrix_sync(af[kc], &Ltf[(ti * 16) * KP + kc * 8], KP);
        for (int tj = 0; tj < ntj; ++tj) {
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> cf;
            wmma::fill_fragment(cf, 0.0f);
            for (int kc = 0; kc < nks; ++kc) {
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                               wmma::row_major> bf;
                wmma::load_matrix_sync(bf, &Utf[(kc * 8) * UCP + tj * 16], UCP);
                wmma::mma_sync(cf, af[kc], bf, cf);
            }
            // (c) Store fragment to Csc, then re-read and subtract from F.
            wmma::store_matrix_sync(&Csc[warp * 256], cf, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int r = e >> 4, c = e & 15;
                const int ii = ti * 16 + r, jj = tj * 16 + c;
                if (ii < uc && jj < uc) F[(long)(nc + ii) * fsz + (nc + jj)] -= Csc[warp * 256 + e];
            }
            __syncwarp();
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

    // (a) Stage L → Ltf and U → Utf, padded with zeros, no explicit TF32 conversion.
    for (int e = t; e < UCP * KP; e += nt) {
        const int i = e / KP, k = e % KP;
        Ltf[e] = (i < uc && k < nc) ? F[(long)(nc + i) * fsz + k] : 0.0f;
    }
    for (int e = t; e < KP * UCP; e += nt) {
        const int k = e / UCP, j = e % UCP;
        Utf[e] = (k < nc && j < uc) ? F[(long)k * fsz + (nc + j)] : 0.0f;
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
                        Utf[(kc * 8 + laneC + 0) * UCP + tj8 * 8 + laneR]);
                    const unsigned b1 = __float_as_uint(
                        Utf[(kc * 8 + laneC + 1) * UCP + tj8 * 8 + laneR]);
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
                    Utf[(kc * 8 + laneC + 0) * UCP + tj8 * 8 + laneR]);
                const unsigned b1 = __float_as_uint(
                    Utf[(kc * 8 + laneC + 1) * UCP + tj8 * 8 + laneR]);
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

// ---------------------------------------------------------------------------------------
//  TF32 PTX K=4       (mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32)
//
// Same 16×8 output tile as the K=8 instruction but with a K-dimension of 4. Selected by the
// dispatcher only when KP_k4 = round_up(nc, 4) is strictly less than KP_k8 = round_up(nc, 8),
// i.e. when nc % 8 ∈ {1..4} — exactly the regime where K=4 halves K-padding waste in the
// staging buffers and in the per-tile FLOP count. Outside that regime the dispatcher routes
// to the K=8 path to avoid the extra K-loop iterations a K=4 instruction would cost.
//
// Per-lane A-matrix register layout (probed):
//   a0 = A[laneR + 0, laneC]    (M_top, K = laneC)        2 .b32 / lane
//   a1 = A[laneR + 8, laneC]    (M_bot, K = laneC)
//   b0 = B[laneC, laneR]        (K, N)                    1 .b32 / lane
//   c0..c3 cover the same 16×8 output positions as the K=8 instruction.
//
// laneR = lane >> 2 (0..7), laneC = lane & 3 (0..3). Each lane covers one K column and two
// adjacent N columns.
__device__ __forceinline__ void trailing_update_mma_tf32_k4_ptx(float* F, int fsz, int nc, int uc,
                                                                  float* Ltf, float* Utf,
                                                                  int t, int nt)
{
    const int UCP = ((uc + 15) / 16) * 16;
    const int KP  = ((nc + 3)  / 4)  * 4;       // K=4 alignment

    // (a) Stage L → Ltf and U → Utf (no explicit TF32 conversion — mma `.tf32` truncates).
    for (int e = t; e < UCP * KP; e += nt) {
        const int i = e / KP, k = e % KP;
        Ltf[e] = (i < uc && k < nc) ? F[(long)(nc + i) * fsz + k] : 0.0f;
    }
    for (int e = t; e < KP * UCP; e += nt) {
        const int k = e / UCP, j = e % UCP;
        Utf[e] = (k < nc && j < uc) ? F[(long)k * fsz + (nc + j)] : 0.0f;
    }
    __syncthreads();

    const int ntj16 = UCP / 16;
    const int ntj8  = UCP / 8;
    const int nks   = KP / 4;                   // K-loop count (mma K=4)
    const int warp  = t >> 5;
    const int nwarp = nt >> 5;
    const int lane  = t & 31;
    const int laneR = lane >> 2;
    const int laneC = lane & 3;                 // single K col per lane (not laneC*2 as in K=8)

    // (b)/(c) One warp per M-strip; iterate N tiles, accumulate over K, drain into F.
    for (int ti = warp; ti < ntj16; ti += nwarp) {
        const int r_top = ti * 16 + laneR;
        const int r_bot = r_top + 8;
        for (int tj8 = 0; tj8 < ntj8; ++tj8) {
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            for (int kc = 0; kc < nks; ++kc) {
                const unsigned a0 = __float_as_uint(Ltf[(ti * 16 + laneR + 0) * KP + kc * 4 + laneC]);
                const unsigned a1 = __float_as_uint(Ltf[(ti * 16 + laneR + 8) * KP + kc * 4 + laneC]);
                const unsigned b0 = __float_as_uint(Utf[(kc * 4 + laneC) * UCP + tj8 * 8 + laneR]);
                asm volatile(
                    "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
                    "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(b0));
            }
            const int col0 = tj8 * 8 + laneC * 2;
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
