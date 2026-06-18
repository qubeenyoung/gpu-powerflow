#pragma once

// FACTORIZE — BIG tier (front too large for shared; stays global-resident). Stages only the L/U
// panels, drains the CB into the parent (fused). TF32 uses the thin-K mma trailing.
// Kernel + staged/TC trailing + dispatch, all here.

#include "factorize/front_ops.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

template <typename T, bool FuseExtend = false>
__device__ __forceinline__ void trailing_update_staged(T* F, int fsz, int nc, int uc, int t,
                                                        int nt, T* sh_L, T* sh_U,
                                                        T* Fp = nullptr, int pfsz = 0,
                                                        const int* asm_local = nullptr, int abase = 0)
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
        const long off = (long)(nc + i) * fsz + (nc + j);
        if constexpr (FuseExtend) {
            atomicAdd(&Fp[(long)asm_local[abase + i] * pfsz + asm_local[abase + j]], F[off] - acc);
        } else {
            F[off] -= acc;
        }
    }
}

// Bounded-shared trailing for the few oversized separator fronts whose full 2·nc·uc staging would
// exceed the opt-in shared budget (large uc, e.g. circuit / 2D-FEM root separators). Stages only the
// strided U panel in column-tiles of width `jt` (shared = nc·jt, independent of uc) and reads the
// row-contiguous L panel straight from global (cheap, coalesced in k). The common power-grid path
// keeps trailing_update_staged unchanged — this is selected only when full_sh > budget.
template <typename T, bool FuseExtend = false>
__device__ __forceinline__ void trailing_update_staged_tiled(T* F, int fsz, int nc, int uc, int t,
                                                             int nt, T* sh_U, int jt,
                                                             T* Fp = nullptr, int pfsz = 0,
                                                             const int* asm_local = nullptr,
                                                             int abase = 0)
{
    for (int j0 = 0; j0 < uc; j0 += jt) {
        const int jw = (uc - j0 < jt) ? (uc - j0) : jt;
        __syncthreads();  // protect sh_U from the previous tile's readers before restaging
        // (a) Stage U columns [j0, j0+jw) into sh_U (stride jt). U is column-strided in global.
        for (int e = t; e < nc * jw; e += nt) {
            const int k = e / jw, jj = e % jw;
            sh_U[k * jt + jj] = F[(long)k * fsz + (nc + j0 + jj)];
        }
        __syncthreads();
        // (b) Each thread owns one (i, j) output. L[i][k] is read row-contiguous from global.
        for (int e = t; e < uc * jw; e += nt) {
            const int i = e / jw, jj = e % jw;
            T acc = T(0);
            for (int k = 0; k < nc; ++k) acc += F[(long)(nc + i) * fsz + k] * sh_U[k * jt + jj];
            const int j = j0 + jj;
            const long off = (long)(nc + i) * fsz + (nc + j);
            if constexpr (FuseExtend) {
                atomicAdd(&Fp[(long)asm_local[abase + i] * pfsz + asm_local[abase + j]],
                          F[off] - acc);
            } else {
                F[off] -= acc;
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
template <bool FuseExtend = false>
__device__ __forceinline__ void trailing_update_tf32_tc(float* F, int fsz, int nc, int uc,
                                                              float* Ltf, float* Utf,
                                                              int t, int nt,
                                                              float* Fp = nullptr,
                                                              int pfsz = 0,
                                                              const int* asm_local = nullptr,
                                                              int abase = 0)
{
    const int uc_pad = ((uc + 15) / 16) * 16;
    const int nc_pad  = ((nc + 7)  / 8)  * 8;
    const int utf_stride = uc_pad + 4;          // Utf column stride padded by 4: 2·uc_pad is a multiple of 32,
                                      // which made the B-read a 4-way shared bank conflict; +4
                                      // spreads laneC across banks 0/8/16/24 → conflict-free.

    // (a) Stage L → Ltf and U → Utf, padded with zeros, no explicit TF32 conversion.
    for (int e = t; e < uc_pad * nc_pad; e += nt) {
        const int i = e / nc_pad, k = e % nc_pad;
        Ltf[e] = (i < uc && k < nc) ? F[(long)(nc + i) * fsz + k] : 0.0f;
    }
    for (int e = t; e < nc_pad * uc_pad; e += nt) {
        const int k = e / uc_pad, j = e % uc_pad;
        Utf[k * utf_stride + j] = (k < nc && j < uc) ? F[(long)k * fsz + (nc + j)] : 0.0f;
    }
    __syncthreads();

    // Tile counts and per-lane index helpers.
    const int m_tiles = uc_pad / 16;            // 16-row tiles (M)
    const int n_tiles  = uc_pad / 8;             // 8-col tiles  (N)
    const int k_tiles   = nc_pad  / 8;             // K-loop count (mma K=8)
    const int warp  = t >> 5;
    const int nwarp = nt >> 5;
    const int lane  = t & 31;
    const int laneR = lane >> 2;           // 0..7
    const int laneC = (lane & 3) * 2;      // 0,2,4,6

    auto store_c = [&](int r, int col, float c) {
        if (r >= uc || col >= uc) return;
        const long off = (long)(nc + r) * fsz + (nc + col);
        if constexpr (FuseExtend) {
            atomicAdd(&Fp[(long)asm_local[abase + r] * pfsz + asm_local[abase + col]],
                      F[off] - c);
        } else {
            F[off] -= c;
        }
    };

    // A-reuse hoisted path. Capped at NTJ8_MAX = 8 N-tiles (uc_pad ≤ 64). The tj8 loop is
    // fully unrolled with an early `break` so the inline-asm operand `c[<const>][.]` binds
    // to dedicated registers; otherwise nvcc would spill the accumulator to local memory
    // and the "+f" operand binding would misbehave.
    constexpr int NTJ8_MAX = 8;
    if (n_tiles <= NTJ8_MAX) {
        for (int ti = warp; ti < m_tiles; ti += nwarp) {
            const int r_top = ti * 16 + laneR;
            const int r_bot = r_top + 8;
            float c[NTJ8_MAX][4];
            #pragma unroll
            for (int j = 0; j < NTJ8_MAX; ++j) { c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f; }
            // (b) Outer K loop: load A once per K-tile, then sweep all N tiles inside.
            for (int kc = 0; kc < k_tiles; ++kc) {
                const float* A_top = &Ltf[(ti * 16 + laneR + 0) * nc_pad + kc * 8 + laneC];
                const float* A_bot = &Ltf[(ti * 16 + laneR + 8) * nc_pad + kc * 8 + laneC];
#ifdef CLS_TF32_OZAKI_TC2
                const Tf32Pair a0 = tf32_ozaki_pair(A_top[0]);
                const Tf32Pair a1 = tf32_ozaki_pair(A_bot[0]);
                const Tf32Pair a2 = tf32_ozaki_pair(A_top[1]);
                const Tf32Pair a3 = tf32_ozaki_pair(A_bot[1]);
#else
                const unsigned a0 = __float_as_uint(A_top[0]);
                const unsigned a1 = __float_as_uint(A_bot[0]);
                const unsigned a2 = __float_as_uint(A_top[1]);
                const unsigned a3 = __float_as_uint(A_bot[1]);
#endif
                #pragma unroll
                for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                    if (tj8 >= n_tiles) break;
                    // B-fragment per (kc, tj8): b0 = B[K_even, N=laneR], b1 = B[K_odd, N=laneR].
#ifdef CLS_TF32_OZAKI_TC2
                    const Tf32Pair b0 = tf32_ozaki_pair(
                        Utf[(kc * 8 + laneC + 0) * utf_stride + tj8 * 8 + laneR]);
                    const Tf32Pair b1 = tf32_ozaki_pair(
                        Utf[(kc * 8 + laneC + 1) * utf_stride + tj8 * 8 + laneR]);
                    CLS_MMA_TF32_OZAKI2(c[tj8][0], c[tj8][1], c[tj8][2], c[tj8][3],
                                        a0, a1, a2, a3, b0, b1);
#else
                    const unsigned b0 = __float_as_uint(
                        Utf[(kc * 8 + laneC + 0) * utf_stride + tj8 * 8 + laneR]);
                    const unsigned b1 = __float_as_uint(
                        Utf[(kc * 8 + laneC + 1) * utf_stride + tj8 * 8 + laneR]);
                    CLS_MMA_TF32_M16N8K8(c[tj8][0], c[tj8][1], c[tj8][2], c[tj8][3],
                                         a0, a1, a2, a3, b0, b1);
#endif
                }
            }
            // (c) Drain accumulators straight into F with uc bounds checks.
            //     Per-lane output positions (laneR, laneC) :
            //       c[tj8][0] → F[r_top, col0],  c[tj8][1] → F[r_top, col1]
            //       c[tj8][2] → F[r_bot, col0],  c[tj8][3] → F[r_bot, col1]
            #pragma unroll
            for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                if (tj8 >= n_tiles) break;
                const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
                store_c(r_top, col0, c[tj8][0]);
                store_c(r_top, col1, c[tj8][1]);
                store_c(r_bot, col0, c[tj8][2]);
                store_c(r_bot, col1, c[tj8][3]);
            }
        }
        return;
    }

    // Fall-through path for uc_pad > 64 (big-tier front strips wider than NTJ8_MAX × 8). The
    // (ti, tj8, kc) ordering reloads A on every tj8, but the absolute A-reuse savings are
    // smaller relative to the per-tile work at this size, so the simpler form wins.
    for (int ti = warp; ti < m_tiles; ti += nwarp) {
        const int r_top = ti * 16 + laneR;
        const int r_bot = r_top + 8;
        for (int tj8 = 0; tj8 < n_tiles; ++tj8) {
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            for (int kc = 0; kc < k_tiles; ++kc) {
                const float* A_top = &Ltf[(ti * 16 + laneR + 0) * nc_pad + kc * 8 + laneC];
                const float* A_bot = &Ltf[(ti * 16 + laneR + 8) * nc_pad + kc * 8 + laneC];
#ifdef CLS_TF32_OZAKI_TC2
                const Tf32Pair a0 = tf32_ozaki_pair(A_top[0]);
                const Tf32Pair a1 = tf32_ozaki_pair(A_bot[0]);
                const Tf32Pair a2 = tf32_ozaki_pair(A_top[1]);
                const Tf32Pair a3 = tf32_ozaki_pair(A_bot[1]);
                const Tf32Pair b0 = tf32_ozaki_pair(
                    Utf[(kc * 8 + laneC + 0) * utf_stride + tj8 * 8 + laneR]);
                const Tf32Pair b1 = tf32_ozaki_pair(
                    Utf[(kc * 8 + laneC + 1) * utf_stride + tj8 * 8 + laneR]);
                CLS_MMA_TF32_OZAKI2(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
                const unsigned a0 = __float_as_uint(A_top[0]);
                const unsigned a1 = __float_as_uint(A_bot[0]);
                const unsigned a2 = __float_as_uint(A_top[1]);
                const unsigned a3 = __float_as_uint(A_bot[1]);
                const unsigned b0 = __float_as_uint(
                    Utf[(kc * 8 + laneC + 0) * utf_stride + tj8 * 8 + laneR]);
                const unsigned b1 = __float_as_uint(
                    Utf[(kc * 8 + laneC + 1) * utf_stride + tj8 * 8 + laneR]);
                CLS_MMA_TF32_M16N8K8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#endif
            }
            const int col0 = tj8 * 8 + laneC;
            const int col1 = col0 + 1;
            store_c(r_top, col0, c0);
            store_c(r_top, col1, c1);
            store_c(r_bot, col0, c2);
            store_c(r_bot, col1, c3);
        }
    }
}

// Unified BIG kernel for every precision (front too large for the shared-resident small/big
// kernels; stays global-resident). Phase-3 trailing stages the L/U panels into shared once and drains
// the contribution block straight into the parent front (fused trail+extend, baked in — removes the
// uncoalesced CB global write+read round-trip; docs note 30/53). UseTC (T = float, TF32) runs the
// TF32 mma trailing on eligible shapes (nc≤32, uc≤256) and scalar otherwise; FP64/FP32 run the
// staged-scalar trailing. Large fronts are always fsz>kFusedMidFrontMax (no lu_mid_front fast path), so the kernel
// runs panel LU -> U-solve -> trailing in order and the fused drain is unconditional.
// Supersedes factor_big_staged<T> and factor_big_tf32_ptx. The L/U staging (2·nc·uc·sizeof(T)) always
// fits the 96 KB opt-in budget on power-grid Jacobians, so there is no non-staged fallback.
template <typename T, bool UseTC>
__global__ void factor_big(int lbegin, int lend, const int* __restrict__ plcols,
                                   const int* __restrict__ front_off,
                                   const int* __restrict__ front_ptr,
                                   const int* __restrict__ ncols,
                                   const int* __restrict__ panel_parent,
                                   const int* __restrict__ asm_ptr,
                                   const int* __restrict__ asm_local, T* frontB,
                                   long front_total, int* sing, int do_extend,
                                   int level_max_nc, int level_max_uc, int uc_pad_max, int nc_pad_max,
                                   bool static_pivoting, double pivot_threshold,
                                   double pivot_shift)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    T* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    T* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int par = panel_parent[p];

    const bool extend_ok = (par >= 0 && do_extend);
    T* Fp = (par >= 0) ? (front + front_off[par]) : nullptr;
    const int pfsz = (par >= 0) ? (front_ptr[par + 1] - front_ptr[par]) : 0;
    const int abase = asm_ptr[p];

    extern __shared__ char smem_big_unified[];
    bool fused;
    if constexpr (UseTC) {
        // T == float (TF32). TC-eligible shapes stage L/U as float and run the mma trailing.
        float* Ltf = reinterpret_cast<float*>(smem_big_unified);
        float* Utf = Ltf + (long)uc_pad_max * nc_pad_max;
        const bool tc = (nc <= kTensorCorePivotColumnCap && uc <= CLS_TC_UC_CAP);
        fused = (extend_ok && fsz > kFusedMidFrontMax && tc);
        (void)level_max_nc; (void)level_max_uc;
        // Big fronts are always fsz>kFusedMidFrontMax, so no fused Phase 1+3 fast path: panel LU, U-solve, trailing.
        lu_panel_factor<float>(F, fsz, nc, t, nt, sing, static_pivoting,
                               pivot_threshold, pivot_shift);       // Phase 1
        u_panel_solve_fewsync<float>(F, fsz, nc, uc, t, nt);        // Phase 2 (barrier-cut)
        if (!tc)        trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);          // Phase 3
        else if (fused) trailing_update_tf32_tc<true>(F, fsz, nc, uc, Ltf, Utf, t, nt,
                                                      Fp, pfsz, asm_local, abase);
        else            trailing_update_tf32_tc<false>(F, fsz, nc, uc, Ltf, Utf, t, nt);
    } else {
        // uc_pad_max carries the trailing column-tile width: 0 = full 2·nc·uc staging (the common
        // power-grid path, unchanged); >0 = bounded-shared tiled staging for oversized fronts whose
        // full staging would exceed the opt-in budget (large uc — circuit / 2D-FEM separators).
        const int trail_jt = uc_pad_max;
        (void)nc_pad_max;
        fused = (extend_ok && fsz > kFusedMidFrontMax);
        lu_panel_factor<T>(F, fsz, nc, t, nt, sing, static_pivoting,
                           pivot_threshold, pivot_shift);          // Phase 1
        u_panel_solve_fewsync<T>(F, fsz, nc, uc, t, nt);           // Phase 2 (barrier-cut)
        if (trail_jt > 0) {
            T* sh_U = reinterpret_cast<T*>(smem_big_unified);    // nc·trail_jt only
            if (fused) trailing_update_staged_tiled<T, true>(F, fsz, nc, uc, t, nt, sh_U, trail_jt,
                                                             Fp, pfsz, asm_local, abase);
            else       trailing_update_staged_tiled<T>(F, fsz, nc, uc, t, nt, sh_U, trail_jt);
        } else {
            T* sh_L = reinterpret_cast<T*>(smem_big_unified);
            T* sh_U = sh_L + (long)level_max_uc * level_max_nc;
            if (fused) trailing_update_staged<T, true>(F, fsz, nc, uc, t, nt, sh_L, sh_U,  // Phase 3
                                                       Fp, pfsz, asm_local, abase);
            else       trailing_update_staged<T>(F, fsz, nc, uc, t, nt, sh_L, sh_U);
        }
    }

    if (fused || !extend_ok) return;
    __syncthreads();
    extend_add<T, T>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

// ---------------------------------------------------------------------------------------
//  Multi-block BIG tier (FP64) — one front spread across many blocks.
//
//  The single-block factor_big is under-fill bound on big-front matrices: a level can hold only
//  a few large separator fronts (ncu: grid=2 on parabolic_fem), so one block per front leaves the
//  GPU >99% idle while each block grinds a uc²·nc trailing serially. Split the big-tier FP64 work
//  into two launches so the embarrassingly-parallel trailing fans out across the whole GPU:
//    (1) factor_big_panel : one block per front, does Phase-1 panel LU + Phase-2 U-solve only.
//    (2) factor_big_trail : a 3D grid (tile, front, batch) — each block owns one TM×TN tile of the
//        front's uc×uc trailing, reads L/U from global (written by kernel 1, ordered by the graph
//        edge between the two launches), and fuses the contribution straight into the parent.
//  Power-grid fronts never reach this tier, so this path is exercised only by circuit/2D-FEM-like
//  matrices with large separators.
// ---------------------------------------------------------------------------------------
template <typename T>
__global__ void factor_big_panel(int lbegin, int lend, const int* __restrict__ plcols,
                                   const int* __restrict__ front_off,
                                   const int* __restrict__ front_ptr,
                                   const int* __restrict__ ncols, T* frontB, long front_total,
                                   int* sing, bool static_pivoting, double pivot_threshold,
                                   double pivot_shift)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    T* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    T* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    lu_panel_factor<T>(F, fsz, nc, t, nt, sing, static_pivoting, pivot_threshold, pivot_shift);
    u_panel_solve_fewsync<T>(F, fsz, nc, uc, t, nt);
}

// Blocked-LU panel split (replaces the one-block-per-front factor_big_panel, which became the
// bottleneck once the trailing was multi-blocked: it did ~1/20 of the trailing's FLOPs but took
// more time — under-fill + serial nc-loop, ncu grid 18-32 / occ 17% / SM 5-12%). The L21/U12 panel
// solves are parallel over the uc rows/cols, so split into:
//   factor_big_pivot  : factor the nc×nc A11 block only (one block/front, small, serial nc).
//   factor_big_panels : multi-block over uc — L21 = A21·U11⁻¹ (per row) and U12 = L11⁻¹·A12 (per
//                         col), forward-substitution against the staged pivot block. Fills the GPU.
template <typename T>
__global__ void factor_big_pivot(int lbegin, int lend, const int* __restrict__ plcols,
                                   const int* __restrict__ front_off,
                                   const int* __restrict__ front_ptr,
                                   const int* __restrict__ ncols, T* frontB, long front_total,
                                   int* sing, bool static_pivoting, double pivot_threshold,
                                   double pivot_shift)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    T* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    T* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    // No-pivot (or static-shift) LU of the nc×nc top-left block only (rows/cols < nc).
    for (int k = 0; k < nc; ++k) {
        const long diag = (long)k * fsz + k;
        T piv = guarded_pivot(F[diag], static_pivoting, pivot_threshold, pivot_shift, sing, t == 0);
        if (t == 0) F[diag] = piv;
        const T inv = T(1) / piv;
        for (int i = k + 1 + t; i < nc; i += nt) {
            const T lik = F[(long)i * fsz + k] * inv;
            F[(long)i * fsz + k] = lik;
            for (int jj = k + 1; jj < nc; ++jj) F[(long)i * fsz + jj] -= lik * F[(long)k * fsz + jj];
        }
        __syncthreads();
    }
}

template <typename T, int TILE>
__global__ void factor_big_panels(int lbegin, int lend, const int* __restrict__ plcols,
                                    const int* __restrict__ front_off,
                                    const int* __restrict__ front_ptr,
                                    const int* __restrict__ ncols, T* frontB, long front_total,
                                    int level_max_nc)
{
    const int fi = lbegin + blockIdx.y;
    if (fi >= lend) return;
    T* front = frontB + (long)blockIdx.z * front_total;
    const int p = plcols[fi];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    const int uc = fsz - nc;
    const int g = blockIdx.x * TILE + threadIdx.x;     // uc index this thread owns (row & col)
    T* F = front + front_off[p];
    // Stage the factored nc×nc pivot block (L11 unit-lower + U11 upper) into shared.
    extern __shared__ char smem_panels[];
    T* P = reinterpret_cast<T*>(smem_panels);          // nc × nc, row-major
    for (int e = threadIdx.x; e < nc * nc; e += TILE) P[e] = F[(long)(e / nc) * fsz + (e % nc)];
    __syncthreads();
    if (g >= uc) return;
    // L21 row g: forward-subst against U11 (cols), in place. L[k] = (A21[k]-Σ_{j<k}L[j]·U11[j][k])/U11[k][k]
    T* Lrow = F + (long)(nc + g) * fsz;                // F[(nc+g), 0..nc)
    for (int k = 0; k < nc; ++k) {
        T s = Lrow[k];
        for (int j = 0; j < k; ++j) s -= Lrow[j] * P[(long)j * nc + k];
        Lrow[k] = s / P[(long)k * nc + k];
    }
    // U12 col g: forward-subst against L11 (unit lower), in place. U[k] = A12[k]-Σ_{i<k}L11[k][i]·U[i]
    for (int k = 0; k < nc; ++k) {
        T s = F[(long)k * fsz + (nc + g)];
        for (int i = 0; i < k; ++i) s -= P[(long)k * nc + i] * F[(long)i * fsz + (nc + g)];
        F[(long)k * fsz + (nc + g)] = s;
    }
    (void)level_max_nc;
}

template <typename T, int TM, int TN>
__global__ void factor_big_trail(int lbegin, int lend, const int* __restrict__ plcols,
                                   const int* __restrict__ front_off,
                                   const int* __restrict__ front_ptr,
                                   const int* __restrict__ ncols,
                                   const int* __restrict__ panel_parent,
                                   const int* __restrict__ asm_ptr,
                                   const int* __restrict__ asm_local, T* frontB, long front_total,
                                   int do_extend, int level_max_nc)
{
    const int fi = lbegin + blockIdx.y;
    if (fi >= lend) return;
    T* front = frontB + (long)blockIdx.z * front_total;
    const int p = plcols[fi];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    const int uc = fsz - nc;
    const int ntx = (uc + TM - 1) / TM, nty = (uc + TN - 1) / TN;
    if (blockIdx.x >= ntx * nty) return;                 // extra tiles for smaller fronts: drop
    const int ti = blockIdx.x / nty, tj = blockIdx.x % nty;
    const int i0 = ti * TM, j0 = tj * TN;
    T* F = front + front_off[p];
    const int par = panel_parent[p];
    const bool fused = (par >= 0 && do_extend && fsz > kFusedMidFrontMax);
    T* Fp = (par >= 0) ? (front + front_off[par]) : nullptr;
    const int pfsz = (par >= 0) ? (front_ptr[par + 1] - front_ptr[par]) : 0;
    const int abase = asm_ptr[p];
    const int t = threadIdx.x, nt = blockDim.x;

    // Stage this tile's L rows (TM×nc) and U cols (nc×TN) into shared for reuse across the nc dot.
    extern __shared__ char smem_trail[];
    T* sL = reinterpret_cast<T*>(smem_trail);            // TM × nc
    T* sU = sL + (long)TM * level_max_nc;                // nc × TN
    for (int e = t; e < TM * nc; e += nt) {
        const int i = e / nc, k = e % nc, gi = i0 + i;
        sL[i * nc + k] = (gi < uc) ? F[(long)(nc + gi) * fsz + k] : T(0);
    }
    for (int e = t; e < nc * TN; e += nt) {
        const int k = e / TN, j = e % TN, gj = j0 + j;
        sU[k * TN + j] = (gj < uc) ? F[(long)k * fsz + (nc + gj)] : T(0);
    }
    __syncthreads();
    for (int e = t; e < TM * TN; e += nt) {
        const int i = e / TN, j = e % TN, gi = i0 + i, gj = j0 + j;
        if (gi >= uc || gj >= uc) continue;
        T acc = T(0);
        for (int k = 0; k < nc; ++k) acc += sL[i * nc + k] * sU[k * TN + j];
        const long off = (long)(nc + gi) * fsz + (nc + gj);
        if (fused) {
            atomicAdd(&Fp[(long)asm_local[abase + gi] * pfsz + asm_local[abase + gj]],
                      F[off] - acc);
        } else {
            F[off] -= acc;
        }
    }
}

// BIG tier — the single global-resident kernel for every precision (fronts too large to be
// shared-resident at all: fsz > whole_front_shared_max). factor_big keeps the front in global memory,
// stages only the L/U panels into shared, runs the trailing (TF32 mma for UseTC / staged scalar for
// FP64/FP32), and drains the contribution block straight into the parent (fused, baked in). The L/U
// staging always fits the opt-in budget on power-grid Jacobians, so there is no non-staged fallback.
// Routed deterministically by tier — no occupancy/opt-in gate.
static void dispatch_factor_big(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                  int b, int e, const int* d_plc, const int* h_plc,
                                  const FrontRangeCaps& caps)
{
    const auto& [max_fsz, max_uc, level_max_nc, level_max_uc] = caps;
    const Precision precision = st.precision;
    constexpr int do_extend = kFactorDoExtend;
    (void)max_fsz; (void)h_plc;
    dim3 grid(e - b, st.batch_count);

    // Blocked-LU big tier (all precisions): pivot(nc×nc, per-front) → L21/U12 panels(multi-block)
    // → trailing(multi-block). Graph edges order the three launches. Only the small pivot block is
    // per-front; everything else fills the GPU. (uc index space uses level_max_uc, unclamped.)
    constexpr int PT = 256;  // panel-solve tile width
    const int p_ntx = (level_max_uc + PT - 1) / PT;
    const dim3 pgrid(p_ntx > 0 ? p_ntx : 1, e - b, st.batch_count);
    if (precision == Precision::FP64) {
        factor_big_pivot<double><<<grid, 64, 0, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols, st.d_front_batch,
            plan.front_total, st.d_sing, st.static_pivoting, st.pivot_threshold, st.pivot_shift);
        factor_big_panels<double, PT><<<pgrid, PT, (size_t)level_max_nc * level_max_nc * sizeof(double), stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols, st.d_front_batch,
            plan.front_total, level_max_nc);
        constexpr int TM = 32, TN = 32;                          // Phase 3: TM×TN trailing tiles
        // grid.x must cover the BIGST front's uc² (level_max_uc is unclamped; max_uc is clamped to
        // the TF32 tile cap and would silently drop tiles beyond it → wrong result).
        const int ntx = (level_max_uc + TM - 1) / TM, nty = (level_max_uc + TN - 1) / TN;
        const dim3 tgrid(ntx * nty > 0 ? ntx * nty : 1, e - b, st.batch_count);
        const size_t sh = (size_t)(TM + TN) * level_max_nc * sizeof(double);  // ≤ 48KB (nc≤64)
        factor_big_trail<double, TM, TN><<<tgrid, 256, sh, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch,
            plan.front_total, do_extend, level_max_nc);
        return;
    }
    // FP32 and TF32: float fronts → the same multi-block scalar-float big-tier path. The TF32 TC
    // trailing lives in the big tier (fsz≤111, its real payoff); for genuinely large fronts the
    // multi-block scalar path fills the GPU at full FP32 accuracy. This also fixes the prior
    // shared-budget faults on big fronts (FP32 fit the full 2·nc·uc staging only by the luck of
    // 4-byte elements; TF32 staged the per-front uc into max_uc-clamped shared and crashed).
    (void)max_uc;
    factor_big_pivot<float><<<grid, 64, 0, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols, st.d_front_batch_f,
        plan.front_total, st.d_sing, st.static_pivoting, st.pivot_threshold, st.pivot_shift);
    factor_big_panels<float, PT><<<pgrid, PT, (size_t)level_max_nc * level_max_nc * sizeof(float), stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols, st.d_front_batch_f,
        plan.front_total, level_max_nc);
    constexpr int TM = 32, TN = 32;                        // Phase 3: TM×TN trailing tiles
    const int ntx = (level_max_uc + TM - 1) / TM, nty = (level_max_uc + TN - 1) / TN;
    const dim3 tgrid(ntx * nty > 0 ? ntx * nty : 1, e - b, st.batch_count);
    const size_t sh = (size_t)(TM + TN) * level_max_nc * sizeof(float);  // ≤ 24KB (nc≤64)
    factor_big_trail<float, TM, TN><<<tgrid, 256, sh, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
        plan.front_total, do_extend, level_max_nc);
}

}  // namespace
}  // namespace custom_linear_solver
