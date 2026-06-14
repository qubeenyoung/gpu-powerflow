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

// Unified BIG kernel for every precision (front too large for the shared-resident mid kernel; stays
// global-resident). Phase-3 trailing stages the L/U panels into shared once and drains the
// contribution block straight into the parent front (fused trail+extend, baked in — removes the
// uncoalesced CB global write+read round-trip; docs note 30/53). UseTC (T = float, TF32) runs the
// TF32 mma trailing on eligible shapes (nc≤32, uc≤256) and scalar otherwise; FP64/FP32 run the
// staged-scalar trailing. Big fronts are always fsz>kFusedSmallFrontMax (no lu_small_front fast path), so the kernel
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
                                   GatherArgs ga)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const long batch_off = (long)blockIdx.y * front_total;
    T* front = frontB + batch_off;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    T* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int par = panel_parent[p];

#ifdef CLS_FACTOR_GATHER
    if (ga.active && !ga.phase_batched) {
        // Fused gather-based assembly directly in the global front (no memset/scatter/extend).
        // (Phase-batched mode skips this: a separate pre-pass already assembled F in global.)
        zero_front<T>(F, fsz * fsz, t, nt);
        __syncthreads();
        if (ga.mode == 1) {
            assemble_outputs<T>(F, ga, blockIdx.y, p, t, nt);
        } else {
            gather_matrix<T>(F, ga, blockIdx.y, p, t, nt);
            gather_children<T>(F, fsz, ga, asm_ptr, asm_local, blockIdx.y, p, t, nt);
        }
        __syncthreads();
    }
    const bool ga_active = ga.active;
#else
    constexpr bool ga_active = false;
#endif

    // Gather path never fuses the CB drain (the parent pulls it from F later) and never extends.
    const bool extend_ok = (par >= 0 && do_extend) && !ga_active;
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
        fused = (extend_ok && fsz > kFusedSmallFrontMax && tc);
        (void)level_max_nc; (void)level_max_uc;
        // Big fronts are always fsz>kFusedSmallFrontMax, so no fused Phase 1+3 fast path: panel LU, U-solve, trailing.
        lu_panel_factor<float>(F, fsz, nc, t, nt, sing);            // Phase 1
        u_panel_solve_fewsync<float>(F, fsz, nc, uc, t, nt);        // Phase 2 (barrier-cut)
        if (!tc)        trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);          // Phase 3
        else if (fused) trailing_update_tf32_tc<true>(F, fsz, nc, uc, Ltf, Utf, t, nt,
                                                      Fp, pfsz, asm_local, abase);
        else            trailing_update_tf32_tc<false>(F, fsz, nc, uc, Ltf, Utf, t, nt);
    } else {
        T* sh_L = reinterpret_cast<T*>(smem_big_unified);
        T* sh_U = sh_L + (long)level_max_uc * level_max_nc;
        fused = (extend_ok && fsz > kFusedSmallFrontMax);
        (void)uc_pad_max; (void)nc_pad_max;
        lu_panel_factor<T>(F, fsz, nc, t, nt, sing);               // Phase 1
        u_panel_solve_fewsync<T>(F, fsz, nc, uc, t, nt);           // Phase 2 (barrier-cut)
        if (fused) trailing_update_staged<T, true>(F, fsz, nc, uc, t, nt, sh_L, sh_U,   // Phase 3
                                                   Fp, pfsz, asm_local, abase);
        else       trailing_update_staged<T>(F, fsz, nc, uc, t, nt, sh_L, sh_U);
    }

#ifdef CLS_FACTOR_GATHER
    if (ga.active) {
        // The CB (Schur) is in F's trailing; copy it to the CB buffer for this front's parent.
        __syncthreads();
        write_cb<T>(ga, blockIdx.y, ga.cb_pos[p], F, fsz, nc, uc, t, nt);
        return;
    }
#endif
    if (fused || !extend_ok) return;
    __syncthreads();
    extend_add<T, T>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

// BIG tier: global-memory kernel. Underfilled levels (level_size × B < num_SMs, e.g. the deep
// levels of a single-system solve) split the scalar path across blocks so the trailing GEMM
// fans out across SMs instead of running one block per front; filled levels keep the fused kernel.
// BIG tier — the single global-resident kernel for every precision (fronts too large for the
// shared-resident mid kernel). factor_big keeps the front in global memory, stages only the L/U
// panels into shared, runs the trailing (TF32 mma for UseTC / staged scalar for FP64/FP32), and
// drains the contribution block straight into the parent (fused, baked in). The L/U staging always
// fits the 96 KB opt-in budget on power-grid Jacobians, so there is no non-staged fallback.
static void dispatch_factor_big(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                int b, int e, const int* d_plc, const int* h_plc,
                                const FrontRangeCaps& caps)
{
    const auto& [max_fsz, max_uc, level_max_nc, level_max_uc] = caps;
    const Precision precision = st.precision;
    constexpr int do_extend = kFactorDoExtend;
    (void)max_fsz; (void)h_plc;
    dim3 grid(e - b, st.batch_count);

    GatherArgs ga = make_gather_args(plan, st);

    if (precision == Precision::FP64) {
        constexpr int T = 128;  // FP64 register pressure caps the block size
        const size_t sh = (size_t)2 * level_max_nc * level_max_uc * sizeof(double);
        factor_big<double, false><<<grid, T, sh, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch,
            plan.front_total, st.d_sing, do_extend, level_max_nc, level_max_uc, 0, 0, ga);
        return;
    }
    if (precision == Precision::TF32) {
        constexpr int bigT = 512;
        const int uc_pad_max = round_up_to_multiple(max_uc, 16);
        const int nc_pad_max = round_up_to_multiple(std::min(level_max_nc, kTensorCorePivotColumnCap), 8);
        const size_t sh = (size_t)(2 * uc_pad_max * nc_pad_max + 4 * nc_pad_max) * sizeof(float);  // +4*nc_pad_max: Utf LDB pad
        factor_big<float, true><<<grid, bigT, sh, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
            plan.front_total, st.d_sing, do_extend, level_max_nc, level_max_uc, uc_pad_max, nc_pad_max, ga);
        return;
    }
    // FP32
    constexpr int bigT = 1024;
    const size_t sh = (size_t)2 * level_max_nc * level_max_uc * sizeof(float);
    factor_big<float, false><<<grid, bigT, sh, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
        plan.front_total, st.d_sing, do_extend, level_max_nc, level_max_uc, 0, 0, ga);
}

}  // namespace
}  // namespace custom_linear_solver
