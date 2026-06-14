#pragma once

// ============================================================================================
// FACTORIZE — TILED-TRAILING path (prototype, 2026-06-12).
//
// Goal 3 (260612_goal.md), generalized from the deprecated big_split_2d to mid+big.
//
// Diagnosis (factorize-bottleneck-3case.md): every mid/big level runs at 1 block/SM (warp
// 25-33%), batch-invariant, because each front is one block (register + whole-front shared/L-U
// staging). At B=1 the under-filled deep levels (few fronts × B < SMs) leave most SMs idle AND
// each resident block is occupancy-starved. The lever (proven by big_split_2d v2: occupancy
// 1->9-10 block/SM, barrier stall 5.34->0.04, usa B=1 1.14x) is to split one front's Phase-3
// trailing across many small thread blocks, each owning one BM×BN output tile and staging only
// its L-rows + U-cols (~8 KB) so many blocks fit per SM.
//
// What this prototype adds over big_split_2d (which was big-only): the dispatch gate
// (schedule.cuh) routes ANY under-filled large level — mid-sized OR big-sized fronts — through
// this path. 13K/25K are mid-dominated with no big tier, so the big-only version never touched
// them. The front always lives in the global front buffer (mid only *stages* it to shared), so
// the same global panel -> tiled trailing structure works for both.
//
// Structure (2-kernel, TF32 only):
//   factor_tiled_panel_tf32  — Phase 1 (panel LU) + Phase 2 (U-solve), 1 block/front, global F.
//   factor_tiled_trail_tf32  — Phase 3 trailing, grid (fronts, B, m_tiles*n_tiles); each block
//                              owns one BM×BN tile of S -= L·U, fused extend-add into the parent.
// Known limitation (big_split_2d README): the per-level 2nd launch + global L/U round-trip is the
// overhead that made the big-only version win only on under-filled single-system cases. The clean
// fix is a cooperative single kernel (grid.sync); this prototype measures the 2-kernel form first
// (with the mid generalization) to decide whether that next step is warranted.
//
// Runtime-gated (no rebuild for A/B): env CLS_TILED_TRAILING=1 enables; CLS_TILED_FILL=N sets the
// under-fill threshold (default = #SMs). Off by default -> identical to production dispatch.
// ============================================================================================

#include <cstdlib>

#include "factorize/front_ops.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// ---- runtime gate (read env once) ----------------------------------------------------------
static bool tiled_trailing_enabled()
{
    static const bool v = [] {
        const char* s = std::getenv("CLS_TILED_TRAILING");
        return s && std::atoi(s) != 0;
    }();
    return v;
}

// Under-fill threshold: a (level × batch) grid smaller than this many blocks leaves SMs idle, so
// the tiled fan-out can help. Default = #SMs (one front per SM is the boundary).
static long tiled_trailing_fill()
{
    static const long v = [] {
        const char* s = std::getenv("CLS_TILED_FILL");
        return s ? std::atol(s) : (long)factor_num_sms();
    }();
    return v;
}

constexpr int kTiledBM = 64;   // output tile rows (M)
constexpr int kTiledBN = 64;   // output tile cols (N)
constexpr int kTiledThreads = 128;

// --- Phase 1 (panel LU) + Phase 2 (U-solve): one block/front, factors L/U in the global front. ---
template <typename T>
__global__ void factor_tiled_panel(int lbegin, int lend, const int* __restrict__ plcols,
        const int* __restrict__ front_off, const int* __restrict__ front_ptr,
        const int* __restrict__ ncols, T* frontB, long front_total, int* sing)
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
    lu_panel_factor<T>(F, fsz, nc, t, nt, sing);   // Phase 1
    u_panel_solve_fewsync<T>(F, fsz, nc, uc, t, nt);   // Phase 2 (barrier-cut)
}

// --- One BM×BN output tile of the TF32 trailing S -= L·U; stages only this tile's L-rows + U-cols.
//     Ported from big_split_2d v2 trailing_tile_tf32 (the version that produced the 1.14x). ---
template <bool FuseExtend>
__device__ __forceinline__ void trailing_tile_tf32(float* F, int fsz, int nc, int uc,
        int r0, int c0, int bm, int cn, float* Lsh, float* Ush, int t, int nt,
        float* Fp, int pfsz, const int* asm_local, int abase)
{
    const int bm_pad = ((bm + 15) / 16) * 16;
    const int cn_pad = ((cn + 7) / 8) * 8;
    const int nc_pad = ((nc + 7) / 8) * 8;
    const int us = cn_pad + 4;                       // Ush column stride (+4 anti bank-conflict)
    for (int e = t; e < bm_pad * nc_pad; e += nt) {
        const int i = e / nc_pad, k = e % nc_pad;
        Lsh[e] = (i < bm && k < nc) ? F[(long)(nc + r0 + i) * fsz + k] : 0.0f;
    }
    for (int e = t; e < nc_pad * cn_pad; e += nt) {
        const int k = e / cn_pad, j = e % cn_pad;
        Ush[k * us + j] = (k < nc && j < cn) ? F[(long)k * fsz + (nc + c0 + j)] : 0.0f;
    }
    __syncthreads();
    const int m_tiles = bm_pad / 16, n_tiles = cn_pad / 8, k_tiles = nc_pad / 8;
    const int warp = t >> 5, nwarp = nt >> 5, lane = t & 31;
    const int laneR = lane >> 2, laneC = (lane & 3) * 2;
    auto store_c = [&](int r, int col, float c) {
        if (r >= bm || col >= cn) return;
        const long off = (long)(nc + r0 + r) * fsz + (nc + c0 + col);
        if constexpr (FuseExtend)
            atomicAdd(&Fp[(long)asm_local[abase + r0 + r] * pfsz + asm_local[abase + c0 + col]], F[off] - c);
        else
            F[off] -= c;
    };
    for (int ti = warp; ti < m_tiles; ti += nwarp) {
        const int r_top = ti * 16 + laneR, r_bot = r_top + 8;
        for (int tj8 = 0; tj8 < n_tiles; ++tj8) {
            float c0a = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            for (int kc = 0; kc < k_tiles; ++kc) {
                const float* A_top = &Lsh[(ti * 16 + laneR + 0) * nc_pad + kc * 8 + laneC];
                const float* A_bot = &Lsh[(ti * 16 + laneR + 8) * nc_pad + kc * 8 + laneC];
#ifdef CLS_TF32_OZAKI_TC2
                const Tf32Pair a0 = tf32_ozaki_pair(A_top[0]); const Tf32Pair a1 = tf32_ozaki_pair(A_bot[0]);
                const Tf32Pair a2 = tf32_ozaki_pair(A_top[1]); const Tf32Pair a3 = tf32_ozaki_pair(A_bot[1]);
                const Tf32Pair b0 = tf32_ozaki_pair(Ush[(kc * 8 + laneC + 0) * us + tj8 * 8 + laneR]);
                const Tf32Pair b1 = tf32_ozaki_pair(Ush[(kc * 8 + laneC + 1) * us + tj8 * 8 + laneR]);
                CLS_MMA_TF32_OZAKI2(c0a, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
                const unsigned a0 = __float_as_uint(A_top[0]); const unsigned a1 = __float_as_uint(A_bot[0]);
                const unsigned a2 = __float_as_uint(A_top[1]); const unsigned a3 = __float_as_uint(A_bot[1]);
                const unsigned b0 = __float_as_uint(Ush[(kc * 8 + laneC + 0) * us + tj8 * 8 + laneR]);
                const unsigned b1 = __float_as_uint(Ush[(kc * 8 + laneC + 1) * us + tj8 * 8 + laneR]);
                CLS_MMA_TF32_M16N8K8(c0a, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#endif
            }
            const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
            store_c(r_top, col0, c0a); store_c(r_top, col1, c1);
            store_c(r_bot, col0, c2);  store_c(r_bot, col1, c3);
        }
    }
}

// --- Phase 3: each block owns one (mt, ntile) output tile. Non-TC fronts (uc>cap / nc>cap) run
//     scalar trailing + extend-add drain once (blockIdx.z==0); TC fronts tile + fused drain. ---
__global__ void factor_tiled_trail_tf32(int lbegin, int lend, const int* __restrict__ plcols,
        const int* __restrict__ front_off, const int* __restrict__ front_ptr,
        const int* __restrict__ ncols, const int* __restrict__ panel_parent,
        const int* __restrict__ asm_ptr, const int* __restrict__ asm_local,
        float* frontB, long front_total, int do_extend, int n_tiles_max, int nc_pad)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int par = panel_parent[p];
    const bool extend_ok = (par >= 0 && do_extend);
    float* Fp = (par >= 0) ? (front + front_off[par]) : nullptr;
    const int pfsz = (par >= 0) ? (front_ptr[par + 1] - front_ptr[par]) : 0;
    const int abase = asm_ptr[p];

    const bool tc = (nc <= kTensorCorePivotColumnCap && uc <= CLS_TC_UC_CAP);
    if (!tc) {   // uc>cap or nc>cap: scalar trailing in one block + extend-add drain (must drain).
        if (blockIdx.z == 0) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
            if (extend_ok) { __syncthreads(); extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt); }
        }
        return;
    }
    const int mt = blockIdx.z / n_tiles_max, ntile = blockIdx.z % n_tiles_max;
    const int r0 = mt * kTiledBM, c0 = ntile * kTiledBN;
    if (r0 >= uc || c0 >= uc) return;                // tile outside this front
    const int bm = min(kTiledBM, uc - r0), cn = min(kTiledBN, uc - c0);
    extern __shared__ char smem_tiled[];
    float* Lsh = reinterpret_cast<float*>(smem_tiled);
    float* Ush = Lsh + kTiledBM * nc_pad;
    if (extend_ok)
        trailing_tile_tf32<true>(F, fsz, nc, uc, r0, c0, bm, cn, Lsh, Ush, t, nt, Fp, pfsz, asm_local, abase);
    else
        trailing_tile_tf32<false>(F, fsz, nc, uc, r0, c0, bm, cn, Lsh, Ush, t, nt, nullptr, 0, nullptr, 0);
}

// --- dispatch: panel kernel (1 block/front) then tiled trailing (fan-out across tiles × SMs). ---
// Caller (schedule.cuh) decides eligibility (enabled + TF32 + under-filled). Returns void; always
// handles the whole [b,e) range. The panel block size follows the big-tier TF32 choice (512).
static void dispatch_factor_tiled_tf32(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                       int b, int e, const int* d_plc, const FrontRangeCaps& caps)
{
    const int max_uc = caps.max_uc;                  // clamped to CLS_TC_UC_CAP in scan_front_range
    const int nc_pad_max = round_up_to_multiple(std::min(caps.level_max_nc, kTensorCorePivotColumnCap), 8);
    constexpr int do_extend = kFactorDoExtend;
    constexpr int panelT = 512;

    // Phase 1+2: panel LU + U-solve, one block/front, into the global front buffer.
    dim3 pgrid(e - b, st.batch_count);
    factor_tiled_panel<float><<<pgrid, panelT, 0, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        st.d_front_batch_f, plan.front_total, st.d_sing);

    // Phase 3: tiled trailing. grid.z spans the front's BM×BN tile grid (sized by the level max uc).
    const int m_tiles_max = (max_uc + kTiledBM - 1) / kTiledBM;
    const int n_tiles_max = (max_uc + kTiledBN - 1) / kTiledBN;
    const size_t sh_tile = (size_t)(kTiledBM * nc_pad_max + nc_pad_max * (kTiledBN + 4)) * sizeof(float);
    dim3 tgrid(e - b, st.batch_count, (unsigned)(m_tiles_max * n_tiles_max));
    factor_tiled_trail_tf32<<<tgrid, kTiledThreads, sh_tile, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
        plan.front_total, do_extend, n_tiles_max, nc_pad_max);
}

// Eligibility test used by the dispatcher: tiled path on only when enabled, TF32, and the level is
// under-filled (level_size × B < threshold). Off by default -> production dispatch unchanged.
static bool tiled_trailing_eligible(const State& st, int level_size)
{
    return tiled_trailing_enabled()
        && st.precision == Precision::TF32
        && (long)level_size * (long)st.batch_count < tiled_trailing_fill();
}

}  // namespace
}  // namespace custom_linear_solver
