#pragma once

// FACTORIZE — host-side level dispatch.
//
// Internal — included only by multifrontal.cu (single TU; CUDA_SEPARABLE_COMPILATION OFF).
//
// Two layers:
//
//   issue_factor_levels        — outer dispatcher. Walks the panel-etree level-by-level
//                                (`plan.num_plevels`). When the plan exposes independent
//                                subtrees, each subtree's pre-spine levels are enqueued on
//                                its own CUDA stream and joined into the main stream before
//                                the spine — a fork/wait/join pattern that lets independent
//                                subtree work overlap on the GPU. With one subtree (or with
//                                multistream disabled) it reduces to a single sequential
//                                level walk on the main stream.
//
//   issue_factor_level_range   — per-range dispatcher. For one (sub)range of panels at one
//                                level it (1) scans the level to find max_fsz / max_uc /
//                                level_max_nc, (2) picks a tier (small / mid / big) by
//                                max_fsz against SMALL_THRESH=32 and MID_THRESH=128, then
//                                (3) selects the kernel variant for the active Precision and
//                                computes the dynamic shared-memory size. The mid tier falls
//                                through to the big tier when the shared budget overflows
//                                the 96 KB per-block cap on sm_86.

#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "multifrontal.hpp"
#include "factorize/kernels.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

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

static int factor_max_block()
{
    static const int v = [] {
        int dev = 0; cudaGetDevice(&dev);
        int mt = 1024;
        cudaDeviceGetAttribute(&mt, cudaDevAttrMaxThreadsPerBlock, dev);
        return mt;
    }();
    return v;
}

static long factor_warp_fill();  // defined below (used by the small-tier sub-group gate)

// Launch helper: instantiates factor_small for a concrete (front type, sub-group size). Keeps
// the runtime SG switch in issue_factor_level_range compact.
template <typename FT, int SG>
static inline void launch_factor_small(int gx, int blk, size_t shb, cudaStream_t stream,
                                       int b, int level_size, int B, int fsz2cap,
                                       const MultifrontalPlan& plan, const int* d_plc,
                                       FT* frontB, int* sing, int do_extend)
{
    factor_small<FT, SG><<<gx, blk, shb, stream>>>(
        b, level_size, B, fsz2cap, d_plc, plan.d_front_off, plan.d_front_ptr,
        plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, frontB,
        plan.front_total, sing, do_extend);
}

// Pick the small-tier sub-group size: SG ∈ {8,16,32}; packing (SG<32) only applies once the
// packed grid still fills the GPU (mirrors solve/dispatch.cuh solve_small_sg).
static int factor_small_sg(int max_fsz, long warps_unpacked)
{
#ifdef CLS_SMALL_SG32_ONLY
    (void)max_fsz; (void)warps_unpacked;
    return 32;
#else
    int sg = (max_fsz <= 8) ? 8 : (max_fsz <= 16 ? 16 : 32);
    if (warps_unpacked / (32 / sg) < factor_warp_fill()) sg = 32;
    return sg;
#endif
}

// Launch the small kernel with (front type fixed, SG resolved at the call site).
template <typename FT>
static inline void launch_factor_small_t(int SG, int gx, int blk, size_t shb, cudaStream_t stream,
                                         int b, int level_size, int B, int fsz2cap,
                                         const MultifrontalPlan& plan, const int* d_plc,
                                         FT* frontB, int* sing, int do_extend)
{
    if (SG == 8)       launch_factor_small<FT, 8>(gx, blk, shb, stream, b, level_size, B, fsz2cap, plan, d_plc, frontB, sing, do_extend);
    else if (SG == 16) launch_factor_small<FT, 16>(gx, blk, shb, stream, b, level_size, B, fsz2cap, plan, d_plc, frontB, sing, do_extend);
    else               launch_factor_small<FT, 32>(gx, blk, shb, stream, b, level_size, B, fsz2cap, plan, d_plc, frontB, sing, do_extend);
}

// ---------------------------------------------------------------------------------------
// Per-range dispatcher: pick the right kernel for fronts in plcols[b..e), launch on `stream`.
//
// Flow:
//   1. Scan plcols[b..e) once to find this level's max_fsz / max_uc / level_max_{nc, uc}.
//      The kernels need these caps for shared-layout sizing and for the WMMA tile bounds.
//   2. Tier selection by max_fsz:
//        max_fsz ≤ SMALL_THRESH (=32)  → small tier (one warp per (front, batch))
//        max_fsz ≤ MID_THRESH   (=128) → mid tier   (one block per (front, batch), shared-resident)
//        otherwise                    → big tier   (one block per (front, batch), global-resident)
//   3. Variant selection by Precision (within tier).
//   4. Shared-memory budget check. The mid tier may overflow the 96 KB sm_86 cap for the
//      largest (fsz, kp_max, mblk) triples; in that case it falls through to the big tier.
static void issue_factor_level_range(const MultifrontalPlan& plan, State& st,
                                     cudaStream_t stream, int b, int e,
                                     const int* d_plc, const int* h_plc)
{
    const Precision prec = st.prec;
    const int B = st.B;
    constexpr int do_extend = 1;
    constexpr int SMALL_THRESH = 32;
    constexpr int SMALL_WARPS = 8;
    constexpr int MID_THRESH = 128;
    constexpr size_t MID_SHARED_BUDGET = 96 * 1024;  // sm_86 max dynamic shared per block
    if (e <= b) return;
    const int level_size = e - b;

    // (1) one host-side pass over the panels in this range to collect dispatch caps.
    int max_fsz = 0, max_uc = 1, level_max_nc = 1, level_max_uc = 1;
    for (int q = b; q < e; ++q) {
        const int pp = h_plc[q];
        const int fsz = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
        const int nc = plan.h_ncols[pp];
        const int uc = fsz - nc;
        if (fsz > max_fsz) max_fsz = fsz;
        if (uc > max_uc && uc <= 256) max_uc = uc;
        if (nc > level_max_nc) level_max_nc = nc;
        if (uc > level_max_uc) level_max_uc = uc;
    }

    // ----- SMALL tier: sub-group-packed kernel -------------------------------------------
    // One sub-group of SG lanes owns one front; FPW = 32/SG fronts pack per warp, SMALL_WARPS
    // warps per block. SG ≈ fsz (from the level's max_fsz) so the dominant tiny (fsz ≤ 16)
    // ranges run 2 fronts/warp with all 16 lanes busy (vs 1 front using ~half a 32-lane warp),
    // raising lane utilisation and per-warp memory-level parallelism on this latency-bound
    // tier. Packing halves/quarters the warp count, so it only pays once the packed grid still
    // fills the GPU — gate on factor_warp_fill() (the same HW occupancy quantity the tier-split
    // uses); otherwise keep the classic one-warp-per-front form (SG=32). Composes with the
    // tier-split: a size-homogeneous small sub-range gets the tightest SG.
    if (max_fsz <= SMALL_THRESH) {
        const long warps_un = (long)level_size * B;        // unpacked (one-warp-per-front) count
        const int SG = factor_small_sg(max_fsz, warps_un), FPW = 32 / SG;
        const int blk = SMALL_WARPS * 32;
        const int gx = (int)(((warps_un + FPW - 1) / FPW + SMALL_WARPS - 1) / SMALL_WARPS);
        const int fsz2cap = max_fsz * max_fsz;
        const size_t elt = (prec == Precision::FP64) ? sizeof(double) : sizeof(float);
        const size_t shb = (size_t)SMALL_WARPS * FPW * fsz2cap * elt;
        if (prec == Precision::FP64)
            launch_factor_small_t<double>(SG, gx, blk, shb, stream, b, level_size, B, fsz2cap,
                                          plan, d_plc, st.d_frontB, st.d_sing, do_extend);
        else
            launch_factor_small_t<float>(SG, gx, blk, shb, stream, b, level_size, B, fsz2cap,
                                         plan, d_plc, st.d_frontBf, st.d_sing, do_extend);
        return;
    }

    // ----- MID tier: shared-resident kernel if the smem budget fits ----------------------
    // Grid is (level_size, B): blockIdx.x indexes the panel within the level, blockIdx.y the
    // batch. Per-variant shared layouts and block sizes follow below.
    if (max_fsz <= MID_THRESH) {
        const int fsz_cap = max_fsz;
        const int ucp_max = round_up_int(max_uc, 16);
        dim3 grid(level_size, B);

        if (prec == Precision::FP16) {
            // FP16 PTX mma.m16n8k8: KP = round_up(level_max_nc, 8), capped at 32 (nc ≤ 32 by
            // the Phase-3 fallback gate). __half staging panels, no Csc (the inline-asm mma
            // drains to per-lane registers). Block size scales with fsz so small mid fronts
            // do not pay the full 256-thread launch cost.
            const int mblk = max_fsz <= 48 ? 64 : (max_fsz <= 80 ? 128 : 256);
            const int kp_max = round_up_int(std::min(level_max_nc, 32), 8);
            const size_t shb = (size_t)fsz_cap * fsz_cap * sizeof(float)        // Fs
                             + (size_t)2 * ucp_max * kp_max * sizeof(__half);   // Lh + Uh
            if (shb <= MID_SHARED_BUDGET) {
                factor_mid_fp16_ptx<<<grid, mblk, shb, stream>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                    plan.front_total, st.d_sing, do_extend, ucp_max, kp_max, fsz_cap);
                return;
            }
        } else if (prec == Precision::TF32) {
            // TF32 PTX hybrid. Picks K=4 vs K=8 per level by the K-padding heuristic
            //
            //     use_k4 ↔ round_up(nc, 4) < round_up(nc, 8)
            //            ↔ level_max_nc % 8 ∈ {1..4}
            //
            // K=4 cuts the staging panel and the per-tile FLOP count by ~50% on levels whose
            // nc lies in the lower half of an 8-step; for the rest the K=8 path avoids the
            // extra K-loop iterations a K=4 instruction would pay. Both PTX paths omit Csc
            // (per-lane accumulators write straight to F).
            const int mblk = max_fsz <= 48 ? 64 : (max_fsz <= 80 ? 128 : 256);
            const int level_nc_clipped = std::min(level_max_nc, 32);
            const int kp_k4 = round_up_int(level_nc_clipped, 4);
            const int kp_k8 = round_up_int(level_nc_clipped, 8);
            const bool use_k4 = (kp_k4 < kp_k8);
            const int kp_max = use_k4 ? kp_k4 : kp_k8;
            const size_t shb = (size_t)fsz_cap * fsz_cap * sizeof(float)        // Fs
                             + (size_t)2 * ucp_max * kp_max * sizeof(float);    // Ltf + Utf
            if (shb <= MID_SHARED_BUDGET) {
                if (use_k4) {
                    factor_mid_tf32_ptx<4><<<grid, mblk, shb, stream>>>(
                        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                        plan.front_total, st.d_sing, do_extend, ucp_max, kp_max, fsz_cap);
                } else {
                    factor_mid_tf32_ptx<8><<<grid, mblk, shb, stream>>>(
                        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                        plan.front_total, st.d_sing, do_extend, ucp_max, kp_max, fsz_cap);
                }
                return;
            }
        } else {
            // FP32 / FP64 staged-scalar mid kernel. Shared layout sized by the level cap of
            // (nc, uc) rather than by single-panel round-ups since this kernel reuses sh_L /
            // sh_U as compact L / U strips (not WMMA-aligned tiles).
            const size_t elt = (prec == Precision::FP64) ? sizeof(double) : sizeof(float);
            const size_t shb_tiled = (size_t)fsz_cap * fsz_cap * elt                  // Fs
                                   + (size_t)2 * level_max_nc * level_max_uc * elt;   // sh_L + sh_U
            // Underfilled levels (block count ≤ num_SMs, e.g. B=1 deep narrow levels) each run a
            // single latency-bound block that uses only a fraction of its SM's warp slots (ncu: 8
            // of 48 warps, 0.3% util). On the otherwise-idle GPU, widening the block to the device
            // max parallelises the per-pivot rank-1 update over more of that SM with no occupancy
            // cost. Once the level fills the GPU with blocks the extra width would only cost
            // occupancy, so keep the default 256. Both the gate (num_SMs) and the wide size
            // (maxThreadsPerBlock) are device attributes, not tuned constants. See docs/05-reports/12.
            const int mblk = ((long)(e - b) * B <= factor_num_sms()) ? factor_max_block() : 256;
            if (shb_tiled <= MID_SHARED_BUDGET) {
                if (prec == Precision::FP64)
                    factor_mid<double><<<grid, mblk, shb_tiled, stream>>>(
                        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB,
                        plan.front_total, st.d_sing, do_extend, fsz_cap, level_max_nc, level_max_uc);
                else
                    factor_mid<float><<<grid, mblk, shb_tiled, stream>>>(
                        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                        plan.front_total, st.d_sing, do_extend, fsz_cap, level_max_nc, level_max_uc);
                return;
            }
        }
        // The mid shared budget did not fit (e.g. FP64 with fsz > ~88, or a wider kp_max with
        // a large fsz_cap). Fall through to the big tier on the same range.
    }

    // ----- BIG tier: global-memory kernel ------------------------------------------------
    dim3 grid(level_size, B);
    if (prec == Precision::FP64) {
        // FP64 uses a smaller block (128) — the scalar trailing on global memory does not
        // amortise the larger occupancy cost of a 1024-thread block at FP64 register pressure.
        constexpr int T = 128;
        factor_big<double><<<grid, T, 0, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB,
            plan.front_total, st.d_sing, do_extend);
        return;
    }
    constexpr int bigT = 1024;
    if (prec == Precision::FP16) {
        // 512-thread block + __launch_bounds__(512, 2) on factor_big_fp16_ptx so two blocks
        // fit per SM on sm_86. Shared scratch is only the __half staging panels (Lh, Uh) —
        // the front stays in global memory and the inline-asm mma needs no Csc readback.
        constexpr int bigT_fp16 = 512;
        const int ucp_max = round_up_int(max_uc, 16);
        const int kp_max = round_up_int(std::min(level_max_nc, 32), 8);
        const size_t shbytes = (size_t)2 * ucp_max * kp_max * sizeof(__half);
        factor_big_fp16_ptx<<<grid, bigT_fp16, shbytes, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
            plan.front_total, st.d_sing, do_extend, ucp_max, kp_max);
        return;
    }
    if (prec == Precision::TF32) {
        // 512-thread block. The matching `__launch_bounds__(512, 2)` on factor_big_tf32_ptx
        // constrains nvcc so two blocks of 512 threads fit per SM on sm_86 (the alternative
        // 1024-thread block would be capped at one resident block by the thread-per-SM limit).
        // PTX path needs no Csc scratch.
        constexpr int bigT_tf32 = 512;
        const int ucp_max = round_up_int(max_uc, 16);
        const int kp_max = round_up_int(std::min(level_max_nc, 32), 8);
        const size_t shbytes = (size_t)2 * ucp_max * kp_max * sizeof(float);
        factor_big_tf32_ptx<<<grid, bigT_tf32, shbytes, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
            plan.front_total, st.d_sing, do_extend, ucp_max, kp_max);
        return;
    }
    // FP32 big: scalar trailing on the global front, no staging.
    factor_big<float><<<grid, bigT, 0, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
        plan.front_total, st.d_sing, do_extend);
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
// Occupancy gate: warp-packing a small-tier range beats the block-per-front kernel only once the
// small fronts × B fill the GPU's warp slots. Below that — notably B=1, the latency regime — the
// block-per-front kernel gives each small front 8× the threads and a mixed range already has enough
// fronts to occupy the device, so keeping the range merged on the larger kernel wins (unsplit
// regressed B=1 factor by up to ~80%). The threshold is a pure hardware quantity (SMs × warps/SM
// via device attrs), so the rule generalizes across matrices and GPUs. See docs/05-reports/11.
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

// Dispatch one (level- or cell-) range described by its (kNumTiers+1) tier boundaries `tb` into
// d_plc/h_plc. Splits into tier-homogeneous sub-launches when the occupancy gate passes, else
// dispatches the whole range on the larger kernel (pre-split behaviour). Same range = independent
// fronts, so the small→mid→big sub-launches are order-free and correct.
static void issue_factor_tiered(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                const int* tb, const int* d_plc, const int* h_plc)
{
    constexpr int NT = MultifrontalPlan::kNumTiers;
    constexpr int NS = MultifrontalPlan::kSmallTiers;
    const long small_cnt = tb[NS] - tb[0];
    const bool mixed = (tb[NT] - tb[0]) > small_cnt;  // a larger-tier front is present
    if (st.tier_split && small_cnt > 0 && mixed && small_cnt * (long)st.B >= factor_warp_fill()) {
        for (int t = 0; t < NT; ++t)
            if (tb[t + 1] > tb[t])
                issue_factor_level_range(plan, st, stream, tb[t], tb[t + 1], d_plc, h_plc);
    } else {
        issue_factor_level_range(plan, st, stream, tb[0], tb[NT], d_plc, h_plc);
    }
}

static void issue_factor_levels(const MultifrontalPlan& plan, State& st, cudaStream_t stream)
{
    constexpr int NT = MultifrontalPlan::kNumTiers;
    const bool use_multistream = st.num_subtree_streams > 1 &&
                                 plan.num_subtrees == st.num_subtree_streams &&
                                 !plan.h_subtree_level_off.empty();
    const int* lvl_tb = plan.h_level_tier_off.data();           // single-stream: per-level ranges
    const int* sub_tb = plan.h_subtree_level_tier_off.data();   // multistream: per-cell ranges
    const bool have_lvl = !plan.h_level_tier_off.empty() && plan.d_plcols_tier;
    const bool have_sub = !plan.h_subtree_level_tier_off.empty();

    if (use_multistream) {
        const int spine_lo = (plan.spine_start_level >= 0) ? plan.spine_start_level : plan.num_plevels;

        // Fork: record an event on the main stream and have each subtree stream wait on it
        // so the subtree work cannot start until any prior main-stream work has retired.
        cudaEventRecord(static_cast<cudaEvent_t>(st.fork_event), stream);
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStreamWaitEvent(static_cast<cudaStream_t>(st.subtree_streams[k]),
                                static_cast<cudaEvent_t>(st.fork_event), 0);
        }

        // Each subtree stream sweeps levels 0..spine_lo-1 over its own panel slice, tier-splitting
        // each (subtree, level) cell under the same occupancy gate as the single-stream walk.
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStream_t ks = static_cast<cudaStream_t>(st.subtree_streams[k]);
            const int* off = plan.h_subtree_level_off.data() + (long)k * plan.num_plevels;
            const int* cnt = plan.h_subtree_level_cnt.data() + (long)k * plan.num_plevels;
            for (int L = 0; L < plan.num_plevels; ++L) {
                if (L >= spine_lo) break;
                if (cnt[L] == 0) continue;
                if (have_sub)
                    issue_factor_tiered(plan, st, ks,
                                        sub_tb + ((long)k * plan.num_plevels + L) * (NT + 1),
                                        plan.d_plcols, plan.h_plcols.data());
                else
                    issue_factor_level_range(plan, st, ks, off[L], off[L] + cnt[L],
                                             plan.d_plcols, plan.h_plcols.data());
            }
            cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
        }

        // Join: main stream waits on every subtree's join event before issuing the spine.
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);
        }

        // Spine levels (if any) on the main stream — cnt=1 chain, nothing to tier-split.
        if (plan.spine_start_level >= 0) {
            for (int L = plan.spine_start_level; L < plan.num_plevels; ++L) {
                const int b = plan.plptr[L], e = plan.plptr[L + 1];
                issue_factor_level_range(plan, st, stream, b, e,
                                         plan.d_plcols, plan.h_plcols.data());
            }
        }
    } else {
        // Single stream: tier-split each level under the occupancy gate (see issue_factor_tiered).
        for (int L = 0; L < plan.num_plevels; ++L) {
            if (have_lvl)
                issue_factor_tiered(plan, st, stream, lvl_tb + (long)L * (NT + 1),
                                    plan.d_plcols_tier, plan.h_plcols_tier.data());
            else
                issue_factor_level_range(plan, st, stream, plan.plptr[L], plan.plptr[L + 1],
                                         plan.d_plcols, plan.h_plcols.data());
        }
    }
}

}  // namespace
}  // namespace custom_linear_solver
