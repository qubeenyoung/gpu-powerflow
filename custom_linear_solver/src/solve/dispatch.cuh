#pragma once

// SOLVE — host-side level dispatch.
//
// Internal — included only by multifrontal.cu (single TU; CUDA_SEPARABLE_COMPILATION OFF).
//
// Two layers of dispatch:
//
//   issue_solve_levels        — outer driver. Forward pass leaves→root, then backward
//                                pass root→leaves. Two execution modes:
//                                  - single stream: walks every level on one stream.
//                                  - multi-stream: forks per-subtree work onto K extra
//                                    streams (the same st.subtree_streams used by factor)
//                                    for the forward sweep, joins, re-forks for the
//                                    backward sweep, joins again.
//
//   per-level routing         — within each pass, levels with max_fsz ≤ kSmallFrontMax go to the
//                                warp-packed small kernel (8 warps/block, one (front,batch)
//                                per warp); larger levels go to the block-per-front kernel
//                                with thread count tuned to the level's max_fsz.

#include <cuda_runtime.h>

#include "level_metrics.hpp"
#include "multifrontal.hpp"
#include "solve/kernels.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// HW warp-slot count (SMs × warps/SM) — the GPU-fill quantity used by the small-tier sub-group
// gate (mirrors factorize/dispatch.cuh factor_warp_fill()).
static long solve_warp_fill()
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

// Small-tier sub-group size: SG ∈ {8,16,32}; packing (SG<32) only applies once the packed grid
// still fills the GPU (mirrors factorize/dispatch.cuh). Shared by the fwd and bwd small launches.
static int solve_small_sg(int max_fsz, long warps_unpacked)
{
#ifdef CLS_SMALL_SG32_ONLY
    (void)max_fsz; (void)warps_unpacked;
    return 32;
#else
    int sg = (max_fsz <= 8) ? 8 : (max_fsz <= 16 ? 16 : 32);
    if (warps_unpacked / (32 / sg) < solve_warp_fill()) sg = 32;
    return sg;
#endif
}

// One launch of a sub-group-packed small solve kernel (forward or backward), with the
// (front type, SG) resolved at the call site. solve_fwd_small / solve_bwd_small share a
// signature; `fwd` selects which, `slab` is the per-sub-group shared slab.
template <typename FT, int SG>
static inline void launch_solve_small(bool fwd, int gx, int blk, size_t shb, cudaStream_t ks,
                                      int b, int level_size, int B, int slab,
                                      const MultifrontalPlan& plan, const int* d_plc,
                                      FT* frontB, FT* yB)
{
    if (fwd)
        solve_fwd_small<FT, SG><<<gx, blk, shb, ks>>>(
            b, level_size, B, slab, d_plc, plan.d_front_off, plan.d_front_ptr,
            plan.d_ncols, plan.d_front_rows, frontB, yB, plan.front_total, plan.num_rows);
    else
        solve_bwd_small<FT, SG><<<gx, blk, shb, ks>>>(
            b, level_size, B, slab, d_plc, plan.d_front_off, plan.d_front_ptr,
            plan.d_ncols, plan.d_front_rows, frontB, yB, plan.front_total, plan.num_rows);
}

template <typename FT>
static inline void launch_solve_small_t(bool fwd, int SG, int gx, int blk, size_t shb,
                                        cudaStream_t ks, int b, int level_size, int B, int slab,
                                        const MultifrontalPlan& plan, const int* d_plc,
                                        FT* frontB, FT* yB)
{
    if (SG == 8)       launch_solve_small<FT, 8>(fwd, gx, blk, shb, ks, b, level_size, B, slab, plan, d_plc, frontB, yB);
    else if (SG == 16) launch_solve_small<FT, 16>(fwd, gx, blk, shb, ks, b, level_size, B, slab, plan, d_plc, frontB, yB);
    else               launch_solve_small<FT, 32>(fwd, gx, blk, shb, ks, b, level_size, B, slab, plan, d_plc, frontB, yB);
}

static void issue_solve_levels(const MultifrontalPlan& plan, State& st, cudaStream_t stream)
{
    const bool float_front = is_fp32_front(st.precision);
    const int B = st.batch_count;
    const int selt = float_front ? (int)sizeof(float) : (int)sizeof(double);

    // Per-level forward dispatch on a caller-chosen stream over a plcols (sub)range.
    auto fwd_level = [&](cudaStream_t ks, int b, int e, const int* d_plc, const int* h_plc) {
        if (e <= b) return;
        const int mfsz = scan_level_metrics(plan, h_plc, b, e).max_fsz;

        // Small tier: sub-group-packed warp kernel (FPW = kWarpSize/SG fronts per warp).
        if (classify_front_tier(mfsz) == FrontTier::kSmall) {
            const long warps_un = (long)(e - b) * B;
            const int SG = solve_small_sg(mfsz, warps_un), FPW = kWarpSize / SG;
            const int blk = kSmallTierWarpsPerBlock * kWarpSize;
            const int gx = (int)(((warps_un + FPW - 1) / FPW + kSmallTierWarpsPerBlock - 1) / kSmallTierWarpsPerBlock);
            const size_t shb = (size_t)kSmallTierWarpsPerBlock * FPW * kMaxPivotColumns * selt;
            if (float_front)
                launch_solve_small_t<float>(/*fwd=*/true, SG, gx, blk, shb, ks, b, e - b, B, kMaxPivotColumns,
                                            plan, d_plc, st.d_front_batch_f, st.d_y_batch_f);
            else
                launch_solve_small_t<double>(/*fwd=*/true, SG, gx, blk, shb, ks, b, e - b, B, kMaxPivotColumns,
                                             plan, d_plc, st.d_front_batch, st.d_y_batch);
            return;
        }

        // Larger levels: one block per (front, batch), thread count tuned to max_fsz.
        const int tsb = mfsz <= 64 ? 64 : (mfsz <= 128 ? 128 : 256);
        const dim3 fg(e - b, B);
        if (float_front)
            solve_fwd<float><<<fg, tsb, 0, ks>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
        else
            solve_fwd<double><<<fg, tsb, 0, ks>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
    };

    auto bwd_level = [&](cudaStream_t ks, int b, int e, const int* d_plc, const int* h_plc) {
        if (e <= b) return;
        const LevelMetrics metrics = scan_level_metrics(plan, h_plc, b, e);
        const int mfsz = metrics.max_fsz;
        const int max_cb = metrics.level_max_uc;

        // Small tier: sub-group-packed warp kernel. slab = kMaxPivotColumns (rhs) + max_cb (x cache).
        if (classify_front_tier(mfsz) == FrontTier::kSmall) {
            const long warps_un = (long)(e - b) * B;
            const int SG = solve_small_sg(mfsz, warps_un), FPW = kWarpSize / SG;
            const int blk = kSmallTierWarpsPerBlock * kWarpSize;
            const int gx = (int)(((warps_un + FPW - 1) / FPW + kSmallTierWarpsPerBlock - 1) / kSmallTierWarpsPerBlock);
            const int slab = kMaxPivotColumns + max_cb;
            const size_t shb = (size_t)kSmallTierWarpsPerBlock * FPW * slab * selt;
            if (float_front)
                launch_solve_small_t<float>(/*fwd=*/false, SG, gx, blk, shb, ks, b, e - b, B, slab,
                                            plan, d_plc, st.d_front_batch_f, st.d_y_batch_f);
            else
                launch_solve_small_t<double>(/*fwd=*/false, SG, gx, blk, shb, ks, b, e - b, B, slab,
                                             plan, d_plc, st.d_front_batch, st.d_y_batch);
            return;
        }

        // Larger levels: one block per (front, batch); max_cb-sized shared for the x cache.
        const int tsb = mfsz <= 64 ? 64 : (mfsz <= 128 ? 128 : 256);
        const dim3 bg(e - b, B);
        if (float_front)
            solve_bwd<float><<<bg, tsb, (size_t)max_cb * sizeof(float), ks>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
        else
            solve_bwd<double><<<bg, tsb, (size_t)max_cb * sizeof(double), ks>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
    };

    // Occupancy-gated tier split of one (level- or cell-) range, for either sweep. Mirrors
    // src/factorize/dispatch.cuh: split a mixed range into tier-homogeneous sub-launches so small
    // fronts go to the warp-packed solve kernel instead of block-per-front, but only once the small
    // fronts × B fill the GPU's warp slots — below that (B=1 latency regime) block-per-front's extra
    // threads/front win, so keep the range merged. Same range = independent fronts → order-free.
    //
    constexpr int NT = MultifrontalPlan::kNumTiers;
    constexpr int NS = MultifrontalPlan::kSmallTiers;
    auto dispatch_tiered = [&](cudaStream_t ks, const int* tb, const int* d_plc, const int* h_plc,
                               bool fwd) {
        const long small_cnt = tb[NS] - tb[0];
        const bool mixed = (tb[NT] - tb[0]) > small_cnt;
        if (st.tier_split && small_cnt > 0 && mixed && small_cnt * (long)B >= solve_warp_fill()) {
            for (int t = 0; t < NT; ++t)
                if (tb[t + 1] > tb[t]) {
                    if (fwd) fwd_level(ks, tb[t], tb[t + 1], d_plc, h_plc);
                    else     bwd_level(ks, tb[t], tb[t + 1], d_plc, h_plc);
                }
        } else {
            if (fwd) fwd_level(ks, tb[0], tb[NT], d_plc, h_plc);
            else     bwd_level(ks, tb[0], tb[NT], d_plc, h_plc);
        }
    };
    const int* lvl_tb = plan.h_level_tier_off.data();
    const int* sub_tb = plan.h_subtree_level_tier_off.data();
    const bool have_lvl = !plan.h_level_tier_off.empty() && plan.d_plcols_tier;
    const bool have_sub = !plan.h_subtree_level_tier_off.empty();

    // Multi-stream: replicate the factor's subtree-fork pattern. Subtree streams sweep
    // their own panel slices through all levels independently, allowing per-subtree level
    // boundaries to slip past each other (no global barrier per level).
    const bool use_multistream = st.num_subtree_streams > 1 &&
                                 plan.num_subtrees == st.num_subtree_streams &&
                                 !plan.h_subtree_level_off.empty();

    if (use_multistream) {
        const int K = st.num_subtree_streams;
        const int spine_lo = (plan.spine_start_level >= 0) ? plan.spine_start_level
                                                           : plan.num_plevels;

        // ---- Forward fork ----
        cudaEventRecord(static_cast<cudaEvent_t>(st.fork_event), stream);
        for (int k = 0; k < K; ++k)
            cudaStreamWaitEvent(static_cast<cudaStream_t>(st.subtree_streams[k]),
                                static_cast<cudaEvent_t>(st.fork_event), 0);

        // Per-subtree forward sweep over pre-spine levels.
        for (int k = 0; k < K; ++k) {
            cudaStream_t ks = static_cast<cudaStream_t>(st.subtree_streams[k]);
            const int* off = plan.h_subtree_level_off.data() + (long)k * plan.num_plevels;
            const int* cnt = plan.h_subtree_level_cnt.data() + (long)k * plan.num_plevels;
            for (int L = 0; L < spine_lo; ++L) {
                if (cnt[L] == 0) continue;
                if (have_sub)
                    dispatch_tiered(ks, sub_tb + ((long)k * plan.num_plevels + L) * (NT + 1),
                                    plan.d_plcols, plan.h_plcols.data(), /*fwd=*/true);
                else
                    fwd_level(ks, off[L], off[L] + cnt[L], plan.d_plcols, plan.h_plcols.data());
            }
            cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
        }

        // ---- Forward join, spine forward on main, then backward re-fork ----
        for (int k = 0; k < K; ++k)
            cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);

        for (int L = spine_lo; L < plan.num_plevels; ++L)
            fwd_level(stream, plan.panel_level_ptr[L], plan.panel_level_ptr[L + 1],
                      plan.d_plcols, plan.h_plcols.data());

        // Backward spine on main first (root → spine bottom), then re-fork for subtrees.
        for (int L = plan.num_plevels - 1; L >= spine_lo; --L)
            bwd_level(stream, plan.panel_level_ptr[L], plan.panel_level_ptr[L + 1],
                      plan.d_plcols, plan.h_plcols.data());

        cudaEventRecord(static_cast<cudaEvent_t>(st.fork_event), stream);
        for (int k = 0; k < K; ++k)
            cudaStreamWaitEvent(static_cast<cudaStream_t>(st.subtree_streams[k]),
                                static_cast<cudaEvent_t>(st.fork_event), 0);

        // Per-subtree backward sweep over pre-spine levels (top → bottom).
        for (int k = 0; k < K; ++k) {
            cudaStream_t ks = static_cast<cudaStream_t>(st.subtree_streams[k]);
            const int* off = plan.h_subtree_level_off.data() + (long)k * plan.num_plevels;
            const int* cnt = plan.h_subtree_level_cnt.data() + (long)k * plan.num_plevels;
            for (int L = spine_lo - 1; L >= 0; --L) {
                if (cnt[L] == 0) continue;
                if (have_sub)
                    dispatch_tiered(ks, sub_tb + ((long)k * plan.num_plevels + L) * (NT + 1),
                                    plan.d_plcols, plan.h_plcols.data(), /*fwd=*/false);
                else
                    bwd_level(ks, off[L], off[L] + cnt[L], plan.d_plcols, plan.h_plcols.data());
            }
            cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
        }

        for (int k = 0; k < K; ++k)
            cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);
        return;
    }

    // ---- Single-stream: per-level occupancy-gated tier split (forward then backward) ----
    for (int L = 0; L < plan.num_plevels; ++L) {
        if (have_lvl)
            dispatch_tiered(stream, lvl_tb + (long)L * (NT + 1),
                            plan.d_plcols_tier, plan.h_plcols_tier.data(), /*fwd=*/true);
        else
            fwd_level(stream, plan.panel_level_ptr[L], plan.panel_level_ptr[L + 1], plan.d_plcols, plan.h_plcols.data());
    }
    for (int L = plan.num_plevels - 1; L >= 0; --L) {
        if (have_lvl)
            dispatch_tiered(stream, lvl_tb + (long)L * (NT + 1),
                            plan.d_plcols_tier, plan.h_plcols_tier.data(), /*fwd=*/false);
        else
            bwd_level(stream, plan.panel_level_ptr[L], plan.panel_level_ptr[L + 1], plan.d_plcols, plan.h_plcols.data());
    }
}

}  // namespace
}  // namespace custom_linear_solver
