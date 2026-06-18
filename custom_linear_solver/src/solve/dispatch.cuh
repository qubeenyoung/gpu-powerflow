#pragma once

// SOLVE — host-side level dispatch.
//
// Internal — included into the factor/solve driver TUs (single TU; CUDA_SEPARABLE_COMPILATION OFF).
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
//   per-tier routing          — within each pass, the small tier (max_fsz ≤ kSmallFrontMax) goes to
//                                the warp-packed kernel (8 warps/block, one (front,batch) per warp);
//                                the mid / big tiers share the block-per-front kernel with
//                                thread count tuned to the level's max_fsz.

#include <cuda_runtime.h>

#include "internal/plan/front_range_caps.hpp"
#include "internal/runtime/state.hpp"
#include "solve/kernels.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

constexpr int kSolveSmallWarpsPerBlock = kSmallTierWarpsPerBlock;

// HW warp-slot count (SMs × warps/SM) — the GPU-fill quantity used by the small-tier sub-group
// gate (mirrors factorize/front_ops.cuh factor_warp_fill()).
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

// Small-tier sub-group size: sub_group_size ∈ {8,16,32}. The triangular recurrences require
// lanes for pivot columns, not every contribution row, so use max_nc rather than max_fsz when
// packing multiple fronts into one warp.
static int solve_small_sg(int max_nc, long warps_unpacked, int B)
{
#ifdef CLS_MID_SG32_ONLY
    (void)max_nc; (void)warps_unpacked; (void)B;
    return 32;
#else
    int sg = (max_nc <= 8) ? 8 : (max_nc <= 16 ? 16 : 32);
    const long packed_warps = (warps_unpacked + (32 / sg) - 1) / (32 / sg);
    if (B == 1 && packed_warps < solve_warp_fill() && packed_warps < 800) sg = 32;
    return sg;
#endif
}

static int solve_regular_threads(int max_front_size, int level_size, int level_max_nc, int B)
{
    if (B == 1) {
        if (max_front_size > 96) return 512;
        if (max_front_size > 80 && level_size <= 128) return 512;
        if (max_front_size > 64 && level_size <= 512 && level_max_nc >= 16) return 512;
        if (max_front_size > 64 && level_size <= 512) return 256;
        if (max_front_size > 48 && level_size <= 512) return 256;
    }
    return max_front_size <= 64 ? 64 : (max_front_size <= 128 ? 128 : 256);
}

static bool solve_range_all_nc(const MultifrontalPlan& plan, const int* h_plc, int b, int e, int nc)
{
    for (int q = b; q < e; ++q)
        if (plan.h_ncols[h_plc[q]] != nc) return false;
    return true;
}

static int solve_regular_fixed_nc_choice(const MultifrontalPlan& plan, const int* h_plc, int b, int e, int level_max_nc, int B)
{
    (void)plan; (void)h_plc; (void)b; (void)e;
    if (B != 1) return 0;
    int choice = 0;
    if (level_max_nc == 8) choice = 8;
    else if (level_max_nc == 10) choice = 10;
    else if (level_max_nc == 14) choice = 14;
    else if (level_max_nc == 16) choice = 16;
    else if (level_max_nc == 20) choice = 20;
    return choice;
}

// One launch of a sub-group-packed small solve kernel (forward or backward), with the
// (front type, sub_group_size) resolved at the call site. solve_fwd_small / solve_bwd_small share a
// signature; `fwd` selects which, `slab` is the per-sub-group shared slab.
template <typename FrontType, int sub_group_size>
static inline void launch_solve_small(bool fwd, int num_blocks, int threads_per_block, size_t shared_bytes, cudaStream_t ks,
                                      int b, int level_size, int B, int slab,
                                      const MultifrontalPlan& plan, const int* d_plc,
                                      FrontType* frontB, FrontType* yB)
{
    if (fwd)
        solve_fwd_small<FrontType, sub_group_size><<<num_blocks, threads_per_block, shared_bytes, ks>>>(
            b, level_size, B, slab, d_plc, plan.d_front_off, plan.d_front_ptr,
            plan.d_ncols, plan.d_front_rows, frontB, yB, plan.front_total, plan.num_rows);
    else
        solve_bwd_small<FrontType, sub_group_size><<<num_blocks, threads_per_block, shared_bytes, ks>>>(
            b, level_size, B, slab, d_plc, plan.d_front_off, plan.d_front_ptr,
            plan.d_ncols, plan.d_front_rows, frontB, yB, plan.front_total, plan.num_rows);
}

template <typename FrontType>
static inline void launch_solve_small_t(bool fwd, int sub_group_size, int num_blocks, int threads_per_block, size_t shared_bytes,
                                        cudaStream_t ks, int b, int level_size, int B, int slab,
                                        const MultifrontalPlan& plan, const int* d_plc,
                                        FrontType* frontB, FrontType* yB)
{
    if (sub_group_size == 8)       launch_solve_small<FrontType, 8>(fwd, num_blocks, threads_per_block, shared_bytes, ks, b, level_size, B, slab, plan, d_plc, frontB, yB);
    else if (sub_group_size == 16) launch_solve_small<FrontType, 16>(fwd, num_blocks, threads_per_block, shared_bytes, ks, b, level_size, B, slab, plan, d_plc, frontB, yB);
    else               launch_solve_small<FrontType, 32>(fwd, num_blocks, threads_per_block, shared_bytes, ks, b, level_size, B, slab, plan, d_plc, frontB, yB);
}

template <typename FrontType>
static inline void launch_solve_fwd_dynamic(dim3 grid, int threads_per_block, size_t shared_bytes,
                                            cudaStream_t ks,
                                            int b, int e, const MultifrontalPlan& plan, const int* d_plc,
                                            FrontType* frontB, FrontType* yB)
{
    solve_fwd<FrontType><<<grid, threads_per_block, shared_bytes, ks>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_front_rows, frontB, yB, plan.front_total, plan.num_rows);
}

// Non-staged fwd/bwd launches for large fronts (staged shared would exceed the default cap).
template <typename FrontType>
static inline void launch_solve_fwd_nostage(dim3 grid, int threads_per_block, cudaStream_t ks,
                                            int b, int e, const MultifrontalPlan& plan, const int* d_plc,
                                            FrontType* frontB, FrontType* yB)
{
    solve_fwd_nostage<FrontType><<<grid, threads_per_block, 0, ks>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_front_rows, frontB, yB, plan.front_total, plan.num_rows);
}

template <typename FrontType>
static inline void launch_solve_bwd_nostage(dim3 grid, int threads_per_block, size_t shared_bytes,
                                            cudaStream_t ks,
                                            int b, int e, const MultifrontalPlan& plan, const int* d_plc,
                                            FrontType* frontB, FrontType* yB)
{
    solve_bwd_nostage<FrontType><<<grid, threads_per_block, shared_bytes, ks>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_front_rows, frontB, yB, plan.front_total, plan.num_rows);
}

// Largest staged-panel shared we allow under the default (non-opt-in) cap; above this the dynamic
// solve falls back to the non-staged kernel (reads F directly). Leaves room for the static sh_piv.
static constexpr size_t kSolveStageSharedCap = 46u * 1024u;

template <typename FrontType, int NC>
static inline void launch_solve_fwd_fixed(bool exact, dim3 grid, int threads_per_block, cudaStream_t ks,
                                          int b, int e, const MultifrontalPlan& plan, const int* d_plc,
                                          FrontType* frontB, FrontType* yB)
{
    if (exact) {
        solve_fwd_exact_nc<FrontType, NC><<<grid, threads_per_block, 0, ks>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
            frontB, yB, plan.front_total, plan.num_rows);
    } else {
        solve_fwd_fixed_nc<FrontType, NC><<<grid, threads_per_block, 0, ks>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_front_rows, frontB, yB, plan.front_total, plan.num_rows);
    }
}

// Forward dispatch for one plcols (sub)range on a caller-chosen stream.
static void fwd_level(const MultifrontalPlan& plan, State& st, cudaStream_t ks, int b, int e,
                      const int* d_plc, const int* h_plc)
{
    if (e <= b) return;
    const int B = st.batch_count;
    const bool float_front = is_fp32_front(st.precision);
    const int element_bytes = float_front ? (int)sizeof(float) : (int)sizeof(double);
    const FrontRangeCaps metrics = scan_front_range(plan, h_plc, b, e);
    const int max_front_size = metrics.max_fsz;

    // Small tier: sub-group-packed warp kernel (fronts_per_warp = kWarpSize/sub_group_size fronts per warp).
    if (classify_front_tier(max_front_size, !float_front) == FrontTier::kSmall) {
        const long warps_unpacked = (long)(e - b) * B;
        const int sub_group_size = solve_small_sg(metrics.level_max_nc, warps_unpacked, B), fronts_per_warp = kWarpSize / sub_group_size;
        const int threads_per_block = kSolveSmallWarpsPerBlock * kWarpSize;
        const int num_blocks = (int)(((warps_unpacked + fronts_per_warp - 1) / fronts_per_warp + kSolveSmallWarpsPerBlock - 1) / kSolveSmallWarpsPerBlock);
        const size_t shared_bytes = (size_t)kSolveSmallWarpsPerBlock * fronts_per_warp * kMaxPivotColumns * element_bytes;
        if (float_front)
            launch_solve_small_t<float>(/*fwd=*/true, sub_group_size, num_blocks, threads_per_block, shared_bytes, ks, b, e - b, B, kMaxPivotColumns,
                                        plan, d_plc, st.d_front_batch_f, st.d_y_batch_f);
        else
            launch_solve_small_t<double>(/*fwd=*/true, sub_group_size, num_blocks, threads_per_block, shared_bytes, ks, b, e - b, B, kMaxPivotColumns,
                                         plan, d_plc, st.d_front_batch, st.d_y_batch);
        return;
    }

    // Larger levels: one block per (front, batch), thread count tuned to max_fsz.
    const int threads_per_block = solve_regular_threads(max_front_size, e - b, metrics.level_max_nc, B);
    const dim3 fg(e - b, B);
    // P3: dynamic shared for the staged L panel (rows × cols[0,nc), upper-bounded by level maxes).
    const size_t fwd_panel_shared = (size_t)max_front_size * metrics.level_max_nc * element_bytes;
    const int fixed_nc = solve_regular_fixed_nc_choice(plan, h_plc, b, e, metrics.level_max_nc, B);
    const bool use_nc8 = fixed_nc == 8;
    const bool use_nc10 = fixed_nc == 10;
    const bool use_nc14 = fixed_nc == 14;
    const bool use_nc16 = fixed_nc == 16;
    const bool use_nc20 = fixed_nc == 20;
    const bool exact_nc8 = use_nc8 && solve_range_all_nc(plan, h_plc, b, e, 8);
    const bool exact_nc10 = use_nc10 && solve_range_all_nc(plan, h_plc, b, e, 10);
    const bool exact_nc14 = use_nc14 && solve_range_all_nc(plan, h_plc, b, e, 14);
    const bool exact_nc16 = false;
    const bool exact_nc20 = use_nc20 && solve_range_all_nc(plan, h_plc, b, e, 20);
    if (float_front) {
        if (use_nc8)
            launch_solve_fwd_fixed<float, 8>(exact_nc8, fg, threads_per_block, ks, b, e, plan, d_plc,
                                             st.d_front_batch_f, st.d_y_batch_f);
        else if (use_nc10)
            launch_solve_fwd_fixed<float, 10>(exact_nc10, fg, threads_per_block, ks, b, e, plan, d_plc,
                                              st.d_front_batch_f, st.d_y_batch_f);
        else if (use_nc14)
            launch_solve_fwd_fixed<float, 14>(exact_nc14, fg, threads_per_block, ks, b, e, plan, d_plc,
                                              st.d_front_batch_f, st.d_y_batch_f);
        else if (use_nc16)
            launch_solve_fwd_fixed<float, 16>(exact_nc16, fg, threads_per_block, ks, b, e, plan, d_plc,
                                              st.d_front_batch_f, st.d_y_batch_f);
        else if (use_nc20)
            launch_solve_fwd_fixed<float, 20>(exact_nc20, fg, threads_per_block, ks, b, e, plan, d_plc,
                                              st.d_front_batch_f, st.d_y_batch_f);
        else if (fwd_panel_shared > kSolveStageSharedCap)
            launch_solve_fwd_nostage<float>(fg, threads_per_block, ks, b, e, plan, d_plc,
                                            st.d_front_batch_f, st.d_y_batch_f);
        else
            launch_solve_fwd_dynamic<float>(fg, threads_per_block, fwd_panel_shared, ks, b, e, plan, d_plc,
                                            st.d_front_batch_f, st.d_y_batch_f);
    } else {
        if (use_nc8)
            launch_solve_fwd_fixed<double, 8>(exact_nc8, fg, threads_per_block, ks, b, e, plan, d_plc,
                                              st.d_front_batch, st.d_y_batch);
        else if (use_nc10)
            launch_solve_fwd_fixed<double, 10>(exact_nc10, fg, threads_per_block, ks, b, e, plan, d_plc,
                                               st.d_front_batch, st.d_y_batch);
        else if (use_nc14)
            launch_solve_fwd_fixed<double, 14>(exact_nc14, fg, threads_per_block, ks, b, e, plan, d_plc,
                                               st.d_front_batch, st.d_y_batch);
        else if (use_nc16)
            launch_solve_fwd_fixed<double, 16>(exact_nc16, fg, threads_per_block, ks, b, e, plan, d_plc,
                                               st.d_front_batch, st.d_y_batch);
        else if (use_nc20)
            launch_solve_fwd_fixed<double, 20>(exact_nc20, fg, threads_per_block, ks, b, e, plan, d_plc,
                                               st.d_front_batch, st.d_y_batch);
        else if (fwd_panel_shared > kSolveStageSharedCap)
            launch_solve_fwd_nostage<double>(fg, threads_per_block, ks, b, e, plan, d_plc,
                                             st.d_front_batch, st.d_y_batch);
        else
            launch_solve_fwd_dynamic<double>(fg, threads_per_block, fwd_panel_shared, ks, b, e, plan, d_plc,
                                             st.d_front_batch, st.d_y_batch);
    }
}

// Backward dispatch for one plcols (sub)range on a caller-chosen stream.
static void bwd_level(const MultifrontalPlan& plan, State& st, cudaStream_t ks, int b, int e,
                      const int* d_plc, const int* h_plc)
{
    if (e <= b) return;
    const int B = st.batch_count;
    const bool float_front = is_fp32_front(st.precision);
    const int element_bytes = float_front ? (int)sizeof(float) : (int)sizeof(double);
    const FrontRangeCaps metrics = scan_front_range(plan, h_plc, b, e);
    const int max_front_size = metrics.max_fsz;
    const int max_contribution = metrics.level_max_uc;
    // P3: regular solve_bwd<T> stages the top-nc U rows (≤ max_fsz·level_max_nc) plus the x cache.
    const size_t bwd_panel_shared = (size_t)(max_front_size * metrics.level_max_nc + max_contribution) * element_bytes;

    // Small tier: sub-group-packed warp kernel. slab = kMaxPivotColumns (rhs) + max_contribution (x cache).
    if (classify_front_tier(max_front_size, !float_front) == FrontTier::kSmall) {
        const long warps_unpacked = (long)(e - b) * B;
        const int sub_group_size = solve_small_sg(metrics.level_max_nc, warps_unpacked, B), fronts_per_warp = kWarpSize / sub_group_size;
        const int threads_per_block = kSolveSmallWarpsPerBlock * kWarpSize;
        const int num_blocks = (int)(((warps_unpacked + fronts_per_warp - 1) / fronts_per_warp + kSolveSmallWarpsPerBlock - 1) / kSolveSmallWarpsPerBlock);
        const int slab = kMaxPivotColumns + max_contribution;
        const size_t shared_bytes = (size_t)kSolveSmallWarpsPerBlock * fronts_per_warp * slab * element_bytes;
        if (float_front)
            launch_solve_small_t<float>(/*fwd=*/false, sub_group_size, num_blocks, threads_per_block, shared_bytes, ks, b, e - b, B, slab,
                                        plan, d_plc, st.d_front_batch_f, st.d_y_batch_f);
        else
            launch_solve_small_t<double>(/*fwd=*/false, sub_group_size, num_blocks, threads_per_block, shared_bytes, ks, b, e - b, B, slab,
                                         plan, d_plc, st.d_front_batch, st.d_y_batch);
        return;
    }

    // Larger levels: one block per (front, batch); max_contribution-sized shared for the x cache.
    const int threads_per_block = solve_regular_threads(max_front_size, e - b, metrics.level_max_nc, B);
    const dim3 bg(e - b, B);
    const int fixed_nc = solve_regular_fixed_nc_choice(plan, h_plc, b, e, metrics.level_max_nc, B);
    const bool use_nc8 = fixed_nc == 8;
    const bool use_nc10 = fixed_nc == 10;
    const bool use_nc14 = fixed_nc == 14;
    const bool use_nc16 = fixed_nc == 16;
    const bool use_nc20 = fixed_nc == 20;
    const bool exact_nc8 = use_nc8 && solve_range_all_nc(plan, h_plc, b, e, 8);
    const bool exact_nc10 = use_nc10 && solve_range_all_nc(plan, h_plc, b, e, 10);
    const bool exact_nc14 = use_nc14 && solve_range_all_nc(plan, h_plc, b, e, 14);
    const bool exact_nc16 = false;
    const bool exact_nc20 = use_nc20 && solve_range_all_nc(plan, h_plc, b, e, 20);
    if (float_front) {
        if (use_nc8) {
            if (exact_nc8)
                solve_bwd_exact_nc<float, 8><<<bg, threads_per_block, (size_t)max_contribution * sizeof(float), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                    st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
            else
                solve_bwd_fixed_nc<float, 8><<<bg, threads_per_block, (size_t)max_contribution * sizeof(float), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
        } else if (use_nc10) {
            if (exact_nc10)
                solve_bwd_exact_nc<float, 10><<<bg, threads_per_block, (size_t)max_contribution * sizeof(float), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                    st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
            else
                solve_bwd_fixed_nc<float, 10><<<bg, threads_per_block, (size_t)max_contribution * sizeof(float), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
        } else if (use_nc14) {
            if (exact_nc14)
                solve_bwd_exact_nc<float, 14><<<bg, threads_per_block, (size_t)max_contribution * sizeof(float), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                    st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
            else
                solve_bwd_fixed_nc<float, 14><<<bg, threads_per_block, (size_t)max_contribution * sizeof(float), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
        } else if (use_nc16) {
            if (exact_nc16)
                solve_bwd_exact_nc<float, 16><<<bg, threads_per_block, (size_t)max_contribution * sizeof(float), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                    st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
            else
                solve_bwd_fixed_nc<float, 16><<<bg, threads_per_block, (size_t)max_contribution * sizeof(float), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
        } else if (use_nc20) {
            if (exact_nc20)
                solve_bwd_exact_nc<float, 20><<<bg, threads_per_block, (size_t)max_contribution * sizeof(float), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                    st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
            else
                solve_bwd_fixed_nc<float, 20><<<bg, threads_per_block, (size_t)max_contribution * sizeof(float), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
        }
        else if (bwd_panel_shared > kSolveStageSharedCap)
            launch_solve_bwd_nostage<float>(bg, threads_per_block, (size_t)max_contribution * sizeof(float),
                                            ks, b, e, plan, d_plc, st.d_front_batch_f, st.d_y_batch_f);
        else
            solve_bwd<float><<<bg, threads_per_block, bwd_panel_shared, ks>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
    } else {
        if (use_nc8) {
            if (exact_nc8)
                solve_bwd_exact_nc<double, 8><<<bg, threads_per_block, (size_t)max_contribution * sizeof(double), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                    st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
            else
                solve_bwd_fixed_nc<double, 8><<<bg, threads_per_block, (size_t)max_contribution * sizeof(double), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_front_rows, st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
        } else if (use_nc10) {
            if (exact_nc10)
                solve_bwd_exact_nc<double, 10><<<bg, threads_per_block, (size_t)max_contribution * sizeof(double), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                    st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
            else
                solve_bwd_fixed_nc<double, 10><<<bg, threads_per_block, (size_t)max_contribution * sizeof(double), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_front_rows, st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
        } else if (use_nc14) {
            if (exact_nc14)
                solve_bwd_exact_nc<double, 14><<<bg, threads_per_block, (size_t)max_contribution * sizeof(double), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                    st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
            else
                solve_bwd_fixed_nc<double, 14><<<bg, threads_per_block, (size_t)max_contribution * sizeof(double), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_front_rows, st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
        } else if (use_nc16) {
            if (exact_nc16)
                solve_bwd_exact_nc<double, 16><<<bg, threads_per_block, (size_t)max_contribution * sizeof(double), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                    st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
            else
                solve_bwd_fixed_nc<double, 16><<<bg, threads_per_block, (size_t)max_contribution * sizeof(double), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_front_rows, st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
        } else if (use_nc20) {
            if (exact_nc20)
                solve_bwd_exact_nc<double, 20><<<bg, threads_per_block, (size_t)max_contribution * sizeof(double), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                    st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
            else
                solve_bwd_fixed_nc<double, 20><<<bg, threads_per_block, (size_t)max_contribution * sizeof(double), ks>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_front_rows, st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
        }
        else if (bwd_panel_shared > kSolveStageSharedCap)
            launch_solve_bwd_nostage<double>(bg, threads_per_block, (size_t)max_contribution * sizeof(double),
                                             ks, b, e, plan, d_plc, st.d_front_batch, st.d_y_batch);
        else
            solve_bwd<double><<<bg, threads_per_block, bwd_panel_shared, ks>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
    }
}

static int spine_max_contribution(const MultifrontalPlan& plan)
{
    int max_contribution = 1;
    for (int p : plan.h_spine_panels) {
        const int fsz = plan.h_front_ptr[p + 1] - plan.h_front_ptr[p];
        const int cb = fsz - plan.h_ncols[p];
        if (cb > max_contribution) max_contribution = cb;
    }
    return max_contribution;
}

static int spine_threads_per_block(const MultifrontalPlan& plan, int B)
{
    int max_front_size = 1;
    for (int p : plan.h_spine_panels) {
        const int fsz = plan.h_front_ptr[p + 1] - plan.h_front_ptr[p];
        if (fsz > max_front_size) max_front_size = fsz;
    }
    if (B == 1 && max_front_size <= 80) return 64;
    if (B == 1 && max_front_size > 96) return 512;
    return max_front_size <= 64 ? 64 : (max_front_size <= 128 ? 128 : 256);
}

static void fwd_spine(const MultifrontalPlan& plan, State& st, cudaStream_t ks)
{
    const int n_spine = (int)plan.h_spine_panels.size();
    if (n_spine <= 0 || !plan.d_spine_panels) return;
    const int spine_threads = spine_threads_per_block(plan, st.batch_count);
    const dim3 grid(1, st.batch_count);
    if (is_fp32_front(st.precision))
        solve_fwd_spine<float><<<grid, spine_threads, 0, ks>>>(
            n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
    else
        solve_fwd_spine<double><<<grid, spine_threads, 0, ks>>>(
            n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_front_rows, st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
}

static void bwd_spine(const MultifrontalPlan& plan, State& st, cudaStream_t ks)
{
    const int n_spine = (int)plan.h_spine_panels.size();
    if (n_spine <= 0 || !plan.d_spine_panels) return;
    const int spine_threads = spine_threads_per_block(plan, st.batch_count);
    const int max_contribution = spine_max_contribution(plan);
    const dim3 grid(1, st.batch_count);
    if (is_fp32_front(st.precision))
        solve_bwd_spine<float><<<grid, spine_threads, (size_t)max_contribution * sizeof(float), ks>>>(
            n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
    else
        solve_bwd_spine<double><<<grid, spine_threads, (size_t)max_contribution * sizeof(double), ks>>>(
            n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_front_rows, st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
}

static void solve_spine_chain(const MultifrontalPlan& plan, State& st, cudaStream_t ks)
{
    const int n_spine = (int)plan.h_spine_panels.size();
    if (n_spine <= 0 || !plan.d_spine_panels) return;
    const int spine_threads = spine_threads_per_block(plan, st.batch_count);
    if (spine_threads > 128) {
        fwd_spine(plan, st, ks);
        bwd_spine(plan, st, ks);
        return;
    }
    const int max_contribution = spine_max_contribution(plan);
    const dim3 grid(1, st.batch_count);
    if (is_fp32_front(st.precision))
        solve_spine<float><<<grid, spine_threads, (size_t)max_contribution * sizeof(float), ks>>>(
            n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f, plan.front_total, plan.num_rows);
    else
        solve_spine<double><<<grid, spine_threads, (size_t)max_contribution * sizeof(double), ks>>>(
            n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_front_rows, st.d_front_batch, st.d_y_batch, plan.front_total, plan.num_rows);
}

// Deterministic per-tier split of one (level- or cell-) range, for either sweep. Each non-empty tier
// sub-range dispatches to its kernel (small -> warp-packed; mid/big -> block-per-front with
// thread count by max_fsz), independent of occupancy. Same range = independent fronts -> order-free.
// (tier_split=false is a debug fallback that merges the whole range onto its largest front's kernel.)
static void dispatch_tiered(const MultifrontalPlan& plan, State& st, cudaStream_t ks,
                            const int* tb, const int* d_plc, const int* h_plc, bool fwd)
{
    constexpr int NT = MultifrontalPlan::kNumTiers;
    if (st.tier_split) {
        for (int t = 0; t < NT; ++t)
            if (tb[t + 1] > tb[t]) {
                if (fwd) fwd_level(plan, st, ks, tb[t], tb[t + 1], d_plc, h_plc);
                else     bwd_level(plan, st, ks, tb[t], tb[t + 1], d_plc, h_plc);
            }
    } else {
        if (fwd) fwd_level(plan, st, ks, tb[0], tb[NT], d_plc, h_plc);
        else     bwd_level(plan, st, ks, tb[0], tb[NT], d_plc, h_plc);
    }
}

// Multi-stream sweep: replicate the factor subtree-fork pattern. Subtree streams sweep their own
// panel slices through all pre-spine levels independently (no global per-level barrier), join, run
// the spine on the main stream, then re-fork for the backward sweep.
static void solve_multistream_sweep(const MultifrontalPlan& plan, State& st, cudaStream_t stream)
{
    constexpr int NT = MultifrontalPlan::kNumTiers;
    const int* sub_tb = plan.h_subtree_level_tier_off.data();
    const bool have_sub = !plan.h_subtree_level_tier_off.empty();
    const int K = st.num_subtree_streams;
    const int spine_lo = (plan.spine_start_level >= 0) ? plan.spine_start_level : plan.num_plevels;

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
                dispatch_tiered(plan, st, ks, sub_tb + ((long)k * plan.num_plevels + L) * (NT + 1),
                                plan.d_plcols, plan.h_plcols.data(), /*fwd=*/true);
            else
                fwd_level(plan, st, ks, off[L], off[L] + cnt[L], plan.d_plcols, plan.h_plcols.data());
        }
        cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
    }

    // ---- Forward join, spine forward on main, then backward re-fork ----
    for (int k = 0; k < K; ++k)
        cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);

    if (spine_lo < plan.num_plevels) solve_spine_chain(plan, st, stream);

    cudaEventRecord(static_cast<cudaEvent_t>(st.fork_event), stream);
    for (int k = 0; k < K; ++k)
        cudaStreamWaitEvent(static_cast<cudaStream_t>(st.subtree_streams[k]),
                            static_cast<cudaEvent_t>(st.fork_event), 0);

    // Per-subtree backward sweep over pre-spine levels (top -> bottom).
    for (int k = 0; k < K; ++k) {
        cudaStream_t ks = static_cast<cudaStream_t>(st.subtree_streams[k]);
        const int* off = plan.h_subtree_level_off.data() + (long)k * plan.num_plevels;
        const int* cnt = plan.h_subtree_level_cnt.data() + (long)k * plan.num_plevels;
        for (int L = spine_lo - 1; L >= 0; --L) {
            if (cnt[L] == 0) continue;
            if (have_sub)
                dispatch_tiered(plan, st, ks, sub_tb + ((long)k * plan.num_plevels + L) * (NT + 1),
                                plan.d_plcols, plan.h_plcols.data(), /*fwd=*/false);
            else
                bwd_level(plan, st, ks, off[L], off[L] + cnt[L], plan.d_plcols, plan.h_plcols.data());
        }
        cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
    }

    for (int k = 0; k < K; ++k)
        cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);
}

// Single-stream sweep: per-level occupancy-gated tier split, forward (leaves->root) then backward.
static void solve_single_stream_sweep(const MultifrontalPlan& plan, State& st, cudaStream_t stream)
{
    constexpr int NT = MultifrontalPlan::kNumTiers;
    const int* lvl_tb = plan.h_level_tier_off.data();
    const bool have_lvl = !plan.h_level_tier_off.empty() && plan.d_plcols_tier;
    const int spine_lo = (plan.spine_start_level >= 0 && !plan.h_spine_panels.empty())
                             ? plan.spine_start_level
                             : plan.num_plevels;
    for (int L = 0; L < spine_lo; ++L) {
        if (have_lvl)
            dispatch_tiered(plan, st, stream, lvl_tb + (long)L * (NT + 1),
                            plan.d_plcols_tier, plan.h_plcols_tier.data(), /*fwd=*/true);
        else
            fwd_level(plan, st, stream, plan.panel_level_ptr[L], plan.panel_level_ptr[L + 1],
                      plan.d_plcols, plan.h_plcols.data());
    }
    if (spine_lo < plan.num_plevels) solve_spine_chain(plan, st, stream);
    for (int L = spine_lo - 1; L >= 0; --L) {
        if (have_lvl)
            dispatch_tiered(plan, st, stream, lvl_tb + (long)L * (NT + 1),
                            plan.d_plcols_tier, plan.h_plcols_tier.data(), /*fwd=*/false);
        else
            bwd_level(plan, st, stream, plan.panel_level_ptr[L], plan.panel_level_ptr[L + 1],
                      plan.d_plcols, plan.h_plcols.data());
    }
}

// Forward (leaves->root) then backward (root->leaves) solve. Multi-stream when the plan exposes
// independent subtrees with matching streams; otherwise a single-stream level walk.
static void issue_solve_levels(const MultifrontalPlan& plan, State& st, cudaStream_t stream)
{
    // With spine fusion active, B=1 no longer pays for subtree stream/event overhead even on
    // USA-scale plans. Keep multistream for true batched solves where each cell exposes B blocks.
    const bool use_multistream = st.batch_count > 1 &&
                                 st.num_subtree_streams > 1 &&
                                 plan.num_subtrees == st.num_subtree_streams &&
                                 !plan.h_subtree_level_off.empty();
    if (use_multistream) solve_multistream_sweep(plan, st, stream);
    else solve_single_stream_sweep(plan, st, stream);
}

}  // namespace
}  // namespace custom_linear_solver
