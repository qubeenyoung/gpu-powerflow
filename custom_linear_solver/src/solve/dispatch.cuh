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
//   per-level routing         — within each pass, levels with max_fsz ≤ SMALL_THRESH go to the
//                                warp-packed small kernel (8 warps/block, one (front,batch)
//                                per warp); larger levels go to the block-per-front kernel
//                                with thread count tuned to the level's max_fsz.

#include <cuda_runtime.h>

#include "multifrontal.hpp"
#include "solve/kernels.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

static void issue_solve_levels(const MultifrontalPlan& plan, State& st, cudaStream_t stream)
{
    const bool float_front = is_fp32_front(st.prec);
    const int B = st.B;
    const int SMALL_THRESH = 32, SMALL_WARPS = 8;
    const int selt = float_front ? (int)sizeof(float) : (int)sizeof(double);

    auto level_max_fsz = [&](int b, int e) {
        int m = 0;
        for (int q = b; q < e; ++q) {
            const int pp = plan.h_plcols[q];
            const int fsz = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
            if (fsz > m) m = fsz;
        }
        return m;
    };

    auto level_max_cb = [&](int b, int e) {
        int m = 1;
        for (int q = b; q < e; ++q) {
            const int pp = plan.h_plcols[q];
            const int cbq = (plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp]) - plan.h_ncols[pp];
            if (cbq > m) m = cbq;
        }
        return m;
    };

    // Per-level forward dispatch on a caller-chosen stream.
    auto fwd_level = [&](cudaStream_t ks, int b, int e) {
        if (e <= b) return;
        if (level_max_fsz(b, e) <= SMALL_THRESH) {
            const long warps = (long)(e - b) * B;
            const int blk = SMALL_WARPS * 32;
            const int gx = (int)((warps + SMALL_WARPS - 1) / SMALL_WARPS);
            const size_t shb = (size_t)SMALL_WARPS * 64 * selt;
            if (float_front)
                solve_fwd_small<float><<<gx, blk, shb, ks>>>(
                    b, e - b, B, 64, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total,
                    plan.n);
            else
                solve_fwd_small<double><<<gx, blk, shb, ks>>>(
                    b, e - b, B, 64, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total,
                    plan.n);
            return;
        }
        const int mf = level_max_fsz(b, e);
        const int tsb = mf <= 64 ? 64 : (mf <= 128 ? 128 : 256);
        const dim3 fg(e - b, B);
        if (float_front)
            solve_fwd<float><<<fg, tsb, 0, ks>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n);
        else
            solve_fwd<double><<<fg, tsb, 0, ks>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total, plan.n);
    };

    auto bwd_level = [&](cudaStream_t ks, int b, int e) {
        if (e <= b) return;
        const int max_cb = level_max_cb(b, e);
        if (level_max_fsz(b, e) <= SMALL_THRESH) {
            const long warps = (long)(e - b) * B;
            const int blk = SMALL_WARPS * 32;
            const int gx = (int)((warps + SMALL_WARPS - 1) / SMALL_WARPS);
            const int slab = 64 + max_cb;
            const size_t shb = (size_t)SMALL_WARPS * slab * selt;
            if (float_front)
                solve_bwd_small<float><<<gx, blk, shb, ks>>>(
                    b, e - b, B, slab, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total,
                    plan.n);
            else
                solve_bwd_small<double><<<gx, blk, shb, ks>>>(
                    b, e - b, B, slab, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total,
                    plan.n);
            return;
        }
        const int mf = level_max_fsz(b, e);
        const int tsb = mf <= 64 ? 64 : (mf <= 128 ? 128 : 256);
        const dim3 bg(e - b, B);
        if (float_front)
            solve_bwd<float><<<bg, tsb, (size_t)max_cb * sizeof(float), ks>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n);
        else
            solve_bwd<double><<<bg, tsb, (size_t)max_cb * sizeof(double), ks>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total, plan.n);
    };

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
            for (int L = 0; L < spine_lo; ++L)
                if (cnt[L] > 0) fwd_level(ks, off[L], off[L] + cnt[L]);
            cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
        }

        // ---- Forward join, spine forward on main, then backward re-fork ----
        for (int k = 0; k < K; ++k)
            cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);

        for (int L = spine_lo; L < plan.num_plevels; ++L)
            fwd_level(stream, plan.plptr[L], plan.plptr[L + 1]);

        // Backward spine on main first (root → spine bottom), then re-fork for subtrees.
        for (int L = plan.num_plevels - 1; L >= spine_lo; --L)
            bwd_level(stream, plan.plptr[L], plan.plptr[L + 1]);

        cudaEventRecord(static_cast<cudaEvent_t>(st.fork_event), stream);
        for (int k = 0; k < K; ++k)
            cudaStreamWaitEvent(static_cast<cudaStream_t>(st.subtree_streams[k]),
                                static_cast<cudaEvent_t>(st.fork_event), 0);

        // Per-subtree backward sweep over pre-spine levels (top → bottom).
        for (int k = 0; k < K; ++k) {
            cudaStream_t ks = static_cast<cudaStream_t>(st.subtree_streams[k]);
            const int* off = plan.h_subtree_level_off.data() + (long)k * plan.num_plevels;
            const int* cnt = plan.h_subtree_level_cnt.data() + (long)k * plan.num_plevels;
            for (int L = spine_lo - 1; L >= 0; --L)
                if (cnt[L] > 0) bwd_level(ks, off[L], off[L] + cnt[L]);
            cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
        }

        for (int k = 0; k < K; ++k)
            cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);
        return;
    }

    // ---- Single-stream fallback ----
    for (int L = 0; L < plan.num_plevels; ++L)
        fwd_level(stream, plan.plptr[L], plan.plptr[L + 1]);
    for (int L = plan.num_plevels - 1; L >= 0; --L)
        bwd_level(stream, plan.plptr[L], plan.plptr[L + 1]);
}

}  // namespace
}  // namespace custom_linear_solver
