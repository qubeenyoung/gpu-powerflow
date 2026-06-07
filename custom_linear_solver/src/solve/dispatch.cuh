#pragma once

// SOLVE — host-side level dispatch.
//
// Internal — included only by multifrontal.cu (single TU; CUDA_SEPARABLE_COMPILATION OFF).
//
// Two layers of dispatch:
//
//   issue_solve_levels        — outer driver. Iterates etree levels forward (L y = b, top
//                                of tree toward root for the solve sense — actually level 0
//                                up through num_plevels) and then backward (root down).
//                                Forward uses `solve_fwd[_small]`; backward `solve_bwd[_small]`.
//
//   per-level routing         — within each pass, levels with max_fsz ≤ SMALL_THRESH go to the
//                                warp-packed small kernel (8 warps/block, one (front,batch)
//                                per warp); larger levels go to the block-per-front kernel
//                                with thread count tuned to the level's max_fsz.
//
// Solve is much lighter than factor — no rank-nc GEMM — so we don't tier-split mid/big; a
// single regular kernel with caller-tuned block size covers everything above SMALL_THRESH.
// No multi-stream is used either (solve work is too small to benefit from concurrent stream
// execution the way factor does; see docs/13).

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

    // ----- Forward pass: L y = b, level 0 → num_plevels-1 (leaves → root) ----------------
    for (int L = 0; L < plan.num_plevels; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        if (level_max_fsz(b, e) <= SMALL_THRESH) {
            const long warps = (long)(e - b) * B;
            const int blk = SMALL_WARPS * 32;
            const int gx = (int)((warps + SMALL_WARPS - 1) / SMALL_WARPS);
            const size_t shb = (size_t)SMALL_WARPS * 64 * selt;  // sh_piv[64] per warp
            if (float_front)
                solve_fwd_small<float><<<gx, blk, shb, stream>>>(
                    b, e - b, B, 64, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total,
                    plan.n);
            else
                solve_fwd_small<double><<<gx, blk, shb, stream>>>(
                    b, e - b, B, 64, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total,
                    plan.n);
            continue;
        }
        // Bigger-front levels: more threads/block hide per-front latency + parallelize the
        // CB update across more lanes.
        const int mf = level_max_fsz(b, e);
        const int tsb = mf <= 64 ? 64 : (mf <= 128 ? 128 : 256);
        const dim3 fg(e - b, B);
        if (float_front)
            solve_fwd<float><<<fg, tsb, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n);
        else
            solve_fwd<double><<<fg, tsb, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total, plan.n);
    }

    // ----- Backward pass: U x = y, level num_plevels-1 → 0 (root → leaves) ---------------
    for (int L = plan.num_plevels - 1; L >= 0; --L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        int max_cb = 1;  // dynamic shared for the bwd x-cache, sized by the level's max CB rows
        for (int q = b; q < e; ++q) {
            const int pp = plan.h_plcols[q];
            const int cbq = (plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp]) - plan.h_ncols[pp];
            if (cbq > max_cb) max_cb = cbq;
        }
        if (level_max_fsz(b, e) <= SMALL_THRESH) {
            const long warps = (long)(e - b) * B;
            const int blk = SMALL_WARPS * 32;
            const int gx = (int)((warps + SMALL_WARPS - 1) / SMALL_WARPS);
            const int slab = 64 + max_cb;
            const size_t shb = (size_t)SMALL_WARPS * slab * selt;
            if (float_front)
                solve_bwd_small<float><<<gx, blk, shb, stream>>>(
                    b, e - b, B, slab, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total,
                    plan.n);
            else
                solve_bwd_small<double><<<gx, blk, shb, stream>>>(
                    b, e - b, B, slab, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total,
                    plan.n);
            continue;
        }
        const int mf = level_max_fsz(b, e);
        const int tsb = mf <= 64 ? 64 : (mf <= 128 ? 128 : 256);
        const dim3 bg(e - b, B);
        if (float_front)
            solve_bwd<float><<<bg, tsb, (size_t)max_cb * sizeof(float), stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n);
        else
            solve_bwd<double><<<bg, tsb, (size_t)max_cb * sizeof(double), stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total, plan.n);
    }
}

}  // namespace
}  // namespace custom_linear_solver
