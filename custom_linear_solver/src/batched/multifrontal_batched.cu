#include "batched/multifrontal_batched.hpp"

#include <cuda_runtime.h>

// Uniform-batch multifrontal factorize + solve. The batched kernels (all front-major:
// gridDim.y = batch; arena = B * front_total) live in the internal batched/*.cuh headers below;
// the build uses CUDA_SEPARABLE_COMPILATION OFF, so every kernel must be defined in the TU that
// launches it (batched_setup). This file is the host orchestration: graph capture, the templated
// factorize/solve bodies, and the public entry points. Templates are instantiated by the switch
// on BatchedState::prec inside batched_setup.
//
// Internal-graph toggle (CLS_INTERNAL_GRAPH, default ON via CMake):
//   ON  - the standalone behavior: batched_setup owns a private stream and captures the factor /
//         solve kernel sequences into replayable CUDA graphs; factorize/solve cudaGraphLaunch
//         them and host-sync for timing.
//   OFF - external/capturable mode (used when an OUTER capture owns the graph, e.g. cuPF's
//         whole-iteration CUDA graph): no private stream, no internal graphs, no host sync. The
//         factor/solve kernels are issued DIRECTLY onto the caller stream (set via
//         batched_set_stream) so the outer cudaStreamBeginCapture records them.
#include "batched/scatter.cuh"        // scatter_batched<FT,VT>
#include "batched/factor_kernels.cuh"  // mf_factor_extend_*_b, mf_invert_pivot_b<FT>
#include "batched/factor_small.cuh"    // mf_factor_small_warp_b<FT> (tiny-front warp-per-front)
#include "tc/factor_tc.cuh"            // mf_factor_extend_tc32_b (FP32-native tensor-core trailing)
#include "tc/trailing_tiled.cuh"       // Σ.14: mf_factor_mid_tiled_b (shared-staged scalar trailing)
#include "tc/factor_no_trailing.cuh"   // Σ.16: NT (no-trailing) clones for direct GEMM-time profiling
#include "batched/solve_kernels.cuh"   // gather_rhs_b, scatter_sol_b, mf_{fwd,bwd}_level_b
#include "batched/solve_small.cuh"     // mf_{fwd,bwd}_small_warp_b<FT> (warp-packed small levels)

namespace custom_linear_solver::batched {

using custom_linear_solver::plan::MultifrontalPlan;

// The front arena is FP32 (and the solve runs in float) for both the pure-FP32 path and the
// FP32-native tensor-core path; FP64/Mixed/TC keep an FP64 front (or master).
static inline bool is_fp32_front(BatchPrecision p)
{
    return p == BatchPrecision::FP32 || p == BatchPrecision::TC32;
}

BatchedState::~BatchedState()
{
    if (factor_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(factor_graph_exec));
    if (solve_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(solve_graph_exec));
    if (d_frontB) cudaFree(d_frontB);
    if (d_frontBf) cudaFree(d_frontBf);
    if (d_yB) cudaFree(d_yB);
    if (d_yBf) cudaFree(d_yBf);
    if (d_sing) cudaFree(d_sing);
    if (fork_event) cudaEventDestroy(static_cast<cudaEvent_t>(fork_event));
    for (int k = 0; k < num_subtree_streams; ++k) {
        if (join_events[k]) cudaEventDestroy(static_cast<cudaEvent_t>(join_events[k]));
        if (subtree_streams[k]) cudaStreamDestroy(static_cast<cudaStream_t>(subtree_streams[k]));
    }
    // Only destroy the stream the solver itself created (internal-graph mode). In external mode the
    // stream is owned by the caller (e.g. cuPF's capture stream) and must outlive this state.
    if (stream && owns_stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

// ---------------------------------------------------------------------------
// Kernel-issue sequences shared by the internal-graph capture (batched_setup, when
// CLS_INTERNAL_GRAPH is defined) and the external/capturable path (batched_factorize/solve issuing
// directly onto a caller stream). Pure kernel launches only — no graph / host-sync here, so they
// are safe both to capture into a graph and to record into an outer stream capture.
// ---------------------------------------------------------------------------
// Issue one contiguous range [b, e) of plcols positions to `stream`. Used both by the original
// single-stream factor loop and the Σ.11 multi-stream subtree dispatch (each subtree's panels
// at level L are a contiguous plcols slice after analyze re-sorts by subtree id).
static void issue_factor_level_range(const MultifrontalPlan& plan, BatchedState& st,
                                     cudaStream_t stream, int b, int e)
{
    const BatchPrecision prec = st.prec;
    const bool pure_fp32 = is_fp32_front(prec);
    const int B = st.B;
    const int T = 128;
    const int do_extend = 1;
    const int SMALL_THRESH = 32;
    const int SMALL_WARPS = 8;
    const int MID_THRESH = (prec == BatchPrecision::TC32) ? 128 : 159;
    const bool small_ok = pure_fp32 || prec == BatchPrecision::FP64;
    // Σ.16-PROFILE: when CLS_PROFILE_NO_TRAILING=1, dispatch the NT clones (no rank-nc trailing
    // update). Measurement only — produces wrong factor. Subtract NT wall from original wall to
    // get the actual trailing-GEMM wall time.
    static const bool s_skip_trailing = []() {
        const char* on = std::getenv("CLS_PROFILE_NO_TRAILING");
        return on && on[0] && on[0] != '0';
    }();
    if (e <= b) return;
    const int level_size = e - b;
    do {
        if (small_ok) {
            int max_fsz = 0, max_uc = 1;
            for (int q = b; q < e; ++q) {
                const int pp = plan.h_plcols[q];
                const int fsz = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
                const int uc = fsz - plan.h_ncols[pp];
                if (fsz > max_fsz) max_fsz = fsz;
                if (uc > max_uc && uc <= 256) max_uc = uc;
            }
            if (max_fsz <= SMALL_THRESH) {
                const long warps = (long)level_size * B;
                const int blk = SMALL_WARPS * 32;
                const int gx = (int)((warps + SMALL_WARPS - 1) / SMALL_WARPS);
                const int fsz2cap = max_fsz * max_fsz;
                const size_t elt = (prec == BatchPrecision::FP64) ? sizeof(double) : sizeof(float);
                const size_t shb = (size_t)SMALL_WARPS * fsz2cap * elt;
                if (prec == BatchPrecision::FP64)
                    mf_factor_small_warp_b<double><<<gx, blk, shb, stream>>>(
                        b, level_size, B, fsz2cap, plan.d_plcols, plan.d_front_off,
                        plan.d_front_ptr, plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr,
                        plan.d_asm_local, st.d_frontB, plan.front_total, st.d_sing, do_extend);
                else if (s_skip_trailing)
                    mf_factor_small_warp_NT_b<<<gx, blk, shb, stream>>>(
                        b, level_size, B, fsz2cap, plan.d_plcols, plan.d_front_off,
                        plan.d_front_ptr, plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr,
                        plan.d_asm_local, st.d_frontBf, plan.front_total, st.d_sing, do_extend);
                else
                    mf_factor_small_warp_b<float><<<gx, blk, shb, stream>>>(
                        b, level_size, B, fsz2cap, plan.d_plcols, plan.d_front_off,
                        plan.d_front_ptr, plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr,
                        plan.d_asm_local, st.d_frontBf, plan.front_total, st.d_sing, do_extend);
                continue;
            }
            // Float-front mid/large levels: shared-resident block-per-front kernel (FP32 scalar
            // trailing, or FP16 tensor-core trailing for TC32). Shared sized to this level's fronts.
            if (pure_fp32 && max_fsz <= MID_THRESH) {
                const int fsz_cap = max_fsz;
                const int ucp_max = ((max_uc + 15) / 16) * 16;
                const bool use_tc = (prec == BatchPrecision::TC32);
                size_t shb = (size_t)fsz_cap * fsz_cap * sizeof(float);
                if (use_tc) shb += (size_t)2 * ucp_max * 32 * sizeof(__half) + 4 * 256 * sizeof(float);
                const int mblk = max_fsz <= 48 ? 64 : (max_fsz <= 80 ? 128 : 256);
                dim3 grid(level_size, B);
                if (use_tc) {
                    mf_factor_mid_tc32_b<true><<<grid, mblk, shb, stream>>>(
                        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                        plan.front_total, st.d_sing, do_extend, ucp_max, fsz_cap);
                    continue;
                }
                // Σ.14: pure-FP32 path also uses the Σ.1 shared-staged trailing (mid_tiled_b)
                // when shared budget fits. Otherwise falls back to mid_tc32_b<false>. This was
                // previously TC-only; applying it here closes the kernel-level gap that was
                // making TC look 17% faster at B=1 (it was just the staged kernel difference).
                // CLS_NO_TILED_TRAILING=1 reverts to legacy non-staged kernel for A/B testing.
                static const bool s_use_tiled_fp32 = []() {
                    const char* off = std::getenv("CLS_NO_TILED_TRAILING");
                    return !(off && off[0] && off[0] != '0');
                }();
                if (s_use_tiled_fp32 && max_fsz >= 48) {
                    int level_max_nc = 1, level_max_uc = 1;
                    for (int q = b; q < e; ++q) {
                        const int pp = plan.h_plcols[q];
                        const int fsz_p = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
                        const int nc_p = plan.h_ncols[pp];
                        if (nc_p > level_max_nc) level_max_nc = nc_p;
                        if ((fsz_p - nc_p) > level_max_uc) level_max_uc = fsz_p - nc_p;
                    }
                    const size_t shb_tiled =
                        (size_t)fsz_cap * fsz_cap * sizeof(float)
                        + 2 * (size_t)level_max_nc * level_max_uc * sizeof(float);
                    if (shb_tiled <= 96 * 1024) {
                        if (s_skip_trailing) {
                            custom_linear_solver::tc::mf_factor_mid_tiled_NT_b<<<grid, 256, shb_tiled, stream>>>(
                                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                                plan.front_total, st.d_sing, do_extend, fsz_cap, level_max_nc, level_max_uc);
                        } else {
                            custom_linear_solver::tc::mf_factor_mid_tiled_b<<<grid, 256, shb_tiled, stream>>>(
                                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                                plan.front_total, st.d_sing, do_extend, fsz_cap, level_max_nc, level_max_uc);
                        }
                        continue;
                    }
                }
                if (s_skip_trailing) {
                    mf_factor_mid_tc32_NT_b<<<grid, mblk, shb, stream>>>(
                        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                        plan.front_total, st.d_sing, do_extend, ucp_max, fsz_cap);
                } else {
                    mf_factor_mid_tc32_b<false><<<grid, mblk, shb, stream>>>(
                        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                        plan.front_total, st.d_sing, do_extend, ucp_max, fsz_cap);
                }
                continue;
            }
            // Float-front BIG separator fronts (fsz > MID_THRESH): too large for the shared cache.
            // Run the global-memory kernel but with many threads -- these top levels have very few
            // fronts (deep, low occupancy), so packing more warps per block hides the long
            // sequential panel/U-solve dependency chain. (Swept: 1024 best for these.) FP64 is not
            // a float front, so it skips this and falls through to the switch below.
            if (pure_fp32) {
                const int bigT = 1024;
                dim3 bgrid(level_size, B);
                if (prec == BatchPrecision::TC32) {
                    const int ucp_max = ((max_uc + 15) / 16) * 16;
                    const size_t shbytes =
                        (size_t)2 * ucp_max * 32 * sizeof(__half) + (bigT / 32) * 256 * sizeof(float);
                    mf_factor_extend_tc32_b<<<bgrid, bigT, shbytes, stream>>>(
                        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                        plan.front_total, st.d_sing, do_extend, ucp_max);
                } else if (s_skip_trailing) {
                    mf_factor_extend_level_NT_b<<<bgrid, bigT, 0, stream>>>(
                        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                        plan.front_total, st.d_sing, do_extend);
                } else {
                    mf_factor_extend_level_b<float><<<bgrid, bigT, 0, stream>>>(
                        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                        plan.front_total, st.d_sing, do_extend);
                }
                continue;
            }
        }
        dim3 grid(level_size, B);
        switch (prec) {
            case BatchPrecision::FP64:
                mf_factor_extend_level_b<double><<<grid, T, 0, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB,
                    plan.front_total, st.d_sing, do_extend);
                break;
            case BatchPrecision::FP32:
                mf_factor_extend_level_b<float><<<grid, T, 0, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                    plan.front_total, st.d_sing, do_extend);
                break;
            case BatchPrecision::Mixed:
                mf_factor_extend_mixed_b<<<grid, T, 0, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB, st.d_frontBf,
                    plan.front_total, st.d_sing, do_extend);
                break;
            case BatchPrecision::TC:
            case BatchPrecision::TC32: {
                // per-level max uc (= dynamic-shared Uh stride) so small-front levels use little shared.
                int max_uc = 1;  // only WMMA-eligible fronts (uc<=256, nc<=32) use the shared staging
                for (int q = b; q < e; ++q) {
                    const int pp = plan.h_plcols[q];
                    const int fsz = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
                    const int nc = plan.h_ncols[pp];
                    const int uc = fsz - nc;
                    if (uc > max_uc && uc <= 256 && nc <= 32) max_uc = uc;
                }
                const int ucp_max = ((max_uc + 15) / 16) * 16;
                const size_t shbytes =
                    (size_t)2 * ucp_max * 32 * sizeof(__half) + 4 * 256 * sizeof(float);
                if (prec == BatchPrecision::TC32)
                    mf_factor_extend_tc32_b<<<grid, 128, shbytes, stream>>>(
                        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                        plan.front_total, st.d_sing, do_extend, ucp_max);
                else
                    mf_factor_extend_mixed_tc_b<<<grid, 128, shbytes, stream>>>(
                        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB,
                        st.d_frontBf, plan.front_total, st.d_sing, do_extend, ucp_max);
                break;
            }
        }
    } while (false);
}

// Σ.11 — wrapper: iterates levels, either single-stream (legacy) or multi-stream subtree
// dispatch (default for batched fp32/fp64/mixed/TC paths). selinv runs on main stream after join.
static void issue_factor_levels(const MultifrontalPlan& plan, BatchedState& st, cudaStream_t stream)
{
    const int B = st.B;
    const BatchPrecision prec = st.prec;
    const bool pure_fp32 = is_fp32_front(prec);

    const char* multi_no = std::getenv("CLS_NO_MULTISTREAM");
    const bool disable_multi = multi_no && multi_no[0] && multi_no[0] != '0';
    const bool use_multistream = !disable_multi && st.num_subtree_streams > 1 &&
                                 plan.num_subtrees == st.num_subtree_streams &&
                                 !plan.h_subtree_level_off.empty();

    if (use_multistream) {
        // Identify spine_lo (lowest spine level); spine work runs on main after join.
        const int spine_lo = (plan.spine_start_level >= 0) ? plan.spine_start_level : plan.num_plevels;
        cudaEventRecord(static_cast<cudaEvent_t>(st.fork_event), stream);
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStreamWaitEvent(static_cast<cudaStream_t>(st.subtree_streams[k]),
                                static_cast<cudaEvent_t>(st.fork_event), 0);
        }
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStream_t ks = static_cast<cudaStream_t>(st.subtree_streams[k]);
            const int* off = plan.h_subtree_level_off.data() + (long)k * plan.num_plevels;
            const int* cnt = plan.h_subtree_level_cnt.data() + (long)k * plan.num_plevels;
            for (int L = 0; L < plan.num_plevels; ++L) {
                if (L >= spine_lo) break;
                if (cnt[L] == 0) continue;
                issue_factor_level_range(plan, st, ks, off[L], off[L] + cnt[L]);
            }
            cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
        }
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);
        }
        // Spine levels (if any) on main stream.
        if (plan.spine_start_level >= 0) {
            for (int L = plan.spine_start_level; L < plan.num_plevels; ++L) {
                const int b = plan.plptr[L], e = plan.plptr[L + 1];
                issue_factor_level_range(plan, st, stream, b, e);
            }
        }
    } else {
        for (int L = 0; L < plan.num_plevels; ++L) {
            const int b = plan.plptr[L], e = plan.plptr[L + 1];
            issue_factor_level_range(plan, st, stream, b, e);
        }
    }

    if (st.selinv) {
        const dim3 ig(plan.num_panels, B);
        if (pure_fp32)
            mf_invert_pivot_b<float><<<ig, 32, 0, stream>>>(
                plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols, st.d_frontBf,
                plan.front_total);
        else
            mf_invert_pivot_b<double><<<ig, 32, 0, stream>>>(
                plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols, st.d_frontB,
                plan.front_total);
    }
}

static void issue_solve_levels(const MultifrontalPlan& plan, BatchedState& st, cudaStream_t stream)
{
    const bool pure_fp32 = is_fp32_front(st.prec);  // float front: pure-FP32 or FP32-native TC
    const int B = st.B;
    const int sel = st.selinv ? 1 : 0;
    // Warp-packed small-level routing (mirrors the factor path): levels whose fronts are all small
    // run WARPS_PER_BLOCK warps per block instead of one 32-thread block per (front,batch), cutting
    // the block-launch/scheduling overhead of the tens of thousands of tiny leaf fronts that
    // dominate the solve.
    const int SMALL_THRESH = 32, SMALL_WARPS = 8;
    const int selt = pure_fp32 ? (int)sizeof(float) : (int)sizeof(double);
    auto level_max_fsz = [&](int b, int e) {
        int m = 0;
        for (int q = b; q < e; ++q) {
            const int pp = plan.h_plcols[q];
            const int fsz = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
            if (fsz > m) m = fsz;
        }
        return m;
    };
    for (int L = 0; L < plan.num_plevels; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        if (level_max_fsz(b, e) <= SMALL_THRESH) {
            const long warps = (long)(e - b) * B;
            const int blk = SMALL_WARPS * 32;
            const int gx = (int)((warps + SMALL_WARPS - 1) / SMALL_WARPS);
            const size_t shb = (size_t)SMALL_WARPS * 64 * selt;  // sh_piv[64] per warp
            if (pure_fp32)
                mf_fwd_small_warp_b<float><<<gx, blk, shb, stream>>>(
                    b, e - b, B, 64, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total,
                    plan.n, sel);
            else
                mf_fwd_small_warp_b<double><<<gx, blk, shb, stream>>>(
                    b, e - b, B, 64, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total,
                    plan.n, sel);
            continue;
        }
        // Bigger-front block levels (few fronts, low occupancy) get more threads/block to hide the
        // per-front latency and parallelize the CB update over more lanes.
        const int mf = level_max_fsz(b, e);
        const int tsb = mf <= 64 ? 64 : (mf <= 128 ? 128 : 256);
        const dim3 fg(e - b, B);
        if (pure_fp32)
            mf_fwd_level_b<float><<<fg, tsb, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n, sel);
        else
            mf_fwd_level_b<double><<<fg, tsb, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total, plan.n, sel);
    }
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
            if (pure_fp32)
                mf_bwd_small_warp_b<float><<<gx, blk, shb, stream>>>(
                    b, e - b, B, slab, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total,
                    plan.n, sel);
            else
                mf_bwd_small_warp_b<double><<<gx, blk, shb, stream>>>(
                    b, e - b, B, slab, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total,
                    plan.n, sel);
            continue;
        }
        const int mf = level_max_fsz(b, e);
        const int tsb = mf <= 64 ? 64 : (mf <= 128 ? 128 : 256);
        const dim3 bg(e - b, B);
        if (pure_fp32)
            mf_bwd_level_b<float><<<bg, tsb, (size_t)max_cb * sizeof(float), stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n, sel);
        else
            mf_bwd_level_b<double><<<bg, tsb, (size_t)max_cb * sizeof(double), stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total, plan.n, sel);
    }
}

bool batched_setup(const MultifrontalPlan& plan, int B, BatchPrecision prec, BatchedState& st)
{
    if (plan.num_panels == 0 || B <= 0) return false;
    st.B = B;
    st.front_total = plan.front_total;
    st.n = plan.n;
    st.prec = prec;
    // selinv = partitioned-inverse trick: factorize end에서 pivot block을 미리 역행렬화해 두면
    // solve의 trsv가 GEMV (완벽히 병렬)로 바뀜. MF_NO_SELINV=1로 끄면 solve는 sequential trsv로
    // 회귀 -- factor side의 mf_invert_pivot_b cost는 사라지지만 solve가 느려짐. trade-off 측정용.
    // selinv = partitioned-inverse trades factor time for solve time. Default OFF for the
    // 1-factor-1-solve power-flow workload; CLS_USE_SELINV=1 to re-enable.
    st.selinv = (std::getenv("CLS_USE_SELINV") != nullptr) &&
                (std::getenv("MF_NO_SELINV") == nullptr);
    const bool pure_fp32 = is_fp32_front(prec);                          // float front (FP32 / TC32)
    const bool need_double = !pure_fp32;                                 // FP64 front or Mixed/TC master
    const bool need_float = pure_fp32 || prec == BatchPrecision::Mixed ||
                            prec == BatchPrecision::TC;                  // FP32 front or Mixed/TC working
    const long fe = (long)B * plan.front_total;
    if (need_double && cudaMalloc(&st.d_frontB, fe * sizeof(double)) != cudaSuccess) return false;
    if (need_float && cudaMalloc(&st.d_frontBf, fe * sizeof(float)) != cudaSuccess) return false;
    // Solve working vector matches the solve precision: float for pure-FP32, double otherwise.
    if (pure_fp32) {
        if (cudaMalloc(&st.d_yBf, (long)B * plan.n * sizeof(float)) != cudaSuccess) return false;
    } else {
        if (cudaMalloc(&st.d_yB, (long)B * plan.n * sizeof(double)) != cudaSuccess) return false;
    }
    if (cudaMalloc(&st.d_sing, sizeof(int)) != cudaSuccess) return false;

    // The warp-packed small-front kernel stages SMALL_WARPS fronts of up to 32x32 in shared; for the
    // FP64 element type that is 8*1024*8 = 64 KB > the 48 KB default, so opt both instantiations in.
    cudaFuncSetAttribute(mf_factor_small_warp_b<float>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    cudaFuncSetAttribute(mf_factor_small_warp_b<double>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    // The shared-resident mid kernels stage the whole front (up to MID_THRESH^2 floats) plus the TC
    // staging into dynamic shared, which exceeds the 48 KB default; opt them in to the sm_86 cap.
    if (is_fp32_front(prec)) {
        cudaFuncSetAttribute(mf_factor_mid_tc32_b<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        cudaFuncSetAttribute(mf_factor_mid_tc32_b<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        // The TC32 big-separator path stages FP16 L/U + per-warp WMMA scratch for up to 32 warps,
        // which can exceed the 48 KB default shared.
        cudaFuncSetAttribute(mf_factor_extend_tc32_b,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        // Σ.14 — staged trailing kernel for FP32 batched path (was TC-only). Shared usage =
        // fsz_cap^2 + 2*nc*uc floats, up to ~96 KB for the largest mid fronts.
        cudaFuncSetAttribute(custom_linear_solver::tc::mf_factor_mid_tiled_b,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        // Σ.16-PROFILE — NT (no-trailing) clones for direct GEMM-time measurement.
        cudaFuncSetAttribute(custom_linear_solver::tc::mf_factor_mid_tiled_NT_b,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        cudaFuncSetAttribute(mf_factor_mid_tc32_NT_b,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        cudaFuncSetAttribute(mf_factor_small_warp_NT_b,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    }

#ifdef CLS_INTERNAL_GRAPH
    // Internal-graph mode: own a private stream and capture the factor / solve kernel sequences
    // into replayable CUDA graphs (default standalone behavior).
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    st.stream = stream;
    st.owns_stream = true;

    // Σ.11 — allocate subtree streams + fork/join events for multi-stream dispatch (default
    // ON; gated by CLS_NO_MULTISTREAM=1).
    const char* multi_no_su = std::getenv("CLS_NO_MULTISTREAM");
    const bool disable_multi_su = multi_no_su && multi_no_su[0] && multi_no_su[0] != '0';
    if (!disable_multi_su && plan.num_subtrees > 1 && plan.num_subtrees <= 8) {
        st.num_subtree_streams = plan.num_subtrees;
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStream_t ks;
            cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking);
            st.subtree_streams[k] = ks;
            cudaEvent_t je;
            cudaEventCreateWithFlags(&je, cudaEventDisableTiming);
            st.join_events[k] = je;
        }
        cudaEvent_t fe;
        cudaEventCreateWithFlags(&fe, cudaEventDisableTiming);
        st.fork_event = fe;
    }

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    issue_factor_levels(plan, st, stream);
    cudaGraph_t g;
    cudaStreamEndCapture(stream, &g);
    cudaGraphExec_t ge;
    cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    cudaGraphDestroy(g);
    st.factor_graph_exec = ge;

    // Capture batched solve graph (the gather/scatter wrappers stay outside, in batched_solve).
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    issue_solve_levels(plan, st, stream);
    cudaGraph_t sg;
    cudaStreamEndCapture(stream, &sg);
    cudaGraphExec_t sge;
    cudaGraphInstantiate(&sge, sg, nullptr, nullptr, 0);
    cudaGraphDestroy(sg);
    st.solve_graph_exec = sge;
    return cudaGetLastError() == cudaSuccess;
#else
    // External/capturable mode: no private stream and no internal graphs. The factor / solve
    // kernels are issued directly onto the caller stream (batched_set_stream) so an outer capture
    // records them. The caller must call batched_set_stream before batched_factorize/solve.
    return true;
#endif
}

void batched_set_stream(BatchedState& st, void* stream)
{
    // External mode: adopt the caller's stream (not owned -> not destroyed by ~BatchedState).
    st.stream = stream;
    st.owns_stream = false;
}

// Shared factorize body, templated on the input CSR value type VT (double or float). The scatter
// casts VT into whichever front the active precision mode consumes.
template <typename VT>
static bool batched_factorize_T(const MultifrontalPlan& plan, BatchedState& st, const VT* d_valuesB,
                                const int* d_o2c, double* kernel_ms)
{
    cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
    const long fe = (long)st.B * plan.front_total;
    const int T = 256;
    const dim3 sgrid((plan.nnz_a + T - 1) / T, st.B);
    // Zero + scatter A into the front the factor consumes: the FP32 front (pure-FP32) or the FP64
    // front / master (all other modes; the FP32 working arena is overwritten per front).
    auto issue_scatter = [&]() {
        cudaMemsetAsync(st.d_sing, 0, sizeof(int), stream);
        if (is_fp32_front(st.prec)) {
            cudaMemsetAsync(st.d_frontBf, 0, fe * sizeof(float), stream);
            scatter_batched<float, VT><<<sgrid, T, 0, stream>>>(plan.nnz_a, plan.front_total, d_o2c,
                                                                plan.d_a_pos, d_valuesB, st.d_frontBf);
        } else {
            cudaMemsetAsync(st.d_frontB, 0, fe * sizeof(double), stream);
            scatter_batched<double, VT><<<sgrid, T, 0, stream>>>(plan.nnz_a, plan.front_total, d_o2c,
                                                                 plan.d_a_pos, d_valuesB, st.d_frontB);
        }
    };
#ifdef CLS_INTERNAL_GRAPH
    cudaEvent_t k0, k1;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);
    issue_scatter();
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(st.factor_graph_exec), stream);
    cudaEventRecord(k1, stream);
    cudaEventSynchronize(k1);
    if (kernel_ms) { float ms = 0; cudaEventElapsedTime(&ms, k0, k1); *kernel_ms = ms; }
    cudaEventDestroy(k0);
    cudaEventDestroy(k1);
    return cudaGetLastError() == cudaSuccess;
#else
    // No graph, no host sync: issue scatter + the factor levels straight onto the caller stream so
    // the outer capture records them. Timing is unavailable here (a sync would break capture).
    (void)kernel_ms;
    issue_scatter();
    issue_factor_levels(plan, st, stream);
    return true;
#endif
}

// Shared solve body, templated on the RHS type RT and the solution type ST (the FP64 working
// vector yB is the same regardless). gather casts RHS->FP64, scatter casts FP64->solution.
template <typename RT, typename ST>
static bool batched_solve_T(const MultifrontalPlan& plan, BatchedState& st, const RT* d_rhsB,
                            ST* d_solB, const int* d_perm, double* kernel_ms)
{
    cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
    const int n = plan.n;
    const int T = 256;
    const dim3 vg((n + T - 1) / T, st.B);
    const bool pure_fp32 = is_fp32_front(st.prec);  // float front: pure-FP32 or FP32-native TC
    // The solve working vector matches the front the factor produced: float for pure-FP32, double
    // otherwise.
    auto issue_gather = [&]() {
        if (pure_fp32) gather_rhs_b<RT, float><<<vg, T, 0, stream>>>(n, d_rhsB, d_perm, st.d_yBf);
        else           gather_rhs_b<RT, double><<<vg, T, 0, stream>>>(n, d_rhsB, d_perm, st.d_yB);
    };
    auto issue_scatter_sol = [&]() {
        if (pure_fp32) scatter_sol_b<float, ST><<<vg, T, 0, stream>>>(n, st.d_yBf, d_perm, d_solB);
        else           scatter_sol_b<double, ST><<<vg, T, 0, stream>>>(n, st.d_yB, d_perm, d_solB);
    };
#ifdef CLS_INTERNAL_GRAPH
    cudaEvent_t k0, k1;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);
    issue_gather();
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(st.solve_graph_exec), stream);
    issue_scatter_sol();
    cudaEventRecord(k1, stream);
    cudaEventSynchronize(k1);
    if (kernel_ms) { float ms = 0; cudaEventElapsedTime(&ms, k0, k1); *kernel_ms = ms; }
    cudaEventDestroy(k0);
    cudaEventDestroy(k1);
    return cudaGetLastError() == cudaSuccess;
#else
    (void)kernel_ms;
    issue_gather();
    issue_solve_levels(plan, st, stream);
    issue_scatter_sol();
    return true;
#endif
}

// FP64-input entry points (existing API).
bool batched_factorize(const MultifrontalPlan& plan, BatchedState& st, const double* d_valuesB,
                       const int* d_o2c, double* kernel_ms)
{
    return batched_factorize_T<double>(plan, st, d_valuesB, d_o2c, kernel_ms);
}

bool batched_solve(const MultifrontalPlan& plan, BatchedState& st, const double* d_rhsB,
                   double* d_solB, const int* d_perm, double* kernel_ms)
{
    return batched_solve_T<double, double>(plan, st, d_rhsB, d_solB, d_perm, kernel_ms);
}

// FP32-input entry points (cuPF Mixed profile: float Jacobian, double RHS, float step).
bool batched_factorize(const MultifrontalPlan& plan, BatchedState& st, const float* d_valuesB,
                       const int* d_o2c, double* kernel_ms)
{
    return batched_factorize_T<float>(plan, st, d_valuesB, d_o2c, kernel_ms);
}

bool batched_solve(const MultifrontalPlan& plan, BatchedState& st, const double* d_rhsB,
                   float* d_solB, const int* d_perm, double* kernel_ms)
{
    return batched_solve_T<double, float>(plan, st, d_rhsB, d_solB, d_perm, kernel_ms);
}

bool batched_solve(const MultifrontalPlan& plan, BatchedState& st, const float* d_rhsB,
                   float* d_solB, const int* d_perm, double* kernel_ms)
{
    return batched_solve_T<float, float>(plan, st, d_rhsB, d_solB, d_perm, kernel_ms);
}

}  // namespace custom_linear_solver::batched
