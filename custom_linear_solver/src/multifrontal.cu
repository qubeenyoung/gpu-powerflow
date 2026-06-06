#include "multifrontal.hpp"

#include <cuda_runtime.h>

// Uniform-batch multifrontal factorize + solve. All kernels are front-major (gridDim.y = batch,
// arena = B * front_total) and live in internal headers below; the build uses
// CUDA_SEPARABLE_COMPILATION OFF, so every launched kernel must be defined in this TU.
//
// Internal-graph toggle (CLS_INTERNAL_GRAPH, default ON via CMake):
//   ON  - standalone behavior: setup owns a private stream and captures the factor /
//         solve kernel sequences into replayable CUDA graphs; factorize/solve cudaGraphLaunch
//         them and host-sync for timing.
//   OFF - external/capturable mode (used when an OUTER capture owns the graph, e.g. cuPF's
//         whole-iteration CUDA graph): no private stream, no internal graphs, no host sync.
#include "factorize/scatter.cuh"          // values gather/scatter
#include "factorize/big.cuh"           // factor_big<T>, factor_big_tc
#include "factorize/small.cuh"         // factor_small<T>
#include "factorize/mid.cuh"           // factor_mid<T> (tiled FP32/FP64), factor_mid_tc
#include "solve/big.cuh"                  // solve_fwd<T>, solve_bwd<T>
#include "solve/small.cuh"                // solve_fwd_small<T>, solve_bwd_small<T>

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

// The front arena is FP32 (and the solve runs in float) for both the pure-FP32 path and the
// FP32-native tensor-core path; only FP64 keeps an FP64 front.
static inline bool is_fp32_front(Precision p)
{
    return p == Precision::FP32 || p == Precision::TC;
}

State::~State()
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
// Kernel-launch sequences. Pure kernel launches only (no graph / host-sync here), so they are
// safe both to capture into the internal CUDA graph (setup with CLS_INTERNAL_GRAPH on)
// and to record into an external stream capture (cuPF's whole-iteration graph).
// ---------------------------------------------------------------------------
//
// Factor dispatch (issue_factor_level_range):
//   For each per-level [b, e) slice of plcols, choose a kernel by precision and front-size tier.
//
//     SMALL (max_fsz ≤ SMALL_THRESH = 32)
//         factor_small<T>        — warp per (front, batch), 8 warps/block
//
//     MID   (SMALL_THRESH < max_fsz ≤ MID_THRESH = 128, shared budget permitting)
//         FP64 / FP32 → factor_mid<T>   — shared-resident front + staged L / U trailing
//         TC          → factor_mid_tc    — shared-resident front + FP16 WMMA trailing
//
//     BIG   (front too big for shared, or precision-specific cases)
//         FP64 → factor_big<double>      —  128 threads/block, global memory
//         FP32 → factor_big<float>       — 1024 threads/block, global memory
//         TC   → factor_big_tc           — 1024 threads/block, FP16 WMMA trailing
//
// MID_THRESH = 128 aligns the boundary with WMMA's 16-wide tiles (128 = 8 × 16). The actual
// mid eligibility for each level is a shared-budget check: front + 2 * nc * uc panels must
// fit in 96 KB; otherwise the level falls through to the big tier. FP64 doubles the per-cell
// cost, so FP64 mid fronts can only fit when fsz ≲ 88 (≈ 96 KB / 8 bytes – staging margin).
static void issue_factor_level_range(const MultifrontalPlan& plan, State& st,
                                     cudaStream_t stream, int b, int e)
{
    const Precision prec = st.prec;
    const bool pure_fp32 = is_fp32_front(prec);
    const int B = st.B;
    constexpr int do_extend = 1;
    constexpr int SMALL_THRESH = 32;
    constexpr int SMALL_WARPS = 8;
    constexpr int MID_THRESH = 128;
    constexpr size_t MID_SHARED_BUDGET = 96 * 1024;  // sm_86 max dynamic shared per block
    if (e <= b) return;
    const int level_size = e - b;

    int max_fsz = 0, max_uc = 1, level_max_nc = 1, level_max_uc = 1;
    for (int q = b; q < e; ++q) {
        const int pp = plan.h_plcols[q];
        const int fsz = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
        const int nc = plan.h_ncols[pp];
        const int uc = fsz - nc;
        if (fsz > max_fsz) max_fsz = fsz;
        if (uc > max_uc && uc <= 256) max_uc = uc;
        if (nc > level_max_nc) level_max_nc = nc;
        if (uc > level_max_uc) level_max_uc = uc;
    }

    // ----- SMALL tier: warp-packed kernel ------------------------------------------------
    if (max_fsz <= SMALL_THRESH) {
        const long warps = (long)level_size * B;
        const int blk = SMALL_WARPS * 32;
        const int gx = (int)((warps + SMALL_WARPS - 1) / SMALL_WARPS);
        const int fsz2cap = max_fsz * max_fsz;
        const size_t elt = (prec == Precision::FP64) ? sizeof(double) : sizeof(float);
        const size_t shb = (size_t)SMALL_WARPS * fsz2cap * elt;
        if (prec == Precision::FP64)
            factor_small<double><<<gx, blk, shb, stream>>>(
                b, level_size, B, fsz2cap, plan.d_plcols, plan.d_front_off,
                plan.d_front_ptr, plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr,
                plan.d_asm_local, st.d_frontB, plan.front_total, st.d_sing, do_extend);
        else
            factor_small<float><<<gx, blk, shb, stream>>>(
                b, level_size, B, fsz2cap, plan.d_plcols, plan.d_front_off,
                plan.d_front_ptr, plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr,
                plan.d_asm_local, st.d_frontBf, plan.front_total, st.d_sing, do_extend);
        return;
    }

    // ----- MID tier: shared-resident kernel if budget fits -------------------------------
    if (max_fsz <= MID_THRESH) {
        const int fsz_cap = max_fsz;
        const int ucp_max = ((max_uc + 15) / 16) * 16;
        dim3 grid(level_size, B);

        if (prec == Precision::TC) {
            const size_t shb = (size_t)fsz_cap * fsz_cap * sizeof(float)
                             + (size_t)2 * ucp_max * 32 * sizeof(__half)
                             + 4 * 256 * sizeof(float);
            if (shb <= MID_SHARED_BUDGET) {
                const int mblk = max_fsz <= 48 ? 64 : (max_fsz <= 80 ? 128 : 256);
                factor_mid_tc<<<grid, mblk, shb, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                    plan.front_total, st.d_sing, do_extend, ucp_max, fsz_cap);
                return;
            }
        } else {
            // FP32 / FP64 tiled mid kernel.
            const size_t elt = (prec == Precision::FP64) ? sizeof(double) : sizeof(float);
            const size_t shb_tiled = (size_t)fsz_cap * fsz_cap * elt
                                   + (size_t)2 * level_max_nc * level_max_uc * elt;
            if (shb_tiled <= MID_SHARED_BUDGET) {
                if (prec == Precision::FP64)
                    factor_mid<double><<<grid, 256, shb_tiled, stream>>>(
                        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB,
                        plan.front_total, st.d_sing, do_extend, fsz_cap, level_max_nc, level_max_uc);
                else
                    factor_mid<float><<<grid, 256, shb_tiled, stream>>>(
                        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                        plan.front_total, st.d_sing, do_extend, fsz_cap, level_max_nc, level_max_uc);
                return;
            }
        }
        // Shared budget exceeded — fall through to the big tier.
    }

    // ----- BIG tier: global-memory kernel ------------------------------------------------
    dim3 grid(level_size, B);
    if (prec == Precision::FP64) {
        constexpr int T = 128;
        factor_big<double><<<grid, T, 0, stream>>>(
            b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB,
            plan.front_total, st.d_sing, do_extend);
        return;
    }
    constexpr int bigT = 1024;
    if (prec == Precision::TC) {
        const int ucp_max = ((max_uc + 15) / 16) * 16;
        const size_t shbytes =
            (size_t)2 * ucp_max * 32 * sizeof(__half) + (bigT / 32) * 256 * sizeof(float);
        factor_big_tc<<<grid, bigT, shbytes, stream>>>(
            b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
            plan.front_total, st.d_sing, do_extend, ucp_max);
        return;
    }
    // FP32 BIG
    factor_big<float><<<grid, bigT, 0, stream>>>(
        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
        plan.front_total, st.d_sing, do_extend);
}

// Iterate etree levels for the factor phase. When the plan has subtree streams, dispatch each
// independent subtree on its own stream and join before the spine levels (which run on the
// main stream). Otherwise iterate levels sequentially on the main stream.
static void issue_factor_levels(const MultifrontalPlan& plan, State& st, cudaStream_t stream)
{
    const bool use_multistream = st.num_subtree_streams > 1 &&
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

}

static void issue_solve_levels(const MultifrontalPlan& plan, State& st, cudaStream_t stream)
{
    const bool pure_fp32 = is_fp32_front(st.prec);  // float front: pure-FP32 or FP32-native TC
    const int B = st.B;
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
        // Bigger-front block levels (few fronts, low occupancy) get more threads/block to hide the
        // per-front latency and parallelize the CB update over more lanes.
        const int mf = level_max_fsz(b, e);
        const int tsb = mf <= 64 ? 64 : (mf <= 128 ? 128 : 256);
        const dim3 fg(e - b, B);
        if (pure_fp32)
            solve_fwd<float><<<fg, tsb, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n);
        else
            solve_fwd<double><<<fg, tsb, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total, plan.n);
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
        if (pure_fp32)
            solve_bwd<float><<<bg, tsb, (size_t)max_cb * sizeof(float), stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n);
        else
            solve_bwd<double><<<bg, tsb, (size_t)max_cb * sizeof(double), stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total, plan.n);
    }
}

bool setup(const MultifrontalPlan& plan, int B, Precision prec, State& st)
{
    if (plan.num_panels == 0 || B <= 0) return false;
    st.B = B;
    st.front_total = plan.front_total;
    st.n = plan.n;
    st.prec = prec;
    const bool pure_fp32 = is_fp32_front(prec);                          // float front (FP32 / TC)
    const long fe = (long)B * plan.front_total;
    if (pure_fp32) {
        if (cudaMalloc(&st.d_frontBf, fe * sizeof(float)) != cudaSuccess) return false;
    } else {
        if (cudaMalloc(&st.d_frontB, fe * sizeof(double)) != cudaSuccess) return false;
    }
    // Solve working vector matches the solve precision: float for pure-FP32, double otherwise.
    if (pure_fp32) {
        if (cudaMalloc(&st.d_yBf, (long)B * plan.n * sizeof(float)) != cudaSuccess) return false;
    } else {
        if (cudaMalloc(&st.d_yB, (long)B * plan.n * sizeof(double)) != cudaSuccess) return false;
    }
    if (cudaMalloc(&st.d_sing, sizeof(int)) != cudaSuccess) return false;

    // All shared-resident kernels can exceed the 48 KB default; opt them in to the sm_86 cap.
    cudaFuncSetAttribute(factor_small<float>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    cudaFuncSetAttribute(factor_small<double>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    cudaFuncSetAttribute(factor_mid<float>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    cudaFuncSetAttribute(factor_mid<double>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    if (is_fp32_front(prec)) {
        cudaFuncSetAttribute(factor_mid_tc,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        cudaFuncSetAttribute(factor_big_tc,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    }

#ifdef CLS_INTERNAL_GRAPH
    // Internal-graph mode: own a private stream and capture the factor / solve kernel sequences
    // into replayable CUDA graphs (default standalone behavior).
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    st.stream = stream;
    st.owns_stream = true;

    // Allocate subtree streams + fork/join events for multi-stream dispatch. One stream per
    // independent subtree (capped at 8).
    if (plan.num_subtrees > 1 && plan.num_subtrees <= 8) {
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

    // Capture batched solve graph (the gather/scatter wrappers stay outside, in solve).
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
    // kernels are issued directly onto the caller stream (set_stream) so an outer capture
    // records them. The caller must call set_stream before factorize/solve.
    return true;
#endif
}

void set_stream(State& st, void* stream)
{
    // External mode: adopt the caller's stream (not owned -> not destroyed by ~State).
    st.stream = stream;
    st.owns_stream = false;
}

// Shared factorize body, templated on the input value type VT. The scatter casts VT into
// whichever front the active precision mode consumes.
template <typename VT>
static bool factorize_impl(const MultifrontalPlan& plan, State& st, const VT* d_valuesB,
                           const int* d_o2c)
{
    cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
    const long fe = (long)st.B * plan.front_total;
    constexpr int T = 256;
    const dim3 sgrid((plan.nnz_a + T - 1) / T, st.B);
    // Zero + scatter A into the front the factor consumes.
    auto issue_scatter = [&]() {
        cudaMemsetAsync(st.d_sing, 0, sizeof(int), stream);
        if (is_fp32_front(st.prec)) {
            cudaMemsetAsync(st.d_frontBf, 0, fe * sizeof(float), stream);
            scatter_values<float, VT><<<sgrid, T, 0, stream>>>(plan.nnz_a, plan.front_total, d_o2c,
                                                               plan.d_a_pos, d_valuesB, st.d_frontBf);
        } else {
            cudaMemsetAsync(st.d_frontB, 0, fe * sizeof(double), stream);
            scatter_values<double, VT><<<sgrid, T, 0, stream>>>(plan.nnz_a, plan.front_total, d_o2c,
                                                                plan.d_a_pos, d_valuesB, st.d_frontB);
        }
    };
#ifdef CLS_INTERNAL_GRAPH
    // Internal-graph mode: scatter onto the private stream, replay the captured factor graph,
    // and sync. Callers measure wall time externally.
    issue_scatter();
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(st.factor_graph_exec), stream);
    cudaStreamSynchronize(stream);
    return cudaGetLastError() == cudaSuccess;
#else
    // External / capturable mode: issue scatter + the factor levels straight onto the caller
    // stream so the outer capture records them. No host sync (it would break capture).
    issue_scatter();
    issue_factor_levels(plan, st, stream);
    return true;
#endif
}

// Shared solve body, templated on the RHS type RT and the solution type ST. gather casts
// RT → (FP32 / FP64) working vector; scatter casts working vector → ST.
template <typename RT, typename ST>
static bool solve_impl(const MultifrontalPlan& plan, State& st, const RT* d_rhsB,
                       ST* d_solB, const int* d_perm)
{
    cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
    const int n = plan.n;
    constexpr int T = 256;
    const dim3 vg((n + T - 1) / T, st.B);
    const bool pure_fp32 = is_fp32_front(st.prec);
    auto issue_gather = [&]() {
        if (pure_fp32) gather_rhs<RT, float><<<vg, T, 0, stream>>>(n, d_rhsB, d_perm, st.d_yBf);
        else           gather_rhs<RT, double><<<vg, T, 0, stream>>>(n, d_rhsB, d_perm, st.d_yB);
    };
    auto issue_scatter_sol = [&]() {
        if (pure_fp32) scatter_sol<float, ST><<<vg, T, 0, stream>>>(n, st.d_yBf, d_perm, d_solB);
        else           scatter_sol<double, ST><<<vg, T, 0, stream>>>(n, st.d_yB, d_perm, d_solB);
    };
#ifdef CLS_INTERNAL_GRAPH
    issue_gather();
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(st.solve_graph_exec), stream);
    issue_scatter_sol();
    cudaStreamSynchronize(stream);
    return cudaGetLastError() == cudaSuccess;
#else
    issue_gather();
    issue_solve_levels(plan, st, stream);
    issue_scatter_sol();
    return true;
#endif
}

// FP64-input entry points.
bool factorize(const MultifrontalPlan& plan, State& st, const double* d_valuesB,
               const int* d_o2c)
{
    return factorize_impl<double>(plan, st, d_valuesB, d_o2c);
}

bool solve(const MultifrontalPlan& plan, State& st, const double* d_rhsB,
           double* d_solB, const int* d_perm)
{
    return solve_impl<double, double>(plan, st, d_rhsB, d_solB, d_perm);
}

// FP32-input overloads (float values / RHS / solution combinations).
bool factorize(const MultifrontalPlan& plan, State& st, const float* d_valuesB,
               const int* d_o2c)
{
    return factorize_impl<float>(plan, st, d_valuesB, d_o2c);
}

bool solve(const MultifrontalPlan& plan, State& st, const double* d_rhsB,
           float* d_solB, const int* d_perm)
{
    return solve_impl<double, float>(plan, st, d_rhsB, d_solB, d_perm);
}

bool solve(const MultifrontalPlan& plan, State& st, const float* d_rhsB,
           float* d_solB, const int* d_perm)
{
    return solve_impl<float, float>(plan, st, d_rhsB, d_solB, d_perm);
}

}  // namespace custom_linear_solver
