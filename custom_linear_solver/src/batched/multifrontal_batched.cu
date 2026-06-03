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
#include "batched/solve_kernels.cuh"   // gather_rhs_b, scatter_sol_b, mf_{fwd,bwd}_level_b

namespace custom_linear_solver::batched {

using custom_linear_solver::plan::MultifrontalPlan;

BatchedState::~BatchedState()
{
    if (factor_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(factor_graph_exec));
    if (solve_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(solve_graph_exec));
    if (d_frontB) cudaFree(d_frontB);
    if (d_frontBf) cudaFree(d_frontBf);
    if (d_yB) cudaFree(d_yB);
    if (d_yBf) cudaFree(d_yBf);
    if (d_sing) cudaFree(d_sing);
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
static void issue_factor_levels(const MultifrontalPlan& plan, BatchedState& st, cudaStream_t stream)
{
    const BatchPrecision prec = st.prec;
    const bool pure_fp32 = (prec == BatchPrecision::FP32);
    const int B = st.B;
    const int T = 128;
    const int do_extend = 1;
    // Per level, gridDim=(level_size, B); kernel chosen by precision.
    for (int L = 0; L < plan.num_plevels; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        dim3 grid(e - b, B);
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
            case BatchPrecision::TC: {
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
                mf_factor_extend_mixed_tc_b<<<grid, 128, shbytes, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB, st.d_frontBf,
                    plan.front_total, st.d_sing, do_extend, ucp_max);
                break;
            }
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
    const bool pure_fp32 = (st.prec == BatchPrecision::FP32);
    const int B = st.B;
    const int sel = st.selinv ? 1 : 0;
    // 32 threads (1 warp) per (front,batch): the warp-parallel pivot solve uses one warp anyway and
    // smaller blocks pack more per SM -> higher occupancy across the many B*fronts blocks (swept).
    const int TS = 32;
    for (int L = 0; L < plan.num_plevels; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        const dim3 fg(e - b, B);
        if (pure_fp32)
            mf_fwd_level_b<float><<<fg, TS, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n, sel);
        else
            mf_fwd_level_b<double><<<fg, TS, 0, stream>>>(
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
        const dim3 bg(e - b, B);
        if (pure_fp32)
            mf_bwd_level_b<float><<<bg, TS, (size_t)max_cb * sizeof(float), stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n, sel);
        else
            mf_bwd_level_b<double><<<bg, TS, (size_t)max_cb * sizeof(double), stream>>>(
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
    st.selinv = true;
    const bool pure_fp32 = (prec == BatchPrecision::FP32);
    const bool need_double = (prec != BatchPrecision::FP32);              // FP64 front or Mixed/TC master
    const bool need_float = (prec == BatchPrecision::FP32 || prec == BatchPrecision::Mixed ||
                             prec == BatchPrecision::TC);                 // FP32 front or Mixed/TC working
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

#ifdef CLS_INTERNAL_GRAPH
    // Internal-graph mode: own a private stream and capture the factor / solve kernel sequences
    // into replayable CUDA graphs (default standalone behavior).
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    st.stream = stream;
    st.owns_stream = true;

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
        if (st.prec == BatchPrecision::FP32) {
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
    const bool pure_fp32 = (st.prec == BatchPrecision::FP32);
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
