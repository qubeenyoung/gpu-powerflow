#include "multifrontal.hpp"

#include <algorithm>
#include <climits>

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
#include "factorize/scatter.cuh"  // scatter_values (input setup)
#include "factorize/dispatch.cuh" // issue_factor_levels + level-range dispatchers
                                  // (transitively pulls in factorize/kernels.cuh, phases.cuh)
#include "solve/permute.cuh"      // gather_rhs, scatter_sol (I/O permutation)
#include "solve/dispatch.cuh"     // issue_solve_levels
                                  // (transitively pulls in solve/kernels.cuh, phases.cuh)

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

// is_fp32_front / round_up_int are inline in multifrontal.hpp (shared with the dispatchers).

State::~State()
{
    if (factor_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(factor_graph_exec));
    if (solve_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(solve_graph_exec));
    if (d_front_batch) cudaFree(d_front_batch);
    if (d_front_batch_f) cudaFree(d_front_batch_f);
    if (d_y_batch) cudaFree(d_y_batch);
    if (d_y_batch_f) cudaFree(d_y_batch_f);
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
// safe both to capture into the internal CUDA graph (setup with CLS_INTERNAL_GRAPH on) and to
// record into an external stream capture (cuPF's whole-iteration graph).
//
// Factor dispatch (issue_factor_levels / _level_range) lives in factorize/dispatch.cuh.
// Solve dispatch (issue_solve_levels) lives in solve/dispatch.cuh.
// ---------------------------------------------------------------------------

bool setup(const MultifrontalPlan& plan, int B, Precision prec, State& st,
           bool use_multistream_subtrees, bool tier_split)
{
    if (plan.num_panels == 0 || B <= 0) return false;
    st.batch_count = B;
    st.front_total = plan.front_total;
    st.num_rows = plan.num_rows;
    st.precision = prec;
    st.tier_split = tier_split;
    // Float-front modes: FP32 / FP16 (PTX) / TF32 (PTX) all run on the f32 arena.
    const bool float_front = is_fp32_front(prec);
    const long fe = (long)B * plan.front_total;
    if (float_front) {
        if (cudaMalloc(&st.d_front_batch_f, fe * sizeof(float)) != cudaSuccess) return false;
    } else {
        if (cudaMalloc(&st.d_front_batch, fe * sizeof(double)) != cudaSuccess) return false;
    }
    // Solve working vector matches the solve precision: float for pure-FP32, double otherwise.
    if (float_front) {
        if (cudaMalloc(&st.d_y_batch_f, (long)B * plan.num_rows * sizeof(float)) != cudaSuccess) return false;
    } else {
        if (cudaMalloc(&st.d_y_batch, (long)B * plan.num_rows * sizeof(double)) != cudaSuccess) return false;
    }
    if (cudaMalloc(&st.d_sing, sizeof(int)) != cudaSuccess) return false;

    // All shared-resident kernels can exceed the 48 KB default; opt them in to the sm_86 cap.
    cudaFuncSetAttribute(factor_small<float, 8>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_small<float, 16>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_small<float, 32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_small<double, 8>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_small<double, 16>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_small<double, 32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_mid<float>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_mid<double>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    if (is_fp32_front(prec)) {
        cudaFuncSetAttribute(factor_mid_fp16_ptx,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
        cudaFuncSetAttribute(factor_big_fp16_ptx,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
        cudaFuncSetAttribute(factor_mid_tf32_ptx<8>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
        cudaFuncSetAttribute(factor_mid_tf32_ptx<4>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
        cudaFuncSetAttribute(factor_big_tf32_ptx,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    }

#ifdef CLS_INTERNAL_GRAPH
    // Internal-graph mode: own a private stream and capture the factor / solve kernel sequences
    // into replayable CUDA graphs (default standalone behavior).
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    st.stream = stream;
    st.owns_stream = true;

    // Allocate subtree streams + fork/join events for multi-stream dispatch. One stream per
    // independent subtree (capped at 8). The caller can opt out via SolverConfig.use_multistream_subtrees;
    // when subtree_streams stays empty, issue_factor_levels falls back to single-stream dispatch.
    if (use_multistream_subtrees && plan.num_subtrees > 1 && plan.num_subtrees <= 8) {
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
    // records them. The caller must call set_stream before factorize/solve. Subtree streams
    // would defeat the outer capture, so multi-stream dispatch is disabled in this mode.
    (void)use_multistream_subtrees;
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
                           const int* d_ordered_value_to_csr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
    const long fe = (long)st.batch_count * plan.front_total;
    constexpr int T = 256;
    const dim3 sgrid((plan.nnz + T - 1) / T, st.batch_count);
    // Zero + scatter A into the front the factor consumes.
    auto issue_scatter = [&]() {
        cudaMemsetAsync(st.d_sing, 0, sizeof(int), stream);
        if (is_fp32_front(st.precision)) {
            cudaMemsetAsync(st.d_front_batch_f, 0, fe * sizeof(float), stream);
            scatter_values<float, VT><<<sgrid, T, 0, stream>>>(plan.nnz, plan.front_total, d_ordered_value_to_csr,
                                                               plan.d_a_pos, d_valuesB, st.d_front_batch_f);
        } else {
            cudaMemsetAsync(st.d_front_batch, 0, fe * sizeof(double), stream);
            scatter_values<double, VT><<<sgrid, T, 0, stream>>>(plan.nnz, plan.front_total, d_ordered_value_to_csr,
                                                                plan.d_a_pos, d_valuesB, st.d_front_batch);
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
    const int n = plan.num_rows;
    constexpr int T = 256;
    const dim3 vg((n + T - 1) / T, st.batch_count);
    const bool float_front = is_fp32_front(st.precision);
    auto issue_gather = [&]() {
        if (float_front) gather_rhs<RT, float><<<vg, T, 0, stream>>>(n, d_rhsB, d_perm, st.d_y_batch_f);
        else           gather_rhs<RT, double><<<vg, T, 0, stream>>>(n, d_rhsB, d_perm, st.d_y_batch);
    };
    auto issue_scatter_sol = [&]() {
        if (float_front) scatter_sol<float, ST><<<vg, T, 0, stream>>>(n, st.d_y_batch_f, d_perm, d_solB);
        else           scatter_sol<double, ST><<<vg, T, 0, stream>>>(n, st.d_y_batch, d_perm, d_solB);
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
               const int* d_ordered_value_to_csr)
{
    return factorize_impl<double>(plan, st, d_valuesB, d_ordered_value_to_csr);
}

bool solve(const MultifrontalPlan& plan, State& st, const double* d_rhsB,
           double* d_solB, const int* d_perm)
{
    return solve_impl<double, double>(plan, st, d_rhsB, d_solB, d_perm);
}

// FP32-input overloads (float values / RHS / solution combinations).
bool factorize(const MultifrontalPlan& plan, State& st, const float* d_valuesB,
               const int* d_ordered_value_to_csr)
{
    return factorize_impl<float>(plan, st, d_valuesB, d_ordered_value_to_csr);
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
