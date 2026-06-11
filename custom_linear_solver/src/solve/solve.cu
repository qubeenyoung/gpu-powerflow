#include "solve/solve.hpp"

#include <cuda_runtime.h>

#include "solve/permute.cuh"    // gather_rhs, scatter_sol (I/O permutation)
#include "solve/dispatch.cuh"   // issue_solve_levels

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

#ifdef CLS_INTERNAL_GRAPH
static void destroy_full_solve_graph(State& st)
{
    if (st.full_solve_graph_exec)
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(st.full_solve_graph_exec));
    st.full_solve_graph_exec = nullptr;
    st.full_solve_rhs = nullptr;
    st.full_solve_solution = nullptr;
    st.full_solve_perm = nullptr;
    st.full_solve_iperm = nullptr;
    st.full_solve_type_tag = 0;
}

template <typename RhsT, typename SolutionT>
static constexpr int full_solve_type_tag()
{
    return (int)sizeof(RhsT) * 16 + (int)sizeof(SolutionT);
}
#endif

}  // namespace

// Shared solve body, templated on the RHS type RhsT and the solution type SolutionT. gather casts
// RhsT → (FP32 / FP64) working vector; scatter casts working vector → SolutionT.
template <typename RhsT, typename SolutionT>
static bool solve_impl(const MultifrontalPlan& plan, State& st, const RhsT* d_rhs_batch,
                       SolutionT* d_solution_batch, const int* d_perm, const int* d_iperm)
{
    cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
    const int n = plan.num_rows;
    constexpr int threads_per_block = 256;
    const dim3 permute_grid((n + threads_per_block - 1) / threads_per_block, st.batch_count);
    constexpr int batched_permute_rows = 256;
    constexpr int batched_permute_batches = 2;
    const dim3 batched_permute_block(batched_permute_rows, batched_permute_batches);
    const dim3 batched_permute_grid((n + batched_permute_rows - 1) / batched_permute_rows,
                                    (st.batch_count + batched_permute_batches - 1) / batched_permute_batches);
    const bool use_batched_permute = st.batch_count >= 16;
    const bool float_front = is_fp32_front(st.precision);
    auto issue_gather = [&]() {
        if (use_batched_permute) {
            if (float_front) gather_rhs_batched<RhsT, float><<<batched_permute_grid, batched_permute_block, 0, stream>>>(n, st.batch_count, d_rhs_batch, d_perm, st.d_y_batch_f);
            else             gather_rhs_batched<RhsT, double><<<batched_permute_grid, batched_permute_block, 0, stream>>>(n, st.batch_count, d_rhs_batch, d_perm, st.d_y_batch);
        } else {
            if (float_front) gather_rhs<RhsT, float><<<permute_grid, threads_per_block, 0, stream>>>(n, d_rhs_batch, d_perm, st.d_y_batch_f);
            else             gather_rhs<RhsT, double><<<permute_grid, threads_per_block, 0, stream>>>(n, d_rhs_batch, d_perm, st.d_y_batch);
        }
    };
    auto issue_scatter_sol = [&]() {
        if (use_batched_permute) {
            if (float_front) scatter_sol_inverse_batched<float, SolutionT><<<batched_permute_grid, batched_permute_block, 0, stream>>>(n, st.batch_count, st.d_y_batch_f, d_iperm, d_solution_batch);
            else             scatter_sol_inverse_batched<double, SolutionT><<<batched_permute_grid, batched_permute_block, 0, stream>>>(n, st.batch_count, st.d_y_batch, d_iperm, d_solution_batch);
        } else {
            if (float_front) scatter_sol_inverse<float, SolutionT><<<permute_grid, threads_per_block, 0, stream>>>(n, st.d_y_batch_f, d_iperm, d_solution_batch);
            else             scatter_sol_inverse<double, SolutionT><<<permute_grid, threads_per_block, 0, stream>>>(n, st.d_y_batch, d_iperm, d_solution_batch);
        }
    };
#ifdef CLS_INTERNAL_GRAPH
    const int type_tag = full_solve_type_tag<RhsT, SolutionT>();
    const bool cache_hit =
        st.full_solve_graph_exec &&
        st.full_solve_rhs == static_cast<const void*>(d_rhs_batch) &&
        st.full_solve_solution == static_cast<void*>(d_solution_batch) &&
        st.full_solve_perm == d_perm &&
        st.full_solve_iperm == d_iperm &&
        st.full_solve_type_tag == type_tag;

    if (!cache_hit) {
        destroy_full_solve_graph(st);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        issue_gather();
        issue_solve_levels(plan, st, stream);
        issue_scatter_sol();
        cudaGraph_t full_graph;
        cudaStreamEndCapture(stream, &full_graph);
        cudaGraphExec_t full_exec;
        cudaGraphInstantiate(&full_exec, full_graph, nullptr, nullptr, 0);
        cudaGraphDestroy(full_graph);
        st.full_solve_graph_exec = full_exec;
        st.full_solve_rhs = static_cast<const void*>(d_rhs_batch);
        st.full_solve_solution = static_cast<void*>(d_solution_batch);
        st.full_solve_perm = d_perm;
        st.full_solve_iperm = d_iperm;
        st.full_solve_type_tag = type_tag;
    }

    cudaGraphLaunch(static_cast<cudaGraphExec_t>(st.full_solve_graph_exec), stream);
    return cudaGetLastError() == cudaSuccess;
#else
    issue_gather();
    issue_solve_levels(plan, st, stream);
    issue_scatter_sol();
    return true;
#endif
}

bool solve(const MultifrontalPlan& plan, State& st, const double* d_rhs_batch,
           double* d_solution_batch, const int* d_perm, const int* d_iperm)
{
    return solve_impl<double, double>(plan, st, d_rhs_batch, d_solution_batch, d_perm, d_iperm);
}

bool solve(const MultifrontalPlan& plan, State& st, const double* d_rhs_batch,
           float* d_solution_batch, const int* d_perm, const int* d_iperm)
{
    return solve_impl<double, float>(plan, st, d_rhs_batch, d_solution_batch, d_perm, d_iperm);
}

bool solve(const MultifrontalPlan& plan, State& st, const float* d_rhs_batch,
           float* d_solution_batch, const int* d_perm, const int* d_iperm)
{
    return solve_impl<float, float>(plan, st, d_rhs_batch, d_solution_batch, d_perm, d_iperm);
}
}  // namespace custom_linear_solver
