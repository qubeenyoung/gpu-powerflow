#include "solve/solve.hpp"

#include <cuda_runtime.h>

#include "solve/permute.cuh"    // gather_rhs, scatter_sol (I/O permutation)
#include "solve/dispatch.cuh"   // issue_solve_levels

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

void issue_solve(const MultifrontalPlan& plan, State& st, void* stream)
{
    issue_solve_levels(plan, st, static_cast<cudaStream_t>(stream));
}

// Shared solve body, templated on the RHS type RhsT and the solution type SolutionT. gather casts
// RhsT → (FP32 / FP64) working vector; scatter casts working vector → SolutionT.
template <typename RhsT, typename SolutionT>
static bool solve_impl(const MultifrontalPlan& plan, State& st, const RhsT* d_rhs_batch,
                       SolutionT* d_solution_batch, const int* d_perm)
{
    cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
    const int n = plan.num_rows;
    constexpr int threads_per_block = 256;
    const dim3 permute_grid((n + threads_per_block - 1) / threads_per_block, st.batch_count);
    const bool float_front = is_fp32_front(st.precision);
    auto issue_gather = [&]() {
        if (float_front) gather_rhs<RhsT, float><<<permute_grid, threads_per_block, 0, stream>>>(n, d_rhs_batch, d_perm, st.d_y_batch_f);
        else           gather_rhs<RhsT, double><<<permute_grid, threads_per_block, 0, stream>>>(n, d_rhs_batch, d_perm, st.d_y_batch);
    };
    auto issue_scatter_sol = [&]() {
        if (float_front) scatter_sol<float, SolutionT><<<permute_grid, threads_per_block, 0, stream>>>(n, st.d_y_batch_f, d_perm, d_solution_batch);
        else           scatter_sol<double, SolutionT><<<permute_grid, threads_per_block, 0, stream>>>(n, st.d_y_batch, d_perm, d_solution_batch);
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

bool solve(const MultifrontalPlan& plan, State& st, const double* d_rhs_batch,
           double* d_solution_batch, const int* d_perm)
{
    return solve_impl<double, double>(plan, st, d_rhs_batch, d_solution_batch, d_perm);
}

bool solve(const MultifrontalPlan& plan, State& st, const double* d_rhs_batch,
           float* d_solution_batch, const int* d_perm)
{
    return solve_impl<double, float>(plan, st, d_rhs_batch, d_solution_batch, d_perm);
}

bool solve(const MultifrontalPlan& plan, State& st, const float* d_rhs_batch,
           float* d_solution_batch, const int* d_perm)
{
    return solve_impl<float, float>(plan, st, d_rhs_batch, d_solution_batch, d_perm);
}
}  // namespace custom_linear_solver
