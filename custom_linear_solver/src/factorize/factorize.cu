#include "factorize/factorize.hpp"

#include <cuda_runtime.h>
#include <vector>

#include "factorize/assemble.cuh"   // assemble_front_values (input assembly)
#include "factorize/schedule.cuh"   // issue_factor_levels (factor tree schedule)

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

// Opt the shared-resident kernels into the sm_86 dynamic-shared cap (they exceed the 48 KB
// default). The PTX tensor-core variants only run on the float-front path.
void register_factor_attributes(Precision precision)
{
    (void)precision;
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
    cudaFuncSetAttribute(factor_mid<float, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_mid<float, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_mid<double, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_big<double, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_big<float, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_big<float, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
}

void issue_factor(const MultifrontalPlan& plan, State& st, void* stream)
{
    issue_factor_levels(plan, st, static_cast<cudaStream_t>(stream));
}

// Shared factorize body, templated on the input value type ValueT. The scatter casts ValueT into
// whichever front the active precision mode consumes.
template <typename ValueT>
static bool factorize_impl(const MultifrontalPlan& plan, State& st, const ValueT* d_values_batch,
                           const int* d_ordered_value_to_csr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
    const long front_elements = (long)st.batch_count * plan.front_total;
    constexpr int threads_per_block = 256;
    const dim3 scatter_grid((plan.nnz + threads_per_block - 1) / threads_per_block, st.batch_count);
    // Zero + scatter A into the front the factor consumes.
    auto issue_scatter = [&]() {
        cudaMemsetAsync(st.d_sing, 0, sizeof(int), stream);
        if (is_fp32_front(st.precision)) {
            cudaMemsetAsync(st.d_front_batch_f, 0, front_elements * sizeof(float), stream);
            assemble_front_values<float, ValueT><<<scatter_grid, threads_per_block, 0, stream>>>(plan.nnz, plan.front_total, d_ordered_value_to_csr,
                                                               plan.d_a_pos, d_values_batch, st.d_front_batch_f);
        } else {
            cudaMemsetAsync(st.d_front_batch, 0, front_elements * sizeof(double), stream);
            assemble_front_values<double, ValueT><<<scatter_grid, threads_per_block, 0, stream>>>(plan.nnz, plan.front_total, d_ordered_value_to_csr,
                                                                plan.d_a_pos, d_values_batch, st.d_front_batch);
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

// FP64-input entry points.
bool factorize(const MultifrontalPlan& plan, State& st, const double* d_values_batch,
               const int* d_ordered_value_to_csr)
{
    return factorize_impl<double>(plan, st, d_values_batch, d_ordered_value_to_csr);
}

// FP32-input overloads (float values / RHS / solution combinations).
bool factorize(const MultifrontalPlan& plan, State& st, const float* d_values_batch,
               const int* d_ordered_value_to_csr)
{
    return factorize_impl<float>(plan, st, d_values_batch, d_ordered_value_to_csr);
}
}  // namespace custom_linear_solver
