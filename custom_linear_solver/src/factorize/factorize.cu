#include <cuda_runtime.h>

#include <vector>

#include "factorize/assemble.cuh"  // AssembleFrontValues (input assembly)
#include "factorize/factorize.hpp"
#include "factorize/schedule.cuh"  // IssueFactorLevels (factor tree schedule)

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

// Opt every shared-resident factor kernel into the dynamic-shared cap; their
// staged-front footprint exceeds the 48 KB default. Tensor-core variants exist
// only on the float-front path.
void RegisterFactorAttributes(Precision precision) {
  (void)precision;
  cudaFuncSetAttribute(FactorSmall<float, 8>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kDynamicSharedMemoryOptInBytes);
  cudaFuncSetAttribute(FactorSmall<float, 16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kDynamicSharedMemoryOptInBytes);
  cudaFuncSetAttribute(FactorSmall<float, 32>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kDynamicSharedMemoryOptInBytes);
  cudaFuncSetAttribute(FactorSmall<double, 8>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kDynamicSharedMemoryOptInBytes);
  cudaFuncSetAttribute(FactorSmall<double, 16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kDynamicSharedMemoryOptInBytes);
  cudaFuncSetAttribute(FactorSmall<double, 32>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kDynamicSharedMemoryOptInBytes);
  cudaFuncSetAttribute(FactorMid<float, true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kDynamicSharedMemoryOptInBytes);
  cudaFuncSetAttribute(FactorMid<float, false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kDynamicSharedMemoryOptInBytes);
  cudaFuncSetAttribute(FactorMid<double, false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kDynamicSharedMemoryOptInBytes);
  cudaFuncSetAttribute(FactorBig<double, false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kDynamicSharedMemoryOptInBytes);
  cudaFuncSetAttribute(FactorBig<float, false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kDynamicSharedMemoryOptInBytes);
  cudaFuncSetAttribute(FactorBig<float, true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kDynamicSharedMemoryOptInBytes);
}

void IssueFactor(const MultifrontalPlan& plan, State& st, void* stream) {
  IssueFactorLevels(plan, st, static_cast<cudaStream_t>(stream));
}

// Shared Factorize body, templated on the input value type ValueT. The scatter
// casts ValueT into whichever front the active precision mode consumes.
template <typename ValueT>
static bool FactorizeImpl(const MultifrontalPlan& plan, State& st,
                          const ValueT* d_values_batch,
                          const int* d_ordered_value_to_csr) {
  cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
  const long front_elements = (long)st.batch_count * plan.front_total;
  constexpr int threads_per_block = 256;
  const dim3 scatter_grid(
      (plan.nnz + threads_per_block - 1) / threads_per_block, st.batch_count);

  // Zero the front buffer, then scatter A into the front the factor consumes.
  auto issue_scatter = [&]() {
    cudaMemsetAsync(st.d_sing, 0, sizeof(int), stream);
    if (IsFp32Front(st.precision)) {
      cudaMemsetAsync(st.d_front_batch_f, 0, front_elements * sizeof(float),
                      stream);
      AssembleFrontValues<float, ValueT>
          <<<scatter_grid, threads_per_block, 0, stream>>>(
              plan.nnz, plan.front_total, d_ordered_value_to_csr, plan.d_a_pos,
              d_values_batch, st.d_front_batch_f);
    } else {
      cudaMemsetAsync(st.d_front_batch, 0, front_elements * sizeof(double),
                      stream);
      AssembleFrontValues<double, ValueT>
          <<<scatter_grid, threads_per_block, 0, stream>>>(
              plan.nnz, plan.front_total, d_ordered_value_to_csr, plan.d_a_pos,
              d_values_batch, st.d_front_batch);
    }
  };
#ifdef CLS_INTERNAL_GRAPH
  // Internal-graph mode: scatter onto the private stream, replay the captured
  // factor graph, and sync. Callers measure wall time externally.
  issue_scatter();
  cudaGraphLaunch(static_cast<cudaGraphExec_t>(st.factor_graph_exec), stream);
  cudaStreamSynchronize(stream);
  return cudaGetLastError() == cudaSuccess;
#else
  // External / capturable mode: issue scatter + the factor levels straight onto
  // the caller stream so the outer capture records them. No host sync (it would
  // break capture).
  issue_scatter();
  IssueFactorLevels(plan, st, stream);
  return true;
#endif
}

// FP64-input entry points.
bool Factorize(const MultifrontalPlan& plan, State& st,
               const double* d_values_batch,
               const int* d_ordered_value_to_csr) {
  return FactorizeImpl<double>(plan, st, d_values_batch,
                               d_ordered_value_to_csr);
}

// FP32-input overloads (float values / RHS / solution combinations).
bool Factorize(const MultifrontalPlan& plan, State& st,
               const float* d_values_batch, const int* d_ordered_value_to_csr) {
  return FactorizeImpl<float>(plan, st, d_values_batch, d_ordered_value_to_csr);
}
}  // namespace custom_linear_solver
