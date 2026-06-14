#include "factorize/factorize.hpp"

#include <cuda_runtime.h>
#include <cstring>
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
    cudaFuncSetAttribute(factor_mid_blocked<float, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_mid_blocked<float, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedMemoryOptInBytes);
    cudaFuncSetAttribute(factor_mid_blocked<double, false>,
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
    // exp_260612 ceiling probe: skip the input memset+scatter to measure assembly's share of
    // factorize wall time (CLS_SKIP_ASSEMBLE=1 → numerically WRONG, timing only).
    static const int skip_asm = [] {
        const char* s = std::getenv("CLS_SKIP_ASSEMBLE"); return s ? std::atoi(s) : 0;  // 1=both,2=scatter-only,3=memset-only
    }();
    // exp_260612 gather-based assembly: when active, the factor kernels assemble each front
    // directly (front_ops.cuh gather_matrix/gather_children), so the global memset + scatter +
    // extend-add are all skipped. Gated to batch fp32 with double input values (the goal regime).
    // Assembly mode: CLS_ASM_MODE = scatter (default) | gather (atomic) | gather_oc (output-centric).
    // CLS_GATHER_ASM=1 stays as a legacy alias for `gather`.
    // Assembly mode: scatter (0) | gather atomic-fused (1) | gather_oc output-centric fused (2) |
    // gather_pb phase-batched atomic (3) | gather_pb_oc phase-batched output-centric (4).
    static const int asm_mode = [] {
        const char* m = std::getenv("CLS_ASM_MODE");
        if (m) {
            if (std::strcmp(m, "gather_pb_oc") == 0) return 4;
            if (std::strcmp(m, "gather_pb") == 0) return 3;
            if (std::strcmp(m, "gather_oc") == 0) return 2;
            if (std::strcmp(m, "gather") == 0) return 1;
            return 0;  // scatter
        }
        const char* g = std::getenv("CLS_GATHER_ASM");
        return (g && std::atoi(g) != 0) ? 1 : 0;
    }();
#ifndef CLS_FACTOR_GATHER
    // Gather assembly is a concluded negative experiment (docs/exp_260612/08): scatter wins
    // structurally. The gather kernel paths are compiled out by default (lean scatter kernels →
    // lower registers → higher occupancy); CLS_ASM_MODE=gather* falls back to scatter. Rebuild with
    // -DCLS_FACTOR_GATHER to restore the gather A/B modes.
    const bool use_gather = false;
    (void)asm_mode;
#else
    const bool use_gather = asm_mode != 0 && is_fp32_front(st.precision) && plan.d_front_nnz_off != nullptr;
#endif
    st.gather_asm = use_gather;
    st.gather_mode = (asm_mode == 2 || asm_mode == 4) ? 1 : 0;   // output-centric kernel path
    st.phase_batched = (asm_mode == 3 || asm_mode == 4);          // separate assemble pre-pass
    if (use_gather) {
        st.cur_values_double = (sizeof(ValueT) == sizeof(double)) ? 1 : 0;
        st.cur_values_d = reinterpret_cast<const double*>(d_values_batch);
        st.cur_values_f = reinterpret_cast<const float*>(d_values_batch);
        st.cur_o2c = d_ordered_value_to_csr;
        st.cur_nnz = plan.nnz;
        const long need = (long)st.batch_count * plan.cb_total;
        if (plan.cb_total > 0 && need > st.cb_alloc_elems) {
            if (st.d_cb_batch_f) cudaFree(st.d_cb_batch_f);
            cudaMalloc(&st.d_cb_batch_f, need * sizeof(float));
            st.cb_alloc_elems = need;
            st.gather_graph_values = nullptr;   // force gather-graph re-capture (cb pointer changed)
        }
    }
    static const bool gather_keep_memset = [] {
        const char* s = std::getenv("CLS_GATHER_KEEP_MEMSET"); return s && std::atoi(s) != 0;
    }();
    // Zero + scatter A into the front the factor consumes.
    auto issue_scatter = [&]() {
        cudaMemsetAsync(st.d_sing, 0, sizeof(int), stream);
        if (use_gather) {   // gather assembles in the factor kernels; optionally still zero (debug)
            // Phase-batched gather needs the whole arena pre-zeroed once (the separate assemble
            // pre-pass gathers into global F via atomics / single writes and relies on 0 fill for
            // the unoccupied positions). One streaming memset is far cheaper than per-front zeroing.
            if ((st.phase_batched || gather_keep_memset) && is_fp32_front(st.precision))
                cudaMemsetAsync(st.d_front_batch_f, 0, front_elements * sizeof(float), stream);
            return;
        }
        if (skip_asm == 1) return;
        if (is_fp32_front(st.precision)) {
            if (skip_asm != 3) cudaMemsetAsync(st.d_front_batch_f, 0, front_elements * sizeof(float), stream);
            if (skip_asm != 2) assemble_front_values<float, ValueT><<<scatter_grid, threads_per_block, 0, stream>>>(plan.nnz, plan.front_total, d_ordered_value_to_csr,
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
    if (use_gather) {
        // The setup-time graph holds the legacy (stage-in + extend) kernels. Capture a gather-
        // specific graph lazily, keyed on the value pointer (re-capture if it changes), then replay
        // it so the gather path keeps the graph's low launch overhead.
        const void* vp = st.cur_values_double ? static_cast<const void*>(st.cur_values_d)
                                              : static_cast<const void*>(st.cur_values_f);
        if (!st.factor_gather_graph_exec || st.gather_graph_values != vp) {
            if (st.factor_gather_graph_exec)
                cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(st.factor_gather_graph_exec));
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            issue_factor_levels(plan, st, stream);
            cudaGraph_t g; cudaStreamEndCapture(stream, &g);
            cudaGraphExec_t ge; cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
            cudaGraphDestroy(g);
            st.factor_gather_graph_exec = ge;
            st.gather_graph_values = vp;
        }
        cudaGraphLaunch(static_cast<cudaGraphExec_t>(st.factor_gather_graph_exec), stream);
    } else {
        cudaGraphLaunch(static_cast<cudaGraphExec_t>(st.factor_graph_exec), stream);
    }
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
