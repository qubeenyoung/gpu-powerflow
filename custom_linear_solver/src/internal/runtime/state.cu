#include "internal/runtime/state.hpp"

#include <cuda_runtime.h>

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

State::~State()
{
    if (factor_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(factor_graph_exec));
    if (full_solve_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(full_solve_graph_exec));
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
}  // namespace custom_linear_solver
