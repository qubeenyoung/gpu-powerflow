#include <cuda_runtime.h>

#include "factorize/factorize.hpp"  // RegisterFactorAttributes, IssueFactor
#include "internal/runtime/state.hpp"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

// Set the per-batch scalar state and allocate the front + Solve-vector arenas.
// Float-front modes (FP32 / TF32) use the f32 arena; FP64 uses the double
// arena.
static bool AllocateState(const MultifrontalPlan& plan, int B,
                          Precision precision, State& st, bool static_pivoting,
                          double pivot_threshold, double pivot_shift) {
  st.batch_count = B;
  st.front_total = plan.front_total;
  st.num_rows = plan.num_rows;
  st.precision = precision;
  st.static_pivoting = static_pivoting;
  st.pivot_threshold = pivot_threshold;
  st.pivot_shift = pivot_shift;
  const bool float_front = IsFp32Front(precision);
  const long front_elements = (long)B * plan.front_total;
  if (float_front) {
    if (cudaMalloc(&st.d_front_batch_f, front_elements * sizeof(float)) !=
        cudaSuccess)
      return false;
    if (cudaMalloc(&st.d_y_batch_f, (long)B * plan.num_rows * sizeof(float)) !=
        cudaSuccess)
      return false;
  } else {
    if (cudaMalloc(&st.d_front_batch, front_elements * sizeof(double)) !=
        cudaSuccess)
      return false;
    if (cudaMalloc(&st.d_y_batch, (long)B * plan.num_rows * sizeof(double)) !=
        cudaSuccess)
      return false;
  }
  if (cudaMalloc(&st.d_sing, sizeof(int)) != cudaSuccess) return false;
  return true;
}

#ifdef CLS_INTERNAL_GRAPH
// Allocate one CUDA stream + join event per independent subtree (capped at
// kMaxSubtreeStreams) plus the shared fork event, for multi-stream dispatch.
// No-op when disabled or when the plan has 0/1 subtrees, in which case dispatch
// falls back to single-stream.
static void CreateSubtreeStreams(const MultifrontalPlan& plan, State& st) {
  // Batched only (B > 1): the single-system schedule runs on one stream. Skip
  // when the plan has 0/1 subtrees or more than the stream cap, in which case
  // dispatch stays single-stream.
  if (UseSingleSystem(st) || plan.num_subtrees <= 1 ||
      plan.num_subtrees > kMaxSubtreeStreams)
    return;
  st.num_subtree_streams = plan.num_subtrees;
  for (int k = 0; k < st.num_subtree_streams; ++k) {
    cudaStream_t subtree_stream;
    cudaStreamCreateWithFlags(&subtree_stream, cudaStreamNonBlocking);
    st.subtree_streams[k] = subtree_stream;
    cudaEvent_t join_evt;
    cudaEventCreateWithFlags(&join_evt, cudaEventDisableTiming);
    st.join_events[k] = join_evt;
  }
  cudaEvent_t fork_evt;
  cudaEventCreateWithFlags(&fork_evt, cudaEventDisableTiming);
  st.fork_event = fork_evt;
}

// Capture the factor kernel sequence into a replayable CUDA graph on `stream`.
// The Solve graph is captured lazily by Solve.cu (it spans gather + Solve
// levels + scatter and is keyed by the I/O pointers), so Setup only owns the
// factor graph.
static void CapturePhaseGraphs(const MultifrontalPlan& plan, State& st,
                               cudaStream_t stream) {
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  IssueFactor(plan, st, stream);
  cudaGraph_t factor_graph;
  cudaStreamEndCapture(stream, &factor_graph);
  cudaGraphExec_t factor_exec;
  cudaGraphInstantiate(&factor_exec, factor_graph, nullptr, nullptr, 0);
  cudaGraphDestroy(factor_graph);
  st.factor_graph_exec = factor_exec;
}
#endif

bool Setup(const MultifrontalPlan& plan, int B, Precision precision, State& st,
           bool static_pivoting, double pivot_threshold, double pivot_shift) {
  if (plan.num_panels == 0 || B <= 0) return false;
  if (!AllocateState(plan, B, precision, st, static_pivoting, pivot_threshold,
                     pivot_shift))
    return false;
  RegisterFactorAttributes(precision);

#ifdef CLS_INTERNAL_GRAPH
  // Internal-graph mode: own a private stream and capture the factor / Solve
  // kernel sequences into replayable CUDA graphs (default standalone behavior).
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  st.stream = stream;
  st.owns_stream = true;

  CreateSubtreeStreams(plan, st);
  CapturePhaseGraphs(plan, st, stream);
  return cudaGetLastError() == cudaSuccess;
#else
  // External/capturable mode: no private stream and no internal graphs. The
  // factor / Solve kernels are issued directly onto the caller stream
  // (SetStream) so an outer capture records them. The caller must call
  // SetStream before Factorize/Solve. Subtree streams would defeat the outer
  // capture, so multi-stream dispatch stays off in this mode.
  return true;
#endif
}

void SetStream(State& st, void* stream) {
  // External mode: adopt the caller's stream (not owned -> not destroyed by
  // ~State).
  st.stream = stream;
  st.owns_stream = false;
}
}  // namespace custom_linear_solver
