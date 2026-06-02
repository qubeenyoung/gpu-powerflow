#pragma once

// ---------------------------------------------------------------------------
// graph_iteration.hpp
//
// CUDA-graph capture/replay of the whole Newton iteration for the custom-solver pipelines.
// cuDSS's cudssExecute is not stream-capturable, but the custom linear solver (built with
// CLS_INTERNAL_GRAPH=OFF, i.e. external/capturable mode) issues its factor/solve as pure kernel
// launches on a caller stream, so the entire iteration body
//   ibus -> mismatch -> norm(device reduce) -> jacobian -> prepare_rhs -> factorize -> solve
//   -> voltage_update
// can be captured ONCE and replayed per Newton step, collapsing ~10 host launches into one
// cudaGraphLaunch. The only data-dependent host control flow — the convergence norm's D2H
// readback + compare — stays OUTSIDE the graph (done after each replay).
//
// Compiled only when both CUDA and the graph feature are enabled.
// ---------------------------------------------------------------------------

#if defined(CUPF_WITH_CUDA) && defined(CUPF_ENABLE_CUDA_GRAPH)

#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/ops/mismatch/cuda_mismatch.hpp"
#include "utils/cuda_utils.hpp"


// Owns a captured Newton-iteration graph and its private capture stream (one per custom pipeline).
// Move-only so it can live inside a pipeline held in the SolverPipeline variant.
struct IterationGraph {
    cudaStream_t    stream = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool            captured = false;
    int             captured_batch = 0;

    IterationGraph() = default;
    IterationGraph(const IterationGraph&) = delete;
    IterationGraph& operator=(const IterationGraph&) = delete;

    IterationGraph(IterationGraph&& o) noexcept
        : stream(o.stream), exec(o.exec), captured(o.captured), captured_batch(o.captured_batch)
    {
        o.stream = nullptr;
        o.exec = nullptr;
        o.captured = false;
        o.captured_batch = 0;
    }

    IterationGraph& operator=(IterationGraph&& o) noexcept
    {
        if (this != &o) {
            reset();
            stream = o.stream;
            exec = o.exec;
            captured = o.captured;
            captured_batch = o.captured_batch;
            o.stream = nullptr;
            o.exec = nullptr;
            o.captured = false;
            o.captured_batch = 0;
        }
        return *this;
    }

    ~IterationGraph() { reset(); }

    void reset()
    {
        if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
        captured = false;
        captured_batch = 0;
    }
};


// Run the NR loop for a graph-enabled custom pipeline. Returns the number of completed iterations.
//
// Loop is rotated so the residual is computed at the END of each step (the "residual-at-bottom"
// Newton form), which makes it match the eager path EXACTLY — same V sequence, same reported norm,
// same iteration count, no wasted step:
//
//   pre-loop (eager, once): ibus -> mismatch -> norm  at the initial V; converged? -> done.
//   captured graph body    : jacobian -> prepare_rhs -> factorize -> solve -> voltage_update
//                            -> ibus -> mismatch -> norm(device)      [one Newton step + next resid]
//   per replay             : launch graph; readback norm at the NEW V; converged? -> break.
//
// The body's top jacobian/solve consume Ibus/V and the residual d_F produced by the PREVIOUS
// replay's trailing ibus/mismatch (or the pre-loop's, for the first step) — carried across replays
// in the persistent device buffers — so J(V)·dx = F(V) holds at the current V each step. The norm
// that triggers convergence is the residual of the V we keep, so there is no extra step.
template <typename Pipeline>
int32_t run_iterations_graph(Pipeline& p, IterationContext& ctx)
{
    using StorageT = std::decay_t<decltype(p.buf)>;
    IterationGraph& g = p.graph;
    const int batch = p.buf.batch_size;

    if (g.stream == nullptr) {
        CUDA_CHECK(cudaStreamCreate(&g.stream));
    }
    // Every op (and the custom solver) launches on g.stream for the duration of this scope.
    ScopedCudaStream scope(g.stream);

    // Pre-loop: residual at the initial V (eager). If already converged, no step is taken.
    ctx.iter = 0;
    p.ibus(ctx);
    p.mismatch(ctx);
    CudaMismatchNormOp<StorageT>{}.run_device(p.buf);
    CudaMismatchNormOp<StorageT>{}.readback(p.buf, ctx);
    if (ctx.converged) return 1;

    // (Re)capture the step+next-residual body on first use or when the batch size changed.
    if (!g.captured || g.captured_batch != batch) {
        if (g.exec) { cudaGraphExecDestroy(g.exec); g.exec = nullptr; }
        // Solver arenas + capture-stream binding happen OUTSIDE the capture (they allocate).
        p.linear_solve.graph_prepare(p.buf, g.stream);

        CUDA_CHECK(cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeThreadLocal));
        p.jacobian(ctx);
        p.prepare_rhs(ctx);
        p.factorize(ctx);
        p.solve(ctx);
        p.voltage_update(ctx);
        p.ibus(ctx);
        p.mismatch(ctx);
        CudaMismatchNormOp<StorageT>{}.run_device(p.buf);  // device L∞ reduction only (capturable)
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamEndCapture(g.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&g.exec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        g.captured = true;
        g.captured_batch = batch;
    }

    // completed counts residual evaluations (= pre-loop + loop iterations), matching the eager
    // path's reporting (k Newton steps -> completed = k + 1). The loop is bounded so total residual
    // evals never exceed max_iter.
    int32_t completed = 1;
    for (int32_t iter = 0; iter + 1 < ctx.config.max_iter; ++iter) {
        ctx.iter = iter;
        ctx.jacobian_updated_this_iter = true;
        ctx.jacobian_age = 0;

        CUDA_CHECK(cudaGraphLaunch(g.exec, g.stream));

        // Readback the norm computed at the END of this replay (residual at the post-step V we
        // keep). copyTo synchronizes g.stream, so the full replay is complete before we read.
        CudaMismatchNormOp<StorageT>{}.readback(p.buf, ctx);
        completed = iter + 2;
        if (ctx.converged) break;
    }
    return completed;
}

#endif  // CUPF_WITH_CUDA && CUPF_ENABLE_CUDA_GRAPH
