// ---------------------------------------------------------------------------
// newton_solver_adjoint.cpp
//
// Backward (adjoint) solve for the implicit power flow: given dL/dVa, dL/dVm at
// the converged state, solve J^T lambda = dL/dx and project lambda onto the
// load gradients. One solve_adjoint_pipeline() overload per pipeline variant;
// the CUDA variants share solve_adjoint_cuda_pipeline(). Each path can reuse a
// cached final-state factorization or, if allowed, refactorize on the spot.
// ---------------------------------------------------------------------------

#include "newton_solver/core/newton_solver_adjoint.hpp"
#include "newton_solver/core/adjoint_math.hpp"
#include "newton_solver/core/csr_transpose.hpp"
#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/ops/ibus/compute_ibus.hpp"
#include "newton_solver/ops/jacobian/fill_jacobian.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>


// ===========================================================================
// Translation-unit-local helpers
// ===========================================================================
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}


#ifdef CUPF_WITH_CUDA
// Batch size per CUDA storage layout (FP64 is single-case).
int32_t cuda_batch_size(const CudaFp64Storage&) { return 1; }
int32_t cuda_batch_size(const CudaFp32Storage& b) { return b.batch_size; }
int32_t cuda_batch_size(const CudaMixedStorage& b) { return b.batch_size; }

// Non-zero count of J per CUDA storage layout.
int32_t cuda_nnz_j(const CudaFp64Storage& b)
{
    // FP64 tracks nnz via the value buffer length; narrow size_t -> int32.
    return static_cast<int32_t>(b.d_J_values.size());
}
int32_t cuda_nnz_j(const CudaFp32Storage& b) { return b.nnz_J; }
int32_t cuda_nnz_j(const CudaMixedStorage& b) { return b.nnz_J; }
#endif  // CUPF_WITH_CUDA

}  // namespace


// ===========================================================================
// CPU / KLU adjoint solve
//
// KLU can solve J^T directly from the cached LU, so no explicit transpose is
// needed. CUDA paths follow below and differ mainly in the transpose handling.
// ===========================================================================
void solve_adjoint_pipeline(CpuFp64Pipeline& p,
                            const double* grad_va,
                            int64_t grad_va_stride,
                            const double* grad_vm,
                            int64_t grad_vm_stride,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            const AdjointOptions& options,
                            const CuDSSOptions&,
                            AdjointResult& result)
{
    validate_adjoint_args(p.buf.n_bus, p.buf.dimF, 1,
                          grad_va, grad_va_stride,
                          grad_vm, grad_vm_stride,
                          batch_size, pv, n_pv, pq, n_pq);
    const auto total_start = Clock::now();

    // Record provenance/diagnostic flags for this (CPU) backend.
    result.backend = "cpu_klu";
    result.transpose_solve_backend = "cpu_klu_tsolve_cached_factorization";
    result.used_adjoint_cache = p.adjoint_cache.has_adjoint_cache;
    result.adjoint_cache_matches_final_state = p.adjoint_cache.adjoint_cache_matches_final_state;
    result.reused_forward_factorization = p.adjoint_cache.reused_forward_factorization;
    result.reused_final_state_factorization = false;
    result.refactorized_for_backward = false;
    result.used_explicit_transpose = false;
    result.includes_host_device_transfer = false;
    result.used_python_scipy = false;
    result.zero_copy = false;
    result.torch_extension_zero_copy = false;
    result.raw_pointer_api_used = false;
    result.current_stream_integrated = false;
    result.jt_symbolic_analyzed_at_initialize = p.adjoint_cache.jt_symbolic_analyzed_at_initialize;
    result.jt_values_transposed_on_device = p.adjoint_cache.jt_values_transposed_on_device;
    result.jt_factorized_during_forward_cache = p.adjoint_cache.jt_factorized_during_forward_cache;
    result.jt_refactorized_during_backward = false;
    result.host_roundtrip_for_jt_transpose = p.adjoint_cache.host_roundtrip_for_jt_transpose;
    result.n_bus = p.buf.n_bus;
    result.batch_size = 1;
    result.dimF = p.buf.dimF;

    // Pack the upstream gradients into the dense dL/dx RHS ordering.
    const std::vector<double> grad_state =
        build_grad_state(grad_va, grad_va_stride, grad_vm, grad_vm_stride,
                         batch_size, pv, n_pv, pq, n_pq);

    // Reuse the cached final-state factorization when it is exact; otherwise
    // refactorize here if the caller allows it.
    const bool cache_ok =
        p.adjoint_cache.has_adjoint_cache &&
        p.adjoint_cache.adjoint_cache_matches_final_state &&
        p.adjoint_cache.factorization_supports_transpose_solve;
    const bool allow_backward_refactor =
        options.allow_refactorize && options.allow_refactorize_for_backward;
    if (!cache_ok) {
        if (options.require_cached_factorization || !allow_backward_refactor) {
            throw std::runtime_error(
                "NewtonSolver::solve_adjoint(): missing exact final-state adjoint cache");
        }
        // Rebuild ibus + Jacobian at the final state and refactorize.
        NRConfig cfg;
        IterationContext ctx{
            .config = cfg,
            .pv = pv, .n_pv = n_pv,
            .pq = pq, .n_pq = n_pq,
        };
        CpuIbusOp{}.run(p.buf, ctx);
        CpuJacobianOpF64{}.run(p.buf, ctx);
        const auto factor_start = Clock::now();
        p.linear_solve.factorize(p.buf, ctx);
        result.factorization_time_ms = elapsed_ms(factor_start, Clock::now());
        result.refactorized_for_backward = true;
        result.jt_refactorized_during_backward = true;
        p.adjoint_cache.has_adjoint_cache = true;
        p.adjoint_cache.adjoint_cache_matches_final_state = true;
        p.adjoint_cache.factorization_supports_transpose_solve = true;
        p.adjoint_cache.refactorized_for_adjoint_cache = true;
        p.adjoint_cache.reused_forward_factorization = false;
        p.adjoint_cache.used_explicit_transpose = false;
        p.adjoint_cache.backend_name = "cpu_klu";
        p.adjoint_cache.transpose_solve_backend_name = "cpu_klu_tsolve_cached_factorization";
        p.adjoint_cache.batch_size = 1;
        p.adjoint_cache.dimF = p.buf.dimF;
    } else {
        result.factorization_time_ms = 0.0;
        result.reused_final_state_factorization = true;
    }

    // Solve J^T lambda = dL/dx using the cached factorization (dimF entries).
    auto solve_start = Clock::now();
    result.lambda.assign(static_cast<std::size_t>(p.buf.dimF), 0.0);  // dimF>0 -> size_t
    p.linear_solve.solve_transpose(grad_state.data(), result.lambda.data(), p.buf.dimF, 1);
    result.solve_time_ms = elapsed_ms(solve_start, Clock::now());
    result.transpose_solve_time_ms = result.solve_time_ms;

    result.used_adjoint_cache = cache_ok;
    result.adjoint_cache_matches_final_state = p.adjoint_cache.adjoint_cache_matches_final_state;
    result.reused_forward_factorization = p.adjoint_cache.reused_forward_factorization;

    // Optional accuracy check, then project lambda onto P/Q load gradients.
    if (options.check_residual) {
        result.jt_residual_norm =
            relative_residual_norm_csc(p.buf.J, result.lambda, grad_state);
    }
    if (options.compute_load_gradients) {
        project_load_gradients(result.lambda, p.buf.n_bus, 1, pv, n_pv, pq, n_pq, result);
    }
    result.total_time_ms = elapsed_ms(total_start, Clock::now());
    result.success = true;
}


// ===========================================================================
// CUDA / cuDSS adjoint solve
//
// cuDSS has no native transpose solve, so these paths rely on an explicit J^T
// factorization (cached during the forward pass, or rebuilt here on fallback).
// The shared template below is instantiated for FP64 / FP32 / Mixed via the
// thin overloads that follow.
// ===========================================================================
#ifdef CUPF_WITH_CUDA
template <typename PipelineT, typename ValueT>
void solve_adjoint_cuda_pipeline(PipelineT& p,
                                 const double* grad_va,
                                 int64_t grad_va_stride,
                                 const double* grad_vm,
                                 int64_t grad_vm_stride,
                                 int32_t batch_size,
                                 const int32_t* pv,
                                 int32_t n_pv,
                                 const int32_t* pq,
                                 int32_t n_pq,
                                 const AdjointOptions& options,
                                 const CuDSSOptions& cudss_options,
                                 const char* backend_name,
                                 AdjointResult& result)
{
    validate_adjoint_args(p.buf.n_bus, p.buf.dimF, cuda_batch_size(p.buf),
                          grad_va, grad_va_stride,
                          grad_vm, grad_vm_stride,
                          batch_size, pv, n_pv, pq, n_pq);
    const auto total_start = Clock::now();

    // Record provenance/diagnostic flags for this (CUDA) backend.
    result.backend = backend_name;
    result.transpose_solve_backend = "cuda_cudss_cached_explicit_transpose_factorization";
    result.used_adjoint_cache = p.adjoint_cache.has_adjoint_cache;
    result.adjoint_cache_matches_final_state = p.adjoint_cache.adjoint_cache_matches_final_state;
    result.reused_forward_factorization = p.adjoint_cache.reused_forward_factorization;
    result.reused_final_state_factorization = false;
    result.refactorized_for_backward = false;
    result.used_explicit_transpose = true;
    result.includes_host_device_transfer = true;
    result.used_python_scipy = false;
    result.zero_copy = false;
    result.torch_extension_zero_copy = false;
    result.raw_pointer_api_used = false;
    result.current_stream_integrated = false;
    result.jt_symbolic_analyzed_at_initialize = p.adjoint_cache.jt_symbolic_analyzed_at_initialize;
    result.jt_values_transposed_on_device = p.adjoint_cache.jt_values_transposed_on_device;
    result.jt_factorized_during_forward_cache = p.adjoint_cache.jt_factorized_during_forward_cache;
    result.jt_refactorized_during_backward = false;
    result.host_roundtrip_for_jt_transpose = p.adjoint_cache.host_roundtrip_for_jt_transpose;
    result.n_bus = p.buf.n_bus;
    result.batch_size = batch_size;
    result.dimF = p.buf.dimF;

    // Pack the upstream gradients into the dense dL/dx RHS ordering.
    const std::vector<double> grad_state =
        build_grad_state(grad_va, grad_va_stride, grad_vm, grad_vm_stride,
                         batch_size, pv, n_pv, pq, n_pq);

    // Reuse the cached explicit-transpose factorization when exact; otherwise
    // rebuild it here (requires both refactorize and transpose-fallback opt-in).
    const bool cache_ok =
        p.adjoint_cache.has_adjoint_cache &&
        p.adjoint_cache.adjoint_cache_matches_final_state &&
        p.linear_solve.has_adjoint_cache();
    const bool allow_backward_refactor =
        options.allow_refactorize && options.allow_refactorize_for_backward;
    if (!cache_ok) {
        if (options.require_cached_factorization || !allow_backward_refactor ||
            !options.allow_explicit_transpose_fallback) {
            throw std::runtime_error(
                "NewtonSolver::solve_adjoint(): missing exact CUDA adjoint cache; cuDSS transpose solve is unsupported, and explicit transpose fallback is not enabled");
        }
        // Rebuild ibus + Jacobian, then build & factorize an explicit J^T.
        NRConfig cfg;
        IterationContext ctx{
            .config = cfg,
            .pv = pv, .n_pv = n_pv,
            .pq = pq, .n_pq = n_pq,
        };
        p.ibus(ctx);
        p.jacobian(ctx);
        p.linear_solve.prepare_adjoint_explicit_transpose_cache(
            p.buf, ctx, result.factorization_time_ms);
        result.refactorized_for_backward = true;
        result.jt_refactorized_during_backward = true;
        p.adjoint_cache.has_adjoint_cache = true;
        p.adjoint_cache.adjoint_cache_matches_final_state = true;
        p.adjoint_cache.factorization_supports_transpose_solve = false;
        p.adjoint_cache.refactorized_for_adjoint_cache = true;
        p.adjoint_cache.reused_forward_factorization = false;
        p.adjoint_cache.used_explicit_transpose = true;
        p.adjoint_cache.includes_host_device_transfer = false;
        p.adjoint_cache.backend_name = backend_name;
        p.adjoint_cache.transpose_solve_backend_name =
            "cuda_cudss_cached_explicit_transpose_factorization";
        p.adjoint_cache.batch_size = batch_size;
        p.adjoint_cache.dimF = p.buf.dimF;
    } else {
        result.factorization_time_ms = 0.0;
        result.reused_final_state_factorization = true;
    }

    // Solve J^T lambda = dL/dx for the whole batch (size_t widening guards the
    // batch_size * dimF allocation against int overflow).
    result.lambda.assign(static_cast<std::size_t>(batch_size) *
                         static_cast<std::size_t>(p.buf.dimF), 0.0);
    p.linear_solve.solve_adjoint_explicit_transpose_host(
        grad_state.data(),
        result.lambda.data(),
        batch_size,
        result.solve_time_ms);
    result.transpose_solve_time_ms = result.solve_time_ms;

    // Optional accuracy check: pull J back to host, transpose it, and measure
    // the relative residual of the adjoint solution against dL/dx.
    if (options.check_residual) {
        const int32_t dim = p.buf.dimF;
        const int32_t nnz = cuda_nnz_j(p.buf);
        // Host CSR mirrors of J (size_t casts size the host vectors).
        std::vector<int32_t> row_ptr(static_cast<std::size_t>(dim + 1));
        std::vector<int32_t> col_idx(static_cast<std::size_t>(nnz));
        std::vector<ValueT> values(static_cast<std::size_t>(batch_size) *
                                   static_cast<std::size_t>(nnz));
        p.buf.d_J_row_ptr.copyTo(row_ptr.data(), row_ptr.size());
        p.buf.d_J_col_idx.copyTo(col_idx.data(), col_idx.size());
        p.buf.d_J_values.copyTo(values.data(), values.size());
        const CsrTransposePattern transpose = build_transpose_pattern(row_ptr, col_idx, dim);
        const std::vector<ValueT> jt_values =
            transpose_batched_values(values, transpose.src_to_transpose_pos, batch_size, nnz);
        result.jt_residual_norm = relative_residual_norm_csr(
            transpose.row_ptr, transpose.col_idx, jt_values,
            result.lambda, grad_state, batch_size, dim, nnz);
    } else {
        result.jt_residual_norm = 0.0;
    }
    // Project lambda onto the P/Q load gradients for the caller.
    if (options.compute_load_gradients) {
        project_load_gradients(result.lambda, p.buf.n_bus, batch_size, pv, n_pv, pq, n_pq, result);
    }
    result.total_time_ms = elapsed_ms(total_start, Clock::now());
    result.success = true;
}

// --- Per-variant dispatch: bind each CUDA pipeline to the shared template ---

void solve_adjoint_pipeline(CudaFp64Pipeline& p,
                            const double* grad_va,
                            int64_t grad_va_stride,
                            const double* grad_vm,
                            int64_t grad_vm_stride,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            const AdjointOptions& options,
                            const CuDSSOptions& cudss_options,
                            AdjointResult& result)
{
    solve_adjoint_cuda_pipeline<CudaFp64Pipeline, double>(
        p, grad_va, grad_va_stride, grad_vm, grad_vm_stride,
        batch_size, pv, n_pv, pq, n_pq, options, cudss_options, "cuda_cudss_fp64", result);
}

#ifdef CUPF_ENABLE_CUSTOM_SOLVER
// The custom FP64 direct solver has no adjoint implementation yet.
void solve_adjoint_pipeline(CudaFp64CustomPipeline&,
                            const double*,
                            int64_t,
                            const double*,
                            int64_t,
                            int32_t,
                            const int32_t*,
                            int32_t,
                            const int32_t*,
                            int32_t,
                            const AdjointOptions&,
                            const CuDSSOptions&,
                            AdjointResult&)
{
    throw std::runtime_error(
        "NewtonSolver::solve_adjoint(): custom CUDA FP64 solver does not implement adjoint solve");
}
#endif

void solve_adjoint_pipeline(CudaFp32Pipeline& p,
                            const double* grad_va,
                            int64_t grad_va_stride,
                            const double* grad_vm,
                            int64_t grad_vm_stride,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            const AdjointOptions& options,
                            const CuDSSOptions& cudss_options,
                            AdjointResult& result)
{
    solve_adjoint_cuda_pipeline<CudaFp32Pipeline, float>(
        p, grad_va, grad_va_stride, grad_vm, grad_vm_stride,
        batch_size, pv, n_pv, pq, n_pq, options, cudss_options, "cuda_cudss_fp32", result);
}

void solve_adjoint_pipeline(CudaMixedPipeline& p,
                            const double* grad_va,
                            int64_t grad_va_stride,
                            const double* grad_vm,
                            int64_t grad_vm_stride,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            const AdjointOptions& options,
                            const CuDSSOptions& cudss_options,
                            AdjointResult& result)
{
    solve_adjoint_cuda_pipeline<CudaMixedPipeline, float>(
        p, grad_va, grad_va_stride, grad_vm, grad_vm_stride,
        batch_size, pv, n_pv, pq, n_pq, options, cudss_options, "cuda_cudss_mixed", result);
}
#endif  // CUPF_WITH_CUDA
