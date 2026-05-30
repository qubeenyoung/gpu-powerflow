// ---------------------------------------------------------------------------
// newton_solver.cpp
//
// NewtonSolver orchestration. Backend-specific math, adjoint solve, and
// Torch interop live in companion core files.
// ---------------------------------------------------------------------------

#include "newton_solver/core/newton_solver.hpp"
#include "newton_solver/core/newton_solver_adjoint.hpp"
#include "newton_solver/core/pipeline.hpp"
#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/ops/jacobian/jacobian_analysis.hpp"
#include "utils/nvtx_trace.hpp"
#include "utils/timer.hpp"

#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>


namespace {

using Clock = std::chrono::steady_clock;

struct StageScope {
    explicit StageScope(const char* label)
        : range(label)
        , timer(label)
    {}

    newton_solver::utils::ScopedNvtxRange range;
    newton_solver::utils::ScopedTimer     timer;
};

void validate_batch_args(int32_t batch_size, int64_t stride, int32_t n_bus, const char* name)
{
    if (batch_size <= 0) {
        throw std::invalid_argument("NewtonSolver::solve_batch(): batch_size must be positive");
    }
    if (stride < n_bus) {
        throw std::invalid_argument(std::string("NewtonSolver::solve_batch(): ") +
                                    name + " stride must be at least n_bus");
    }
}

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

#ifdef CUPF_WITH_CUDA
int32_t cuda_batch_size(const CudaFp64Buffers&) { return 1; }
int32_t cuda_batch_size(const CudaFp32Buffers& b) { return b.batch_size; }
int32_t cuda_batch_size(const CudaMixedBuffers& b) { return b.batch_size; }
#endif

}  // namespace


// ---------------------------------------------------------------------------
// Constructor: assemble the pipeline from NewtonOptions.
// ---------------------------------------------------------------------------
NewtonSolver::NewtonSolver(const NewtonOptions& options)
    : options_(options)
{
    if (options.backend == BackendKind::CPU) {
        pipeline_ = std::make_unique<SolverPipeline>(
            SolverPipeline{CpuFp64Pipeline{}});
        return;
    }

#ifdef CUPF_WITH_CUDA
    if (options.backend == BackendKind::CUDA) {
        if (options.cuda_linear_solver == CudaLinearSolverKind::Custom &&
            options.compute != ComputePolicy::FP64) {
            throw std::invalid_argument(
                "NewtonSolver: custom CUDA linear solver는 FP64 단일 케이스만 지원합니다.");
        }
        if (options.compute == ComputePolicy::FP64) {
#ifdef CUPF_ENABLE_CUSTOM_SOLVER
            if (options.cuda_linear_solver == CudaLinearSolverKind::Custom) {
                pipeline_ = std::make_unique<SolverPipeline>(
                    SolverPipeline{CudaFp64CustomPipeline{}});
                return;
            }
#else
            if (options.cuda_linear_solver == CudaLinearSolverKind::Custom) {
                throw std::invalid_argument(
                    "NewtonSolver: custom CUDA linear solver를 요청했지만 "
                    "cuPF가 CUPF_ENABLE_CUSTOM_SOLVER 없이 빌드되었습니다.");
            }
#endif
            pipeline_ = std::make_unique<SolverPipeline>(
                SolverPipeline{CudaFp64Pipeline{options.cudss}});
            return;
        }
        if (options.compute == ComputePolicy::FP32) {
            pipeline_ = std::make_unique<SolverPipeline>(
                SolverPipeline{CudaFp32Pipeline{options.cudss}});
            return;
        }
        if (options.compute == ComputePolicy::Mixed) {
            pipeline_ = std::make_unique<SolverPipeline>(
                SolverPipeline{CudaMixedPipeline{options.cudss}});
            return;
        }
    }
#else
    if (options.backend == BackendKind::CUDA) {
        throw std::invalid_argument(
            "NewtonSolver: CUDA backend를 요청했지만 cuPF가 CUDA 없이 빌드되었습니다.");
    }
#endif

    throw std::invalid_argument(
        "NewtonSolver: 지원하지 않는 backend/compute 조합입니다.");
}


NewtonSolver::~NewtonSolver() = default;


// ---------------------------------------------------------------------------
// initialize: Jacobian analysis -> pipeline::initialize
// ---------------------------------------------------------------------------
void NewtonSolver::initialize(
    const YbusView& ybus,
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq)
{
    StageScope total("NR.initialize.total");

    JacobianPattern pattern;
    JacobianScatterMap scatter_map;
    {
        StageScope stage("NR.initialize.jacobian_analysis");
        const JacobianIndexing indexing =
            make_jacobian_indexing(ybus.rows, pv, n_pv, pq, n_pq);
        pattern = JacobianPatternGenerator().generate(ybus, indexing);
        scatter_map = JacobianMapBuilder().build(ybus, indexing, pattern);
    }

    InitializeContext ctx{
        .ybus  = ybus,
        .maps  = scatter_map,
        .J     = pattern,
        .n_bus = ybus.rows,
        .pv    = pv, .n_pv = n_pv,
        .pq    = pq, .n_pq = n_pq,
    };

    {
        StageScope stage("NR.initialize.pipeline");
        std::visit([&](auto& p) { p.initialize(ctx); }, pipeline_->v);
    }

    initialized_ = true;
}


// ---------------------------------------------------------------------------
// solve: single-case wrapper around solve_batch
// ---------------------------------------------------------------------------
void NewtonSolver::solve(
    const YbusView&          ybus,
    const std::complex<double>* sbus,
    const std::complex<double>* V0,
    const int32_t*              pv, int32_t n_pv,
    const int32_t*              pq, int32_t n_pq,
    const NRConfig&             config,
    const SolveOptions&         solve_options,
    NRResult&                result)
{
    NRBatchResult batch_result;
    solve_batch(ybus, sbus, ybus.rows, V0, ybus.rows, 1,
                pv, n_pv, pq, n_pq, config, solve_options, batch_result);

    result = {};
    result.V = std::move(batch_result.V);
    result.iterations = batch_result.iterations.empty() ? 0 : batch_result.iterations[0];
    result.final_mismatch = batch_result.final_mismatch.empty() ? 0.0 : batch_result.final_mismatch[0];
    result.converged = !batch_result.converged.empty() && batch_result.converged[0] != 0;
}


void NewtonSolver::solve_batch(
    const YbusView&          ybus,
    const std::complex<double>* sbus,
    int64_t                     sbus_stride,
    const std::complex<double>* V0,
    int64_t                     V0_stride,
    int32_t                     batch_size,
    const int32_t*              pv, int32_t n_pv,
    const int32_t*              pq, int32_t n_pq,
    const NRConfig&             config,
    const SolveOptions&         solve_options,
    NRBatchResult&           result)
{
    StageScope total("NR.solve.total");

    if (!initialized_) {
        throw std::runtime_error(
            "NewtonSolver::solve_batch(): initialize()를 먼저 호출해야 합니다.");
    }

    validate_batch_args(batch_size, sbus_stride, ybus.rows, "sbus");
    validate_batch_args(batch_size, V0_stride, ybus.rows, "V0");

    if (batch_size != 1) {
        const bool supported = std::visit(
            [](const auto& p) { return p.batch_supported; }, pipeline_->v);
        if (!supported) {
            throw std::runtime_error(
                "NewtonSolver::solve_batch(): batch_size > 1 is currently supported "
                "only by CUDA FP32/Mixed pipelines.");
        }
    }

    result = {};
    result.n_bus = ybus.rows;
    result.batch_size = batch_size;

    {
        StageScope stage("NR.solve.upload");
        std::visit([&](auto& p) { p.adjoint_cache = AdjointCache{}; }, pipeline_->v);
        SolveContext upload_ctx{
            .ybus   = &ybus,
            .sbus   = sbus,
            .V0     = V0,
            .config = &config,
            .batch_size          = batch_size,
            .sbus_stride         = sbus_stride,
            .V0_stride           = V0_stride,
            .ybus_values_batched = false,
            .ybus_value_stride   = ybus.nnz,
        };
        std::visit([&](auto& p) { p.upload(upload_ctx); }, pipeline_->v);
    }

    IterationContext iter_ctx{
        .config = config,
        .pv     = pv, .n_pv = n_pv,
        .pq     = pq, .n_pq = n_pq,
    };

    const int32_t iterations = run_iteration_stages(iter_ctx);

    if (solve_options.prepare_adjoint_cache) {
        StageScope stage("NR.solve.prepare_adjoint_cache");
        prepare_adjoint_cache(iter_ctx, solve_options, iter_ctx.normF);
    }

    {
        StageScope stage("NR.solve.download");
        std::visit([&](auto& p) { p.download_batch(result); }, pipeline_->v);
    }

    result.n_bus = ybus.rows;
    result.batch_size = batch_size;
    result.iterations.assign(static_cast<std::size_t>(batch_size), iterations);

    if (result.final_mismatch.size() != static_cast<std::size_t>(batch_size)) {
        result.final_mismatch.assign(static_cast<std::size_t>(batch_size), iter_ctx.normF);
    }
    result.converged.resize(static_cast<std::size_t>(batch_size));
    for (int32_t b = 0; b < batch_size; ++b) {
        const double norm =
            result.final_mismatch.empty()
                ? iter_ctx.normF
                : result.final_mismatch[static_cast<std::size_t>(b)];
        result.converged[static_cast<std::size_t>(b)] =
            static_cast<uint8_t>(norm <= config.tolerance ? 1 : 0);
    }
}


void NewtonSolver::solve_adjoint(const double*        grad_va,
                                 int64_t              grad_va_stride,
                                 const double*        grad_vm,
                                 int64_t              grad_vm_stride,
                                 int32_t              batch_size,
                                 const int32_t*       pv, int32_t n_pv,
                                 const int32_t*       pq, int32_t n_pq,
                                 const AdjointOptions& options,
                                 AdjointResult&       result)
{
    StageScope total("NR.adjoint.total");

    if (!initialized_) {
        throw std::runtime_error(
            "NewtonSolver::solve_adjoint(): initialize() and solve()/solve_batch() must be called first");
    }
    if (options.reuse_forward_factorization && !options.allow_refactorize) {
        throw std::runtime_error(
            "NewtonSolver::solve_adjoint(): forward factorization reuse is not available for the final converged state; "
            "enable allow_refactorize");
    }

    result = {};
    std::visit([&](auto& p) {
        solve_adjoint_pipeline(p,
                               grad_va,
                               grad_va_stride,
                               grad_vm,
                               grad_vm_stride,
                               batch_size,
                               pv, n_pv,
                               pq, n_pq,
                               options,
                               options_.cudss,
                               result);
    }, pipeline_->v);
}


void NewtonSolver::prepare_adjoint_cache(IterationContext& ctx,
                                         const SolveOptions& solve_options,
                                         double final_mismatch_norm)
{
    if (solve_options.adjoint_cache_mode == AdjointCacheMode::None) {
        return;
    }
    if (solve_options.adjoint_cache_mode == AdjointCacheMode::ReuseLastNewtonFactorizationIfExact &&
        !ctx.jacobian_updated_this_iter) {
        throw std::runtime_error(
            "NewtonSolver::prepare_adjoint_cache(): last Newton factorization is not known to be exact at final state");
    }

    std::visit([&](auto& p) {
        const auto start = Clock::now();
        p.ibus(ctx);
        p.jacobian(ctx);

        p.adjoint_cache = AdjointCache{};
        p.adjoint_cache.has_adjoint_cache = true;
        p.adjoint_cache.adjoint_cache_matches_final_state = true;
        p.adjoint_cache.final_mismatch_norm = final_mismatch_norm;
        p.adjoint_cache.batch_size = [&]() -> int32_t {
            if constexpr (std::is_same_v<std::decay_t<decltype(p)>, CpuFp64Pipeline>) {
                return 1;
            } else {
#ifdef CUPF_WITH_CUDA
                return cuda_batch_size(p.buf);
#else
                return 1;
#endif
            }
        }();
        p.adjoint_cache.dimF = p.buf.dimF;
        p.adjoint_cache.reused_forward_factorization = false;
        p.adjoint_cache.refactorized_for_adjoint_cache = true;

        if constexpr (std::is_same_v<std::decay_t<decltype(p)>, CpuFp64Pipeline>) {
            p.linear_solve.factorize(p.buf, ctx);
            p.adjoint_cache.factorization_supports_transpose_solve = true;
            p.adjoint_cache.used_explicit_transpose = false;
            p.adjoint_cache.includes_host_device_transfer = false;
            p.adjoint_cache.jt_symbolic_analyzed_at_initialize = true;
            p.adjoint_cache.jt_values_transposed_on_device = false;
            p.adjoint_cache.jt_factorized_during_forward_cache = true;
            p.adjoint_cache.host_roundtrip_for_jt_transpose = false;
            p.adjoint_cache.backend_name = "cpu_klu";
            p.adjoint_cache.transpose_solve_backend_name =
                "cpu_klu_tsolve_cached_factorization";
        }
#ifdef CUPF_WITH_CUDA
        else {
            if (!solve_options.allow_explicit_transpose_fallback) {
                throw std::runtime_error(
                    "NewtonSolver::prepare_adjoint_cache(): cuDSS transpose solve is unsupported; "
                    "enable allow_explicit_transpose_fallback to cache an explicit J^T factorization");
            }
            double factor_ms = 0.0;
            p.linear_solve.prepare_adjoint_explicit_transpose_cache(p.buf, ctx, factor_ms);
            p.adjoint_cache.factorization_time_ms = factor_ms;
            p.adjoint_cache.factorization_supports_transpose_solve = false;
            p.adjoint_cache.used_explicit_transpose = true;
            p.adjoint_cache.includes_host_device_transfer = false;
            p.adjoint_cache.jt_symbolic_analyzed_at_initialize =
                p.linear_solve.has_adjoint_symbolic_analysis();
            p.adjoint_cache.jt_values_transposed_on_device = true;
            p.adjoint_cache.jt_factorized_during_forward_cache = true;
            p.adjoint_cache.host_roundtrip_for_jt_transpose = false;
            if constexpr (std::is_same_v<std::decay_t<decltype(p)>, CudaFp64Pipeline>) {
                p.adjoint_cache.backend_name = "cuda_cudss_fp64";
#ifdef CUPF_ENABLE_CUSTOM_SOLVER
            } else if constexpr (std::is_same_v<std::decay_t<decltype(p)>, CudaFp64CustomPipeline>) {
                p.adjoint_cache.backend_name = "cuda_custom_fp64";
#endif
            } else if constexpr (std::is_same_v<std::decay_t<decltype(p)>, CudaFp32Pipeline>) {
                p.adjoint_cache.backend_name = "cuda_cudss_fp32";
            } else {
                p.adjoint_cache.backend_name = "cuda_cudss_mixed";
            }
            p.adjoint_cache.transpose_solve_backend_name =
                "cuda_cudss_cached_explicit_transpose_factorization";
        }
#endif

        if (p.adjoint_cache.factorization_time_ms == 0.0) {
            p.adjoint_cache.factorization_time_ms = elapsed_ms(start, Clock::now());
        }
    }, pipeline_->v);
}


// ---------------------------------------------------------------------------
// run_iteration_stages: NR loop - ibus -> mismatch -> norm -> jac -> solve -> update
// ---------------------------------------------------------------------------
int32_t NewtonSolver::run_iteration_stages(IterationContext& ctx)
{
    int32_t completed = 0;

    for (int32_t iter = 0; iter < ctx.config.max_iter; ++iter) {
        StageScope total("NR.iteration.total");
        ctx.iter = iter;
        ctx.jacobian_updated_this_iter = false;
        completed = iter + 1;

        std::visit([&](auto& p) {
            { StageScope s("NR.iteration.ibus");          p.ibus(ctx); }
            { StageScope s("NR.iteration.mismatch");      p.mismatch(ctx); }
            { StageScope s("NR.iteration.mismatch_norm"); p.mismatch_norm(ctx); }
        }, pipeline_->v);

        if (ctx.converged) break;

        std::visit([&](auto& p) {
            { StageScope s("NR.iteration.jacobian");       p.jacobian(ctx); }
        }, pipeline_->v);
        ctx.jacobian_updated_this_iter = true;
        ctx.jacobian_age = 0;

        std::visit([&](auto& p) {
            { StageScope s("NR.iteration.prepare_rhs");    p.prepare_rhs(ctx); }
            { StageScope s("NR.iteration.factorize");      p.factorize(ctx); }
            { StageScope s("NR.iteration.solve");          p.solve(ctx); }
            { StageScope s("NR.iteration.voltage_update"); p.voltage_update(ctx); }
        }, pipeline_->v);
    }

    return completed;
}
