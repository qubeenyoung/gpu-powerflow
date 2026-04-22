// ---------------------------------------------------------------------------
// newton_solver.cpp
//
// NewtonSolver의 구현.
//
// public I/O는 항상 FP64다. backend·precision 결정은 생성자 시점에 완료되며,
// 이후 hot path는 std::visit를 통해 pipeline으로 위임한다.
// ---------------------------------------------------------------------------

#include "newton_solver/core/newton_solver.hpp"
#include "newton_solver/core/pipeline.hpp"
#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/ops/jacobian/jacobian_analysis.hpp"
#include "utils/nvtx_trace.hpp"
#include "utils/timer.hpp"

#include <stdexcept>
#include <variant>


namespace {

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

}  // namespace


// ---------------------------------------------------------------------------
// Constructor: assemble the pipeline from NewtonOptions.
// ---------------------------------------------------------------------------
NewtonSolver::NewtonSolver(const NewtonOptions& options)
{
    if (options.backend == BackendKind::CPU) {
        pipeline_ = std::make_unique<SolverPipeline>(
            SolverPipeline{CpuFp64Pipeline{}});
        return;
    }

#ifdef CUPF_WITH_CUDA
    if (options.backend == BackendKind::CUDA) {
        if (options.compute == ComputePolicy::FP64) {
            pipeline_ = std::make_unique<SolverPipeline>(
                SolverPipeline{CudaFp64Pipeline{options.cudss}});
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
// initialize: Jacobian analysis → pipeline::initialize (prepare + KLU/cuDSS)
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
    NRResult&                result)
{
    NRBatchResult batch_result;
    solve_batch(ybus, sbus, ybus.rows, V0, ybus.rows, 1,
                pv, n_pv, pq, n_pq, config, batch_result);

    result = {};
    result.V = std::move(batch_result.V);
    result.iterations    = batch_result.iterations.empty()     ? 0    : batch_result.iterations[0];
    result.final_mismatch= batch_result.final_mismatch.empty() ? 0.0  : batch_result.final_mismatch[0];
    result.converged     = !batch_result.converged.empty() && batch_result.converged[0] != 0;
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
    NRBatchResult&           result)
{
    StageScope total("NR.solve.total");

    if (!initialized_) {
        throw std::runtime_error(
            "NewtonSolver::solve_batch(): initialize()를 먼저 호출해야 합니다.");
    }

    validate_batch_args(batch_size, sbus_stride, ybus.rows, "sbus");
    validate_batch_args(batch_size, V0_stride,   ybus.rows, "V0");

    if (batch_size != 1) {
        const bool supported = std::visit(
            [](const auto& p) { return p.batch_supported; }, pipeline_->v);
        if (!supported) {
            throw std::runtime_error(
                "NewtonSolver::solve_batch(): batch_size > 1 is currently supported "
                "only by the CUDA Mixed pipeline.");
        }
    }

    result = {};
    result.n_bus       = ybus.rows;
    result.batch_size  = batch_size;

    {
        StageScope stage("NR.solve.upload");
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

    {
        StageScope stage("NR.solve.download");
        std::visit([&](auto& p) { p.download_batch(result); }, pipeline_->v);
    }

    result.n_bus      = ybus.rows;
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


// ---------------------------------------------------------------------------
// run_iteration_stages: NR loop — ibus → mismatch → norm → jac → solve → update
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
