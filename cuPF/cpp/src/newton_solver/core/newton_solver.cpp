// ---------------------------------------------------------------------------
// newton_solver.cpp
//
// NewtonSolver의 구현.
//
// public I/O는 항상 FP64다. Mixed 모드에서 내부 FP32 연산이 사용되더라도
// 이 파일에는 precision 분기가 없다. backend·precision 결정은 모두 생성자
// 시점에 PlanBuilder가 완료하며, 이후 hot path는 인터페이스 호출만 한다.
// ---------------------------------------------------------------------------

#include "newton_solver/core/newton_solver.hpp"
#include "newton_solver/core/contexts.hpp"
#include "newton_solver/core/plan_builder.hpp"
#include "utils/nvtx_trace.hpp"
#include "utils/timer.hpp"

#include <stdexcept>


namespace {

// ---------------------------------------------------------------------------
// StageScope: 하나의 NR stage에 대한 NVTX range + 타이머를 묶어 관리한다.
// 스택에 올라가고 소멸 시 자동으로 range·타이머를 닫는다.
// ---------------------------------------------------------------------------
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
// 생성자: NewtonOptions로부터 ExecutionPlan을 조립한다.
//
// JacobianBuilder 알고리즘은 NewtonOptions에서 선택한다.
// 기본값은 EdgeBased이며, VertexBased는 CUDA Jacobian에서 warp-per-row 커널을 사용한다.
// ---------------------------------------------------------------------------
NewtonSolver::NewtonSolver(const NewtonOptions& options)
    : options_(options)
    , plan_(PlanBuilder::build(options))
    , jac_builder_(options.jacobian_builder)
{}


NewtonSolver::~NewtonSolver() = default;


// ---------------------------------------------------------------------------
// analyze: Ybus 희소 구조를 분석하고 solve()에 필요한 내부 상태를 준비한다.
//
// 단계:
//   1. JacobianBuilder::analyze() — 희소 맵과 Jacobian CSR 패턴 생성
//   2. IStorage::prepare()         — device-side 디스크립터 초기화
//   3. ILinearSolveOp::analyze()   — solver symbolic 분석
//
// 네트워크 위상(pv/pq 집합)이 바뀌지 않는 한 한 번만 호출하면 된다.
// ---------------------------------------------------------------------------
void NewtonSolver::analyze(
    const YbusViewF64& ybus,
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq)
{
    StageScope total("NR.analyze.total");

    JacobianBuilder::Result analysis;
    {
        StageScope stage("NR.analyze.jacobian_builder");
        analysis = jac_builder_.analyze(ybus, pv, n_pv, pq, n_pq);
    }

    AnalyzeContext ctx{
        .ybus  = ybus,
        .maps  = analysis.maps,
        .J     = analysis.J,
        .n_bus = ybus.rows,
        .pv    = pv, .n_pv = n_pv,
        .pq    = pq, .n_pq = n_pq,
    };

    run_analyze_stages(ctx);
    analyzed_ = true;
}


// ---------------------------------------------------------------------------
// solve: Newton-Raphson 반복을 실행하고 result를 채운다.
//
// 단계:
//   1. IStorage::upload()           — Ybus 값, V0, Sbus를 device로 올린다
//   2. run_iteration_stages()       — NR 반복 루프 (mismatch→J→solve→dV)
//   3. IStorage::download_result()  — 최종 전압을 host로 내린다
// ---------------------------------------------------------------------------
void NewtonSolver::solve(
    const YbusViewF64&          ybus,
    const std::complex<double>* sbus,
    const std::complex<double>* V0,
    const int32_t*              pv, int32_t n_pv,
    const int32_t*              pq, int32_t n_pq,
    const NRConfig&             config,
    NRResultF64&                result)
{
    NRBatchResultF64 batch_result;
    solve_batch(ybus,
                sbus,
                ybus.rows,
                V0,
                ybus.rows,
                1,
                pv,
                n_pv,
                pq,
                n_pq,
                config,
                batch_result);

    result = {};
    result.V = std::move(batch_result.V);
    result.iterations = batch_result.iterations.empty() ? 0 : batch_result.iterations[0];
    result.final_mismatch = batch_result.final_mismatch.empty() ? 0.0 : batch_result.final_mismatch[0];
    result.converged = !batch_result.converged.empty() && batch_result.converged[0] != 0;
}


void NewtonSolver::solve_batch(
    const YbusViewF64&          ybus,
    const std::complex<double>* sbus,
    int64_t                     sbus_stride,
    const std::complex<double>* V0,
    int64_t                     V0_stride,
    int32_t                     batch_size,
    const int32_t*              pv, int32_t n_pv,
    const int32_t*              pq, int32_t n_pq,
    const NRConfig&             config,
    NRBatchResultF64&           result)
{
    StageScope total("NR.solve.total");

    if (!analyzed_) {
        throw std::runtime_error(
            "NewtonSolver::solve_batch(): analyze()를 먼저 호출해야 합니다.");
    }

    validate_batch_args(batch_size, sbus_stride, ybus.rows, "sbus");
    validate_batch_args(batch_size, V0_stride, ybus.rows, "V0");

    result = {};
    result.n_bus = ybus.rows;
    result.batch_size = batch_size;

    // 현재 실제 B>1 실행은 CUDA Mixed optimized path에서만 활성화되어 있다.
    // CPU/FP64 CUDA 경로는 single-case compatibility path로 남긴다.
    if (batch_size != 1 &&
        !(plan_.storage->backend() == BackendKind::CUDA &&
          plan_.storage->compute() == ComputePolicy::Mixed)) {
        throw std::runtime_error(
            "NewtonSolver::solve_batch(): batch_size > 1 is currently supported "
            "only by the CUDA Mixed path.");
    }

    {
        StageScope stage("NR.solve.upload");
        SolveContext upload_ctx{
            .ybus   = &ybus,
            .sbus   = sbus,
            .V0     = V0,
            .config = &config,
            .batch_size = batch_size,
            .sbus_stride = sbus_stride,
            .V0_stride = V0_stride,
            .ybus_values_batched = false,
            .ybus_value_stride = ybus.nnz,
        };
        plan_.storage->upload(upload_ctx);
    }

    IterationContext iter_ctx{
        .storage = *plan_.storage,
        .config  = config,
        .pv      = pv, .n_pv = n_pv,
        .pq      = pq, .n_pq = n_pq,
    };

    const int32_t iterations = run_iteration_stages(iter_ctx);

    {
        StageScope stage("NR.solve.download");
        plan_.storage->download_batch_result(result);
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
            result.final_mismatch.empty() ? iter_ctx.normF : result.final_mismatch[static_cast<std::size_t>(b)];
        result.converged[static_cast<std::size_t>(b)] =
            static_cast<uint8_t>(norm <= config.tolerance ? 1 : 0);
    }
}


// ---------------------------------------------------------------------------
// run_analyze_stages: storage 준비 → linear_solve symbolic 분석
// ---------------------------------------------------------------------------
void NewtonSolver::run_analyze_stages(const AnalyzeContext& ctx)
{
    {
        StageScope stage("NR.analyze.storage_prepare");
        plan_.storage->prepare(ctx);
    }
    {
        StageScope stage("NR.analyze.linear_solve");
        plan_.linear_solve->analyze(ctx);
    }
}


// ---------------------------------------------------------------------------
// run_iteration_stages: NR 반복 루프
//
// 각 반복은 4개의 stage로 구성된다.
//   mismatch     — F = S_calculated - S_specified, normF 계산
//   jacobian     — Jacobian J 채우기
//   linear_solve — 선형계 풀이
//   voltage_update — dx 적용, V cache 재구성
//
// mismatch 단계 후 수렴하면 루프를 즉시 종료한다.
// ---------------------------------------------------------------------------
int32_t NewtonSolver::run_iteration_stages(IterationContext& ctx)
{
    int32_t completed = 0;

    for (int32_t iter = 0; iter < ctx.config.max_iter; ++iter) {
        StageScope total("NR.iteration.total");
        ctx.iter = iter;
        ctx.jacobian_updated_this_iter = false;
        completed = iter + 1;

        {
            StageScope stage("NR.iteration.mismatch");
            plan_.mismatch->run(ctx);
        }
        if (ctx.converged) {
            break;
        }

        {
            StageScope stage("NR.iteration.jacobian");
            plan_.jacobian->run(ctx);
            ctx.jacobian_updated_this_iter = true;
            ctx.jacobian_age = 0;
        }
        {
            StageScope stage("NR.iteration.linear_solve");
            plan_.linear_solve->run(ctx);
        }
        {
            StageScope stage("NR.iteration.voltage_update");
            plan_.voltage_update->run(ctx);
        }
    }

    return completed;
}
