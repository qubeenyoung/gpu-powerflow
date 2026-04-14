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
    StageScope total("NR.solve.total");

    if (!analyzed_) {
        throw std::runtime_error(
            "NewtonSolver::solve(): analyze()를 먼저 호출해야 합니다.");
    }

    result = {};

    {
        StageScope stage("NR.solve.upload");
        SolveContext upload_ctx{
            .ybus   = &ybus,
            .sbus   = sbus,
            .V0     = V0,
            .config = &config,
        };
        plan_.storage->upload(upload_ctx);
    }

    IterationContext iter_ctx{
        .storage = *plan_.storage,
        .config  = config,
        .pv      = pv, .n_pv = n_pv,
        .pq      = pq, .n_pq = n_pq,
    };

    result.iterations = run_iteration_stages(iter_ctx);
    result.converged  = iter_ctx.converged;

    {
        StageScope stage("NR.solve.download");
        plan_.storage->download_result(result);
    }

    result.final_mismatch = iter_ctx.normF;
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
//   mismatch     — F = S_specified - S_calculated, normF 계산
//   jacobian     — Jacobian J 채우기
//   linear_solve — J·dx = -F 풀기
//   voltage_update — dx 적용, V 재구성
//
// mismatch 단계 후 수렴하면 루프를 즉시 종료한다.
// ---------------------------------------------------------------------------
int32_t NewtonSolver::run_iteration_stages(IterationContext& ctx)
{
    int32_t completed = 0;

    for (int32_t iter = 0; iter < ctx.config.max_iter; ++iter) {
        StageScope total("NR.iteration.total");
        ctx.iter = iter;
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
