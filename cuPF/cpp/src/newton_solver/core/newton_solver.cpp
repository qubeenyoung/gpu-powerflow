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
// run_iteration_stages: NR 반복 루프 선택
// ---------------------------------------------------------------------------
int32_t NewtonSolver::run_iteration_stages(IterationContext& ctx)
{
    switch (options_.algorithm) {
        case NewtonAlgorithm::Standard:
            return run_standard_iteration_stages(ctx);
        case NewtonAlgorithm::Modified:
            return run_modified_iteration_stages(ctx);
    }

    throw std::invalid_argument("NewtonSolver: unsupported NewtonAlgorithm");
}


// ---------------------------------------------------------------------------
// run_standard_iteration_stages: 기존 NR 반복 루프
//
// 각 반복은 4개의 stage로 구성된다.
//   mismatch     — F = S_specified - S_calculated, normF 계산
//   jacobian     — Jacobian J 채우기
//   linear_solve — factorize 후 J·dx = -F 풀기
//   voltage_update — dx 적용, V 재구성
//
// mismatch 단계 후 수렴하면 루프를 즉시 종료한다.
// ---------------------------------------------------------------------------
int32_t NewtonSolver::run_standard_iteration_stages(IterationContext& ctx)
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
            StageScope linear("NR.iteration.linear");
            {
                StageScope stage("NR.iteration.linear_factorize");
                plan_.linear_solve->factorize(ctx);
            }
            {
                StageScope stage("NR.iteration.linear_solve");
                plan_.linear_solve->solve(ctx);
            }
        }
        {
            StageScope stage("NR.iteration.voltage_update");
            plan_.voltage_update->run(ctx);
        }
    }

    return completed;
}


// ---------------------------------------------------------------------------
// run_modified_iteration_stages: factorization reuse 스케줄
//
// 초기 V0에서 F0/J0를 만든 뒤, outer pass마다 existing Jacobian을 한 번
// factorize한다. 이후 같은 factorization으로
//   solve(Fk)   → voltage_update(Vk+1) → mismatch(Fk+1)
//   solve(Fk+1) → voltage_update(Vk+2) → mismatch(Fk+2)
// 를 최대 두 번 수행하고, 루프 마지막에서 다음 factorize에 사용할 Jacobian을
// 갱신한다.
//
// 따라서 operator 타이밍에서 linear_factorize count는 linear_solve count의
// 절반(수렴/최대반복에 따라 올림)이 된다.
// ---------------------------------------------------------------------------
int32_t NewtonSolver::run_modified_iteration_stages(IterationContext& ctx)
{
    if (ctx.config.max_iter <= 0) {
        return 0;
    }

    int32_t completed = 0;

    ctx.iter = 0;
    {
        StageScope stage("NR.iteration.mismatch");
        plan_.mismatch->run(ctx);
        completed = 1;
    }
    if (ctx.converged) {
        return completed;
    }

    ctx.iter = completed - 1;
    {
        StageScope stage("NR.iteration.jacobian");
        plan_.jacobian->run(ctx);
    }

    int32_t corrections = 0;
    while (!ctx.converged && corrections < ctx.config.max_iter) {
        for (int32_t reuse = 0;
             reuse < 2 && !ctx.converged && corrections < ctx.config.max_iter;
             ++reuse) {
            StageScope total("NR.iteration.total");

            // The current F was produced by the latest mismatch evaluation.
            ctx.iter = completed - 1;

            if (reuse == 0) {
                {
                    StageScope linear("NR.iteration.linear");
                    {
                        StageScope stage("NR.iteration.linear_factorize");
                        plan_.linear_solve->factorize(ctx);
                    }
                    {
                        StageScope stage("NR.iteration.linear_solve");
                        plan_.linear_solve->solve(ctx);
                    }
                }
            } else {
                {
                    StageScope linear("NR.iteration.linear");
                    {
                        StageScope stage("NR.iteration.linear_solve");
                        plan_.linear_solve->solve(ctx);
                    }
                }
            }

            {
                StageScope stage("NR.iteration.voltage_update");
                plan_.voltage_update->run(ctx);
            }
            ++corrections;

            ctx.iter = completed;
            {
                StageScope stage("NR.iteration.mismatch");
                plan_.mismatch->run(ctx);
                completed = ctx.iter + 1;
            }
        }

        if (!ctx.converged && corrections < ctx.config.max_iter) {
            ctx.iter = completed - 1;
            {
                StageScope stage("NR.iteration.jacobian");
                plan_.jacobian->run(ctx);
            }
        }
    }

    return completed;
}
