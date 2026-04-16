#pragma once

#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/core/jacobian_types.hpp"


// Forward declarations
struct IterationContext;
struct AnalyzeContext;
struct SolveContext;


// ---------------------------------------------------------------------------
// IStorage: 한 ExecutionPlan에 필요한 버퍼·라이브러리 핸들·디스크립터를 소유.
//
// 책임:
//   - host/device 버퍼 할당과 생명주기 관리
//   - cuSPARSE / cuDSS / Eigen / KLU 핸들과 디스크립터
//   - analyze 단계 메모리 레이아웃 준비
//   - solve 초기화 입력 업로드 (Ybus 값, V0, Sbus)
//
// "메모리 + 라이브러리 상태" 이다. 계산 로직은 구현하지 않는다.
// ---------------------------------------------------------------------------
class IStorage {
public:
    virtual ~IStorage() = default;

    virtual BackendKind   backend() const = 0;
    virtual ComputePolicy compute() const = 0;

    // JacobianBuilder::analyze() 결과를 받아 device-side 구조를 초기화한다.
    // solve() 전에 한 번만 호출된다.
    virtual void prepare(const AnalyzeContext& ctx) = 0;

    // 매 solve() 시작 시 호출. Ybus 값, V0, Sbus를 device로 올린다.
    virtual void upload(const SolveContext& ctx) = 0;

    // NR 루프 완료 후 최종 전압 벡터를 host로 내려 result에 채운다.
    virtual void download_result(NRResultF64& result) const = 0;
};


// ---------------------------------------------------------------------------
// Op 인터페이스: NR stage solver의 연산자 계약을 정의한다.
//
// 각 구체 Op는 특정 Storage 타입과 결합되어 생성된다.
// NR 핫 패스에서는 인터페이스 호출만 수행하며, backend 분기가 없다.
// ---------------------------------------------------------------------------

class IMismatchOp {
public:
    virtual ~IMismatchOp() = default;

    // F = mismatch(V, Ybus, Sbus), normF를 계산하고 ctx.converged를 설정한다.
    virtual void run(IterationContext& ctx) = 0;
};

class IJacobianOp {
public:
    virtual ~IJacobianOp() = default;

    // J 값을 초기화하고 Ybus 원소를 Jacobian에 산포(scatter)한다.
    virtual void run(IterationContext& ctx) = 0;
};

class ILinearSolveOp {
public:
    virtual ~ILinearSolveOp() = default;

    // Jacobian 희소 구조에 대한 symbolic 분석을 실행한다.
    // NewtonSolver::analyze() 내에서 storage->prepare() 이후 한 번 호출된다.
    virtual void analyze(const AnalyzeContext& ctx) = 0;

    // 현재 Jacobian 값으로 numeric factorization / refactorization을 수행한다.
    virtual void factorize(IterationContext& ctx) = 0;

    // 현재 mismatch F로 b = -F를 준비하고 factorized Jacobian으로 J·dx = b를 푼다.
    virtual void solve(IterationContext& ctx) = 0;

    // 기존 호출부 호환용: factorize와 solve를 연속 실행한다.
    virtual void run(IterationContext& ctx)
    {
        factorize(ctx);
        solve(ctx);
    }
};

class IVoltageUpdateOp {
public:
    virtual ~IVoltageUpdateOp() = default;

    // dx를 적용해 Va·Vm을 갱신하고 복소 전압 V를 재구성한다.
    virtual void run(IterationContext& ctx) = 0;
};
