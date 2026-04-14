// ---------------------------------------------------------------------------
// cpu_klu.cpp — CPU FP64 선형 솔버 구현 (Eigen::KLU)
//
// SuiteSparse KLU를 Eigen 래퍼를 통해 사용하는 희소 직접 솔버.
//
// analyze() 단계 (한 번만 호출):
//   1. lu.analyzePattern(J) — KLU symbolic analysis
//      Jacobian CSC 구조만 사용하므로 analyze 시점에 J.values는 미확정이어도 됨.
//      결과는 내부 KLU symbolic 객체에 저장; storage_.has_klu_symbolic = true 표시.
//
// run() 단계 (매 NR 반복):
//   1. lu.factorize(J) — KLU numeric factorization (LU 분해)
//   2. dx = lu.solve(-F) — forward/backward substitution → dx = J⁻¹·(-F)
//
// 선형 시스템: J · dx = -F   →   dx = -J⁻¹ · F
// ---------------------------------------------------------------------------

#include "cpu_klu.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "utils/timer.hpp"

#include <Eigen/Sparse>

#include <stdexcept>


namespace {

using CpuRealVectorF64 = Eigen::Matrix<double, Eigen::Dynamic, 1>;

}  // namespace


CpuLinearSolveKLU::CpuLinearSolveKLU(IStorage& storage)
    : storage_(static_cast<CpuFp64Storage&>(storage)) {}


// KLU symbolic analysis: Jacobian 희소 패턴 분석 (한 번만 수행).
// J.values는 이 시점에 확정되지 않아도 된다; 구조(indptr/indices)만 사용.
void CpuLinearSolveKLU::analyze(const AnalyzeContext& ctx)
{
    (void)ctx;

    if (storage_.J.rows() != storage_.dimF || storage_.J.cols() != storage_.dimF) {
        throw std::runtime_error("CpuLinearSolveKLU::analyze: Jacobian shape does not match dimF");
    }
    if (storage_.J.nonZeros() <= 0) {
        throw std::runtime_error("CpuLinearSolveKLU::analyze: Jacobian sparsity is empty");
    }

    newton_solver::utils::ScopedTimer timer("CPU.analyze.kluSymbolic");
    storage_.lu.analyzePattern(storage_.J);
    if (storage_.lu.info() != Eigen::Success) {
        throw std::runtime_error("CpuLinearSolveKLU::analyze: KLU symbolic analysis failed");
    }
    storage_.has_klu_symbolic = true;
}


// 매 NR 반복 호출: numeric factorize → solve.
// dx = J⁻¹ · (-F) 를 Eigen::Map 으로 storage_ 버퍼에 직접 기록.
void CpuLinearSolveKLU::run(IterationContext& ctx)
{
    (void)ctx;

    if (storage_.J.rows() != storage_.dimF || storage_.J.cols() != storage_.dimF) {
        throw std::runtime_error("CpuLinearSolveKLU::run: Jacobian shape does not match dimF");
    }
    if (storage_.J.nonZeros() <= 0) {
        throw std::runtime_error("CpuLinearSolveKLU::run: Jacobian is empty");
    }
    if (!storage_.has_klu_symbolic) {
        throw std::runtime_error("CpuLinearSolveKLU::run: linear solver analyze was not completed");
    }

    {
        // numeric factorization: 현재 J.values 기준 LU 분해
        newton_solver::utils::ScopedTimer timer("CPU.solve.factorize");
        storage_.lu.factorize(storage_.J);
        if (storage_.lu.info() != Eigen::Success) {
            throw std::runtime_error("CpuLinearSolveKLU::run: KLU numeric factorization failed");
        }
    }

    // Eigen::Map: 외부 버퍼를 Eigen 벡터처럼 참조 (복사 없음)
    Eigen::Map<const CpuRealVectorF64> F(storage_.F.data(), storage_.dimF);
    Eigen::Map<CpuRealVectorF64> dx(storage_.dx.data(), storage_.dimF);

    {
        // triangular solve: dx = J⁻¹·(-F) — forward/backward substitution
        newton_solver::utils::ScopedTimer timer("CPU.solve.solve");
        dx = storage_.lu.solve(-F);
        if (storage_.lu.info() != Eigen::Success) {
            throw std::runtime_error("CpuLinearSolveKLU::run: KLU solve failed");
        }
    }
}
