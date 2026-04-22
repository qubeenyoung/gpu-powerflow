#pragma once

#include <Eigen/KLUSupport>
#include <Eigen/Sparse>

#include <memory>

struct CpuFp64Buffers;
struct InitializeContext;
struct IterationContext;

using CpuJacobianMatrixF64 = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;


// ---------------------------------------------------------------------------
// CpuLinearSolveKLU: KLU 기반 희소 직접 솔버 (CPU FP64 경로).
//
// KLU symbolic/numeric 상태를 직접 소유한다. 이전에 CpuFp64Buffers에
// 있던 lu, has_klu_symbolic 필드가 여기로 이동되었다.
// ---------------------------------------------------------------------------
struct CpuLinearSolveKLU {
    void initialize(CpuFp64Buffers& buf, const InitializeContext& ctx);
    void prepare_rhs(CpuFp64Buffers& buf, IterationContext& ctx);
    void factorize(CpuFp64Buffers& buf, IterationContext& ctx);
    void solve(CpuFp64Buffers& buf, IterationContext& ctx);

private:
    std::unique_ptr<Eigen::KLU<CpuJacobianMatrixF64>> lu_;
    bool has_symbolic_ = false;
};
