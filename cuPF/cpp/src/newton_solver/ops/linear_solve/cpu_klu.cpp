// ---------------------------------------------------------------------------
// cpu_klu.cpp — CPU FP64 선형 솔버 구현 (Eigen::KLU)
// ---------------------------------------------------------------------------

#include "cpu_klu.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"

#include <Eigen/Sparse>
#include <stdexcept>


namespace {
using CpuRealVectorF64 = Eigen::Matrix<double, Eigen::Dynamic, 1>;
}


void CpuLinearSolveKLU::initialize(CpuFp64Buffers& buf, const InitializeContext& ctx)
{
    (void)ctx;

    if (buf.J.rows() != buf.dimF || buf.J.cols() != buf.dimF) {
        throw std::runtime_error("CpuLinearSolveKLU::initialize: Jacobian shape does not match dimF");
    }
    if (buf.J.nonZeros() <= 0) {
        throw std::runtime_error("CpuLinearSolveKLU::initialize: Jacobian sparsity is empty");
    }

    lu_ = std::make_unique<Eigen::KLU<CpuJacobianMatrixF64>>();
    lu_->analyzePattern(buf.J);
    if (lu_->info() != Eigen::Success) {
        throw std::runtime_error("CpuLinearSolveKLU::initialize: KLU symbolic analysis failed");
    }
    has_symbolic_ = true;
}


void CpuLinearSolveKLU::prepare_rhs(CpuFp64Buffers& buf, IterationContext& ctx)
{
    (void)buf;
    (void)ctx;
}


void CpuLinearSolveKLU::factorize(CpuFp64Buffers& buf, IterationContext& ctx)
{
    (void)ctx;

    if (buf.J.rows() != buf.dimF || buf.J.cols() != buf.dimF) {
        throw std::runtime_error("CpuLinearSolveKLU::factorize: Jacobian shape does not match dimF");
    }
    if (buf.J.nonZeros() <= 0) {
        throw std::runtime_error("CpuLinearSolveKLU::factorize: Jacobian is empty");
    }
    if (!has_symbolic_ || !lu_) {
        throw std::runtime_error("CpuLinearSolveKLU::factorize: initialize() was not called");
    }

    lu_->factorize(buf.J);
    if (lu_->info() != Eigen::Success) {
        throw std::runtime_error("CpuLinearSolveKLU::factorize: KLU numeric factorization failed");
    }
}


void CpuLinearSolveKLU::solve(CpuFp64Buffers& buf, IterationContext& ctx)
{
    (void)ctx;

    Eigen::Map<const CpuRealVectorF64> F(buf.F.data(), buf.dimF);
    Eigen::Map<CpuRealVectorF64>      dx(buf.dx.data(), buf.dimF);

    dx = lu_->solve(F);
    if (lu_->info() != Eigen::Success) {
        throw std::runtime_error("CpuLinearSolveKLU::solve: KLU solve failed");
    }
}
