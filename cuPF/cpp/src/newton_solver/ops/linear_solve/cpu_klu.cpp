// ---------------------------------------------------------------------------
// cpu_klu.cpp — CPU FP64 선형 솔버 구현 (SuiteSparse KLU)
// ---------------------------------------------------------------------------

#include "cpu_klu.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "utils/dump.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <stdexcept>


namespace {
using CpuRealVectorF64 = Eigen::Matrix<double, Eigen::Dynamic, 1>;
}


CpuLinearSolveKLU::CpuLinearSolveKLU()
{
    klu_defaults(&common_);
}


CpuLinearSolveKLU::~CpuLinearSolveKLU()
{
    release();
}


CpuLinearSolveKLU::CpuLinearSolveKLU(CpuLinearSolveKLU&& other) noexcept
    : symbolic_(other.symbolic_)
    , numeric_(other.numeric_)
    , common_(other.common_)
    , has_symbolic_(other.has_symbolic_)
{
    other.symbolic_ = nullptr;
    other.numeric_ = nullptr;
    other.has_symbolic_ = false;
}


CpuLinearSolveKLU& CpuLinearSolveKLU::operator=(CpuLinearSolveKLU&& other) noexcept
{
    if (this != &other) {
        release();
        symbolic_ = other.symbolic_;
        numeric_ = other.numeric_;
        common_ = other.common_;
        has_symbolic_ = other.has_symbolic_;
        other.symbolic_ = nullptr;
        other.numeric_ = nullptr;
        other.has_symbolic_ = false;
    }
    return *this;
}


void CpuLinearSolveKLU::release()
{
    if (numeric_) {
        klu_free_numeric(&numeric_, &common_);
        numeric_ = nullptr;
    }
    if (symbolic_) {
        klu_free_symbolic(&symbolic_, &common_);
        symbolic_ = nullptr;
    }
    has_symbolic_ = false;
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
    if (!buf.J.isCompressed()) {
        throw std::runtime_error("CpuLinearSolveKLU::initialize: Jacobian must be compressed CSC");
    }

    release();
    klu_defaults(&common_);
    symbolic_ = klu_analyze(
        buf.dimF,
        const_cast<int32_t*>(buf.J.outerIndexPtr()),
        const_cast<int32_t*>(buf.J.innerIndexPtr()),
        &common_);
    if (!symbolic_) {
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
    if (!has_symbolic_ || !symbolic_) {
        throw std::runtime_error("CpuLinearSolveKLU::factorize: initialize() was not called");
    }
    if (!buf.J.isCompressed()) {
        throw std::runtime_error("CpuLinearSolveKLU::factorize: Jacobian must be compressed CSC");
    }

    if (numeric_) {
        klu_free_numeric(&numeric_, &common_);
        numeric_ = nullptr;
    }
    numeric_ = klu_factor(
        const_cast<int32_t*>(buf.J.outerIndexPtr()),
        const_cast<int32_t*>(buf.J.innerIndexPtr()),
        const_cast<double*>(buf.J.valuePtr()),
        symbolic_,
        &common_);
    if (!numeric_) {
        throw std::runtime_error("CpuLinearSolveKLU::factorize: KLU numeric factorization failed");
    }
}


void CpuLinearSolveKLU::solve(CpuFp64Buffers& buf, IterationContext& ctx)
{
    (void)ctx;

    if (!numeric_ || !symbolic_) {
        throw std::runtime_error("CpuLinearSolveKLU::solve: factorize() must be called first");
    }

    std::copy(buf.F.begin(), buf.F.end(), buf.dx.begin());
    const int ok = klu_solve(symbolic_, numeric_, buf.dimF, 1, buf.dx.data(), &common_);
    if (!ok) {
        throw std::runtime_error("CpuLinearSolveKLU::solve: KLU solve failed");
    }
    newton_solver::utils::dumpVector("dx", ctx.iter, buf.dx);
}


void CpuLinearSolveKLU::solve_transpose(const double* rhs,
                                        double* solution,
                                        int32_t dim,
                                        int32_t nrhs)
{
    if (!numeric_ || !symbolic_) {
        throw std::runtime_error("CpuLinearSolveKLU::solve_transpose: factorize() must be called first");
    }
    if (rhs == nullptr || solution == nullptr || dim <= 0 || nrhs <= 0) {
        throw std::invalid_argument("CpuLinearSolveKLU::solve_transpose: invalid arguments");
    }

    std::copy(rhs, rhs + static_cast<std::size_t>(dim) * static_cast<std::size_t>(nrhs), solution);
    const int ok = klu_tsolve(symbolic_, numeric_, dim, nrhs, solution, &common_);
    if (!ok) {
        throw std::runtime_error("CpuLinearSolveKLU::solve_transpose: KLU transpose solve failed");
    }
}
