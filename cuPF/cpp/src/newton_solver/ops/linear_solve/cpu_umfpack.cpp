#include "cpu_umfpack.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "utils/dump.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>


CpuLinearSolveUMFPACK::CpuLinearSolveUMFPACK()
{
    umfpack_di_defaults(control_);
}


CpuLinearSolveUMFPACK::~CpuLinearSolveUMFPACK()
{
    release();
}


CpuLinearSolveUMFPACK::CpuLinearSolveUMFPACK(CpuLinearSolveUMFPACK&& other) noexcept
    : symbolic_(other.symbolic_)
    , numeric_(other.numeric_)
    , ap_(other.ap_)
    , ai_(other.ai_)
    , ax_(other.ax_)
    , dim_(other.dim_)
    , has_symbolic_(other.has_symbolic_)
{
    std::copy(other.control_, other.control_ + UMFPACK_CONTROL, control_);
    std::copy(other.info_, other.info_ + UMFPACK_INFO, info_);
    other.symbolic_ = nullptr;
    other.numeric_ = nullptr;
    other.ap_ = nullptr;
    other.ai_ = nullptr;
    other.ax_ = nullptr;
    other.dim_ = 0;
    other.has_symbolic_ = false;
}


CpuLinearSolveUMFPACK& CpuLinearSolveUMFPACK::operator=(CpuLinearSolveUMFPACK&& other) noexcept
{
    if (this != &other) {
        release();
        symbolic_ = other.symbolic_;
        numeric_ = other.numeric_;
        ap_ = other.ap_;
        ai_ = other.ai_;
        ax_ = other.ax_;
        dim_ = other.dim_;
        has_symbolic_ = other.has_symbolic_;
        std::copy(other.control_, other.control_ + UMFPACK_CONTROL, control_);
        std::copy(other.info_, other.info_ + UMFPACK_INFO, info_);
        other.symbolic_ = nullptr;
        other.numeric_ = nullptr;
        other.ap_ = nullptr;
        other.ai_ = nullptr;
        other.ax_ = nullptr;
        other.dim_ = 0;
        other.has_symbolic_ = false;
    }
    return *this;
}


void CpuLinearSolveUMFPACK::release()
{
    if (numeric_) {
        umfpack_di_free_numeric(&numeric_);
        numeric_ = nullptr;
    }
    if (symbolic_) {
        umfpack_di_free_symbolic(&symbolic_);
        symbolic_ = nullptr;
    }
    ap_ = nullptr;
    ai_ = nullptr;
    ax_ = nullptr;
    dim_ = 0;
    has_symbolic_ = false;
}


void CpuLinearSolveUMFPACK::check_status(int status, const char* where) const
{
    if (status < 0) {
        throw std::runtime_error(std::string(where) + ": UMFPACK status " + std::to_string(status));
    }
}


void CpuLinearSolveUMFPACK::initialize(CpuFp64Storage& buf, const InitializeContext& ctx)
{
    (void)ctx;

    if (buf.J.rows() != buf.dimF || buf.J.cols() != buf.dimF) {
        throw std::runtime_error("CpuLinearSolveUMFPACK::initialize: Jacobian shape does not match dimF");
    }
    if (buf.J.nonZeros() <= 0) {
        throw std::runtime_error("CpuLinearSolveUMFPACK::initialize: Jacobian sparsity is empty");
    }
    if (!buf.J.isCompressed()) {
        throw std::runtime_error("CpuLinearSolveUMFPACK::initialize: Jacobian must be compressed CSC");
    }

    release();
    umfpack_di_defaults(control_);
    const int status = umfpack_di_symbolic(
        buf.dimF,
        buf.dimF,
        const_cast<int32_t*>(buf.J.outerIndexPtr()),
        const_cast<int32_t*>(buf.J.innerIndexPtr()),
        const_cast<double*>(buf.J.valuePtr()),
        &symbolic_,
        control_,
        info_);
    check_status(status, "CpuLinearSolveUMFPACK::initialize");
    if (!symbolic_) {
        throw std::runtime_error("CpuLinearSolveUMFPACK::initialize: symbolic analysis failed");
    }
    has_symbolic_ = true;
}


void CpuLinearSolveUMFPACK::prepare_rhs(CpuFp64Storage& buf, IterationContext& ctx)
{
    (void)buf;
    (void)ctx;
}


void CpuLinearSolveUMFPACK::factorize(CpuFp64Storage& buf, IterationContext& ctx)
{
    (void)ctx;

    if (buf.J.rows() != buf.dimF || buf.J.cols() != buf.dimF) {
        throw std::runtime_error("CpuLinearSolveUMFPACK::factorize: Jacobian shape does not match dimF");
    }
    if (buf.J.nonZeros() <= 0) {
        throw std::runtime_error("CpuLinearSolveUMFPACK::factorize: Jacobian is empty");
    }
    if (!has_symbolic_ || !symbolic_) {
        throw std::runtime_error("CpuLinearSolveUMFPACK::factorize: initialize() was not called");
    }
    if (!buf.J.isCompressed()) {
        throw std::runtime_error("CpuLinearSolveUMFPACK::factorize: Jacobian must be compressed CSC");
    }

    if (numeric_) {
        umfpack_di_free_numeric(&numeric_);
        numeric_ = nullptr;
    }
    const int status = umfpack_di_numeric(
        const_cast<int32_t*>(buf.J.outerIndexPtr()),
        const_cast<int32_t*>(buf.J.innerIndexPtr()),
        const_cast<double*>(buf.J.valuePtr()),
        symbolic_,
        &numeric_,
        control_,
        info_);
    check_status(status, "CpuLinearSolveUMFPACK::factorize");
    if (!numeric_) {
        throw std::runtime_error("CpuLinearSolveUMFPACK::factorize: numeric factorization failed");
    }
    ap_ = buf.J.outerIndexPtr();
    ai_ = buf.J.innerIndexPtr();
    ax_ = buf.J.valuePtr();
    dim_ = buf.dimF;
}


void CpuLinearSolveUMFPACK::solve(CpuFp64Storage& buf, IterationContext& ctx)
{
    if (!numeric_) {
        throw std::runtime_error("CpuLinearSolveUMFPACK::solve: factorize() must be called first");
    }

    const int status = umfpack_di_solve(
        UMFPACK_A,
        const_cast<int32_t*>(buf.J.outerIndexPtr()),
        const_cast<int32_t*>(buf.J.innerIndexPtr()),
        const_cast<double*>(buf.J.valuePtr()),
        buf.dx.data(),
        buf.F.data(),
        numeric_,
        control_,
        info_);
    check_status(status, "CpuLinearSolveUMFPACK::solve");
    newton_solver::utils::dumpVector("dx", ctx.iter, buf.dx);
}


void CpuLinearSolveUMFPACK::solve_transpose(const double* rhs,
                                            double* solution,
                                            int32_t dim,
                                            int32_t nrhs)
{
    if (!numeric_) {
        throw std::runtime_error("CpuLinearSolveUMFPACK::solve_transpose: factorize() must be called first");
    }
    if (rhs == nullptr || solution == nullptr || dim <= 0 || nrhs <= 0) {
        throw std::invalid_argument("CpuLinearSolveUMFPACK::solve_transpose: invalid arguments");
    }
    if (nrhs != 1) {
        throw std::invalid_argument("CpuLinearSolveUMFPACK::solve_transpose: only one RHS is currently supported");
    }

    if (ap_ == nullptr || ai_ == nullptr || ax_ == nullptr || dim_ != dim) {
        throw std::runtime_error("CpuLinearSolveUMFPACK::solve_transpose: active matrix is unavailable");
    }

    const int status = umfpack_di_solve(
        UMFPACK_At,
        const_cast<int32_t*>(ap_),
        const_cast<int32_t*>(ai_),
        const_cast<double*>(ax_),
        solution,
        const_cast<double*>(rhs),
        numeric_,
        control_,
        info_);
    check_status(status, "CpuLinearSolveUMFPACK::solve_transpose");
}
