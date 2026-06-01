// ---------------------------------------------------------------------------
// cpu_umfpack.cpp — CPU FP64 linear solver (SuiteSparse UMFPACK)
//
// Alternative to the KLU backend (cpu_klu.cpp) with the same
// initialize -> factorize -> solve interface. UMFPACK is also a CSC direct LU
// solver and natively supports the transpose solve (UMFPACK_At), so the CPU
// adjoint path reuses the SAME factorization with no explicit J^T (like KLU).
//
// UMFPACK keeps two opaque handles (symbolic_, numeric_) and two plain value
// arrays (control_ = input options, info_ = output stats). symbolic depends only
// on the sparsity (computed once); numeric is redone each iteration as J changes.
// ---------------------------------------------------------------------------

#include "cpu_umfpack.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "utils/dump.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>


// Load UMFPACK's default control settings into control_.
CpuLinearSolveUMFPACK::CpuLinearSolveUMFPACK()
{
    umfpack_di_defaults(control_);
}


CpuLinearSolveUMFPACK::~CpuLinearSolveUMFPACK()
{
    release();
}


// Move: transfer the opaque handles + cached matrix pointers and null the
// source; control_/info_ are fixed-size value arrays, so they are copied.
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


// Free both UMFPACK handles and forget the cached matrix arrays.
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


// UMFPACK returns negative codes for errors (positive = warnings); turn a
// genuine error into an exception tagged with the call site.
void CpuLinearSolveUMFPACK::check_status(int status, const char* where) const
{
    if (status < 0) {
        throw std::runtime_error(std::string(where) + ": UMFPACK status " + std::to_string(status));
    }
}


// One-time symbolic analysis (ordering + symbolic factorization) from the
// Jacobian sparsity only, reused across iterations. const_cast: UMFPACK's C API
// takes non-const pointers but only reads the pattern/values here.
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


// No-op for UMFPACK: solve() reads buf.F directly, nothing to stage here.
void CpuLinearSolveUMFPACK::prepare_rhs(CpuFp64Storage& buf, IterationContext& ctx)
{
    (void)buf;
    (void)ctx;
}


// Numeric LU for the current Jacobian values, reusing the symbolic analysis.
// Re-run every iteration as J changes. The matrix arrays (ap_/ai_/ax_) are
// cached so solve_transpose() can hand them back to UMFPACK_At later (the
// numeric object alone is not enough — the transpose solve re-reads the matrix).
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


// Forward solve J dx = F via UMFPACK_A. umfpack_di_solve writes the solution
// into its X argument (buf.dx) and reads B from buf.F — no in-place copy needed
// (unlike KLU).
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


// Transpose solve J^T x = rhs via UMFPACK_At, reusing the SAME cached numeric
// factorization as the forward solve (native transpose — no explicit J^T, like
// KLU). Needs the matrix arrays again, taken from the factorize()-time cache.
// Only a single RHS (nrhs == 1) is currently supported.
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
