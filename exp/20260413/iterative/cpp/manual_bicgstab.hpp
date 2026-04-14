#pragma once

#include "iterative_probe_common.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace exp_20260413::iterative::probe {

struct ManualSolveResult {
    Vector x;
    bool success = false;
    int32_t iterations = 0;
    double estimated_error = std::numeric_limits<double>::quiet_NaN();
};

// Shared BiCGSTAB loop for the hand-written preconditioners.
// The preconditioner type only needs a solve(rhs) method that applies M^{-1}.
template <typename Preconditioner>
ManualSolveResult run_bicgstab(const SparseMatrix& matrix,
                               const Vector& rhs,
                               const Preconditioner& preconditioner,
                               double tolerance,
                               int32_t max_iter)
{
    ManualSolveResult result;
    result.x = Vector::Zero(rhs.size());

    const double rhs_norm = std::max(vector_inf_norm(rhs), std::numeric_limits<double>::min());
    const double atol = tolerance * rhs_norm;

    Vector r = rhs - matrix * result.x;
    double residual_norm = vector_inf_norm(r);
    result.estimated_error = residual_norm / rhs_norm;
    if (residual_norm <= atol) {
        result.success = true;
        return result;
    }

    const Vector r_hat = r;
    Vector p = Vector::Zero(rhs.size());
    Vector v = Vector::Zero(rhs.size());

    double rho_prev = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    for (int32_t iter = 0; iter < max_iter; ++iter) {
        const double rho = r_hat.dot(r);
        if (!std::isfinite(rho) || std::abs(rho) <= std::numeric_limits<double>::min()) {
            break;
        }

        const double beta = (rho / rho_prev) * (alpha / omega);
        p = r + beta * (p - omega * v);

        const Vector p_hat = preconditioner.solve(p);
        v = matrix * p_hat;

        const double denom_alpha = r_hat.dot(v);
        if (!std::isfinite(denom_alpha) || std::abs(denom_alpha) <= std::numeric_limits<double>::min()) {
            break;
        }
        alpha = rho / denom_alpha;

        const Vector s = r - alpha * v;
        const double s_norm = vector_inf_norm(s);
        if (s_norm <= atol) {
            result.x += alpha * p_hat;
            result.iterations = iter + 1;
            result.estimated_error = s_norm / rhs_norm;
            result.success = true;
            return result;
        }

        const Vector s_hat = preconditioner.solve(s);
        const Vector t = matrix * s_hat;

        const double tt = t.dot(t);
        if (!std::isfinite(tt) || tt <= std::numeric_limits<double>::min()) {
            break;
        }
        omega = t.dot(s) / tt;
        if (!std::isfinite(omega) || std::abs(omega) <= std::numeric_limits<double>::min()) {
            break;
        }

        result.x += alpha * p_hat + omega * s_hat;
        r = s - omega * t;

        residual_norm = vector_inf_norm(r);
        result.iterations = iter + 1;
        result.estimated_error = residual_norm / rhs_norm;
        if (residual_norm <= atol) {
            result.success = true;
            return result;
        }

        rho_prev = rho;
    }

    return result;
}

template <typename Preconditioner, typename SetupFunction>
ProbeResult solve_with_manual_preconditioner(const std::string& solver_name,
                                             const SparseMatrix& matrix,
                                             const Vector& rhs,
                                             const Snapshot& snapshot,
                                             const SolverOptions& options,
                                             const std::filesystem::path& snapshot_dir,
                                             SetupFunction setup)
{
    ProbeResult result = make_probe_result(snapshot_dir, solver_name, snapshot);

    Preconditioner preconditioner;
    const auto setup_start = Clock::now();
    const bool setup_ok = setup(preconditioner);
    const auto setup_end = Clock::now();
    result.setup_sec = std::chrono::duration<double>(setup_end - setup_start).count();

    Vector x = Vector::Zero(rhs.size());
    if (setup_ok) {
        const auto solve_start = Clock::now();
        const ManualSolveResult solve =
            run_bicgstab(matrix, rhs, preconditioner, options.tolerance, options.max_iter);
        const auto solve_end = Clock::now();
        result.solve_sec = std::chrono::duration<double>(solve_end - solve_start).count();
        x = solve.x;
        result.success = solve.success;
        result.iterations = solve.iterations;
        result.estimated_error = solve.estimated_error;
    }

    finalize_probe_result(result, snapshot, x);
    return result;
}

}  // namespace exp_20260413::iterative::probe
