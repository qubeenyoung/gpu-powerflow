#pragma once

#include "iterative_probe_common.hpp"

#include <Eigen/IterativeLinearSolvers>

#include <chrono>

namespace exp_20260413::iterative::probe {

// Small adapter for Eigen's BiCGSTAB variants. The solver-specific files choose
// the preconditioner; this function only records timings and residuals uniformly.
template <typename Solver>
ProbeResult solve_with_eigen_bicgstab(Solver& solver,
                                      const std::string& solver_name,
                                      const SparseMatrix& matrix,
                                      const Vector& rhs,
                                      const Snapshot& snapshot,
                                      const SolverOptions& options,
                                      const std::filesystem::path& snapshot_dir)
{
    ProbeResult result = make_probe_result(snapshot_dir, solver_name, snapshot);

    solver.setTolerance(options.tolerance);
    solver.setMaxIterations(options.max_iter);

    const auto setup_start = Clock::now();
    solver.compute(matrix);
    const auto setup_end = Clock::now();
    result.setup_sec = std::chrono::duration<double>(setup_end - setup_start).count();

    const auto solve_start = Clock::now();
    const Vector x = solver.solve(rhs);
    const auto solve_end = Clock::now();
    result.solve_sec = std::chrono::duration<double>(solve_end - solve_start).count();

    result.success = solver.info() == Eigen::Success;
    result.iterations = static_cast<int>(solver.iterations());
    result.estimated_error = solver.error();
    finalize_probe_result(result, snapshot, x);
    return result;
}

}  // namespace exp_20260413::iterative::probe
