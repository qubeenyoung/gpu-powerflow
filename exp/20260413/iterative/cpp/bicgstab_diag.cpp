#include "eigen_bicgstab_runner.hpp"
#include "iterative_solvers.hpp"

#include <Eigen/IterativeLinearSolvers>

namespace exp_20260413::iterative::probe {

ProbeResult solve_bicgstab_diag(const SparseMatrix& matrix,
                                const Vector& rhs,
                                const Snapshot& snapshot,
                                const SolverOptions& options,
                                const std::filesystem::path& snapshot_dir)
{
    Eigen::BiCGSTAB<SparseMatrix, Eigen::DiagonalPreconditioner<double>> solver;

    return solve_with_eigen_bicgstab(
        solver, "bicgstab_diag", matrix, rhs, snapshot, options, snapshot_dir);
}

}  // namespace exp_20260413::iterative::probe
