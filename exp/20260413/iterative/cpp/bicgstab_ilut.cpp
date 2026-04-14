#include "eigen_bicgstab_runner.hpp"
#include "iterative_solvers.hpp"

#include <Eigen/IterativeLinearSolvers>

namespace exp_20260413::iterative::probe {

ProbeResult solve_bicgstab_ilut(const SparseMatrix& matrix,
                                const Vector& rhs,
                                const Snapshot& snapshot,
                                const SolverOptions& options,
                                const std::filesystem::path& snapshot_dir)
{
    Eigen::BiCGSTAB<SparseMatrix, Eigen::IncompleteLUT<double>> solver;
    solver.preconditioner().setDroptol(options.ilut_drop_tol);
    solver.preconditioner().setFillfactor(options.ilut_fill_factor);

    return solve_with_eigen_bicgstab(
        solver, "bicgstab_ilut", matrix, rhs, snapshot, options, snapshot_dir);
}

}  // namespace exp_20260413::iterative::probe
