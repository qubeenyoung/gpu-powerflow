#include "iterative_solvers.hpp"

#include <stdexcept>

namespace exp_20260413::iterative::probe {

bool is_supported_solver(std::string_view solver_name)
{
    return solver_name == "bicgstab_ilut" ||
           solver_name == "bicgstab_ilu0" ||
           solver_name == "bicgstab_ilu1" ||
           solver_name == "bicgstab_block_jacobi" ||
           solver_name == "bicgstab_diag" ||
           solver_name == "bicgstab_identity"
#ifdef ITERATIVE_WITH_HYPRE
           || solver_name == "bicgstab_hypre_boomeramg" ||
           solver_name == "fgmres_hypre_boomeramg"
#endif
        ;
}

std::string supported_solver_names()
{
    std::string names = "bicgstab_ilut|bicgstab_ilu0|bicgstab_ilu1|"
                        "bicgstab_block_jacobi|bicgstab_diag|bicgstab_identity";
#ifdef ITERATIVE_WITH_HYPRE
    names += "|bicgstab_hypre_boomeramg|fgmres_hypre_boomeramg";
#endif
    return names;
}

ProbeResult solve_snapshot(const SparseMatrix& matrix,
                           const Vector& rhs,
                           const Snapshot& snapshot,
                           const SolverOptions& options,
                           const std::filesystem::path& snapshot_dir)
{
    if (options.solver == "bicgstab_ilut") {
        return solve_bicgstab_ilut(matrix, rhs, snapshot, options, snapshot_dir);
    }
    if (options.solver == "bicgstab_ilu0") {
        return solve_bicgstab_ilu0(matrix, rhs, snapshot, options, snapshot_dir);
    }
    if (options.solver == "bicgstab_ilu1") {
        return solve_bicgstab_ilu1(matrix, rhs, snapshot, options, snapshot_dir);
    }
    if (options.solver == "bicgstab_block_jacobi") {
        return solve_bicgstab_block_jacobi(matrix, rhs, snapshot, options, snapshot_dir);
    }
    if (options.solver == "bicgstab_diag") {
        return solve_bicgstab_diag(matrix, rhs, snapshot, options, snapshot_dir);
    }
    if (options.solver == "bicgstab_identity") {
        return solve_bicgstab_identity(matrix, rhs, snapshot, options, snapshot_dir);
    }
#ifdef ITERATIVE_WITH_HYPRE
    if (options.solver == "bicgstab_hypre_boomeramg") {
        return solve_bicgstab_hypre_boomeramg(matrix, rhs, snapshot, options, snapshot_dir);
    }
    if (options.solver == "fgmres_hypre_boomeramg") {
        return solve_fgmres_hypre_boomeramg(matrix, rhs, snapshot, options, snapshot_dir);
    }
#endif

    throw std::runtime_error("Unknown solver: " + options.solver);
}

}  // namespace exp_20260413::iterative::probe
