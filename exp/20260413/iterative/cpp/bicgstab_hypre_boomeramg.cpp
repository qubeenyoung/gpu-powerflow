#include "hypre_boomeramg_common.hpp"
#include "iterative_solvers.hpp"

#include <HYPRE_parcsr_ls.h>
#include <mpi.h>

#include <limits>

namespace exp_20260413::iterative::probe {

ProbeResult solve_bicgstab_hypre_boomeramg(const SparseMatrix& matrix,
                                           const Vector& rhs,
                                           const Snapshot& snapshot,
                                           const SolverOptions& options,
                                           const std::filesystem::path& snapshot_dir)
{
    initialize_hypre_runtime();
    clear_hypre_errors();

    ProbeResult result = make_probe_result(snapshot_dir, "bicgstab_hypre_boomeramg", snapshot);
    Vector x = Vector::Zero(rhs.size());
    HYPRE_Solver solver = nullptr;
    HYPRE_Solver preconditioner = nullptr;

    const auto setup_start = Clock::now();
    HypreIjMatrix hypre_matrix(matrix);
    HypreIjVector hypre_rhs(rhs);
    HypreIjVector hypre_x(static_cast<int32_t>(rhs.size()), 0.0);

    preconditioner = create_boomeramg_preconditioner();
    check_hypre(HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver),
                "HYPRE_ParCSRBiCGSTABCreate");
    check_hypre(HYPRE_ParCSRBiCGSTABSetTol(solver, options.tolerance),
                "HYPRE_ParCSRBiCGSTABSetTol");
    check_hypre(HYPRE_ParCSRBiCGSTABSetMaxIter(solver, options.max_iter),
                "HYPRE_ParCSRBiCGSTABSetMaxIter");
    check_hypre(HYPRE_ParCSRBiCGSTABSetLogging(solver, 1), "HYPRE_ParCSRBiCGSTABSetLogging");
    check_hypre(HYPRE_ParCSRBiCGSTABSetPrintLevel(solver, 0),
                "HYPRE_ParCSRBiCGSTABSetPrintLevel");
    check_hypre(HYPRE_ParCSRBiCGSTABSetPrecond(
                    solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, preconditioner),
                "HYPRE_ParCSRBiCGSTABSetPrecond");
    check_hypre(HYPRE_ParCSRBiCGSTABSetup(
                    solver, hypre_matrix.parcsr(), hypre_rhs.parvector(), hypre_x.parvector()),
                "HYPRE_ParCSRBiCGSTABSetup");
    const auto setup_end = Clock::now();
    result.setup_sec = std::chrono::duration<double>(setup_end - setup_start).count();

    const auto solve_start = Clock::now();
    const HYPRE_Int solve_code = HYPRE_ParCSRBiCGSTABSolve(
        solver, hypre_matrix.parcsr(), hypre_rhs.parvector(), hypre_x.parvector());
    if (solve_code != 0) {
        clear_hypre_errors();
    }
    const auto solve_end = Clock::now();
    result.solve_sec = std::chrono::duration<double>(solve_end - solve_start).count();

    HYPRE_Int iterations = 0;
    HYPRE_Real estimated_error = 0.0;
    const HYPRE_Int get_iter_code = HYPRE_ParCSRBiCGSTABGetNumIterations(solver, &iterations);
    if (get_iter_code != 0) {
        clear_hypre_errors();
        iterations = options.max_iter;
    }
    const HYPRE_Int get_error_code =
        HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver, &estimated_error);
    if (get_error_code != 0) {
        clear_hypre_errors();
        estimated_error = std::numeric_limits<double>::quiet_NaN();
    }

    x = hypre_x.values();
    result.iterations = static_cast<int>(iterations);
    result.estimated_error = estimated_error;

    HYPRE_ParCSRBiCGSTABDestroy(solver);
    HYPRE_BoomerAMGDestroy(preconditioner);

    finalize_probe_result(result, snapshot, x);
    result.success = result.relative_residual_inf <= options.tolerance;
    return result;
}

}  // namespace exp_20260413::iterative::probe
