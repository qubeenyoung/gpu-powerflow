#include "hypre_boomeramg_common.hpp"
#include "iterative_solvers.hpp"

#include <HYPRE_parcsr_ls.h>
#include <mpi.h>

#include <limits>

namespace exp_20260413::iterative::probe {

ProbeResult solve_fgmres_hypre_boomeramg(const SparseMatrix& matrix,
                                         const Vector& rhs,
                                         const Snapshot& snapshot,
                                         const SolverOptions& options,
                                         const std::filesystem::path& snapshot_dir)
{
    constexpr int kKrylovDimension = 30;

    initialize_hypre_runtime();
    clear_hypre_errors();

    ProbeResult result = make_probe_result(snapshot_dir, "fgmres_hypre_boomeramg", snapshot);
    Vector x = Vector::Zero(rhs.size());
    HYPRE_Solver solver = nullptr;
    HYPRE_Solver preconditioner = nullptr;

    const auto setup_start = Clock::now();
    HypreIjMatrix hypre_matrix(matrix);
    HypreIjVector hypre_rhs(rhs);
    HypreIjVector hypre_x(static_cast<int32_t>(rhs.size()), 0.0);

    preconditioner = create_boomeramg_preconditioner();
    check_hypre(HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &solver),
                "HYPRE_ParCSRFlexGMRESCreate");
    check_hypre(HYPRE_ParCSRFlexGMRESSetKDim(solver, kKrylovDimension),
                "HYPRE_ParCSRFlexGMRESSetKDim");
    check_hypre(HYPRE_ParCSRFlexGMRESSetTol(solver, options.tolerance),
                "HYPRE_ParCSRFlexGMRESSetTol");
    check_hypre(HYPRE_ParCSRFlexGMRESSetMaxIter(solver, options.max_iter),
                "HYPRE_ParCSRFlexGMRESSetMaxIter");
    check_hypre(HYPRE_ParCSRFlexGMRESSetLogging(solver, 1), "HYPRE_ParCSRFlexGMRESSetLogging");
    check_hypre(HYPRE_ParCSRFlexGMRESSetPrintLevel(solver, 0),
                "HYPRE_ParCSRFlexGMRESSetPrintLevel");
    check_hypre(HYPRE_ParCSRFlexGMRESSetPrecond(
                    solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, preconditioner),
                "HYPRE_ParCSRFlexGMRESSetPrecond");
    check_hypre(HYPRE_ParCSRFlexGMRESSetup(
                    solver, hypre_matrix.parcsr(), hypre_rhs.parvector(), hypre_x.parvector()),
                "HYPRE_ParCSRFlexGMRESSetup");
    const auto setup_end = Clock::now();
    result.setup_sec = std::chrono::duration<double>(setup_end - setup_start).count();

    const auto solve_start = Clock::now();
    const HYPRE_Int solve_code = HYPRE_ParCSRFlexGMRESSolve(
        solver, hypre_matrix.parcsr(), hypre_rhs.parvector(), hypre_x.parvector());
    if (solve_code != 0) {
        clear_hypre_errors();
    }
    const auto solve_end = Clock::now();
    result.solve_sec = std::chrono::duration<double>(solve_end - solve_start).count();

    HYPRE_Int iterations = 0;
    HYPRE_Real estimated_error = 0.0;
    const HYPRE_Int get_iter_code = HYPRE_ParCSRFlexGMRESGetNumIterations(solver, &iterations);
    if (get_iter_code != 0) {
        clear_hypre_errors();
        iterations = options.max_iter;
    }
    const HYPRE_Int get_error_code =
        HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm(solver, &estimated_error);
    if (get_error_code != 0) {
        clear_hypre_errors();
        estimated_error = std::numeric_limits<double>::quiet_NaN();
    }

    x = hypre_x.values();
    result.iterations = static_cast<int>(iterations);
    result.estimated_error = estimated_error;

    HYPRE_ParCSRFlexGMRESDestroy(solver);
    HYPRE_BoomerAMGDestroy(preconditioner);

    finalize_probe_result(result, snapshot, x);
    result.success = result.relative_residual_inf <= options.tolerance;
    return result;
}

}  // namespace exp_20260413::iterative::probe
