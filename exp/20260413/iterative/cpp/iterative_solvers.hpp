#pragma once

#include "iterative_probe_common.hpp"

#include <string>
#include <string_view>

namespace exp_20260413::iterative::probe {

bool is_supported_solver(std::string_view solver_name);
std::string supported_solver_names();

ProbeResult solve_snapshot(const SparseMatrix& matrix,
                           const Vector& rhs,
                           const Snapshot& snapshot,
                           const SolverOptions& options,
                           const std::filesystem::path& snapshot_dir);

ProbeResult solve_bicgstab_ilut(const SparseMatrix& matrix,
                                const Vector& rhs,
                                const Snapshot& snapshot,
                                const SolverOptions& options,
                                const std::filesystem::path& snapshot_dir);

ProbeResult solve_bicgstab_ilu0(const SparseMatrix& matrix,
                                const Vector& rhs,
                                const Snapshot& snapshot,
                                const SolverOptions& options,
                                const std::filesystem::path& snapshot_dir);

ProbeResult solve_bicgstab_ilu1(const SparseMatrix& matrix,
                                const Vector& rhs,
                                const Snapshot& snapshot,
                                const SolverOptions& options,
                                const std::filesystem::path& snapshot_dir);

ProbeResult solve_bicgstab_block_jacobi(const SparseMatrix& matrix,
                                        const Vector& rhs,
                                        const Snapshot& snapshot,
                                        const SolverOptions& options,
                                        const std::filesystem::path& snapshot_dir);

ProbeResult solve_bicgstab_diag(const SparseMatrix& matrix,
                                const Vector& rhs,
                                const Snapshot& snapshot,
                                const SolverOptions& options,
                                const std::filesystem::path& snapshot_dir);

ProbeResult solve_bicgstab_identity(const SparseMatrix& matrix,
                                    const Vector& rhs,
                                    const Snapshot& snapshot,
                                    const SolverOptions& options,
                                    const std::filesystem::path& snapshot_dir);

#ifdef ITERATIVE_WITH_HYPRE
ProbeResult solve_bicgstab_hypre_boomeramg(const SparseMatrix& matrix,
                                           const Vector& rhs,
                                           const Snapshot& snapshot,
                                           const SolverOptions& options,
                                           const std::filesystem::path& snapshot_dir);

ProbeResult solve_fgmres_hypre_boomeramg(const SparseMatrix& matrix,
                                         const Vector& rhs,
                                         const Snapshot& snapshot,
                                         const SolverOptions& options,
                                         const std::filesystem::path& snapshot_dir);
#endif

}  // namespace exp_20260413::iterative::probe
