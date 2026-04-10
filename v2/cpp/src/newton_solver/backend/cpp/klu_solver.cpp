#include "cpu_backend_impl.hpp"
#include "utils/timer.hpp"

#include <Eigen/Sparse>
#include <Eigen/KLUSupport>


// ---------------------------------------------------------------------------
// solveLinearSystem
//
// Numerically factorizes J (via KLU) and solves J·dx = -F.
//
// The symbolic factorization (column ordering) was already done once in
// analyze(). Here we only redo the numeric factorization each iteration
// because J values change while J's sparsity pattern stays fixed.
//
// Output dx is written into the caller's buffer (same length as F: n_pv + 2*n_pq).
// ---------------------------------------------------------------------------
void CpuNewtonSolverBackend::solveLinearSystem(const double* F, double* dx)
{
    auto& im = *impl_;

    {
        newton_solver::utils::ScopedTimer timer("CPU.solve.factorize");
        im.lu.factorize(im.J);
    }

    // Solve J·dx = -F
    const int32_t dimF = im.J.rows();
    Eigen::Map<const Eigen::VectorXd> F_vec(F, dimF);
    Eigen::Map<Eigen::VectorXd> dx_map(dx, dimF);
    {
        newton_solver::utils::ScopedTimer timer("CPU.solve.solve");
        dx_map = im.lu.solve(-F_vec);
    }
}
