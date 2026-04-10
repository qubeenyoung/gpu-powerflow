#include "naive_cpu_backend_impl.hpp"

#include <superlu/slu_ddefs.h>

#include <stdexcept>
#include <string>
#include <vector>


// ---------------------------------------------------------------------------
// solveLinearSystem
//
// One-shot SuperLU dgssv: symbolic ordering + numeric LU + triangular solve
// in a single call, with no reuse of any previous factorization.
//
// This mirrors scipy.sparse.linalg.spsolve() in the current Python baseline:
//   - Python environment has no scikits.umfpack installed.
//   - pplinsolve() default path calls spsolve() directly (PF_LIN_SOLVER_NR = "").
//   - spsolve() uses SuperLU internally when UMFPACK is unavailable.
//   - options.Fact = DOFACT → full symbolic + numeric each call, same as spsolve().
//
// Input layout:
//   F[0..n_pv-1]          = Re(mis[pv])
//   F[n_pv..n_pvpq-1]     = Re(mis[pq])
//   F[n_pvpq..dimF-1]     = Im(mis[pq])
//
// SuperLU interface:
//   A  — input sparse matrix (CCS / CSC, non-owning view of im.J)
//   B  — dense RHS on entry (-F), solution on exit
//   L, U — output LU factors (allocated internally, freed here)
// ---------------------------------------------------------------------------
void NaiveCpuNewtonSolverBackend::solveLinearSystem(const double* F, double* dx)
{
    auto& im = *impl_;
    const int n   = im.J.rows();
    const int nnz = im.J.nonZeros();

    // ------------------------------------------------------------------
    // Wrap Eigen CSC Jacobian as SuperLU NCS (Non-CSC) matrix.
    //
    // Eigen ColMajor: outerIndexPtr = col_ptr (size n+1),
    //                 innerIndexPtr = row_ind (size nnz),
    //                 valuePtr      = vals    (size nnz).
    // All are 0-based, matching SuperLU's default index base.
    //
    // dCreate_CompCol_Matrix does NOT copy data — it stores pointers.
    // Destroy_SuperMatrix_Store only frees the Store struct, not the arrays.
    // ------------------------------------------------------------------
    SuperMatrix A;
    dCreate_CompCol_Matrix(
        &A, n, n, nnz,
        const_cast<double*>(im.J.valuePtr()),
        const_cast<int*>(im.J.innerIndexPtr()),   // row indices
        const_cast<int*>(im.J.outerIndexPtr()),   // col pointers
        SLU_NC, SLU_D, SLU_GE);

    // ------------------------------------------------------------------
    // Dense RHS: B = -F.
    // dCreate_Dense_Matrix stores the pointer; solution overwrites in-place.
    // ------------------------------------------------------------------
    std::vector<double> rhs(n);
    for (int i = 0; i < n; ++i) rhs[i] = -F[i];

    SuperMatrix B;
    dCreate_Dense_Matrix(&B, n, 1, rhs.data(), n, SLU_DN, SLU_D, SLU_GE);

    // ------------------------------------------------------------------
    // Options: DOFACT = full symbolic + numeric factorization every call.
    // Equivalent to spsolve() one-shot (no structure reuse across calls).
    // ------------------------------------------------------------------
    superlu_options_t options;
    set_default_options(&options);
    options.Fact = DOFACT;

    // ------------------------------------------------------------------
    // Solve: A * x = B  →  B overwritten with x = A^{-1} * B = -J^{-1} * F
    // L and U factors are allocated internally by dgssv.
    // ------------------------------------------------------------------
    SuperMatrix L, U;
    std::vector<int> perm_c(n), perm_r(n);

    SuperLUStat_t stat;
    StatInit(&stat);

    int info = 0;
    dgssv(&options, &A, perm_c.data(), perm_r.data(),
          &L, &U, &B, &stat, &info);

    StatFree(&stat);

    if (info != 0) {
        Destroy_SuperMatrix_Store(&A);
        Destroy_SuperMatrix_Store(&B);
        Destroy_SuperNode_Matrix(&L);
        Destroy_CompCol_Matrix(&U);
        throw std::runtime_error(
            "NaiveCpuNewtonSolverBackend: SuperLU dgssv failed, info=" +
            std::to_string(info) + ". Jacobian may be singular.");
    }

    // ------------------------------------------------------------------
    // Copy solution from B (in-place after dgssv).
    // B.Store is a DNformat; nzval holds the dense column-major values.
    // ------------------------------------------------------------------
    const auto* dn  = static_cast<DNformat*>(B.Store);
    const auto* sol = static_cast<double*>(dn->nzval);
    for (int i = 0; i < n; ++i) dx[i] = sol[i];

    // ------------------------------------------------------------------
    // Free SuperLU-allocated factor storage.
    // Destroy_SuperMatrix_Store frees only the Store struct metadata,
    // NOT the underlying Eigen/rhs arrays — correct for non-owning views.
    // ------------------------------------------------------------------
    Destroy_SuperMatrix_Store(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_Matrix(&L);
    Destroy_CompCol_Matrix(&U);
}
