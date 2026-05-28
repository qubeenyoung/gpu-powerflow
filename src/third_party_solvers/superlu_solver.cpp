#include "third_party_solvers/superlu_solver.hpp"

#include <stdexcept>
#include <vector>

#include <slu_ddefs.h>

#include "third_party_solvers/utils.hpp"
#include "tools/timer.hpp"

namespace sparse_direct::solver {
namespace {

std::vector<int_t> to_superlu_index_array(const std::vector<matrix::Index>& values)
{
    std::vector<int_t> converted(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        converted[i] = static_cast<int_t>(values[i]);
    }
    return converted;
}

class SuperLuSolver final : public LinearSolver {
public:
    std::string name() const override
    {
        return "superlu";
    }

    SolverRun solve(
        const matrix::CsrMatrix&,
        const matrix::CscMatrix& csc,
        const std::vector<matrix::Value>& b) override
    {
        SolverRun result;

        SuperMatrix A;
        SuperMatrix B;
        SuperMatrix AC;
        SuperMatrix L;
        SuperMatrix U;
        GlobalLU_t global_lu;
        SuperLUStat_t stat;

        bool matrix_created = false;
        bool reordered_matrix_created = false;
        bool rhs_created = false;
        bool factors_created = false;
        bool stat_created = false;

        try {
            utils::require_square(csc, "SuperLU");
            utils::require_rhs_size(csc.rows, b.size());

            std::vector<int_t> row_idx = to_superlu_index_array(csc.row_idx);
            std::vector<int_t> col_ptr = to_superlu_index_array(csc.col_ptr);
            std::vector<double> matrix_values = csc.values;
            std::vector<double> rhs_values = b;

            dCreate_CompCol_Matrix(
                &A,
                csc.rows,
                csc.cols,
                static_cast<int_t>(csc.nnz()),
                matrix_values.data(),
                row_idx.data(),
                col_ptr.data(),
                SLU_NC,
                SLU_D,
                SLU_GE);
            matrix_created = true;

            dCreate_Dense_Matrix(
                &B,
                csc.rows,
                1,
                rhs_values.data(),
                csc.rows,
                SLU_DN,
                SLU_D,
                SLU_GE);
            rhs_created = true;

            superlu_options_t options;
            set_default_options(&options);
            options.PrintStat = NO;
            options.Fact = DOFACT;
            options.Equil = NO;

            std::vector<int> perm_c(static_cast<std::size_t>(csc.cols));
            std::vector<int> perm_r(static_cast<std::size_t>(csc.rows));
            std::vector<int> etree(static_cast<std::size_t>(csc.cols));
            int_t info = 0;

            StatInit(&stat);
            stat_created = true;

            timer::Stopwatch phase_timer;
            get_perm_c(options.ColPerm, &A, perm_c.data());
            sp_preorder(&options, &A, perm_c.data(), etree.data(), &AC);
            result.analysis_ms = phase_timer.elapsed_ms();
            reordered_matrix_created = true;

            const int relax = sp_ienv(2);
            const int panel_size = sp_ienv(1);
            phase_timer.reset();
            dgstrf(
                &options,
                &AC,
                relax,
                panel_size,
                etree.data(),
                nullptr,
                0,
                perm_c.data(),
                perm_r.data(),
                &L,
                &U,
                &global_lu,
                &stat,
                &info);
            result.factor_ms = phase_timer.elapsed_ms();
            factors_created = (info == 0);

            if (!factors_created) {
                throw std::runtime_error("dgstrf factorization failed with info " + std::to_string(info));
            }

            int solve_info = 0;
            phase_timer.reset();
            dgstrs(
                NOTRANS,
                &L,
                &U,
                perm_c.data(),
                perm_r.data(),
                &B,
                &stat,
                &solve_info);
            result.solve_ms = phase_timer.elapsed_ms();

            if (solve_info != 0) {
                throw std::runtime_error("dgstrs solve failed with info " + std::to_string(solve_info));
            }

            result.x = rhs_values;

            result.success = true;
            result.message = "ok";
        } catch (const std::exception& error) {
            result.success = false;
            result.message = error.what();
            result.x.clear();
        }

        if (factors_created) {
            Destroy_SuperNode_Matrix(&L);
            Destroy_CompCol_Matrix(&U);
        }
        if (reordered_matrix_created) {
            Destroy_CompCol_Permuted(&AC);
        }
        if (stat_created) {
            StatFree(&stat);
        }
        if (rhs_created) {
            Destroy_SuperMatrix_Store(&B);
        }
        if (matrix_created) {
            Destroy_SuperMatrix_Store(&A);
        }

        return result;
    }
};

}  // namespace

std::unique_ptr<LinearSolver> make_superlu_solver()
{
    return std::make_unique<SuperLuSolver>();
}

}  // namespace sparse_direct::solver
