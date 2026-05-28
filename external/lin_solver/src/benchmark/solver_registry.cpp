#include "benchmark/solver_registry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "mysolver/factorize/own_pipeline.hpp"
#include "mysolver/solver.hpp"
#include "third_party_solvers/utils.hpp"
#include "tools/timer.hpp"

namespace sparse_direct::solver {
namespace {

// Componentwise backward error (matches tools/compute_error): no x_true needed,
// so the adapter can decide whether to trust the own-numeric result.
double componentwise_berr(const matrix::CsrMatrix& a, const std::vector<matrix::Value>& b,
                          const std::vector<double>& x)
{
    double berr = 0.0;
    for (matrix::Index row = 0; row < a.rows; ++row) {
        double ax = 0.0;
        double den = std::fabs(b[row]);
        for (matrix::Index p = a.row_ptr[row]; p < a.row_ptr[row + 1]; ++p) {
            const double aij = a.values[p];
            const double xj = x[a.col_idx[p]];
            ax += aij * xj;
            den += std::fabs(aij) * std::fabs(xj);
        }
        const double num = std::fabs(ax - b[row]);
        if (den > 0.0) {
            berr = std::max(berr, num / den);
        } else if (num > 0.0) {
            return std::numeric_limits<double>::infinity();
        }
    }
    return berr;
}

// --solver mysolver. M3: try mysolver's own numeric factorization first; if its
// backward error clears the accuracy gate, use it. Otherwise fall back to the
// KLU-backed path (still mysolver's own ordering). This is gate-safe: a bad
// own-numeric result is never accepted.
class MysolverSolver final : public LinearSolver {
public:
    std::string name() const override
    {
        return "mysolver";
    }

    SolverRun solve(
        const matrix::CsrMatrix& csr,
        const matrix::CscMatrix& csc,
        const std::vector<matrix::Value>& b) override
    {
        SolverRun result;

        try {
            utils::require_square(csc, name());
            utils::require_rhs_size(csc.rows, b.size());

            // Own numeric pipeline first (no KLU).
            std::vector<double> x_own;
            mysolver::OwnSolveStats st;
            if (mysolver::try_own_solve(csc, b, x_own, &st) &&
                componentwise_berr(csr, b, x_own) <= 1e-10) {
                result.x = std::move(x_own);
                result.analysis_ms = st.analysis_ms;
                result.factor_ms = st.factor_ms;
                result.solve_ms = st.solve_ms;
                result.success = true;
                result.message = "ok (M3 own-numeric)";
                return result;
            }

            // Fallback: mysolver ordering + KLU numeric.
            timer::Stopwatch phase_timer;
            mysolver::AnalyzeResult analysis = mysolver::analyze(csc);
            result.analysis_ms = phase_timer.elapsed_ms();

            phase_timer.reset();
            mysolver::FactorState factor;
            mysolver::factorize(csc, analysis, &factor);
            result.factor_ms = phase_timer.elapsed_ms();

            phase_timer.reset();
            mysolver::solve(factor, analysis, b, result.x);
            result.solve_ms = phase_timer.elapsed_ms();

            result.success = true;
            result.message = std::string("ok (klu-fallback ") +
                             (factor.ordering == 3 ? "metis-nd" : "amd") + ")";
        } catch (const std::exception& error) {
            result.success = false;
            result.message = error.what();
            result.x.clear();
        }

        return result;
    }
};

}  // namespace

std::unique_ptr<LinearSolver> make_mysolver_solver()
{
    return std::make_unique<MysolverSolver>();
}

}  // namespace sparse_direct::solver
