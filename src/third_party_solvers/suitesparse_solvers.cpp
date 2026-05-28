#include "third_party_solvers/suitesparse_solvers.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include <suitesparse/klu.h>
#include <suitesparse/umfpack.h>

#include "third_party_solvers/utils.hpp"
#include "tools/timer.hpp"

namespace sparse_direct::solver {
namespace {

class KluSolver final : public LinearSolver {
public:
    std::string name() const override
    {
        return "klu";
    }

    SolverRun solve(
        const matrix::CsrMatrix&,
        const matrix::CscMatrix& csc,
        const std::vector<matrix::Value>& b) override
    {
        SolverRun result;

        try {
            utils::require_square(csc, name());
            utils::require_rhs_size(csc.rows, b.size());

            klu_common common;
            klu_defaults(&common);

            timer::Stopwatch phase_timer;
            klu_symbolic* symbolic = klu_analyze(
                csc.cols,
                const_cast<int*>(csc.col_ptr.data()),
                const_cast<int*>(csc.row_idx.data()),
                &common);
            result.analysis_ms = phase_timer.elapsed_ms();
            if (!symbolic) {
                throw std::runtime_error("klu_analyze failed");
            }

            phase_timer.reset();
            klu_numeric* numeric = klu_factor(
                const_cast<int*>(csc.col_ptr.data()),
                const_cast<int*>(csc.row_idx.data()),
                const_cast<double*>(csc.values.data()),
                symbolic,
                &common);
            result.factor_ms = phase_timer.elapsed_ms();
            if (!numeric) {
                klu_free_symbolic(&symbolic, &common);
                throw std::runtime_error("klu_factor failed");
            }

            result.x = b;
            phase_timer.reset();
            const int ok = klu_solve(symbolic, numeric, csc.cols, 1, result.x.data(), &common);
            result.solve_ms = phase_timer.elapsed_ms();

            const long lnz = numeric->lnz;
            const long unz = numeric->unz;
            klu_free_numeric(&numeric, &common);
            klu_free_symbolic(&symbolic, &common);

            if (!ok) {
                throw std::runtime_error("klu_solve failed");
            }

            result.success = true;
            result.message = "ok lnz+unz=" + std::to_string(lnz + unz);
        } catch (const std::exception& error) {
            result.success = false;
            result.message = error.what();
            result.x.clear();
        }

        return result;
    }
};

class UmfpackSolver final : public LinearSolver {
public:
    std::string name() const override
    {
        return "umfpack";
    }

    SolverRun solve(
        const matrix::CsrMatrix&,
        const matrix::CscMatrix& csc,
        const std::vector<matrix::Value>& b) override
    {
        SolverRun result;

        try {
            utils::require_square(csc, name());
            utils::require_rhs_size(csc.rows, b.size());

            double control[UMFPACK_CONTROL];
            double info[UMFPACK_INFO];
            umfpack_di_defaults(control);

            void* symbolic = nullptr;
            timer::Stopwatch phase_timer;
            int status = umfpack_di_symbolic(
                csc.rows,
                csc.cols,
                const_cast<int*>(csc.col_ptr.data()),
                const_cast<int*>(csc.row_idx.data()),
                const_cast<double*>(csc.values.data()),
                &symbolic,
                control,
                info);
            result.analysis_ms = phase_timer.elapsed_ms();
            if (status != UMFPACK_OK) {
                throw std::runtime_error("umfpack_di_symbolic failed with status " + std::to_string(status));
            }

            void* numeric = nullptr;
            phase_timer.reset();
            status = umfpack_di_numeric(
                const_cast<int*>(csc.col_ptr.data()),
                const_cast<int*>(csc.row_idx.data()),
                const_cast<double*>(csc.values.data()),
                symbolic,
                &numeric,
                control,
                info);
            result.factor_ms = phase_timer.elapsed_ms();
            umfpack_di_free_symbolic(&symbolic);
            if (status != UMFPACK_OK) {
                throw std::runtime_error("umfpack_di_numeric failed with status " + std::to_string(status));
            }

            result.x.assign(static_cast<std::size_t>(csc.cols), 0.0);
            phase_timer.reset();
            status = umfpack_di_solve(
                UMFPACK_A,
                const_cast<int*>(csc.col_ptr.data()),
                const_cast<int*>(csc.row_idx.data()),
                const_cast<double*>(csc.values.data()),
                result.x.data(),
                const_cast<double*>(b.data()),
                numeric,
                control,
                info);
            result.solve_ms = phase_timer.elapsed_ms();
            umfpack_di_free_numeric(&numeric);

            if (status != UMFPACK_OK) {
                throw std::runtime_error("umfpack_di_solve failed with status " + std::to_string(status));
            }

            result.success = true;
            result.message = "ok";
        } catch (const std::exception& error) {
            result.success = false;
            result.message = error.what();
            result.x.clear();
        }

        return result;
    }
};

bool wants_solver(const std::vector<std::string>& requested_names, const std::string& name)
{
    for (const std::string& requested : requested_names) {
        if (requested == "all" || requested == name) {
            return true;
        }
    }
    return false;
}

}  // namespace

std::vector<std::unique_ptr<LinearSolver>> make_suitesparse_solvers(
    const std::vector<std::string>& requested_names)
{
    std::vector<std::unique_ptr<LinearSolver>> solvers;

    if (wants_solver(requested_names, "klu")) {
        solvers.push_back(std::make_unique<KluSolver>());
    }
    if (wants_solver(requested_names, "umfpack")) {
        solvers.push_back(std::make_unique<UmfpackSolver>());
    }

    return solvers;
}

}  // namespace sparse_direct::solver
