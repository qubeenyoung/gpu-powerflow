#include "third_party_solvers/strumpack_solver.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <StrumpackSparseSolver.h>

#include "third_party_solvers/utils.hpp"
#include "tools/timer.hpp"

namespace sparse_direct::solver {
namespace {

void check_strumpack_status(STRUMPACK_RETURN_CODE code, const std::string& phase_name)
{
    if (code != STRUMPACK_SUCCESS) {
        throw std::runtime_error(
            "STRUMPACK " + phase_name + " failed with code " + std::to_string(static_cast<int>(code)));
    }
}

class StrumpackSolver final : public LinearSolver {
public:
    StrumpackSolver(std::string solver_name, bool use_gpu)
        : solver_name_(std::move(solver_name)),
          use_gpu_(use_gpu)
    {
    }

    std::string name() const override
    {
        return solver_name_;
    }

    SolverRun solve(
        const matrix::CsrMatrix& csr,
        const matrix::CscMatrix&,
        const std::vector<matrix::Value>& b) override
    {
        SolverRun result;
        STRUMPACK_SparseSolver solver;
        bool initialized = false;

        try {
            utils::require_square(csr, "STRUMPACK");
            utils::require_rhs_size(csr.rows, b.size());

            int argc = 1;
            char program_name[] = "benchmark";
            char* argv[] = {program_name, nullptr};

            STRUMPACK_init_mt(&solver, STRUMPACK_DOUBLE, STRUMPACK_MT, argc, argv, 0);
            initialized = true;
            STRUMPACK_set_verbose(solver, 0);
            STRUMPACK_set_compression(solver, STRUMPACK_NONE);
            STRUMPACK_set_Krylov_solver(solver, STRUMPACK_DIRECT);
            STRUMPACK_set_reordering_method(solver, STRUMPACK_METIS);
            STRUMPACK_enable_METIS_NodeNDP(solver);
            STRUMPACK_set_matching(solver, STRUMPACK_MATCHING_MAX_DIAGONAL_PRODUCT_SCALING);
            if (use_gpu_) {
                STRUMPACK_enable_gpu(solver);
            } else {
                STRUMPACK_disable_gpu(solver);
            }

            int matrix_size = csr.rows;
            std::vector<int> row_ptr(csr.row_ptr.begin(), csr.row_ptr.end());
            std::vector<int> col_idx(csr.col_idx.begin(), csr.col_idx.end());
            std::vector<double> values = csr.values;

            STRUMPACK_set_csr_matrix(solver, &matrix_size, row_ptr.data(), col_idx.data(), values.data(), 0);

            timer::Stopwatch phase_timer;
            check_strumpack_status(STRUMPACK_reorder(solver), "reorder");
            result.analysis_ms = phase_timer.elapsed_ms();

            phase_timer.reset();
            check_strumpack_status(STRUMPACK_factor(solver), "factorization");
            result.factor_ms = phase_timer.elapsed_ms();

            result.x.assign(static_cast<std::size_t>(csr.cols), 0.0);
            phase_timer.reset();
            check_strumpack_status(STRUMPACK_solve(solver, b.data(), result.x.data(), 0), "solve");
            result.solve_ms = phase_timer.elapsed_ms();

            result.success = true;
            result.message = "ok";
        } catch (const std::exception& error) {
            result.success = false;
            result.message = error.what();
            result.x.clear();
        }

        if (initialized) {
            STRUMPACK_destroy(&solver);
        }

        return result;
    }

private:
    std::string solver_name_;
    bool use_gpu_;
};

}  // namespace

std::unique_ptr<LinearSolver> make_strumpack_cpu_solver()
{
    return std::make_unique<StrumpackSolver>("strumpack-cpu", false);
}

std::unique_ptr<LinearSolver> make_strumpack_gpu_solver()
{
    return std::make_unique<StrumpackSolver>("strumpack-gpu", true);
}

}  // namespace sparse_direct::solver
