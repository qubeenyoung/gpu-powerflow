#include "third_party_solvers/pardiso_solver.hpp"

#include <stdexcept>
#include <vector>

#include <mkl.h>

#include "third_party_solvers/utils.hpp"
#include "tools/timer.hpp"

namespace sparse_direct::solver {
namespace {

std::vector<MKL_INT> make_one_based_ptr(const std::vector<matrix::Index>& zero_based_ptr)
{
    std::vector<MKL_INT> one_based(zero_based_ptr.size());
    for (std::size_t i = 0; i < zero_based_ptr.size(); ++i) {
        one_based[i] = static_cast<MKL_INT>(zero_based_ptr[i] + 1);
    }
    return one_based;
}

std::vector<MKL_INT> make_one_based_indices(const std::vector<matrix::Index>& zero_based_indices)
{
    std::vector<MKL_INT> one_based(zero_based_indices.size());
    for (std::size_t i = 0; i < zero_based_indices.size(); ++i) {
        one_based[i] = static_cast<MKL_INT>(zero_based_indices[i] + 1);
    }
    return one_based;
}

class PardisoSolver final : public LinearSolver {
public:
    std::string name() const override
    {
        return "pardiso";
    }

    SolverRun solve(
        const matrix::CsrMatrix& csr,
        const matrix::CscMatrix&,
        const std::vector<matrix::Value>& b) override
    {
        SolverRun result;

        try {
            utils::require_square(csr, "PARDISO");
            utils::require_rhs_size(csr.rows, b.size());

            void* pt[64] = {};
            MKL_INT iparm[64] = {};
            MKL_INT maxfct = 1;
            MKL_INT mnum = 1;
            MKL_INT mtype = 11;  // real unsymmetric matrix
            MKL_INT n = static_cast<MKL_INT>(csr.rows);
            MKL_INT nrhs = 1;
            MKL_INT msglvl = 0;
            MKL_INT error = 0;

            pardisoinit(pt, &mtype, iparm);
            iparm[1] = 2;   // METIS fill-reducing ordering.
            iparm[9] = 13;  // Default nonsymmetric pivot perturbation.
            iparm[10] = 1;  // Enable nonsymmetric scaling.
            iparm[12] = 1;  // Enable weighted matching.
            iparm[34] = 0;  // one-based CSR indices

            std::vector<MKL_INT> row_ptr = make_one_based_ptr(csr.row_ptr);
            std::vector<MKL_INT> col_idx = make_one_based_indices(csr.col_idx);
            std::vector<double> values = csr.values;
            std::vector<double> rhs = b;
            result.x.assign(static_cast<std::size_t>(csr.cols), 0.0);
            std::vector<MKL_INT> perm(static_cast<std::size_t>(csr.rows), 0);

            MKL_INT phase = 11;
            timer::Stopwatch phase_timer;
            pardiso(
                pt,
                &maxfct,
                &mnum,
                &mtype,
                &phase,
                &n,
                values.data(),
                row_ptr.data(),
                col_idx.data(),
                perm.data(),
                &nrhs,
                iparm,
                &msglvl,
                rhs.data(),
                result.x.data(),
                &error);
            result.analysis_ms = phase_timer.elapsed_ms();
            if (error != 0) {
                throw std::runtime_error("PARDISO analysis failed with error " + std::to_string(error));
            }

            phase = 22;
            phase_timer.reset();
            pardiso(
                pt,
                &maxfct,
                &mnum,
                &mtype,
                &phase,
                &n,
                values.data(),
                row_ptr.data(),
                col_idx.data(),
                perm.data(),
                &nrhs,
                iparm,
                &msglvl,
                rhs.data(),
                result.x.data(),
                &error);
            result.factor_ms = phase_timer.elapsed_ms();
            if (error != 0) {
                throw std::runtime_error("PARDISO factorization failed with error " + std::to_string(error));
            }

            phase = 33;
            phase_timer.reset();
            pardiso(
                pt,
                &maxfct,
                &mnum,
                &mtype,
                &phase,
                &n,
                values.data(),
                row_ptr.data(),
                col_idx.data(),
                perm.data(),
                &nrhs,
                iparm,
                &msglvl,
                rhs.data(),
                result.x.data(),
                &error);
            result.solve_ms = phase_timer.elapsed_ms();
            if (error != 0) {
                throw std::runtime_error("PARDISO solve failed with error " + std::to_string(error));
            }

            phase = -1;
            pardiso(
                pt,
                &maxfct,
                &mnum,
                &mtype,
                &phase,
                &n,
                values.data(),
                row_ptr.data(),
                col_idx.data(),
                perm.data(),
                &nrhs,
                iparm,
                &msglvl,
                rhs.data(),
                result.x.data(),
                &error);

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

}  // namespace

std::unique_ptr<LinearSolver> make_pardiso_solver()
{
    return std::make_unique<PardisoSolver>();
}

}  // namespace sparse_direct::solver
