#include "third_party_solvers/mumps_solver.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <dmumps_c.h>

#include "third_party_solvers/utils.hpp"
#include "tools/timer.hpp"

namespace sparse_direct::solver {
namespace {

#define MUMPS_ICNTL(I) icntl[(I)-1]

constexpr MUMPS_INT kJobInit = -1;
constexpr MUMPS_INT kJobEnd = -2;
constexpr MUMPS_INT kUseCommWorld = -987654;

struct MumpsTripletMatrix {
    std::vector<MUMPS_INT> rows;
    std::vector<MUMPS_INT> cols;
    std::vector<double> values;
};

MumpsTripletMatrix make_one_based_triplets(const matrix::CsrMatrix& csr)
{
    MumpsTripletMatrix triplets;
    triplets.rows.reserve(csr.values.size());
    triplets.cols.reserve(csr.values.size());
    triplets.values.reserve(csr.values.size());

    for (matrix::Index row = 0; row < csr.rows; ++row) {
        for (matrix::Index pos = csr.row_ptr[row]; pos < csr.row_ptr[row + 1]; ++pos) {
            triplets.rows.push_back(static_cast<MUMPS_INT>(row + 1));
            triplets.cols.push_back(static_cast<MUMPS_INT>(csr.col_idx[pos] + 1));
            triplets.values.push_back(csr.values[pos]);
        }
    }

    return triplets;
}

void check_mumps_status(const DMUMPS_STRUC_C& mumps, const std::string& phase_name)
{
    if (mumps.infog[0] < 0) {
        throw std::runtime_error(
            "MUMPS " + phase_name +
            " failed: INFOG(1)=" + std::to_string(mumps.infog[0]) +
            " INFOG(2)=" + std::to_string(mumps.infog[1]));
    }
}

class MumpsSolver final : public LinearSolver {
public:
    MumpsSolver(std::string solver_name, bool use_gpu)
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
        DMUMPS_STRUC_C mumps;
        std::memset(&mumps, 0, sizeof(mumps));
        bool initialized_mumps = false;

        try {
            utils::require_square(csr, "MUMPS");
            utils::require_rhs_size(csr.rows, b.size());

            utils::ensure_mpi_initialized();

            MumpsTripletMatrix triplets = make_one_based_triplets(csr);
            result.x = b;

            mumps.comm_fortran = kUseCommWorld;
            mumps.par = 1;
            mumps.sym = 0;
            mumps.job = kJobInit;
            dmumps_c(&mumps);
            initialized_mumps = true;

            mumps.MUMPS_ICNTL(1) = -1;
            mumps.MUMPS_ICNTL(2) = -1;
            mumps.MUMPS_ICNTL(3) = -1;
            mumps.MUMPS_ICNTL(4) = 0;
            mumps.MUMPS_ICNTL(51) = use_gpu_ ? 1 : 0;
            if (use_gpu_) {
                // MUMPS 5.9 async GPU pinning can enqueue null-size pin requests
                // in this container; keep GPU offload, but use unpinned transfers.
                mumps.keep[421] = 0;  // KEEP(422): memory pinning mode.
            }

            mumps.n = static_cast<MUMPS_INT>(csr.rows);
            mumps.nz = static_cast<MUMPS_INT>(triplets.values.size());
            mumps.nnz = static_cast<MUMPS_INT8>(triplets.values.size());
            mumps.irn = triplets.rows.data();
            mumps.jcn = triplets.cols.data();
            mumps.a = triplets.values.data();
            mumps.rhs = result.x.data();
            mumps.nrhs = 1;
            mumps.lrhs = static_cast<MUMPS_INT>(csr.rows);

            timer::Stopwatch phase_timer;
            mumps.job = 1;
            dmumps_c(&mumps);
            result.analysis_ms = phase_timer.elapsed_ms();
            check_mumps_status(mumps, "analysis");

            phase_timer.reset();
            mumps.job = 2;
            dmumps_c(&mumps);
            result.factor_ms = phase_timer.elapsed_ms();
            check_mumps_status(mumps, "factorization");

            phase_timer.reset();
            mumps.job = 3;
            dmumps_c(&mumps);
            result.solve_ms = phase_timer.elapsed_ms();
            check_mumps_status(mumps, "solve");

            result.success = true;
            result.message = "ok";
        } catch (const std::exception& error) {
            result.success = false;
            result.message = error.what();
            result.x.clear();
        }

        if (initialized_mumps) {
            mumps.job = kJobEnd;
            dmumps_c(&mumps);
        }

        return result;
    }

private:
    std::string solver_name_;
    bool use_gpu_;
};

}  // namespace

std::unique_ptr<LinearSolver> make_mumps_cpu_solver()
{
    return std::make_unique<MumpsSolver>("mumps-cpu", false);
}

std::unique_ptr<LinearSolver> make_mumps_gpu_solver()
{
    return std::make_unique<MumpsSolver>("mumps-gpu", true);
}

}  // namespace sparse_direct::solver
