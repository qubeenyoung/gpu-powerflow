#include "third_party_solvers/pastix_solver.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pastix.h>

#include "third_party_solvers/utils.hpp"
#include "tools/timer.hpp"

namespace sparse_direct::solver {
namespace {

void check_pastix_status(int status, const std::string& phase_name)
{
    if (status != 0) {
        throw std::runtime_error(
            "PaStiX " + phase_name + " failed with status " + std::to_string(status));
    }
}

struct PastixCscStorage {
    std::vector<spm_int_t> col_ptr;
    std::vector<spm_int_t> row_idx;
    std::vector<double> values;
};

PastixCscStorage copy_csc_for_pastix(const matrix::CscMatrix& csc)
{
    PastixCscStorage storage;
    storage.col_ptr.reserve(csc.col_ptr.size());
    storage.row_idx.reserve(csc.row_idx.size());
    storage.values = csc.values;

    for (matrix::Index offset : csc.col_ptr) {
        storage.col_ptr.push_back(static_cast<spm_int_t>(offset));
    }
    for (matrix::Index row : csc.row_idx) {
        storage.row_idx.push_back(static_cast<spm_int_t>(row));
    }

    return storage;
}

spmatrix_t make_pastix_spm(const matrix::CscMatrix& csc, PastixCscStorage& storage)
{
    spmatrix_t spm;
    spmInit(&spm);

    spm.mtxtype = SpmGeneral;
    spm.flttype = SpmDouble;
    spm.fmttype = SpmCSC;
    spm.baseval = 0;
    spm.n = static_cast<spm_int_t>(csc.cols);
    spm.nnz = static_cast<spm_int_t>(csc.nnz());
    spm.dof = 1;
    spm.layout = SpmColMajor;
    spm.replicated = 1;

    spm.colptr = storage.col_ptr.data();
    spm.rowptr = storage.row_idx.data();
    spm.values = storage.values.data();

    spmUpdateComputedFields(&spm);
    return spm;
}

class PastixSolver final : public LinearSolver {
public:
    PastixSolver(std::string solver_name, bool use_gpu)
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
        const matrix::CscMatrix& csc,
        const std::vector<matrix::Value>& b) override
    {
        SolverRun result;
        pastix_data_t* pastix_data = nullptr;
        spmatrix_t corrected_spm;
        bool corrected_spm_created = false;
        spmInit(&corrected_spm);

        try {
            utils::require_square(csr, "PaStiX");
            utils::require_square(csc, "PaStiX");
            utils::require_rhs_size(csr.rows, b.size());
            utils::ensure_mpi_initialized();

            std::vector<pastix_int_t> iparm(IPARM_SIZE);
            std::vector<double> dparm(DPARM_SIZE);
            pastixInitParam(iparm.data(), dparm.data());

            iparm[IPARM_VERBOSE] = PastixVerboseNot;
            iparm[IPARM_ORDERING] = PastixOrderMetis;
            iparm[IPARM_ORDERING_DEFAULT] = 1;
            iparm[IPARM_FACTORIZATION] = PastixFactLU;
            iparm[IPARM_SCHEDULER] = use_gpu_ ? PastixSchedStarPU : PastixSchedDynamic;
            iparm[IPARM_GPU_NBR] = use_gpu_ ? 1 : 0;
            iparm[IPARM_THREAD_NBR] = -1;
            dparm[DPARM_EPSILON_MAGN_CTRL] =
                utils::finite_env_or("PASTIX_EPSILON_MAGN_CTRL", 0.0);

            pastixInit(&pastix_data, static_cast<PASTIX_Comm>(MPI_COMM_WORLD), iparm.data(), dparm.data());
            if (pastix_data == nullptr) {
                throw std::runtime_error("PaStiX initialization failed");
            }

            PastixCscStorage storage = copy_csc_for_pastix(csc);
            spmatrix_t spm = make_pastix_spm(csc, storage);
            spmatrix_t* active_spm = &spm;
            if (spmCheckAndCorrect(&spm, &corrected_spm) != 0) {
                active_spm = &corrected_spm;
                corrected_spm_created = true;
            }
            result.x = b;

            timer::Stopwatch phase_timer;
            check_pastix_status(pastix_task_analyze(pastix_data, active_spm), "analysis");
            result.analysis_ms = phase_timer.elapsed_ms();

            phase_timer.reset();
            check_pastix_status(pastix_task_numfact(pastix_data, active_spm), "factorization");
            result.factor_ms = phase_timer.elapsed_ms();

            phase_timer.reset();
            check_pastix_status(
                pastix_task_solve(pastix_data, active_spm->nexp, 1, result.x.data(), active_spm->nexp),
                "solve");
            result.solve_ms = phase_timer.elapsed_ms();

            result.success = true;
            result.message = "ok";
        } catch (const std::exception& error) {
            result.success = false;
            result.message = error.what();
            result.x.clear();
        }

        if (pastix_data != nullptr) {
            pastixFinalize(&pastix_data);
        }
        if (corrected_spm_created) {
            spmExit(&corrected_spm);
        }

        return result;
    }

private:
    std::string solver_name_;
    bool use_gpu_;
};

}  // namespace

std::unique_ptr<LinearSolver> make_pastix_cpu_solver()
{
    return std::make_unique<PastixSolver>("pastix-cpu", false);
}

std::unique_ptr<LinearSolver> make_pastix_gpu_solver()
{
    return std::make_unique<PastixSolver>("pastix-gpu", true);
}

}  // namespace sparse_direct::solver
