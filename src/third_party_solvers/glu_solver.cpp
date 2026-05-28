#include "third_party_solvers/glu_solver.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "third_party_solvers/utils.hpp"
#include "tools/timer.hpp"

#include "nicslu.h"
#include "numeric.h"
#include "symbolic.h"
#include "type.h"

extern "C" int my_DumpA(
    SNicsLU* nicslu,
    double** ax,
    unsigned int** ai,
    unsigned int** ap);

namespace sparse_direct::solver {
namespace {

int preprocess_from_csc(
    unsigned int n,
    unsigned int nnz,
    const double* ax_in,
    const unsigned int* ai_in,
    const unsigned int* ap_in,
    SNicsLU* nicslu,
    double** ax_out,
    unsigned int** ai_out,
    unsigned int** ap_out)
{
    *ax_out = nullptr;
    *ai_out = nullptr;
    *ap_out = nullptr;

    int ret = NicsLU_Initialize(nicslu);
    if (ret != NICS_OK) {
        return ret;
    }

    double* ax_copy = static_cast<double*>(std::malloc(sizeof(double) * nnz));
    uint__t* ai_copy = static_cast<uint__t*>(std::malloc(sizeof(uint__t) * nnz));
    uint__t* ap_copy = static_cast<uint__t*>(std::malloc(sizeof(uint__t) * (n + 1)));
    if (!ax_copy || !ai_copy || !ap_copy) {
        std::free(ax_copy);
        std::free(ai_copy);
        std::free(ap_copy);
        return -1;
    }

    std::memcpy(ax_copy, ax_in, sizeof(double) * nnz);
    for (unsigned int i = 0; i < nnz; ++i) {
        ai_copy[i] = static_cast<uint__t>(ai_in[i]);
    }
    for (unsigned int i = 0; i <= n; ++i) {
        ap_copy[i] = static_cast<uint__t>(ap_in[i]);
    }

    ret = NicsLU_CreateMatrix(nicslu, static_cast<uint__t>(n), static_cast<uint__t>(nnz), ax_copy, ai_copy, ap_copy);
    if (ret != NICS_OK) {
        std::free(ax_copy);
        std::free(ai_copy);
        std::free(ap_copy);
        return ret;
    }

    nicslu->cfgi[0] = 1;
    nicslu->cfgf[1] = 0;

    ret = NicsLU_Analyze(nicslu);
    if (ret != NICS_OK) {
        return ret;
    }

    ret = my_DumpA(nicslu, ax_out, ai_out, ap_out);
    if (ret != 0) {
        return ret;
    }

    return 0;
}

class Glu3Solver final : public LinearSolver {
public:
    std::string name() const override
    {
        return "glu3-gpu";
    }

    SolverRun solve(
        const matrix::CsrMatrix&,
        const matrix::CscMatrix& csc,
        const std::vector<matrix::Value>& b) override
    {
        SolverRun result;
        SNicsLU* nicslu = nullptr;
        double* ax_out = nullptr;
        unsigned int* ai_out = nullptr;
        unsigned int* ap_out = nullptr;

        try {
            utils::require_square(csc, "GLU3");
            utils::require_rhs_size(csc.rows, b.size());
            if (csc.rows <= 0 || csc.nnz() <= 0) {
                throw std::runtime_error("GLU3 requires a non-empty sparse matrix");
            }

            utils::require_cuda_device();

            std::vector<unsigned int> row_idx = utils::to_unsigned_indices(csc.row_idx);
            std::vector<unsigned int> col_ptr = utils::to_unsigned_indices(csc.col_ptr);

            nicslu = static_cast<SNicsLU*>(std::malloc(sizeof(SNicsLU)));
            if (nicslu == nullptr) {
                throw std::bad_alloc();
            }

            const int preprocess_status = preprocess_from_csc(
                static_cast<unsigned int>(csc.cols),
                static_cast<unsigned int>(csc.nnz()),
                csc.values.data(),
                row_idx.data(),
                col_ptr.data(),
                nicslu,
                &ax_out,
                &ai_out,
                &ap_out);
            if (preprocess_status != 0) {
                throw std::runtime_error("GLU3 preprocessing failed with status " + std::to_string(preprocess_status));
            }

            std::ostringstream out_stream;
            std::ostringstream err_stream;
            timer::Stopwatch phase_timer;
            Symbolic_Matrix symbolic(nicslu->n, out_stream, err_stream);
            symbolic.fill_in(ai_out, ap_out);
            symbolic.csr();
            symbolic.predictLU(ai_out, ap_out, ax_out);
            symbolic.leveling();
            result.analysis_ms = phase_timer.elapsed_ms();

            std::free(ax_out);
            std::free(ai_out);
            std::free(ap_out);
            ax_out = nullptr;
            ai_out = nullptr;
            ap_out = nullptr;

            phase_timer.reset();
            LUonDevice(symbolic, out_stream, err_stream, true);
            utils::synchronize_cuda("cudaDeviceSynchronize GLU3 factorization");
            result.factor_ms = phase_timer.elapsed_ms();

            const std::string factor_stderr = err_stream.str();
            if (!factor_stderr.empty()) {
                bool nonfinite_factor = false;
                for (REAL value : symbolic.val) {
                    if (!std::isfinite(static_cast<double>(value))) {
                        nonfinite_factor = true;
                        break;
                    }
                }
                if (nonfinite_factor) {
                    throw std::runtime_error("GLU3 factorization produced non-finite factors");
                }
            }

            std::vector<REAL> rhs(static_cast<std::size_t>(csc.rows));
            for (std::size_t i = 0; i < b.size(); ++i) {
                rhs[i] = static_cast<REAL>(b[i]);
            }

            phase_timer.reset();
            std::vector<REAL> x_float = symbolic.solve(nicslu, rhs);
            utils::synchronize_cuda("cudaDeviceSynchronize GLU3 solve");
            result.solve_ms = phase_timer.elapsed_ms();

            result.x.resize(x_float.size());
            for (std::size_t i = 0; i < x_float.size(); ++i) {
                result.x[i] = static_cast<double>(x_float[i]);
            }

            result.success = true;
            result.message = "ok";
        } catch (const std::exception& error) {
            result.success = false;
            result.message = error.what();
            result.x.clear();
        }

        std::free(ax_out);
        std::free(ai_out);
        std::free(ap_out);
        if (nicslu != nullptr) {
            NicsLU_Destroy(nicslu);
            std::free(nicslu);
        }

        return result;
    }
};

}  // namespace

std::unique_ptr<LinearSolver> make_glu3_gpu_solver()
{
    return std::make_unique<Glu3Solver>();
}

}  // namespace sparse_direct::solver
