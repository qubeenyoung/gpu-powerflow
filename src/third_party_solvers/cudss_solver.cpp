#include "third_party_solvers/cudss_solver.hpp"

#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <cudss.h>

#include "third_party_solvers/utils.hpp"
#include "tools/timer.hpp"

namespace sparse_direct::solver {
namespace {

void check_cudss(cudssStatus_t status, const std::string& action)
{
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(action + " failed with cuDSS status " + std::to_string(static_cast<int>(status)));
    }
}

template <typename T>
T* copy_vector_to_device(const std::vector<T>& host_values)
{
    T* device_values = nullptr;
    const std::size_t bytes = host_values.size() * sizeof(T);
    utils::check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_values), bytes), "cudaMalloc");
    utils::check_cuda(cudaMemcpy(device_values, host_values.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return device_values;
}

const char* cudss_threading_layer()
{
    const char* env_value = std::getenv("CUDSS_THREADING_LIB");
    if (env_value != nullptr && env_value[0] != '\0') {
        return env_value;
    }
    return "/opt/nvidia/cudss/lib/libcudss_mtlayer_gomp.so.0";
}

class CudssSolver final : public LinearSolver {
public:
    std::string name() const override
    {
        return "cudss-gpu";
    }

    SolverRun solve(
        const matrix::CsrMatrix& csr,
        const matrix::CscMatrix&,
        const std::vector<matrix::Value>& b) override
    {
        SolverRun result;

        int64_t* d_row_offsets = nullptr;
        int64_t* d_col_idx = nullptr;
        double* d_values = nullptr;
        double* d_rhs = nullptr;
        double* d_x = nullptr;

        cudssHandle_t handle = nullptr;
        cudssConfig_t config = nullptr;
        cudssData_t data = nullptr;
        cudssMatrix_t matrix = nullptr;
        cudssMatrix_t rhs = nullptr;
        cudssMatrix_t solution = nullptr;

        try {
            utils::require_square(csr, "cuDSS");
            utils::require_rhs_size(csr.rows, b.size());
            utils::require_cuda_device();

            std::vector<int64_t> row_offsets(csr.row_ptr.size());
            for (std::size_t i = 0; i < csr.row_ptr.size(); ++i) {
                row_offsets[i] = csr.row_ptr[i];
            }
            std::vector<int64_t> col_idx(csr.col_idx.size());
            for (std::size_t i = 0; i < csr.col_idx.size(); ++i) {
                col_idx[i] = csr.col_idx[i];
            }

            d_row_offsets = copy_vector_to_device(row_offsets);
            d_col_idx = copy_vector_to_device(col_idx);
            d_values = copy_vector_to_device(csr.values);
            d_rhs = copy_vector_to_device(b);
            utils::check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_x), b.size() * sizeof(double)), "cudaMalloc solution");
            utils::check_cuda(cudaMemset(d_x, 0, b.size() * sizeof(double)), "cudaMemset solution");

            check_cudss(cudssCreate(&handle), "cudssCreate");
            check_cudss(cudssSetThreadingLayer(handle, cudss_threading_layer()), "cudssSetThreadingLayer");
            check_cudss(cudssConfigCreate(&config), "cudssConfigCreate");
            int use_matching = 1;
            check_cudss(
                cudssConfigSet(
                    config,
                    CUDSS_CONFIG_USE_MATCHING,
                    &use_matching,
                    sizeof(use_matching)),
                "cudssConfigSet USE_MATCHING");
            int host_nthreads = utils::positive_env_or(
                "CUDSS_HOST_NTHREADS",
                static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
            check_cudss(
                cudssConfigSet(
                    config,
                    CUDSS_CONFIG_HOST_NTHREADS,
                    &host_nthreads,
                    sizeof(host_nthreads)),
                "cudssConfigSet HOST_NTHREADS");
            check_cudss(cudssDataCreate(handle, &data), "cudssDataCreate");

            check_cudss(
                cudssMatrixCreateCsr(
                    &matrix,
                    csr.rows,
                    csr.cols,
                    csr.nnz(),
                    d_row_offsets,
                    nullptr,
                    d_col_idx,
                    d_values,
                    CUDA_R_64I,
                    CUDA_R_64F,
                    CUDSS_MTYPE_GENERAL,
                    CUDSS_MVIEW_FULL,
                    CUDSS_BASE_ZERO),
                "cudssMatrixCreateCsr");

            check_cudss(
                cudssMatrixCreateDn(
                    &rhs,
                    csr.rows,
                    1,
                    csr.rows,
                    d_rhs,
                    CUDA_R_64F,
                    CUDSS_LAYOUT_COL_MAJOR),
                "cudssMatrixCreateDn rhs");

            check_cudss(
                cudssMatrixCreateDn(
                    &solution,
                    csr.cols,
                    1,
                    csr.cols,
                    d_x,
                    CUDA_R_64F,
                    CUDSS_LAYOUT_COL_MAJOR),
                "cudssMatrixCreateDn solution");
            utils::synchronize_cuda("cudaDeviceSynchronize setup");

            timer::Stopwatch phase_timer;
            check_cudss(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, matrix, solution, rhs), "cuDSS analysis");
            utils::synchronize_cuda("cudaDeviceSynchronize analysis");
            result.analysis_ms = phase_timer.elapsed_ms();

            // cy340: CUDSS_REPEAT=N times the FACTORIZATION and SOLVE phases warm (reusing the same
            // handle/symbolic, like an NR re-factorization loop) and reports the MEDIAN -- matching
            // our gpu_mf_bench warm 7-median methodology for a fair warm-vs-warm comparison. Default
            // N=1 = the original single (cold) call.
            const char* rep_s = std::getenv("CUDSS_REPEAT");
            const int reps = (rep_s && std::atoi(rep_s) > 0) ? std::atoi(rep_s) : 1;
            auto median_of = [](std::vector<double>& v) {
                std::sort(v.begin(), v.end());
                return v[v.size() / 2];
            };
            std::vector<double> fts, sts;
            for (int r = 0; r < reps; ++r) {
                phase_timer.reset();
                check_cudss(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, matrix, solution, rhs), "cuDSS factorization");
                utils::synchronize_cuda("cudaDeviceSynchronize factorization");
                fts.push_back(phase_timer.elapsed_ms());
            }
            result.factor_ms = median_of(fts);

            for (int r = 0; r < reps; ++r) {
                phase_timer.reset();
                check_cudss(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, matrix, solution, rhs), "cuDSS solve");
                utils::synchronize_cuda("cudaDeviceSynchronize solve");
                sts.push_back(phase_timer.elapsed_ms());
            }
            result.solve_ms = median_of(sts);

            result.x.assign(static_cast<std::size_t>(csr.cols), 0.0);
            utils::check_cuda(cudaMemcpy(result.x.data(), d_x, result.x.size() * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

            result.success = true;
            result.message = "ok";
        } catch (const std::exception& error) {
            result.success = false;
            result.message = error.what();
            result.x.clear();
        }

        if (solution) cudssMatrixDestroy(solution);
        if (rhs) cudssMatrixDestroy(rhs);
        if (matrix) cudssMatrixDestroy(matrix);
        if (data) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);

        if (d_x) cudaFree(d_x);
        if (d_rhs) cudaFree(d_rhs);
        if (d_values) cudaFree(d_values);
        if (d_col_idx) cudaFree(d_col_idx);
        if (d_row_offsets) cudaFree(d_row_offsets);

        return result;
    }
};

}  // namespace

std::unique_ptr<LinearSolver> make_cudss_gpu_solver()
{
    return std::make_unique<CudssSolver>();
}

}  // namespace sparse_direct::solver
