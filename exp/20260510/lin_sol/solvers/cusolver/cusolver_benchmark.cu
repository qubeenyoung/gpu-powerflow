#include "benchmark_common.hpp"

#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <iostream>
#include <type_traits>

namespace {

void cuda_check(cudaError_t status, const char* expr)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(status));
    }
}

void cusolver_check(cusolverStatus_t status, const char* expr)
{
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(expr) + ": cuSolver status " + std::to_string(static_cast<int>(status)));
    }
}

void cusparse_check(cusparseStatus_t status, const char* expr)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(expr) + ": cuSPARSE status " + std::to_string(static_cast<int>(status)));
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr)
#define CUSOLVER_CHECK(expr) cusolver_check((expr), #expr)
#define CUSPARSE_CHECK(expr) cusparse_check((expr), #expr)

template <typename T>
std::vector<T> convert(const std::vector<double>& values)
{
    std::vector<T> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<T>(values[i]);
    }
    return out;
}

float event_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

template <typename T>
cusolverStatus_t csrlsvqr(
    cusolverSpHandle_t handle,
    int n,
    int nnz,
    cusparseMatDescr_t descr,
    const T* val,
    const int* row,
    const int* col,
    const T* rhs,
    T tol,
    int reorder,
    T* x,
    int* singularity);

template <>
cusolverStatus_t csrlsvqr<float>(
    cusolverSpHandle_t handle,
    int n,
    int nnz,
    cusparseMatDescr_t descr,
    const float* val,
    const int* row,
    const int* col,
    const float* rhs,
    float tol,
    int reorder,
    float* x,
    int* singularity)
{
    return cusolverSpScsrlsvqr(handle, n, nnz, descr, val, row, col, rhs, tol, reorder, x, singularity);
}

template <>
cusolverStatus_t csrlsvqr<double>(
    cusolverSpHandle_t handle,
    int n,
    int nnz,
    cusparseMatDescr_t descr,
    const double* val,
    const int* row,
    const int* col,
    const double* rhs,
    double tol,
    int reorder,
    double* x,
    int* singularity)
{
    return cusolverSpDcsrlsvqr(handle, n, nnz, descr, val, row, col, rhs, tol, reorder, x, singularity);
}

template <typename T>
linbench::Result run_solver(const linbench::CliOptions& opt)
{
    const auto total_start = linbench::Clock::now();
    double format_convert_ms = 0.0;
    const auto load_start = linbench::Clock::now();
    linbench::CsrMatrix A = linbench::load_matrix_market_csr(opt.matrix, &format_convert_ms);
    std::vector<double> rhs_double = linbench::load_vector(opt.rhs);
    std::vector<double> x_ref = linbench::load_vector(opt.xref);
    linbench::Meta meta = linbench::load_meta(opt.meta);
    const auto load_stop = linbench::Clock::now();
    if (A.rows != A.cols || A.rows != static_cast<int>(rhs_double.size()) || A.cols != static_cast<int>(x_ref.size())) {
        throw std::runtime_error("matrix/vector dimensions are inconsistent");
    }

    std::vector<T> values = convert<T>(A.values);
    std::vector<T> rhs = convert<T>(rhs_double);
    std::vector<T> x_host(static_cast<std::size_t>(A.cols), T{});

    CUDA_CHECK(cudaFree(nullptr));
    std::size_t free_before = 0;
    std::size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_mem));

    int* d_row = nullptr;
    int* d_col = nullptr;
    T* d_val = nullptr;
    T* d_rhs = nullptr;
    T* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row, static_cast<std::size_t>(A.rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col, static_cast<std::size_t>(A.nnz) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, static_cast<std::size_t>(A.nnz) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_rhs, static_cast<std::size_t>(A.rows) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_x, static_cast<std::size_t>(A.cols) * sizeof(T)));

    cudaEvent_t e0 = nullptr;
    cudaEvent_t e1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_row, A.row_ptr.data(), static_cast<std::size_t>(A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, A.col_idx.data(), static_cast<std::size_t>(A.nnz) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, values.data(), static_cast<std::size_t>(A.nnz) * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs, rhs.data(), static_cast<std::size_t>(A.rows) * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, static_cast<std::size_t>(A.cols) * sizeof(T)));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    const double h2d_ms = event_ms(e0, e1);

    cusolverSpHandle_t handle = nullptr;
    cusparseMatDescr_t descr = nullptr;
    CUSOLVER_CHECK(cusolverSpCreate(&handle));
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    auto solve_once = [&]() -> std::pair<double, int> {
        int singularity = -1;
        CUDA_CHECK(cudaMemset(d_x, 0, static_cast<std::size_t>(A.cols) * sizeof(T)));
        CUDA_CHECK(cudaEventRecord(e0));
        CUSOLVER_CHECK(csrlsvqr<T>(
            handle,
            A.rows,
            A.nnz,
            descr,
            d_val,
            d_row,
            d_col,
            d_rhs,
            static_cast<T>(1e-12),
            1,
            d_x,
            &singularity));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        return {event_ms(e0, e1), singularity};
    };

    int last_singularity = -1;
    for (int i = 0; i < opt.warmup; ++i) {
        last_singularity = solve_once().second;
    }
    std::vector<double> solve_ms;
    solve_ms.reserve(static_cast<std::size_t>(opt.repeats));
    for (int i = 0; i < opt.repeats; ++i) {
        auto [ms, singularity] = solve_once();
        solve_ms.push_back(ms);
        last_singularity = singularity;
    }

    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(x_host.data(), d_x, static_cast<std::size_t>(A.cols) * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    const double d2h_ms = event_ms(e0, e1);

    std::size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_mem));
    const double used_mb = static_cast<double>(free_before > free_after ? free_before - free_after : 0) / (1024.0 * 1024.0);

    std::vector<double> x(x_host.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<double>(x_host[i]);
    }
    std::vector<double> r = linbench::residual(A, x, rhs_double);
    const double rel_res = linbench::norm2(r) / std::max(linbench::norm2(rhs_double), std::numeric_limits<double>::min());
    const double rel_err = linbench::relative_error(x, x_ref);
    linbench::Stats s_stats = linbench::make_stats(solve_ms);

    if (descr) CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));
    if (handle) CUSOLVER_CHECK(cusolverSpDestroy(handle));
    CUDA_CHECK(cudaFree(d_row));
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    const auto total_stop = linbench::Clock::now();
    linbench::Result result;
    result.solver_name = "cuSolverSP";
    result.solver_version = std::to_string(CUSOLVER_VER_MAJOR) + "." + std::to_string(CUSOLVER_VER_MINOR) + "." + std::to_string(CUSOLVER_VER_PATCH) + "." + std::to_string(CUSOLVER_VER_BUILD);
    result.library_path = CUSOLVER_LIBRARY_PATH_STR;
    result.dtype = opt.dtype;
    result.case_name = meta.case_name;
    result.iteration = meta.iteration;
    result.matrix_rows = A.rows;
    result.matrix_cols = A.cols;
    result.nnz = A.nnz;
    result.repeat_count = opt.repeats;
    result.warmup_count = opt.warmup;
    result.load_ms = linbench::elapsed_ms(load_start, load_stop);
    result.format_convert_ms = format_convert_ms;
    result.h2d_ms = h2d_ms;
    result.analysis_ms = 0.0;
    result.factorization_ms = 0.0;
    result.solve_ms = s_stats.mean;
    result.d2h_ms = d2h_ms;
    result.total_solver_ms = s_stats.mean;
    result.total_end_to_end_ms = linbench::elapsed_ms(total_start, total_stop);
    result.peak_gpu_memory_mb = used_mb;
    result.relative_residual_2 = rel_res;
    result.relative_error_to_x_ref_2 = rel_err;
    result.converged = rel_res < (std::is_same<T, double>::value ? 1e-8 : 1e-4) && last_singularity < 0;
    result.num_iterations = 1;
    result.gpu_resident_after_initial_load = "yes_for_QR_API";
    result.notes = "Uses cuSolverSP csrlsvqr for general sparse systems. The CUDA 12.8 LU and Cholesky sparse solve APIs are marked deprecated in headers with cuDSS as the replacement; QR is monolithic and does not expose separate analysis/factorization timings.";
    result.timing_stats["solve_ms"] = s_stats;
    result.timing_stats["total_solver_ms"] = s_stats;
    result.extra_numbers["singularity"] = static_cast<double>(last_singularity);
    result.extra_strings["cusolverRf_note"] = "cuSolverRF headers are available but require externally supplied LU factors; this wrapper benchmarks cuSolverSP QR because it accepts the raw general CSR Jacobian.";
    return result;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        linbench::CliOptions opt = linbench::parse_cli(argc, argv);
        linbench::Result result = opt.dtype == "fp64" ? run_solver<double>(opt) : run_solver<float>(opt);
        linbench::write_result_json(opt.out, result);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "cuSolverSP benchmark failed: " << exc.what() << "\n";
        return 2;
    }
}
