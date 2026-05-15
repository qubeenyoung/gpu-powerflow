#include "benchmark_common.hpp"

#include <cuda_runtime.h>
#include <cudss.h>

#include <iostream>
#include <type_traits>

namespace {

void cuda_check(cudaError_t status, const char* expr)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(status));
    }
}

void cudss_check(cudssStatus_t status, const char* expr)
{
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(expr) + ": cuDSS status " + std::to_string(static_cast<int>(status)));
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr)
#define CUDSS_CHECK(expr) cudss_check((expr), #expr)

template <typename T>
cudaDataType_t cuda_value_type();

template <>
cudaDataType_t cuda_value_type<double>() { return CUDA_R_64F; }

template <>
cudaDataType_t cuda_value_type<float>() { return CUDA_R_32F; }

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
    T* d_rhs_original = nullptr;
    T* d_rhs = nullptr;
    T* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row, static_cast<std::size_t>(A.rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col, static_cast<std::size_t>(A.nnz) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, static_cast<std::size_t>(A.nnz) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_rhs_original, static_cast<std::size_t>(A.rows) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_rhs, static_cast<std::size_t>(A.rows) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_x, static_cast<std::size_t>(A.cols) * sizeof(T)));

    cudaEvent_t e0 = nullptr;
    cudaEvent_t e1 = nullptr;
    cudaEvent_t e2 = nullptr;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));

    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_row, A.row_ptr.data(), static_cast<std::size_t>(A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, A.col_idx.data(), static_cast<std::size_t>(A.nnz) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, values.data(), static_cast<std::size_t>(A.nnz) * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs_original, rhs.data(), static_cast<std::size_t>(A.rows) * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs, d_rhs_original, static_cast<std::size_t>(A.rows) * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, static_cast<std::size_t>(A.cols) * sizeof(T)));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    const double h2d_ms = event_ms(e0, e1);

    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matrix = nullptr;
    cudssMatrix_t rhs_matrix = nullptr;
    cudssMatrix_t x_matrix = nullptr;

    CUDSS_CHECK(cudssCreate(&handle));
    CUDSS_CHECK(cudssConfigCreate(&config));
    CUDSS_CHECK(cudssDataCreate(handle, &data));
    CUDSS_CHECK(cudssMatrixCreateCsr(
        &matrix,
        A.rows,
        A.cols,
        A.nnz,
        d_row,
        nullptr,
        d_col,
        d_val,
        CUDA_R_32I,
        cuda_value_type<T>(),
        CUDSS_MTYPE_GENERAL,
        CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&rhs_matrix, A.rows, 1, A.rows, d_rhs, cuda_value_type<T>(), CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&x_matrix, A.cols, 1, A.cols, d_x, cuda_value_type<T>(), CUDSS_LAYOUT_COL_MAJOR));

    CUDA_CHECK(cudaEventRecord(e0));
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, matrix, x_matrix, rhs_matrix));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    const double analysis_ms = event_ms(e0, e1);

    auto factor_and_solve = [&]() -> std::pair<double, double> {
        CUDA_CHECK(cudaMemcpy(d_rhs, d_rhs_original, static_cast<std::size_t>(A.rows) * sizeof(T), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_x, 0, static_cast<std::size_t>(A.cols) * sizeof(T)));
        CUDA_CHECK(cudaEventRecord(e0));
        CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, matrix, x_matrix, rhs_matrix));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, matrix, x_matrix, rhs_matrix));
        CUDA_CHECK(cudaEventRecord(e2));
        CUDA_CHECK(cudaEventSynchronize(e2));
        return {event_ms(e0, e1), event_ms(e1, e2)};
    };

    for (int i = 0; i < opt.warmup; ++i) {
        (void)factor_and_solve();
    }
    std::vector<double> factor_ms;
    std::vector<double> solve_ms;
    factor_ms.reserve(static_cast<std::size_t>(opt.repeats));
    solve_ms.reserve(static_cast<std::size_t>(opt.repeats));
    for (int i = 0; i < opt.repeats; ++i) {
        auto [f_ms, s_ms] = factor_and_solve();
        factor_ms.push_back(f_ms);
        solve_ms.push_back(s_ms);
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
    linbench::Stats f_stats = linbench::make_stats(factor_ms);
    linbench::Stats s_stats = linbench::make_stats(solve_ms);
    std::vector<double> total_solver_samples;
    for (std::size_t i = 0; i < factor_ms.size(); ++i) {
        total_solver_samples.push_back(factor_ms[i] + solve_ms[i]);
    }
    linbench::Stats total_stats = linbench::make_stats(total_solver_samples);

    if (matrix) CUDSS_CHECK(cudssMatrixDestroy(matrix));
    if (rhs_matrix) CUDSS_CHECK(cudssMatrixDestroy(rhs_matrix));
    if (x_matrix) CUDSS_CHECK(cudssMatrixDestroy(x_matrix));
    if (data) CUDSS_CHECK(cudssDataDestroy(handle, data));
    if (config) CUDSS_CHECK(cudssConfigDestroy(config));
    if (handle) CUDSS_CHECK(cudssDestroy(handle));
    CUDA_CHECK(cudaFree(d_row));
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_rhs_original));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2));

    const auto total_stop = linbench::Clock::now();
    linbench::Result result;
    result.solver_name = "cuDSS";
    result.solver_version = std::to_string(CUDSS_VERSION_MAJOR) + "." + std::to_string(CUDSS_VERSION_MINOR) + "." + std::to_string(CUDSS_VERSION_PATCH);
    result.library_path = CUDSS_LIBRARY_PATH_STR;
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
    result.analysis_ms = analysis_ms;
    result.factorization_ms = f_stats.mean;
    result.solve_ms = s_stats.mean;
    result.d2h_ms = d2h_ms;
    result.total_solver_ms = analysis_ms + total_stats.mean;
    result.total_end_to_end_ms = linbench::elapsed_ms(total_start, total_stop);
    result.peak_gpu_memory_mb = used_mb;
    result.relative_residual_2 = rel_res;
    result.relative_error_to_x_ref_2 = rel_err;
    result.converged = rel_res < (std::is_same<T, double>::value ? 1e-8 : 1e-4);
    result.num_iterations = 1;
    result.gpu_resident_after_initial_load = "yes";
    result.notes = "cuDSS sparse direct LU path; analysis is run once and reused across repeated numeric factorization/solve calls for the same sparsity pattern.";
    result.timing_stats["factorization_ms"] = f_stats;
    result.timing_stats["solve_ms"] = s_stats;
    result.timing_stats["total_solver_ms"] = total_stats;
    result.extra_strings["symbolic_analysis_reuse"] = "yes_same_pattern";
    result.extra_numbers["numeric_factor_plus_solve_ms"] = total_stats.mean;
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
        std::cerr << "cuDSS benchmark failed: " << exc.what() << "\n";
        return 2;
    }
}
