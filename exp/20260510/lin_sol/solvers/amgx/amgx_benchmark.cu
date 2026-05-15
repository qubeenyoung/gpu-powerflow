#include "benchmark_common.hpp"

#include <amgx_c.h>
#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>

namespace {

void cuda_check(cudaError_t status, const char* expr)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(status));
    }
}

void amgx_check(AMGX_RC status, const char* expr)
{
    if (status != AMGX_RC_OK) {
        char msg[4096] = {};
        AMGX_get_error_string(status, msg, sizeof(msg));
        throw std::runtime_error(std::string(expr) + ": " + msg);
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr)
#define AMGX_CHECK(expr) amgx_check((expr), #expr)

template <typename T>
std::vector<T> convert(const std::vector<double>& values)
{
    std::vector<T> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<T>(values[i]);
    }
    return out;
}

template <typename T>
AMGX_Mode amgx_mode();

template <>
AMGX_Mode amgx_mode<double>() { return AMGX_mode_dDDI; }

template <>
AMGX_Mode amgx_mode<float>() { return AMGX_mode_dFFI; }

std::string status_name(AMGX_SOLVE_STATUS status)
{
    switch (status) {
        case AMGX_SOLVE_SUCCESS: return "success";
        case AMGX_SOLVE_FAILED: return "failed";
        case AMGX_SOLVE_DIVERGED: return "diverged";
        case AMGX_SOLVE_NOT_CONVERGED: return "not_converged";
        default: return "unknown";
    }
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
    if (opt.config.empty()) {
        throw std::runtime_error("AMGx wrapper requires --config path");
    }

    std::vector<T> values = convert<T>(A.values);
    std::vector<T> rhs = convert<T>(rhs_double);
    std::vector<T> x_host(static_cast<std::size_t>(A.cols), T{});

    CUDA_CHECK(cudaFree(nullptr));
    std::size_t free_before = 0;
    std::size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_mem));

    const auto h2d_start = linbench::Clock::now();
    AMGX_CHECK(AMGX_initialize());
    AMGX_CHECK(AMGX_initialize_plugins());
    AMGX_config_handle cfg = nullptr;
    AMGX_resources_handle rsrc = nullptr;
    AMGX_matrix_handle matrix = nullptr;
    AMGX_vector_handle rhs_vec = nullptr;
    AMGX_vector_handle x_vec = nullptr;
    AMGX_solver_handle solver = nullptr;
    AMGX_CHECK(AMGX_config_create_from_file(&cfg, opt.config.string().c_str()));
    AMGX_CHECK(AMGX_resources_create_simple(&rsrc, cfg));
    AMGX_CHECK(AMGX_matrix_create(&matrix, rsrc, amgx_mode<T>()));
    AMGX_CHECK(AMGX_vector_create(&rhs_vec, rsrc, amgx_mode<T>()));
    AMGX_CHECK(AMGX_vector_create(&x_vec, rsrc, amgx_mode<T>()));
    AMGX_CHECK(AMGX_matrix_upload_all(matrix, A.rows, A.nnz, 1, 1, A.row_ptr.data(), A.col_idx.data(), values.data(), nullptr));
    AMGX_CHECK(AMGX_vector_upload(rhs_vec, A.rows, 1, rhs.data()));
    AMGX_CHECK(AMGX_vector_set_zero(x_vec, A.rows, 1));
    const auto h2d_stop = linbench::Clock::now();

    AMGX_CHECK(AMGX_solver_create(&solver, rsrc, amgx_mode<T>(), cfg));
    const auto setup_start = linbench::Clock::now();
    AMGX_CHECK(AMGX_solver_setup(solver, matrix));
    const auto setup_stop = linbench::Clock::now();

    auto solve_once = [&]() -> double {
        AMGX_CHECK(AMGX_vector_set_zero(x_vec, A.rows, 1));
        const auto s0 = linbench::Clock::now();
        AMGX_CHECK(AMGX_solver_solve_with_0_initial_guess(solver, rhs_vec, x_vec));
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto s1 = linbench::Clock::now();
        return linbench::elapsed_ms(s0, s1);
    };

    for (int i = 0; i < opt.warmup; ++i) {
        (void)solve_once();
    }
    std::vector<double> solve_ms;
    solve_ms.reserve(static_cast<std::size_t>(opt.repeats));
    for (int i = 0; i < opt.repeats; ++i) {
        solve_ms.push_back(solve_once());
    }

    int iterations = -1;
    AMGX_SOLVE_STATUS solve_status = AMGX_SOLVE_FAILED;
    double final_residual = std::numeric_limits<double>::quiet_NaN();
    AMGX_CHECK(AMGX_solver_get_iterations_number(solver, &iterations));
    AMGX_CHECK(AMGX_solver_get_status(solver, &solve_status));
    if (iterations >= 0) {
        (void)AMGX_solver_get_iteration_residual(solver, iterations, 0, &final_residual);
    }

    const auto d2h_start = linbench::Clock::now();
    AMGX_CHECK(AMGX_vector_download(x_vec, x_host.data()));
    const auto d2h_stop = linbench::Clock::now();

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

    char* version = nullptr;
    char* date = nullptr;
    char* time = nullptr;
    (void)AMGX_get_build_info_strings(&version, &date, &time);
    int api_major = 0;
    int api_minor = 0;
    (void)AMGX_get_api_version(&api_major, &api_minor);
    const std::string version_text = std::string(version ? version : "unknown") + " api " + std::to_string(api_major) + "." + std::to_string(api_minor);

    if (solver) AMGX_CHECK(AMGX_solver_destroy(solver));
    if (x_vec) AMGX_CHECK(AMGX_vector_destroy(x_vec));
    if (rhs_vec) AMGX_CHECK(AMGX_vector_destroy(rhs_vec));
    if (matrix) AMGX_CHECK(AMGX_matrix_destroy(matrix));
    if (rsrc) AMGX_CHECK(AMGX_resources_destroy(rsrc));
    if (cfg) AMGX_CHECK(AMGX_config_destroy(cfg));
    AMGX_CHECK(AMGX_finalize_plugins());
    AMGX_CHECK(AMGX_finalize());

    const auto total_stop = linbench::Clock::now();
    linbench::Result result;
    result.solver_name = "AMGx";
    result.solver_version = version_text;
    result.library_path = AMGX_LIBRARY_PATH_STR;
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
    result.h2d_ms = linbench::elapsed_ms(h2d_start, h2d_stop);
    result.analysis_ms = linbench::elapsed_ms(setup_start, setup_stop);
    result.factorization_ms = 0.0;
    result.solve_ms = s_stats.mean;
    result.d2h_ms = linbench::elapsed_ms(d2h_start, d2h_stop);
    result.total_solver_ms = result.analysis_ms + result.solve_ms;
    result.total_end_to_end_ms = linbench::elapsed_ms(total_start, total_stop);
    result.peak_gpu_memory_mb = used_mb;
    result.relative_residual_2 = rel_res;
    result.relative_error_to_x_ref_2 = rel_err;
    result.converged = solve_status == AMGX_SOLVE_SUCCESS;
    result.num_iterations = iterations;
    result.gpu_resident_after_initial_load = "yes";
    result.notes = "AMGx iterative GMRES/AMG configuration; setup is reported as analysis_ms and solve_ms is repeated solve time from zero initial guess.";
    result.timing_stats["solve_ms"] = s_stats;
    result.extra_strings["solver_configuration"] = opt.config.string();
    result.extra_strings["preconditioner"] = "AMG";
    result.extra_strings["solve_status"] = status_name(solve_status);
    result.extra_numbers["final_reported_residual"] = final_residual;
    result.extra_numbers["convergence_tolerance"] = 1e-8;
    result.extra_numbers["max_iterations"] = 200.0;
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
        std::cerr << "AMGx benchmark failed: " << exc.what() << "\n";
        return 2;
    }
}
