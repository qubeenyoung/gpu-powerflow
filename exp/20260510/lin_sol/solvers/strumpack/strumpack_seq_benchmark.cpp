#include "benchmark_common.hpp"

#include <StrumpackConfig.h>
#include <StrumpackSparseSolver.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <type_traits>

namespace {

template <typename T>
struct SolveResult {
    double analysis_ms = 0.0;
    double factorization_ms = 0.0;
    double solve_ms = 0.0;
    int code = 0;
    std::vector<T> x;
};

template <typename T>
SolveResult<T> solve_once(const linbench::CsrMatrix& A, const std::vector<double>& rhs)
{
    strumpack::StrumpackSparseSolver<T, int> solver(false, true);
    solver.options().enable_gpu();

    std::vector<T> values;
    values.reserve(A.values.size());
    for (double v : A.values) {
        values.push_back(static_cast<T>(v));
    }

    std::vector<T> rhs_t;
    rhs_t.reserve(rhs.size());
    for (double v : rhs) {
        rhs_t.push_back(static_cast<T>(v));
    }
    std::vector<T> x(rhs_t.size(), T{});

    SolveResult<T> result;
    const auto analysis_start = linbench::Clock::now();
    solver.set_csr_matrix(A.rows, A.row_ptr.data(), A.col_idx.data(), values.data(), false);
    const auto reorder_code = solver.reorder();
    const auto analysis_stop = linbench::Clock::now();

    const auto factor_start = linbench::Clock::now();
    const auto factor_code = solver.factor();
    const auto factor_stop = linbench::Clock::now();

    const auto solve_start = linbench::Clock::now();
    const auto solve_code = solver.solve(rhs_t.data(), x.data());
    const auto solve_stop = linbench::Clock::now();

    result.analysis_ms = linbench::elapsed_ms(analysis_start, analysis_stop);
    result.factorization_ms = linbench::elapsed_ms(factor_start, factor_stop);
    result.solve_ms = linbench::elapsed_ms(solve_start, solve_stop);
    result.code = std::max({
        reorder_code == strumpack::ReturnCode::SUCCESS ? 0 : 1,
        factor_code == strumpack::ReturnCode::SUCCESS ? 0 : 1,
        solve_code == strumpack::ReturnCode::SUCCESS ? 0 : 1,
    });
    result.x = std::move(x);
    return result;
}

template <typename T>
linbench::Result run_strumpack_seq(const linbench::CliOptions& opt)
{
    const auto total_start = linbench::Clock::now();
    double format_convert_ms = 0.0;
    const auto load_start = linbench::Clock::now();
    linbench::CsrMatrix A = linbench::load_matrix_market_csr(opt.matrix, &format_convert_ms);
    std::vector<double> rhs = linbench::load_vector(opt.rhs);
    std::vector<double> x_ref = linbench::load_vector(opt.xref);
    linbench::Meta meta = linbench::load_meta(opt.meta);
    const auto load_stop = linbench::Clock::now();

    if (A.rows != A.cols || A.rows != static_cast<int>(rhs.size()) ||
        A.cols != static_cast<int>(x_ref.size())) {
        throw std::runtime_error("matrix/vector dimensions are inconsistent");
    }

    for (int i = 0; i < opt.warmup; ++i) {
        auto warmup = solve_once<T>(A, rhs);
        if (warmup.code != 0) {
            throw std::runtime_error("STRUMPACK sequential warmup failed");
        }
    }

    std::vector<double> analysis_ms;
    std::vector<double> factorization_ms;
    std::vector<double> solve_ms;
    std::vector<T> last_x_t;
    int last_code = 0;
    for (int i = 0; i < opt.repeats; ++i) {
        auto run = solve_once<T>(A, rhs);
        analysis_ms.push_back(run.analysis_ms);
        factorization_ms.push_back(run.factorization_ms);
        solve_ms.push_back(run.solve_ms);
        last_code = run.code;
        last_x_t = std::move(run.x);
        if (last_code != 0) {
            break;
        }
    }

    std::vector<double> x_solution;
    x_solution.reserve(last_x_t.size());
    for (T v : last_x_t) {
        x_solution.push_back(static_cast<double>(v));
    }

    linbench::Stats analysis_stats = linbench::make_stats(analysis_ms);
    linbench::Stats factor_stats = linbench::make_stats(factorization_ms);
    linbench::Stats solve_stats = linbench::make_stats(solve_ms);

    std::size_t free_mem = 0;
    std::size_t total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
        free_mem = 0;
        total_mem = 0;
    }

    const auto total_stop = linbench::Clock::now();
    linbench::Result result;
    result.solver_name = "STRUMPACK-Sequential";
    result.solver_version =
        std::to_string(STRUMPACK_VERSION_MAJOR) + "." +
        std::to_string(STRUMPACK_VERSION_MINOR) + "." +
        std::to_string(STRUMPACK_VERSION_PATCH);
    result.library_path = STRUMPACK_LIBRARY_PATH_STR;
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
    result.h2d_ms = 0.0;
    result.analysis_ms = analysis_stats.mean;
    result.factorization_ms = factor_stats.mean;
    result.solve_ms = solve_stats.mean;
    result.d2h_ms = 0.0;
    result.total_solver_ms = result.analysis_ms + result.factorization_ms + result.solve_ms;
    result.total_end_to_end_ms = linbench::elapsed_ms(total_start, total_stop);
    result.peak_gpu_memory_mb = total_mem > free_mem
        ? static_cast<double>(total_mem - free_mem) / (1024.0 * 1024.0)
        : std::numeric_limits<double>::quiet_NaN();
    result.num_iterations = 1;
    result.gpu_resident_after_initial_load = "host_csr_with_internal_gpu_offload";
    result.notes =
        "CUDA-enabled STRUMPACK sequential SparseSolver path. Matrix and vectors are host-side CSR/dense arrays; "
        "solver.options().enable_gpu() enables internal GPU offload where the STRUMPACK build supports it.";
    result.timing_stats["analysis_ms"] = analysis_stats;
    result.timing_stats["factorization_ms"] = factor_stats;
    result.timing_stats["solve_ms"] = solve_stats;
    result.extra_strings["solver_type"] = "sequential sparse direct";
    result.extra_strings["CUDA_enabled"] = "yes";
    result.extra_strings["MPI_required"] = "linked_library_only";
    result.extra_strings["data_residency"] = "host_input_output_with_internal_gpu_offload";
    result.extra_strings["strumpack_have_magma"] = have_magma() ? "yes" : "no";
    result.extra_strings["strumpack_have_kblas"] = have_kblas() ? "yes" : "no";
    result.extra_strings["strumpack_have_slate"] = have_slate() ? "yes" : "no";
    result.extra_numbers["strumpack_return_code"] = static_cast<double>(last_code);

    std::vector<double> r = linbench::residual(A, x_solution, rhs);
    result.relative_residual_2 =
        linbench::norm2(r) / std::max(linbench::norm2(rhs), std::numeric_limits<double>::min());
    result.relative_error_to_x_ref_2 = linbench::relative_error(x_solution, x_ref);
    const double tol = std::is_same<T, double>::value ? 1e-8 : 1e-4;
    result.converged = last_code == 0 && std::isfinite(result.relative_residual_2) &&
        result.relative_residual_2 < tol;
    return result;
}

linbench::Result failure_result(const linbench::CliOptions& opt, const std::string& message)
{
    linbench::Meta meta = linbench::load_meta(opt.meta);
    linbench::Result result;
    result.solver_name = "STRUMPACK-Sequential";
    result.solver_version = "8.0.0";
    result.library_path = STRUMPACK_LIBRARY_PATH_STR;
    result.build_status = "runtime_failed";
    result.dtype = opt.dtype;
    result.case_name = meta.case_name;
    result.iteration = meta.iteration;
    result.matrix_rows = meta.matrix_rows;
    result.matrix_cols = meta.matrix_cols;
    result.nnz = meta.nnz;
    result.repeat_count = opt.repeats;
    result.warmup_count = opt.warmup;
    result.gpu_resident_after_initial_load = "unknown";
    result.notes = message;
    result.extra_strings["CUDA_enabled"] = "yes";
    result.extra_strings["MPI_required"] = "linked_library_only";
    return result;
}

}  // namespace

int main(int argc, char** argv)
{
    int rc = 0;
    try {
        linbench::CliOptions opt = linbench::parse_cli(argc, argv);
        linbench::Result result = opt.dtype == "fp64"
            ? run_strumpack_seq<double>(opt)
            : run_strumpack_seq<float>(opt);
        linbench::write_result_json(opt.out, result);
    } catch (const std::exception& exc) {
        try {
            linbench::CliOptions opt = linbench::parse_cli(argc, argv);
            linbench::write_result_json(opt.out, failure_result(opt, exc.what()));
            std::cerr << "STRUMPACK sequential benchmark failed: " << exc.what() << "\n";
        } catch (const std::exception& nested) {
            std::cerr << "STRUMPACK sequential benchmark failed before result creation: "
                      << exc.what() << "; nested: " << nested.what() << "\n";
        }
        rc = 2;
    }
    return rc;
}
