#include "benchmark_common.hpp"

#include <cuda_runtime.h>
#include <ginkgo/ginkgo.hpp>

#include <iostream>
#include <memory>
#include <sstream>
#include <type_traits>

namespace {

void cuda_check(cudaError_t status, const char* expr)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(status));
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr)

template <typename T>
std::vector<T> convert(const std::vector<double>& values)
{
    std::vector<T> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<T>(values[i]);
    }
    return out;
}

struct SolverConfig {
    std::string solver = "gmres";
    std::string preconditioner = "jacobi";
    int max_iterations = 1000;
    double tolerance = 1e-10;
};

SolverConfig load_config(const linbench::CliOptions& opt)
{
    SolverConfig cfg;
    if (opt.dtype == "fp32") {
        cfg.tolerance = 1e-5;
    }
    if (opt.config.empty()) {
        return cfg;
    }
    std::string text = opt.config.string();
    if (std::filesystem::exists(opt.config)) {
        text = linbench::read_file(opt.config);
    }
    auto get_string = [&](const std::string& key, const std::string& fallback) {
        const std::string value = linbench::json_string_value(text, key);
        return value.empty() ? fallback : value;
    };
    auto get_int = [&](const std::string& key, int fallback) {
        const int value = linbench::json_int_value(text, key);
        return value == 0 ? fallback : value;
    };
    auto get_double = [&](const std::string& key, double fallback) {
        const std::regex re("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
        std::smatch m;
        if (std::regex_search(text, m, re)) {
            return std::stod(m[1].str());
        }
        return fallback;
    };
    cfg.solver = get_string("solver", cfg.solver);
    cfg.preconditioner = get_string("preconditioner", cfg.preconditioner);
    cfg.max_iterations = get_int("max_iterations", cfg.max_iterations);
    cfg.tolerance = get_double("tolerance", cfg.tolerance);
    std::transform(cfg.solver.begin(), cfg.solver.end(), cfg.solver.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::transform(cfg.preconditioner.begin(), cfg.preconditioner.end(), cfg.preconditioner.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return cfg;
}

template <typename T>
std::unique_ptr<gko::LinOpFactory> make_solver_factory(
    const SolverConfig& cfg,
    const std::shared_ptr<gko::CudaExecutor>& exec)
{
    using index_type = int;
    using bj = gko::preconditioner::Jacobi<T, index_type>;
    auto criteria = gko::stop::Iteration::build()
                        .with_max_iters(static_cast<gko::size_type>(cfg.max_iterations));
    auto residual = gko::stop::ResidualNorm<T>::build()
                        .with_reduction_factor(static_cast<gko::remove_complex<T>>(cfg.tolerance));
    if (cfg.solver == "bicgstab") {
        return gko::solver::Bicgstab<T>::build()
            .with_criteria(criteria, residual)
            .with_preconditioner(bj::build().on(exec))
            .on(exec);
    }
    return gko::solver::Gmres<T>::build()
        .with_criteria(criteria, residual)
        .with_preconditioner(bj::build().on(exec))
        .on(exec);
}

template <typename T>
linbench::Result run_solver(const linbench::CliOptions& opt)
{
    const auto total_start = linbench::Clock::now();
    SolverConfig cfg = load_config(opt);
    double format_convert_ms = 0.0;
    const auto load_start = linbench::Clock::now();
    linbench::CsrMatrix A_host = linbench::load_matrix_market_csr(opt.matrix, &format_convert_ms);
    std::vector<double> rhs_double = linbench::load_vector(opt.rhs);
    std::vector<double> x_ref = linbench::load_vector(opt.xref);
    linbench::Meta meta = linbench::load_meta(opt.meta);
    const auto load_stop = linbench::Clock::now();
    if (A_host.rows != A_host.cols || A_host.rows != static_cast<int>(rhs_double.size()) ||
        A_host.cols != static_cast<int>(x_ref.size())) {
        throw std::runtime_error("matrix/vector dimensions are inconsistent");
    }

    std::vector<T> values = convert<T>(A_host.values);
    std::vector<T> rhs_values = convert<T>(rhs_double);
    std::vector<T> zeros(static_cast<std::size_t>(A_host.cols), T{});

    CUDA_CHECK(cudaFree(nullptr));
    std::size_t free_before = 0;
    std::size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_mem));

    auto host_exec = gko::ReferenceExecutor::create();
    auto exec = gko::CudaExecutor::create(0, host_exec);
    using matrix_type = gko::matrix::Csr<T, int>;
    using vector_type = gko::matrix::Dense<T>;
    using value_array = gko::array<T>;
    using index_array = gko::array<int>;

    const auto h2d_start = linbench::Clock::now();
    auto A = gko::share(matrix_type::create(
        exec,
        gko::dim<2>(static_cast<gko::size_type>(A_host.rows), static_cast<gko::size_type>(A_host.cols)),
        value_array::view(host_exec, static_cast<gko::size_type>(values.size()), values.data()),
        index_array::view(host_exec, static_cast<gko::size_type>(A_host.col_idx.size()), A_host.col_idx.data()),
        index_array::view(host_exec, static_cast<gko::size_type>(A_host.row_ptr.size()), A_host.row_ptr.data())));
    auto host_b = vector_type::create(
        host_exec,
        gko::dim<2>(static_cast<gko::size_type>(A_host.rows), 1),
        value_array::view(host_exec, static_cast<gko::size_type>(rhs_values.size()), rhs_values.data()),
        1);
    auto host_x0 = vector_type::create(
        host_exec,
        gko::dim<2>(static_cast<gko::size_type>(A_host.cols), 1),
        value_array::view(host_exec, static_cast<gko::size_type>(zeros.size()), zeros.data()),
        1);
    auto b = gko::clone(exec, host_b);
    auto x = gko::clone(exec, host_x0);
    exec->synchronize();
    const auto h2d_stop = linbench::Clock::now();
    const double h2d_ms = linbench::elapsed_ms(h2d_start, h2d_stop);

    auto factory = make_solver_factory<T>(cfg, exec);
    const auto analysis_start = linbench::Clock::now();
    auto solver = factory->generate(A);
    exec->synchronize();
    const auto analysis_stop = linbench::Clock::now();
    const double analysis_ms = linbench::elapsed_ms(analysis_start, analysis_stop);

    auto solve_once = [&]() {
        x->fill(T{});
        exec->synchronize();
        const auto start = linbench::Clock::now();
        solver->apply(b, x);
        exec->synchronize();
        const auto stop = linbench::Clock::now();
        return linbench::elapsed_ms(start, stop);
    };

    for (int i = 0; i < opt.warmup; ++i) {
        solve_once();
    }
    std::vector<double> solve_ms;
    solve_ms.reserve(static_cast<std::size_t>(opt.repeats));
    for (int i = 0; i < opt.repeats; ++i) {
        solve_ms.push_back(solve_once());
    }

    auto final_solver = factory->generate(A);
    std::shared_ptr<const gko::log::Convergence<T>> logger(gko::log::Convergence<T>::create());
    final_solver->add_logger(logger);
    x->fill(T{});
    exec->synchronize();
    final_solver->apply(b, x);
    exec->synchronize();

    const auto d2h_start = linbench::Clock::now();
    auto x_host = gko::clone(host_exec, x);
    exec->synchronize();
    const auto d2h_stop = linbench::Clock::now();
    const double d2h_ms = linbench::elapsed_ms(d2h_start, d2h_stop);

    std::size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_mem));
    const double used_mb = static_cast<double>(free_before > free_after ? free_before - free_after : 0) / (1024.0 * 1024.0);

    std::vector<double> x_double(static_cast<std::size_t>(A_host.cols));
    const T* x_values = x_host->get_const_values();
    for (std::size_t i = 0; i < x_double.size(); ++i) {
        x_double[i] = static_cast<double>(x_values[i]);
    }
    std::vector<double> r = linbench::residual(A_host, x_double, rhs_double);
    const double rhs_norm = linbench::norm2(rhs_double);
    const double rel_res = linbench::norm2(r) / std::max(rhs_norm, std::numeric_limits<double>::min());
    const double rel_err = linbench::relative_error(x_double, x_ref);
    linbench::Stats solve_stats = linbench::make_stats(solve_ms);

    std::ostringstream version;
    version << gko::version_info::get();
    const auto total_stop = linbench::Clock::now();
    linbench::Result result;
    result.solver_name = cfg.solver == "bicgstab" ? "Ginkgo-BiCGSTAB-Jacobi" : "Ginkgo-GMRES-Jacobi";
    result.solver_version = version.str();
    result.library_path = GINKGO_LIBRARY_PATH_STR;
    result.dtype = opt.dtype;
    result.case_name = meta.case_name;
    result.iteration = meta.iteration;
    result.matrix_rows = A_host.rows;
    result.matrix_cols = A_host.cols;
    result.nnz = A_host.nnz;
    result.repeat_count = opt.repeats;
    result.warmup_count = opt.warmup;
    result.load_ms = linbench::elapsed_ms(load_start, load_stop);
    result.format_convert_ms = format_convert_ms;
    result.h2d_ms = h2d_ms;
    result.analysis_ms = analysis_ms;
    result.factorization_ms = 0.0;
    result.solve_ms = solve_stats.mean;
    result.d2h_ms = d2h_ms;
    result.total_solver_ms = analysis_ms + solve_stats.mean;
    result.total_end_to_end_ms = linbench::elapsed_ms(total_start, total_stop);
    result.peak_gpu_memory_mb = used_mb;
    result.relative_residual_2 = rel_res;
    result.relative_error_to_x_ref_2 = rel_err;
    result.converged = rel_res < (std::is_same<T, double>::value ? 1e-8 : 1e-4);
    result.num_iterations = static_cast<int>(logger->get_num_iterations());
    result.gpu_resident_after_initial_load = "yes_for_matrix_vector_and_iterations";
    result.notes = "Ginkgo CUDA executor benchmark using CPU wall-clock timing around synchronized CUDA operations. Analysis is solver generation plus Jacobi setup; no direct factorization phase is exposed for this iterative configuration.";
    result.timing_stats["analysis_ms"] = linbench::make_stats({analysis_ms});
    result.timing_stats["solve_ms"] = solve_stats;
    result.timing_stats["total_solver_ms"] = linbench::make_stats({analysis_ms + solve_stats.mean});
    result.extra_strings["solver_configuration"] = cfg.solver;
    result.extra_strings["preconditioner"] = cfg.preconditioner;
    result.extra_numbers["convergence_tolerance"] = cfg.tolerance;
    result.extra_numbers["max_iterations"] = static_cast<double>(cfg.max_iterations);
    result.extra_numbers["actual_iterations"] = static_cast<double>(result.num_iterations);
    result.extra_numbers["final_residual"] = rel_res;
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
        std::cerr << "Ginkgo benchmark failed: " << exc.what() << "\n";
        return 2;
    }
}
