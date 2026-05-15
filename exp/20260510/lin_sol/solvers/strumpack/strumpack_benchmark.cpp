#include "benchmark_common.hpp"

#include <mpi.h>

#include <StrumpackConfig.h>
#include <StrumpackSparseSolverMPIDist.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <type_traits>

namespace {

template <typename T>
MPI_Datatype mpi_type();

template <>
MPI_Datatype mpi_type<double>()
{
    return MPI_DOUBLE;
}

template <>
MPI_Datatype mpi_type<float>()
{
    return MPI_FLOAT;
}

struct RowPartition {
    std::vector<int> dist;
    int first = 0;
    int last = 0;
    int local_rows = 0;
};

RowPartition make_partition(int rows, int rank, int nprocs)
{
    RowPartition p;
    p.dist.resize(static_cast<std::size_t>(nprocs + 1), 0);
    for (int r = 0; r <= nprocs; ++r) {
        p.dist[static_cast<std::size_t>(r)] = rows * r / nprocs;
    }
    p.first = p.dist[static_cast<std::size_t>(rank)];
    p.last = p.dist[static_cast<std::size_t>(rank + 1)];
    p.local_rows = p.last - p.first;
    return p;
}

template <typename T>
struct LocalCsr {
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<T> values;
    std::vector<T> rhs;
};

template <typename T>
LocalCsr<T> make_local_csr(const linbench::CsrMatrix& A, const std::vector<double>& rhs, const RowPartition& part)
{
    LocalCsr<T> local;
    local.row_ptr.assign(static_cast<std::size_t>(part.local_rows + 1), 0);
    for (int row = part.first; row < part.last; ++row) {
        const int local_row = row - part.first;
        const int begin = A.row_ptr[static_cast<std::size_t>(row)];
        const int end = A.row_ptr[static_cast<std::size_t>(row + 1)];
        local.row_ptr[static_cast<std::size_t>(local_row + 1)] =
            local.row_ptr[static_cast<std::size_t>(local_row)] + (end - begin);
        for (int p = begin; p < end; ++p) {
            local.col_idx.push_back(A.col_idx[static_cast<std::size_t>(p)]);
            local.values.push_back(static_cast<T>(A.values[static_cast<std::size_t>(p)]));
        }
        local.rhs.push_back(static_cast<T>(rhs[static_cast<std::size_t>(row)]));
    }
    return local;
}

struct PhaseTiming {
    double ms = 0.0;
    int code = 0;
};

template <typename Func>
PhaseTiming timed_mpi_phase(Func&& func)
{
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = linbench::Clock::now();
    const int local_code = func();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto stop = linbench::Clock::now();
    const double local_ms = linbench::elapsed_ms(start, stop);
    double global_ms = 0.0;
    int global_code = 0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&local_code, &global_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return {global_ms, global_code};
}

template <typename T>
std::vector<double> gather_solution(const std::vector<T>& local_x, const RowPartition& part, int rank, int nprocs)
{
    std::vector<int> counts(static_cast<std::size_t>(nprocs), 0);
    std::vector<int> displs(static_cast<std::size_t>(nprocs), 0);
    for (int p = 0; p < nprocs; ++p) {
        counts[static_cast<std::size_t>(p)] =
            part.dist[static_cast<std::size_t>(p + 1)] - part.dist[static_cast<std::size_t>(p)];
        displs[static_cast<std::size_t>(p)] = part.dist[static_cast<std::size_t>(p)];
    }

    std::vector<T> global_x_t;
    if (rank == 0) {
        global_x_t.resize(static_cast<std::size_t>(part.dist.back()));
    }
    MPI_Gatherv(local_x.data(), static_cast<int>(local_x.size()), mpi_type<T>(),
                rank == 0 ? global_x_t.data() : nullptr, counts.data(), displs.data(), mpi_type<T>(),
                0, MPI_COMM_WORLD);

    std::vector<double> global_x;
    if (rank == 0) {
        global_x.reserve(global_x_t.size());
        for (T v : global_x_t) {
            global_x.push_back(static_cast<double>(v));
        }
    }
    return global_x;
}

template <typename T>
linbench::Result unsupported_result(const linbench::CliOptions& opt, const std::string& reason)
{
    linbench::Meta meta = linbench::load_meta(opt.meta);
    linbench::Result result;
    result.solver_name = "STRUMPACK";
    result.solver_version = "8.0.0";
    result.library_path = STRUMPACK_LIBRARY_PATH_STR;
    result.build_status = "unsupported";
    result.dtype = opt.dtype;
    result.case_name = meta.case_name;
    result.iteration = meta.iteration;
    result.matrix_rows = meta.matrix_rows;
    result.matrix_cols = meta.matrix_cols;
    result.nnz = meta.nnz;
    result.repeat_count = opt.repeats;
    result.warmup_count = opt.warmup;
    result.gpu_resident_after_initial_load = "unsupported";
    result.notes = reason;
    result.extra_strings["CUDA_enabled"] = "yes";
    result.extra_strings["MPI_required"] = "yes";
    return result;
}

template <typename T>
linbench::Result run_strumpack(const linbench::CliOptions& opt, int rank, int nprocs)
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

    const auto partition_start = linbench::Clock::now();
    RowPartition part = make_partition(A.rows, rank, nprocs);
    LocalCsr<T> local = make_local_csr<T>(A, rhs, part);
    const auto partition_stop = linbench::Clock::now();

    auto solve_once = [&]() {
        strumpack::StrumpackSparseSolverMPIDist<T, int> solver(MPI_COMM_WORLD, rank == 0);
        solver.options().enable_gpu();
        std::vector<T> local_x(static_cast<std::size_t>(part.local_rows), T{});

        const auto set_matrix_timing = timed_mpi_phase([&]() {
            solver.set_distributed_csr_matrix(
                part.local_rows,
                local.row_ptr.data(),
                local.col_idx.data(),
                local.values.data(),
                part.dist.data(),
                false);
            return 0;
        });
        const auto reorder_timing = timed_mpi_phase([&]() {
            return solver.reorder() == strumpack::ReturnCode::SUCCESS ? 0 : 1;
        });
        const auto factor_timing = timed_mpi_phase([&]() {
            return solver.factor() == strumpack::ReturnCode::SUCCESS ? 0 : 1;
        });
        const auto solve_timing = timed_mpi_phase([&]() {
            return solver.solve(local.rhs.data(), local_x.data()) == strumpack::ReturnCode::SUCCESS ? 0 : 1;
        });

        const int code = std::max({set_matrix_timing.code, reorder_timing.code, factor_timing.code, solve_timing.code});
        return std::make_tuple(
            set_matrix_timing.ms + reorder_timing.ms,
            factor_timing.ms,
            solve_timing.ms,
            code,
            std::move(local_x));
    };

    for (int i = 0; i < opt.warmup; ++i) {
        auto [analysis_ms, factor_ms, solve_ms, code, x] = solve_once();
        (void)analysis_ms;
        (void)factor_ms;
        (void)solve_ms;
        (void)x;
        if (code != 0) {
            throw std::runtime_error("STRUMPACK warmup failed");
        }
    }

    std::vector<double> analysis_ms;
    std::vector<double> factorization_ms;
    std::vector<double> solve_ms;
    int last_code = 0;
    std::vector<T> last_local_x;
    for (int i = 0; i < opt.repeats; ++i) {
        auto [analysis, factor, solve, code, x] = solve_once();
        analysis_ms.push_back(analysis);
        factorization_ms.push_back(factor);
        solve_ms.push_back(solve);
        last_code = code;
        last_local_x = std::move(x);
        if (code != 0) {
            break;
        }
    }

    std::vector<double> x_solution = gather_solution(last_local_x, part, rank, nprocs);
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
    result.solver_name = "STRUMPACK";
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
    result.format_convert_ms = format_convert_ms + linbench::elapsed_ms(partition_start, partition_stop);
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
    result.gpu_resident_after_initial_load = "cpu_gpu_hybrid_mpi_dist";
    result.notes = "CUDA-enabled STRUMPACK MPIDist direct solve. Input and output vectors are host-side/distributed by MPI; STRUMPACK GPU offload is enabled internally, so timings include host, MPI, and GPU-offload overhead.";
    result.timing_stats["analysis_ms"] = analysis_stats;
    result.timing_stats["factorization_ms"] = factor_stats;
    result.timing_stats["solve_ms"] = solve_stats;
    result.extra_numbers["mpi_ranks"] = static_cast<double>(nprocs);
    result.extra_numbers["strumpack_return_code"] = static_cast<double>(last_code);
    result.extra_strings["solver_type"] = "distributed sparse direct";
    result.extra_strings["CUDA_enabled"] = "yes";
    result.extra_strings["MPI_required"] = "yes";
    result.extra_strings["data_residency"] = "host_input_output_with_internal_gpu_offload";

    if (rank == 0) {
        std::vector<double> r = linbench::residual(A, x_solution, rhs);
        result.relative_residual_2 = linbench::norm2(r) /
            std::max(linbench::norm2(rhs), std::numeric_limits<double>::min());
        result.relative_error_to_x_ref_2 = linbench::relative_error(x_solution, x_ref);
        const double tol = std::is_same<T, double>::value ? 1e-8 : 1e-4;
        result.converged = last_code == 0 && std::isfinite(result.relative_residual_2) && result.relative_residual_2 < tol;
    }

    return result;
}

linbench::Result failure_result(const linbench::CliOptions& opt, const std::string& message)
{
    linbench::Meta meta = linbench::load_meta(opt.meta);
    linbench::Result result;
    result.solver_name = "STRUMPACK";
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
    result.extra_strings["MPI_required"] = "yes";
    return result;
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int rc = 0;
    try {
        linbench::CliOptions opt = linbench::parse_cli(argc, argv);
        linbench::Result result = opt.dtype == "fp64"
            ? run_strumpack<double>(opt, rank, nprocs)
            : run_strumpack<float>(opt, rank, nprocs);
        if (rank == 0) {
            linbench::write_result_json(opt.out, result);
        }
    } catch (const std::exception& exc) {
        if (rank == 0) {
            try {
                linbench::CliOptions opt = linbench::parse_cli(argc, argv);
                linbench::write_result_json(opt.out, failure_result(opt, exc.what()));
                std::cerr << "STRUMPACK benchmark failed: " << exc.what() << "\n";
            } catch (const std::exception& nested) {
                std::cerr << "STRUMPACK benchmark failed before result creation: "
                          << exc.what() << "; nested: " << nested.what() << "\n";
            }
        }
        rc = 2;
    }

    MPI_Finalize();
    return rc;
}
