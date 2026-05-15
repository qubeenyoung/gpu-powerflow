#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/reorder/metis_partition.hpp"
#include "cuiter/solver/cpu_block_ilu0_pilot.hpp"
#include "cupf_minimal/case_data.hpp"
#include "cupf_minimal/direct_cudss_solver.hpp"
#include "cupf_minimal/jacobian_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

struct CliOptions {
    std::filesystem::path jf_root = "raw/cupf_jf_dumps";
    std::filesystem::path case_root =
        "/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps";
    std::filesystem::path output_dir = "results";
    std::vector<std::string> cases = {
        "case2383wp",
        "case3120sp",
        "case9241pegase",
        "case13659pegase",
        "case6468rte",
    };
    std::vector<int32_t> iterations = {1};
    std::vector<int32_t> block_sizes = {16, 32};
    std::string iterative_solver = "bicgstab";
    int32_t bicgstab_iters = 2;
    int32_t gmres_iters = 4;
    int32_t top_n = 20;
    double rel_eps = 1.0e-30;
    double diag_shift_scale = 1.0e-8;
    bool allow_missing = false;
};

struct LinearMetadata {
    int32_t n_bus = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
    std::vector<int32_t> index_to_bus;
    std::vector<int32_t> index_field;
    std::vector<int32_t> theta_index;
    std::vector<int32_t> vmag_index;
    std::vector<int32_t> p_row;
    std::vector<int32_t> q_row;
};

struct SolverDx {
    std::string solver;
    std::string ordering;
    std::vector<double> dx;
    std::vector<double> error;
    std::vector<double> residual_error;
    bool failed = false;
    std::string stop_reason;
};

struct BusErrorRow {
    std::string case_name;
    int32_t iteration = 0;
    int32_t block_size = 0;
    std::string solver;
    int32_t bus_id = 0;
    double theta_abs_error = kNan;
    double theta_rel_error = kNan;
    double vmag_abs_error = kNan;
    double vmag_rel_error = kNan;
    double combined_dx_error_norm = 0.0;
    double dx_cudss_theta = kNan;
    double dx_iter_theta = kNan;
    double dx_cudss_vmag = kNan;
    double dx_iter_vmag = kNan;
};

struct EquationErrorRow {
    std::string case_name;
    int32_t iteration = 0;
    int32_t block_size = 0;
    std::string solver;
    int32_t bus_id = 0;
    double p_error_abs = kNan;
    double q_error_abs = kNan;
    double combined_pq_error_norm = 0.0;
};

struct BlockErrorRow {
    std::string case_name;
    int32_t iteration = 0;
    int32_t block_size_target = 0;
    std::string solver;
    std::string ordering;
    int32_t block_id = 0;
    int32_t block_size = 0;
    double dx_error_norm = 0.0;
    double dx_cudss_norm = 0.0;
    double dx_error_ratio = 0.0;
    double residual_error_norm_on_block_rows = 0.0;
};

struct ComparisonRow {
    std::string case_name;
    int32_t iteration = 0;
    int32_t block_size = 0;
    std::string category;
    int32_t rank = 0;
    int32_t bus_id = 0;
    double improvement_dx = 0.0;
    double improvement_residual = 0.0;
    double dx_error_bj_norm = 0.0;
    double dx_error_bilu_norm = 0.0;
    double residual_error_bj_norm = 0.0;
    double residual_error_bilu_norm = 0.0;
    double theta_abs_error_bj = kNan;
    double theta_abs_error_bilu = kNan;
    double vmag_abs_error_bj = kNan;
    double vmag_abs_error_bilu = kNan;
    double p_error_bj_abs = kNan;
    double p_error_bilu_abs = kNan;
    double q_error_bj_abs = kNan;
    double q_error_bilu_abs = kNan;
};

struct SummaryRow {
    std::string case_name;
    int32_t iteration = 0;
    int32_t block_size = 0;
    std::string solver;
    double dx_error_norm = 0.0;
    double dx_error_ratio = 0.0;
    double theta_error_norm = 0.0;
    double theta_error_ratio = 0.0;
    double vmag_error_norm = 0.0;
    double vmag_error_ratio = 0.0;
    double residual_error_norm = 0.0;
};

bool is_block_jacobi_solver(const std::string& solver);
bool is_block_ilu0_solver(const std::string& solver);

std::vector<std::string> split_list(const std::string& text)
{
    std::vector<std::string> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(item);
        }
    }
    return values;
}

std::vector<int32_t> parse_int_list(const std::string& text)
{
    std::vector<int32_t> values;
    for (const std::string& item : split_list(text)) {
        values.push_back(std::stoi(item));
    }
    return values;
}

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --jf-root PATH\n"
        << "  --case-root PATH\n"
        << "  --output-dir PATH\n"
        << "  --cases a,b,c\n"
        << "  --iterations 1 or 0,1,2\n"
        << "  --block-sizes 16,32\n"
        << "  --iterative-solver bicgstab|gmres\n"
        << "  --bicgstab-iters 2\n"
        << "  --gmres-iters 4\n"
        << "  --top-n 20\n"
        << "  --rel-eps FLOAT\n"
        << "  --block-ilu-diag-shift-scale FLOAT\n"
        << "  --allow-missing\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--jf-root" && i + 1 < argc) {
            options.jf_root = argv[++i];
        } else if (arg == "--case-root" && i + 1 < argc) {
            options.case_root = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            options.output_dir = argv[++i];
        } else if (arg == "--cases" && i + 1 < argc) {
            options.cases = split_list(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            options.iterations = parse_int_list(argv[++i]);
        } else if (arg == "--block-sizes" && i + 1 < argc) {
            options.block_sizes = parse_int_list(argv[++i]);
        } else if (arg == "--iterative-solver" && i + 1 < argc) {
            options.iterative_solver = argv[++i];
        } else if (arg == "--bicgstab-iters" && i + 1 < argc) {
            options.bicgstab_iters = std::stoi(argv[++i]);
        } else if (arg == "--gmres-iters" && i + 1 < argc) {
            options.gmres_iters = std::stoi(argv[++i]);
        } else if (arg == "--top-n" && i + 1 < argc) {
            options.top_n = std::stoi(argv[++i]);
        } else if (arg == "--rel-eps" && i + 1 < argc) {
            options.rel_eps = std::stod(argv[++i]);
        } else if (arg == "--block-ilu-diag-shift-scale" && i + 1 < argc) {
            options.diag_shift_scale = std::stod(argv[++i]);
        } else if (arg == "--allow-missing") {
            options.allow_missing = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.cases.empty() || options.iterations.empty() || options.block_sizes.empty()) {
        throw std::runtime_error("case, iteration, and block-size lists must be nonempty");
    }
    if (options.iterative_solver != "bicgstab" && options.iterative_solver != "gmres") {
        throw std::runtime_error("--iterative-solver must be bicgstab or gmres");
    }
    return options;
}

void expect_token(std::istream& in,
                  const std::string& expected,
                  const std::filesystem::path& path)
{
    std::string token;
    if (!(in >> token) || token != expected) {
        throw std::runtime_error("expected token '" + expected + "' in " + path.string());
    }
}

cuiter::CsrMatrix load_cupf_csr_dump(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open matrix file: " + path.string());
    }
    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "csr_matrix") {
        throw std::runtime_error("matrix file is not a cuPF csr_matrix dump: " + path.string());
    }

    cuiter::CsrMatrix matrix;
    int32_t nnz = 0;
    expect_token(in, "rows", path);
    in >> matrix.rows;
    expect_token(in, "cols", path);
    in >> matrix.cols;
    expect_token(in, "nnz", path);
    in >> nnz;
    matrix.row_ptr.resize(static_cast<std::size_t>(matrix.rows + 1));
    matrix.col_idx.resize(static_cast<std::size_t>(nnz));
    matrix.values.resize(static_cast<std::size_t>(nnz));
    expect_token(in, "row_ptr", path);
    for (int32_t i = 0; i <= matrix.rows; ++i) {
        in >> matrix.row_ptr[static_cast<std::size_t>(i)];
    }
    expect_token(in, "col_idx", path);
    for (int32_t i = 0; i < nnz; ++i) {
        in >> matrix.col_idx[static_cast<std::size_t>(i)];
    }
    expect_token(in, "values", path);
    for (int32_t i = 0; i < nnz; ++i) {
        in >> matrix.values[static_cast<std::size_t>(i)];
    }
    if (!in || matrix.rows <= 0 || matrix.rows != matrix.cols ||
        matrix.row_ptr.front() != 0 || matrix.row_ptr.back() != nnz) {
        throw std::runtime_error("malformed CSR dump: " + path.string());
    }
    return matrix;
}

std::vector<double> load_cupf_vector_dump(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open vector file: " + path.string());
    }
    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "vector") {
        throw std::runtime_error("vector file is not a cuPF vector dump: " + path.string());
    }
    expect_token(in, "size", path);
    int32_t n = 0;
    in >> n;
    expect_token(in, "values", path);
    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    for (int32_t i = 0; i < n; ++i) {
        int32_t index = 0;
        double value = 0.0;
        in >> index >> value;
        if (!in || index < 0 || index >= n) {
            throw std::runtime_error("malformed vector dump: " + path.string());
        }
        values[static_cast<std::size_t>(index)] = value;
    }
    return values;
}

std::filesystem::path jacobian_path(const std::filesystem::path& jf_root,
                                    const std::string& case_name,
                                    int32_t iteration)
{
    const auto case_dir = jf_root / case_name;
    const auto direct = case_dir / ("J" + std::to_string(iteration) + ".txt");
    if (std::filesystem::exists(direct)) {
        return direct;
    }
    return case_dir / "repeat_00" / ("jacobian_iter" + std::to_string(iteration) + ".txt");
}

std::filesystem::path rhs_path(const std::filesystem::path& jf_root,
                               const std::string& case_name,
                               int32_t iteration)
{
    const auto case_dir = jf_root / case_name;
    const std::vector<std::filesystem::path> candidates = {
        case_dir / ("F" + std::to_string(iteration) + ".txt"),
        case_dir / "repeat_00" / ("residual_iter" + std::to_string(iteration) + ".txt"),
        case_dir / "repeat_00" / ("residual_before_update_iter" + std::to_string(iteration) + ".txt"),
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return candidates.front();
}

double norm2(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}

double ratio(double numerator, double denominator)
{
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

std::string format_double(double value)
{
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }
    std::ostringstream out;
    out << std::setprecision(12) << value;
    return out.str();
}

std::vector<double> spmv(const cuiter::CsrMatrix& matrix, const std::vector<double>& x)
{
    std::vector<double> y(static_cast<std::size_t>(matrix.rows), 0.0);
    for (int32_t row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            sum += matrix.values[static_cast<std::size_t>(pos)] *
                   x[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(pos)])];
        }
        y[static_cast<std::size_t>(row)] = sum;
    }
    return y;
}

LinearMetadata make_linear_metadata(const cupf_minimal::DumpCaseData& case_data)
{
    const cupf_minimal::JacobianIndexing indexing =
        cupf_minimal::make_jacobian_indexing(case_data.rows,
                                             case_data.pv.data(),
                                             static_cast<int32_t>(case_data.pv.size()),
                                             case_data.pq.data(),
                                             static_cast<int32_t>(case_data.pq.size()));
    LinearMetadata metadata;
    metadata.n_bus = case_data.rows;
    metadata.n_pvpq = indexing.n_pvpq;
    metadata.n_pq = indexing.n_pq;
    const int32_t n = metadata.n_pvpq + metadata.n_pq;
    metadata.index_to_bus.assign(static_cast<std::size_t>(n), -1);
    metadata.index_field.assign(static_cast<std::size_t>(n), -1);
    metadata.theta_index.assign(static_cast<std::size_t>(case_data.rows), -1);
    metadata.vmag_index.assign(static_cast<std::size_t>(case_data.rows), -1);
    metadata.p_row.assign(static_cast<std::size_t>(case_data.rows), -1);
    metadata.q_row.assign(static_cast<std::size_t>(case_data.rows), -1);

    for (int32_t i = 0; i < indexing.n_pvpq; ++i) {
        const int32_t bus = indexing.pvpq[static_cast<std::size_t>(i)];
        metadata.index_to_bus[static_cast<std::size_t>(i)] = bus;
        metadata.index_field[static_cast<std::size_t>(i)] = 0;
        metadata.theta_index[static_cast<std::size_t>(bus)] = i;
        metadata.p_row[static_cast<std::size_t>(bus)] = i;
    }
    for (int32_t i = 0; i < indexing.n_pq; ++i) {
        const int32_t bus = case_data.pq[static_cast<std::size_t>(i)];
        const int32_t index = indexing.n_pvpq + i;
        metadata.index_to_bus[static_cast<std::size_t>(index)] = bus;
        metadata.index_field[static_cast<std::size_t>(index)] = 1;
        metadata.vmag_index[static_cast<std::size_t>(bus)] = index;
        metadata.q_row[static_cast<std::size_t>(bus)] = index;
    }
    return metadata;
}

std::vector<double> solve_cudss_dx(const cuiter::CsrMatrix& matrix,
                                   const std::vector<double>& rhs)
{
    cuiter::DeviceBuffer<int32_t> d_row_ptr;
    cuiter::DeviceBuffer<int32_t> d_col_idx;
    cuiter::DeviceBuffer<double> d_values;
    cuiter::DeviceBuffer<double> d_rhs;
    cuiter::DeviceBuffer<double> d_x;
    d_row_ptr.assign(matrix.row_ptr.data(), matrix.row_ptr.size());
    d_col_idx.assign(matrix.col_idx.data(), matrix.col_idx.size());
    d_values.assign(matrix.values.data(), matrix.values.size());
    d_rhs.assign(rhs.data(), rhs.size());
    d_x.resize(rhs.size());
    d_x.memset_zero();

    cupf_minimal::DirectCudssSolver solver;
    solver.initialize(matrix,
                      d_row_ptr.data(),
                      d_col_idx.data(),
                      d_values.data(),
                      d_rhs.data(),
                      d_x.data());
    solver.analyze();
    solver.factorize();
    solver.solve();

    std::vector<double> dx(rhs.size(), 0.0);
    d_x.copy_to(dx.data(), dx.size());
    return dx;
}

SolverDx solve_iterative(const cuiter::CsrMatrix& matrix,
                         const std::vector<double>& rhs,
                         const std::vector<double>& dx_cudss,
                         int32_t block_size,
                         bool use_block_ilu0,
                         const CliOptions& options)
{
    cuiter::cpu_pilot::CpuBlockIlu0Options solve_options;
    solve_options.block_size = block_size;
    solve_options.bicgstab_iters = options.bicgstab_iters;
    solve_options.gmres_iters = options.gmres_iters;
    solve_options.diag_shift_scale = options.diag_shift_scale;
    solve_options.use_block_ilu0 = use_block_ilu0;
    solve_options.use_block_coloring_order = use_block_ilu0;
    solve_options.use_gmres = options.iterative_solver == "gmres";

    const cuiter::cpu_pilot::CpuBlockIlu0Result result =
        cuiter::cpu_pilot::solve(matrix, rhs, solve_options);

    SolverDx out;
    const std::string solver_prefix =
        options.iterative_solver == "gmres" ? "gmres" : "bicgstab";
    out.solver = solver_prefix + (use_block_ilu0 ? "_block_ilu0" : "_block_jacobi");
    out.ordering = use_block_ilu0 ? "block_coloring" : "current_metis_block_order";
    out.dx = result.solution;
    out.failed = result.factor_failed;
    out.stop_reason = result.stop_reason;
    out.error.assign(dx_cudss.size(), 0.0);
    for (std::size_t i = 0; i < dx_cudss.size(); ++i) {
        out.error[i] = out.dx[i] - dx_cudss[i];
    }
    out.residual_error = spmv(matrix, out.error);
    return out;
}

std::vector<int32_t> block_of_old_indices(const cuiter::CsrMatrix& matrix,
                                          const cuiter::MetisPermutation& permutation)
{
    std::vector<int32_t> block_of_new(static_cast<std::size_t>(matrix.rows), -1);
    for (int32_t block = 0; block < static_cast<int32_t>(permutation.block_sizes.size()); ++block) {
        const int32_t begin = permutation.block_starts[static_cast<std::size_t>(block)];
        const int32_t end = begin + permutation.block_sizes[static_cast<std::size_t>(block)];
        for (int32_t index = begin; index < end; ++index) {
            block_of_new[static_cast<std::size_t>(index)] = block;
        }
    }
    std::vector<int32_t> block_of_old(static_cast<std::size_t>(matrix.rows), -1);
    for (int32_t old_index = 0; old_index < matrix.rows; ++old_index) {
        const int32_t new_index = permutation.old_to_new[static_cast<std::size_t>(old_index)];
        block_of_old[static_cast<std::size_t>(old_index)] =
            block_of_new[static_cast<std::size_t>(new_index)];
    }
    return block_of_old;
}

cuiter::MetisPermutation build_solver_permutation(const cuiter::CsrMatrix& matrix,
                                                  int32_t block_size,
                                                  bool use_block_ilu0)
{
    if (use_block_ilu0) {
        return cuiter::cpu_pilot::detail::build_colored_block_permutation(matrix, block_size);
    }
    return cuiter::build_metis_permutation(matrix, block_size);
}

BusErrorRow make_bus_error_row(const std::string& case_name,
                               int32_t iteration,
                               int32_t block_size,
                               const SolverDx& solver,
                               const LinearMetadata& metadata,
                               const std::vector<double>& dx_cudss,
                               int32_t bus,
                               double rel_eps)
{
    BusErrorRow row;
    row.case_name = case_name;
    row.iteration = iteration;
    row.block_size = block_size;
    row.solver = solver.solver;
    row.bus_id = bus;
    const int32_t theta = metadata.theta_index[static_cast<std::size_t>(bus)];
    const int32_t vmag = metadata.vmag_index[static_cast<std::size_t>(bus)];
    double sq = 0.0;
    if (theta >= 0) {
        row.theta_abs_error = std::abs(solver.error[static_cast<std::size_t>(theta)]);
        row.theta_rel_error = row.theta_abs_error /
                              (std::abs(dx_cudss[static_cast<std::size_t>(theta)]) + rel_eps);
        row.dx_cudss_theta = dx_cudss[static_cast<std::size_t>(theta)];
        row.dx_iter_theta = solver.dx[static_cast<std::size_t>(theta)];
        sq += row.theta_abs_error * row.theta_abs_error;
    }
    if (vmag >= 0) {
        row.vmag_abs_error = std::abs(solver.error[static_cast<std::size_t>(vmag)]);
        row.vmag_rel_error = row.vmag_abs_error /
                             (std::abs(dx_cudss[static_cast<std::size_t>(vmag)]) + rel_eps);
        row.dx_cudss_vmag = dx_cudss[static_cast<std::size_t>(vmag)];
        row.dx_iter_vmag = solver.dx[static_cast<std::size_t>(vmag)];
        sq += row.vmag_abs_error * row.vmag_abs_error;
    }
    row.combined_dx_error_norm = std::sqrt(sq);
    return row;
}

EquationErrorRow make_equation_error_row(const std::string& case_name,
                                         int32_t iteration,
                                         int32_t block_size,
                                         const SolverDx& solver,
                                         const LinearMetadata& metadata,
                                         int32_t bus)
{
    EquationErrorRow row;
    row.case_name = case_name;
    row.iteration = iteration;
    row.block_size = block_size;
    row.solver = solver.solver;
    row.bus_id = bus;
    const int32_t p = metadata.p_row[static_cast<std::size_t>(bus)];
    const int32_t q = metadata.q_row[static_cast<std::size_t>(bus)];
    double sq = 0.0;
    if (p >= 0) {
        row.p_error_abs = std::abs(solver.residual_error[static_cast<std::size_t>(p)]);
        sq += row.p_error_abs * row.p_error_abs;
    }
    if (q >= 0) {
        row.q_error_abs = std::abs(solver.residual_error[static_cast<std::size_t>(q)]);
        sq += row.q_error_abs * row.q_error_abs;
    }
    row.combined_pq_error_norm = std::sqrt(sq);
    return row;
}

std::vector<BlockErrorRow> make_block_error_rows(const std::string& case_name,
                                                 int32_t iteration,
                                                 int32_t block_size,
                                                 const SolverDx& solver,
                                                 const cuiter::CsrMatrix& matrix,
                                                 const std::vector<double>& dx_cudss,
                                                 bool use_block_ilu0)
{
    const cuiter::MetisPermutation permutation =
        build_solver_permutation(matrix, block_size, use_block_ilu0);
    const std::vector<int32_t> block_of_old = block_of_old_indices(matrix, permutation);
    const int32_t num_blocks = static_cast<int32_t>(permutation.block_sizes.size());
    std::vector<double> error_sq(static_cast<std::size_t>(num_blocks), 0.0);
    std::vector<double> cudss_sq(static_cast<std::size_t>(num_blocks), 0.0);
    std::vector<double> residual_sq(static_cast<std::size_t>(num_blocks), 0.0);

    for (int32_t i = 0; i < matrix.rows; ++i) {
        const int32_t block = block_of_old[static_cast<std::size_t>(i)];
        const double e = solver.error[static_cast<std::size_t>(i)];
        const double d = dx_cudss[static_cast<std::size_t>(i)];
        const double r = solver.residual_error[static_cast<std::size_t>(i)];
        error_sq[static_cast<std::size_t>(block)] += e * e;
        cudss_sq[static_cast<std::size_t>(block)] += d * d;
        residual_sq[static_cast<std::size_t>(block)] += r * r;
    }

    std::vector<BlockErrorRow> rows;
    rows.reserve(static_cast<std::size_t>(num_blocks));
    for (int32_t block = 0; block < num_blocks; ++block) {
        BlockErrorRow row;
        row.case_name = case_name;
        row.iteration = iteration;
        row.block_size_target = block_size;
        row.solver = solver.solver;
        row.ordering = solver.ordering;
        row.block_id = block;
        row.block_size = permutation.block_sizes[static_cast<std::size_t>(block)];
        row.dx_error_norm = std::sqrt(error_sq[static_cast<std::size_t>(block)]);
        row.dx_cudss_norm = std::sqrt(cudss_sq[static_cast<std::size_t>(block)]);
        row.dx_error_ratio = ratio(row.dx_error_norm, row.dx_cudss_norm);
        row.residual_error_norm_on_block_rows =
            std::sqrt(residual_sq[static_cast<std::size_t>(block)]);
        rows.push_back(row);
    }
    return rows;
}

SummaryRow make_summary(const std::string& case_name,
                        int32_t iteration,
                        int32_t block_size,
                        const SolverDx& solver,
                        const LinearMetadata& metadata,
                        const std::vector<double>& dx_cudss)
{
    SummaryRow row;
    row.case_name = case_name;
    row.iteration = iteration;
    row.block_size = block_size;
    row.solver = solver.solver;
    row.dx_error_norm = norm2(solver.error);
    row.dx_error_ratio = ratio(row.dx_error_norm, norm2(dx_cudss));
    row.residual_error_norm = norm2(solver.residual_error);
    double theta_error_sq = 0.0;
    double theta_cudss_sq = 0.0;
    double vmag_error_sq = 0.0;
    double vmag_cudss_sq = 0.0;
    for (int32_t i = 0; i < static_cast<int32_t>(metadata.index_field.size()); ++i) {
        const double e = solver.error[static_cast<std::size_t>(i)];
        const double d = dx_cudss[static_cast<std::size_t>(i)];
        if (metadata.index_field[static_cast<std::size_t>(i)] == 0) {
            theta_error_sq += e * e;
            theta_cudss_sq += d * d;
        } else {
            vmag_error_sq += e * e;
            vmag_cudss_sq += d * d;
        }
    }
    row.theta_error_norm = std::sqrt(theta_error_sq);
    row.theta_error_ratio = ratio(row.theta_error_norm, std::sqrt(theta_cudss_sq));
    row.vmag_error_norm = std::sqrt(vmag_error_sq);
    row.vmag_error_ratio = ratio(row.vmag_error_norm, std::sqrt(vmag_cudss_sq));
    return row;
}

void add_top_comparisons(const std::string& case_name,
                         int32_t iteration,
                         int32_t block_size,
                         const LinearMetadata& metadata,
                         const std::vector<BusErrorRow>& bus_rows,
                         const std::vector<EquationErrorRow>& equation_rows,
                         int32_t top_n,
                         std::vector<ComparisonRow>& out)
{
    struct Item {
        int32_t bus = 0;
        double dx_bj = 0.0;
        double dx_bilu = 0.0;
        double res_bj = 0.0;
        double res_bilu = 0.0;
        double theta_bj = kNan;
        double theta_bilu = kNan;
        double vmag_bj = kNan;
        double vmag_bilu = kNan;
        double p_bj = kNan;
        double p_bilu = kNan;
        double q_bj = kNan;
        double q_bilu = kNan;
    };
    std::vector<Item> items(static_cast<std::size_t>(metadata.n_bus));
    for (int32_t bus = 0; bus < metadata.n_bus; ++bus) {
        items[static_cast<std::size_t>(bus)].bus = bus;
    }
    for (const BusErrorRow& row : bus_rows) {
        if (row.case_name != case_name || row.iteration != iteration ||
            row.block_size != block_size) {
            continue;
        }
        Item& item = items[static_cast<std::size_t>(row.bus_id)];
        if (is_block_jacobi_solver(row.solver)) {
            item.dx_bj = row.combined_dx_error_norm;
            item.theta_bj = row.theta_abs_error;
            item.vmag_bj = row.vmag_abs_error;
        } else if (is_block_ilu0_solver(row.solver)) {
            item.dx_bilu = row.combined_dx_error_norm;
            item.theta_bilu = row.theta_abs_error;
            item.vmag_bilu = row.vmag_abs_error;
        }
    }
    for (const EquationErrorRow& row : equation_rows) {
        if (row.case_name != case_name || row.iteration != iteration ||
            row.block_size != block_size) {
            continue;
        }
        Item& item = items[static_cast<std::size_t>(row.bus_id)];
        if (is_block_jacobi_solver(row.solver)) {
            item.res_bj = row.combined_pq_error_norm;
            item.p_bj = row.p_error_abs;
            item.q_bj = row.q_error_abs;
        } else if (is_block_ilu0_solver(row.solver)) {
            item.res_bilu = row.combined_pq_error_norm;
            item.p_bilu = row.p_error_abs;
            item.q_bilu = row.q_error_abs;
        }
    }

    auto emit = [&](std::string category, auto improvement_fn) {
        std::vector<Item> ranked = items;
        std::sort(ranked.begin(), ranked.end(), [&](const Item& lhs, const Item& rhs) {
            return improvement_fn(lhs) > improvement_fn(rhs);
        });
        const int32_t count = std::min<int32_t>(top_n, static_cast<int32_t>(ranked.size()));
        for (int32_t i = 0; i < count; ++i) {
            const Item& item = ranked[static_cast<std::size_t>(i)];
            ComparisonRow row;
            row.case_name = case_name;
            row.iteration = iteration;
            row.block_size = block_size;
            row.category = category;
            row.rank = i + 1;
            row.bus_id = item.bus;
            row.improvement_dx = item.dx_bj - item.dx_bilu;
            row.improvement_residual = item.res_bj - item.res_bilu;
            row.dx_error_bj_norm = item.dx_bj;
            row.dx_error_bilu_norm = item.dx_bilu;
            row.residual_error_bj_norm = item.res_bj;
            row.residual_error_bilu_norm = item.res_bilu;
            row.theta_abs_error_bj = item.theta_bj;
            row.theta_abs_error_bilu = item.theta_bilu;
            row.vmag_abs_error_bj = item.vmag_bj;
            row.vmag_abs_error_bilu = item.vmag_bilu;
            row.p_error_bj_abs = item.p_bj;
            row.p_error_bilu_abs = item.p_bilu;
            row.q_error_bj_abs = item.q_bj;
            row.q_error_bilu_abs = item.q_bilu;
            out.push_back(row);
        }
    };
    emit("improves_dx", [](const Item& item) { return item.dx_bj - item.dx_bilu; });
    emit("worsens_dx", [](const Item& item) { return item.dx_bilu - item.dx_bj; });
    emit("improves_residual", [](const Item& item) { return item.res_bj - item.res_bilu; });
    emit("worsens_residual", [](const Item& item) { return item.res_bilu - item.res_bj; });
}

void write_bus_errors(const std::filesystem::path& path, const std::vector<BusErrorRow>& rows)
{
    std::ofstream out(path);
    out << "case,iteration,block_size,solver,bus_id,theta_abs_error,theta_rel_error,"
           "vmag_abs_error,vmag_rel_error,combined_dx_error_norm,dx_cudss_theta,"
           "dx_iter_theta,dx_cudss_vmag,dx_iter_vmag\n";
    for (const BusErrorRow& row : rows) {
        out << row.case_name << ',' << row.iteration << ',' << row.block_size << ','
            << row.solver << ',' << row.bus_id << ','
            << format_double(row.theta_abs_error) << ','
            << format_double(row.theta_rel_error) << ','
            << format_double(row.vmag_abs_error) << ','
            << format_double(row.vmag_rel_error) << ','
            << format_double(row.combined_dx_error_norm) << ','
            << format_double(row.dx_cudss_theta) << ','
            << format_double(row.dx_iter_theta) << ','
            << format_double(row.dx_cudss_vmag) << ','
            << format_double(row.dx_iter_vmag) << '\n';
    }
}

void write_equation_errors(const std::filesystem::path& path,
                           const std::vector<EquationErrorRow>& rows)
{
    std::ofstream out(path);
    out << "case,iteration,block_size,solver,bus_id,P_error_abs,Q_error_abs,"
           "combined_PQ_error_norm\n";
    for (const EquationErrorRow& row : rows) {
        out << row.case_name << ',' << row.iteration << ',' << row.block_size << ','
            << row.solver << ',' << row.bus_id << ','
            << format_double(row.p_error_abs) << ','
            << format_double(row.q_error_abs) << ','
            << format_double(row.combined_pq_error_norm) << '\n';
    }
}

void write_block_errors(const std::filesystem::path& path,
                        const std::vector<BlockErrorRow>& rows)
{
    std::ofstream out(path);
    out << "case,iteration,block_size_target,solver,ordering,block_id,block_size,"
           "dx_error_norm,dx_cudss_norm,dx_error_ratio,residual_error_norm_on_block_rows\n";
    for (const BlockErrorRow& row : rows) {
        out << row.case_name << ',' << row.iteration << ',' << row.block_size_target << ','
            << row.solver << ',' << row.ordering << ',' << row.block_id << ','
            << row.block_size << ',' << format_double(row.dx_error_norm) << ','
            << format_double(row.dx_cudss_norm) << ','
            << format_double(row.dx_error_ratio) << ','
            << format_double(row.residual_error_norm_on_block_rows) << '\n';
    }
}

void write_comparisons(const std::filesystem::path& path,
                       const std::vector<ComparisonRow>& rows)
{
    std::ofstream out(path);
    out << "case,iteration,block_size,category,rank,bus_id,improvement_dx,"
           "improvement_residual,dx_error_bj_norm,dx_error_bilu_norm,"
           "residual_error_bj_norm,residual_error_bilu_norm,theta_abs_error_bj,"
           "theta_abs_error_bilu,vmag_abs_error_bj,vmag_abs_error_bilu,"
           "P_error_bj_abs,P_error_bilu_abs,Q_error_bj_abs,Q_error_bilu_abs\n";
    for (const ComparisonRow& row : rows) {
        out << row.case_name << ',' << row.iteration << ',' << row.block_size << ','
            << row.category << ',' << row.rank << ',' << row.bus_id << ','
            << format_double(row.improvement_dx) << ','
            << format_double(row.improvement_residual) << ','
            << format_double(row.dx_error_bj_norm) << ','
            << format_double(row.dx_error_bilu_norm) << ','
            << format_double(row.residual_error_bj_norm) << ','
            << format_double(row.residual_error_bilu_norm) << ','
            << format_double(row.theta_abs_error_bj) << ','
            << format_double(row.theta_abs_error_bilu) << ','
            << format_double(row.vmag_abs_error_bj) << ','
            << format_double(row.vmag_abs_error_bilu) << ','
            << format_double(row.p_error_bj_abs) << ','
            << format_double(row.p_error_bilu_abs) << ','
            << format_double(row.q_error_bj_abs) << ','
            << format_double(row.q_error_bilu_abs) << '\n';
    }
}

double mean_of(const std::vector<SummaryRow>& rows,
               const std::string& solver,
               double SummaryRow::*member)
{
    double sum = 0.0;
    int32_t count = 0;
    for (const SummaryRow& row : rows) {
        if (row.solver == solver && std::isfinite(row.*member)) {
            sum += row.*member;
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : kNan;
}

bool is_block_jacobi_solver(const std::string& solver)
{
    return solver.find("block_jacobi") != std::string::npos;
}

bool is_block_ilu0_solver(const std::string& solver)
{
    return solver.find("block_ilu0") != std::string::npos;
}

double mean_of_family(const std::vector<SummaryRow>& rows,
                      bool (*matches)(const std::string&),
                      double SummaryRow::*member)
{
    double sum = 0.0;
    int32_t count = 0;
    for (const SummaryRow& row : rows) {
        if (matches(row.solver) && std::isfinite(row.*member)) {
            sum += row.*member;
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : kNan;
}

std::vector<SummaryRow> sorted_by_metric(std::vector<SummaryRow> rows,
                                         const std::string& solver,
                                         double SummaryRow::*member)
{
    rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const SummaryRow& row) {
                   return row.solver != solver;
               }),
               rows.end());
    std::sort(rows.begin(), rows.end(), [&](const SummaryRow& lhs, const SummaryRow& rhs) {
        return lhs.*member > rhs.*member;
    });
    return rows;
}

std::vector<SummaryRow> sorted_by_family(std::vector<SummaryRow> rows,
                                         bool (*matches)(const std::string&),
                                         double SummaryRow::*member)
{
    rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const SummaryRow& row) {
                   return !matches(row.solver);
               }),
               rows.end());
    std::sort(rows.begin(), rows.end(), [&](const SummaryRow& lhs, const SummaryRow& rhs) {
        return lhs.*member > rhs.*member;
    });
    return rows;
}

void write_report(const std::filesystem::path& path,
                  const std::vector<SummaryRow>& summaries,
                  const std::vector<BlockErrorRow>& block_rows,
                  const std::vector<ComparisonRow>& comparison_rows)
{
    std::ofstream out(path);
    const double bj_dx =
        mean_of_family(summaries, is_block_jacobi_solver, &SummaryRow::dx_error_ratio);
    const double bilu_dx =
        mean_of_family(summaries, is_block_ilu0_solver, &SummaryRow::dx_error_ratio);
    const double bj_theta =
        mean_of_family(summaries, is_block_jacobi_solver, &SummaryRow::theta_error_ratio);
    const double bilu_theta =
        mean_of_family(summaries, is_block_ilu0_solver, &SummaryRow::theta_error_ratio);
    const double bj_vmag =
        mean_of_family(summaries, is_block_jacobi_solver, &SummaryRow::vmag_error_ratio);
    const double bilu_vmag =
        mean_of_family(summaries, is_block_ilu0_solver, &SummaryRow::vmag_error_ratio);
    const double bj_res =
        mean_of_family(summaries, is_block_jacobi_solver, &SummaryRow::residual_error_norm);
    const double bilu_res =
        mean_of_family(summaries, is_block_ilu0_solver, &SummaryRow::residual_error_norm);

    int32_t dx_better = 0;
    int32_t residual_better = 0;
    int32_t pairs = 0;
    for (const SummaryRow& bj : summaries) {
        if (!is_block_jacobi_solver(bj.solver)) {
            continue;
        }
        const auto it = std::find_if(summaries.begin(), summaries.end(), [&](const SummaryRow& row) {
            return is_block_ilu0_solver(row.solver) &&
                   row.case_name == bj.case_name &&
                   row.iteration == bj.iteration &&
                   row.block_size == bj.block_size;
        });
        if (it == summaries.end()) {
            continue;
        }
        ++pairs;
        dx_better += it->dx_error_norm < bj.dx_error_norm ? 1 : 0;
        residual_better += it->residual_error_norm < bj.residual_error_norm ? 1 : 0;
    }

    const std::vector<SummaryRow> largest_bilu_dx =
        sorted_by_family(summaries, is_block_ilu0_solver, &SummaryRow::dx_error_ratio);
    const std::vector<SummaryRow> largest_bilu_res =
        sorted_by_family(summaries, is_block_ilu0_solver, &SummaryRow::residual_error_norm);

    std::vector<BlockErrorRow> largest_blocks = block_rows;
    largest_blocks.erase(std::remove_if(largest_blocks.begin(),
                                        largest_blocks.end(),
                                        [](const BlockErrorRow& row) {
                                            return !is_block_ilu0_solver(row.solver);
                                        }),
                         largest_blocks.end());
    std::sort(largest_blocks.begin(), largest_blocks.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.dx_error_norm > rhs.dx_error_norm;
    });

    std::vector<ComparisonRow> top_improve_res;
    for (const ComparisonRow& row : comparison_rows) {
        if (row.category == "improves_residual" && row.rank <= 1) {
            top_improve_res.push_back(row);
        }
    }
    std::sort(top_improve_res.begin(), top_improve_res.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.improvement_residual > rhs.improvement_residual;
    });

    out << "# dx Difference Localization Diagnostic\n\n";
    out << "## Answers\n\n";
    out << "1. Block ILU closer to cuDSS than block-Jacobi: `"
        << dx_better << "/" << pairs << "` case/iteration/block-size pairs by total dx error. "
        << "Mean dx error ratio BJ=`" << format_double(bj_dx)
        << "`, BILU=`" << format_double(bilu_dx) << "`.\n";
    out << "2. Theta versus |V|: mean theta error ratio BJ=`" << format_double(bj_theta)
        << "`, BILU=`" << format_double(bilu_theta)
        << "`; mean |V| error ratio BJ=`" << format_double(bj_vmag)
        << "`, BILU=`" << format_double(bilu_vmag) << "`.\n";
    out << "3. Largest remaining dx errors under BILU:\n";
    for (std::size_t i = 0; i < std::min<std::size_t>(5, largest_bilu_dx.size()); ++i) {
        const SummaryRow& row = largest_bilu_dx[i];
        out << "   - `" << row.case_name << "` J" << row.iteration << " bs"
            << row.block_size << ": dx_error_ratio=`"
            << format_double(row.dx_error_ratio) << "`.\n";
    }
    out << "4. Largest block-local BILU dx errors:\n";
    for (std::size_t i = 0; i < std::min<std::size_t>(5, largest_blocks.size()); ++i) {
        const BlockErrorRow& row = largest_blocks[i];
        out << "   - `" << row.case_name << "` J" << row.iteration << " bs"
            << row.block_size_target << " block " << row.block_id
            << ": dx_error_norm=`" << format_double(row.dx_error_norm)
            << "`, residual_error_norm=`"
            << format_double(row.residual_error_norm_on_block_rows) << "`.\n";
    }
    out << "5. Equation-space residual error reduction: `"
        << residual_better << "/" << pairs << "` pairs improved. Mean residual-error norm BJ=`"
        << format_double(bj_res) << "`, BILU=`" << format_double(bilu_res) << "`.\n\n";

    out << "## Strongest equation-space improvements\n\n";
    for (std::size_t i = 0; i < std::min<std::size_t>(5, top_improve_res.size()); ++i) {
        const ComparisonRow& row = top_improve_res[i];
        out << "- `" << row.case_name << "` J" << row.iteration << " bs" << row.block_size
            << " bus " << row.bus_id << ": residual improvement=`"
            << format_double(row.improvement_residual) << "`, dx improvement=`"
            << format_double(row.improvement_dx) << "`.\n";
    }
    out << "\n## Notes\n\n";
    out << "- BJ uses the selected fixed-iteration Krylov solver with current unknown-level "
           "METIS block-Jacobi.\n";
    out << "- BILU uses the selected fixed-iteration Krylov solver with CPU pilot block ILU(0), "
           "block coloring order.\n";
    out << "- `residual_error = J * (dx_iter - dx_cudss)`; this localizes the equation-space "
           "difference from the direct Newton correction.\n";
    out << "- Bus IDs are internal dump bus indices.\n";
}

void analyze_one(const CliOptions& options,
                 const std::string& case_name,
                 int32_t iteration,
                 int32_t block_size,
                 const cuiter::CsrMatrix& matrix,
                 const std::vector<double>& rhs,
                 const LinearMetadata& metadata,
                 const std::vector<double>& dx_cudss,
                 std::vector<BusErrorRow>& bus_rows,
                 std::vector<EquationErrorRow>& equation_rows,
                 std::vector<BlockErrorRow>& block_rows,
                 std::vector<ComparisonRow>& comparison_rows,
                 std::vector<SummaryRow>& summaries)
{
    const SolverDx bj = solve_iterative(matrix, rhs, dx_cudss, block_size, false, options);
    const SolverDx bilu = solve_iterative(matrix, rhs, dx_cudss, block_size, true, options);
    const std::vector<SolverDx> solvers = {bj, bilu};

    const std::size_t bus_start = bus_rows.size();
    const std::size_t equation_start = equation_rows.size();
    for (const SolverDx& solver : solvers) {
        summaries.push_back(make_summary(case_name,
                                         iteration,
                                         block_size,
                                         solver,
                                         metadata,
                                         dx_cudss));
        for (int32_t bus = 0; bus < metadata.n_bus; ++bus) {
            bus_rows.push_back(make_bus_error_row(case_name,
                                                  iteration,
                                                  block_size,
                                                  solver,
                                                  metadata,
                                                  dx_cudss,
                                                  bus,
                                                  options.rel_eps));
            equation_rows.push_back(make_equation_error_row(case_name,
                                                            iteration,
                                                            block_size,
                                                            solver,
                                                            metadata,
                                                            bus));
        }
        const bool use_block_ilu0 = is_block_ilu0_solver(solver.solver);
        std::vector<BlockErrorRow> solver_blocks = make_block_error_rows(case_name,
                                                                         iteration,
                                                                         block_size,
                                                                         solver,
                                                                         matrix,
                                                                         dx_cudss,
                                                                         use_block_ilu0);
        block_rows.insert(block_rows.end(), solver_blocks.begin(), solver_blocks.end());
    }

    std::vector<BusErrorRow> new_bus_rows(bus_rows.begin() + static_cast<long>(bus_start),
                                          bus_rows.end());
    std::vector<EquationErrorRow> new_equation_rows(
        equation_rows.begin() + static_cast<long>(equation_start),
        equation_rows.end());
    add_top_comparisons(case_name,
                        iteration,
                        block_size,
                        metadata,
                        new_bus_rows,
                        new_equation_rows,
                        options.top_n,
                        comparison_rows);
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        std::filesystem::create_directories(options.output_dir);

        std::vector<BusErrorRow> bus_rows;
        std::vector<BlockErrorRow> block_rows;
        std::vector<EquationErrorRow> equation_rows;
        std::vector<ComparisonRow> comparison_rows;
        std::vector<SummaryRow> summaries;

        for (const std::string& case_name : options.cases) {
            const cupf_minimal::DumpCaseData case_data =
                cupf_minimal::load_dump_case(options.case_root / case_name);
            const LinearMetadata metadata = make_linear_metadata(case_data);
            for (int32_t iteration : options.iterations) {
                const auto j_path = jacobian_path(options.jf_root, case_name, iteration);
                const auto f_path = rhs_path(options.jf_root, case_name, iteration);
                if (!std::filesystem::exists(j_path) || !std::filesystem::exists(f_path)) {
                    if (options.allow_missing) {
                        std::cerr << "[skip] missing J/F for " << case_name << " J"
                                  << iteration << '\n';
                        continue;
                    }
                    throw std::runtime_error("missing J/F for " + case_name + " J" +
                                             std::to_string(iteration));
                }
                std::cout << "[case] " << case_name << " J" << iteration << '\n';
                const cuiter::CsrMatrix matrix = load_cupf_csr_dump(j_path);
                const std::vector<double> rhs = load_cupf_vector_dump(f_path);
                if (matrix.rows != static_cast<int32_t>(rhs.size()) ||
                    matrix.rows != static_cast<int32_t>(metadata.index_to_bus.size())) {
                    throw std::runtime_error("dimension mismatch for " + case_name);
                }
                const std::vector<double> dx_cudss = solve_cudss_dx(matrix, rhs);
                for (int32_t block_size : options.block_sizes) {
                    std::cout << "  [bs=" << block_size << "] " << options.iterative_solver
                              << " BJ vs block ILU0\n";
                    analyze_one(options,
                                case_name,
                                iteration,
                                block_size,
                                matrix,
                                rhs,
                                metadata,
                                dx_cudss,
                                bus_rows,
                                equation_rows,
                                block_rows,
                                comparison_rows,
                                summaries);
                }
            }
        }

        write_bus_errors(options.output_dir / "dx_diff_bus_errors.csv", bus_rows);
        write_block_errors(options.output_dir / "dx_diff_block_errors.csv", block_rows);
        write_equation_errors(options.output_dir / "dx_diff_equation_errors.csv", equation_rows);
        write_comparisons(options.output_dir / "dx_diff_bj_vs_bilu.csv", comparison_rows);
        write_report(options.output_dir / "dx_diff_report.md",
                     summaries,
                     block_rows,
                     comparison_rows);

        std::cout << "[done] wrote dx-difference localization diagnostics to "
                  << options.output_dir << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
